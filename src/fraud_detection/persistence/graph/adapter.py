"""Production Azure Cosmos DB for Gremlin evidence adapter."""

import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, TypeVar, cast

from gremlin_python.driver.client import Client
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.driver.serializer import GraphSONSerializersV2d0
from pydantic import ValidationError

from fraud_detection.config import Settings
from fraud_detection.persistence.graph.errors import (
    GremlinAdapterError,
    GremlinConfigurationError,
    GremlinEvidenceConflictError,
)
from fraud_detection.persistence.graph.models import GraphEdge, GraphVertex
from fraud_detection.ports.health import DependencyHealth, DependencyState

T = TypeVar("T")
TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class CosmosGremlinAdapter:
    """Bounded parameterized access to the approved evidence graph."""

    def __init__(self, settings: Settings, *, client: Any | None = None) -> None:
        if not settings.graph_configured:
            raise GremlinConfigurationError("Cosmos Gremlin settings are incomplete")
        self._settings = settings
        if client is None:
            endpoint = cast(Any, settings.gremlin_endpoint).get_secret_value()
            credential = cast(Any, settings.gremlin_key).get_secret_value()
            client = Client(
                endpoint,
                "g",
                username=cast(str, settings.gremlin_username),
                password=credential,
                message_serializer=GraphSONSerializersV2d0(),
                pool_size=1,
            )
        self._client = client

    @staticmethod
    def _status_code(error: GremlinServerError) -> int | None:
        status = getattr(error, "status", None)
        if isinstance(status, Mapping):
            code = status.get("code")
            return int(code) if code is not None else None
        code = getattr(error, "status_code", None)
        return int(code) if code is not None else None

    def _call(self, operation: Callable[[], T], operation_name: str) -> T:
        attempts = self._settings.gremlin_max_retries + 1
        for attempt in range(attempts):
            try:
                return operation()
            except GremlinServerError as exc:
                status = self._status_code(exc)
                if status not in TRANSIENT_STATUS_CODES or attempt + 1 == attempts:
                    raise GremlinAdapterError(
                        f"Gremlin {operation_name} failed with status {status or 'unknown'}"
                    ) from exc
                time.sleep(min(0.25 * (2**attempt), 2.0))
            except (FutureTimeoutError, OSError, TimeoutError) as exc:
                if attempt + 1 == attempts:
                    raise GremlinAdapterError(f"Gremlin {operation_name} failed") from exc
                time.sleep(min(0.25 * (2**attempt), 2.0))
        raise GremlinAdapterError(f"Gremlin {operation_name} exhausted retries")

    def _submit(self, query: str, bindings: dict[str, Any], operation_name: str) -> list[Any]:
        def execute() -> list[Any]:
            result_set = self._client.submit(
                query,
                bindings=bindings,
                request_options={
                    "evaluationTimeout": int(self._settings.gremlin_request_timeout_seconds * 1_000)
                },
            )
            return cast(
                list[Any],
                result_set.all().result(timeout=self._settings.gremlin_request_timeout_seconds),
            )

        return self._call(execute, operation_name)

    @staticmethod
    def _flatten_properties(properties: Mapping[Any, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key, value in properties.items():
            normalized = value[0] if isinstance(value, list) and len(value) == 1 else value
            flattened[str(key)] = normalized
        return flattened

    @classmethod
    def _normalize(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise GremlinAdapterError("Gremlin returned an unsupported result shape")
        result = dict(value)
        properties = result.pop("properties", {})
        if isinstance(properties, Mapping):
            result.update(cls._flatten_properties(properties))
        return {str(key): item for key, item in result.items()}

    def validate_graph(self) -> None:
        self._submit("g.V().limit(1).count()", {}, "graph validation")

    async def readiness(self) -> DependencyHealth:
        try:
            self.validate_graph()
        except GremlinAdapterError:
            return DependencyHealth(
                name="cosmos_gremlin",
                state=DependencyState.UNAVAILABLE,
                required=True,
                detail="configured Cosmos Gremlin graph is unavailable",
            )
        return DependencyHealth(
            name="cosmos_gremlin",
            state=DependencyState.HEALTHY,
            required=True,
            detail="bounded Gremlin traversal succeeded",
        )

    def read_vertex(self, vertex_id: str, pk: str) -> Mapping[str, Any] | None:
        query = (
            "g.V().hasId(vertex_id).has('pk',pk)"
            ".project('id','label','properties')"
            ".by(id()).by(label()).by(valueMap())"
        )
        values = self._submit(query, {"vertex_id": vertex_id, "pk": pk}, "vertex point read")
        return self._normalize(values[0]) if values else None

    def read_edge(self, edge_id: str) -> Mapping[str, Any] | None:
        query = (
            "g.E().hasId(edge_id).project('id','label','properties')"
            ".by(id()).by(label()).by(valueMap())"
        )
        values = self._submit(query, {"edge_id": edge_id}, "edge point read")
        return self._normalize(values[0]) if values else None

    @staticmethod
    def _property_chain(
        base: str, payload: Mapping[str, Any], excluded: set[str]
    ) -> tuple[str, dict[str, Any]]:
        query = base
        bindings: dict[str, Any] = {}
        index = 0
        for key in sorted(set(payload) - excluded):
            value = payload[key]
            if value is None:
                continue
            binding = f"property_{index}"
            query += f".property('{key}',{binding})"
            bindings[binding] = value
            index += 1
        return query, bindings

    def upsert_vertex(self, vertex: Mapping[str, Any]) -> str:
        try:
            expected = GraphVertex.model_validate(dict(vertex))
        except ValidationError as exc:
            raise GremlinAdapterError("graph vertex contract validation failed") from exc
        existing = self.read_vertex(expected.id, expected.pk)
        if existing is not None:
            if existing.get("content_digest") == expected.content_digest:
                return "existing"
            raise GremlinEvidenceConflictError(
                f"vertex {expected.id} has conflicting immutable evidence"
            )
        base = "g.addV(vertex_label).property('id',vertex_id).property('pk',vertex_pk)"
        query, properties = self._property_chain(base, expected.payload(), {"id", "pk", "label"})
        bindings = {
            "vertex_label": expected.label,
            "vertex_id": expected.id,
            "vertex_pk": expected.pk,
            **properties,
        }
        try:
            self._submit(query, bindings, "vertex create")
        except GremlinAdapterError:
            concurrent = self.read_vertex(expected.id, expected.pk)
            if (
                concurrent is not None
                and concurrent.get("content_digest") == expected.content_digest
            ):
                return "existing"
            raise
        return "created"

    def upsert_edge(self, edge: Mapping[str, Any]) -> str:
        try:
            expected = GraphEdge.model_validate(dict(edge))
        except ValidationError as exc:
            raise GremlinAdapterError("graph edge contract validation failed") from exc
        existing = self.read_edge(expected.id)
        if existing is not None:
            if existing.get("content_digest") == expected.content_digest:
                return "existing"
            raise GremlinEvidenceConflictError(
                f"edge {expected.id} has conflicting immutable evidence"
            )
        base = (
            "g.V().hasId(out_id).has('pk',out_pk).as('source')"
            ".V().hasId(in_id).has('pk',in_pk)"
            ".addE(edge_label).from('source').property('id',edge_id)"
        )
        query, properties = self._property_chain(base, expected.payload(), {"id", "label"})
        bindings = {
            "out_id": expected.out_id,
            "out_pk": expected.out_pk,
            "in_id": expected.in_id,
            "in_pk": expected.in_pk,
            "edge_label": expected.label,
            "edge_id": expected.id,
            **properties,
        }
        try:
            self._submit(query, bindings, "edge create")
        except GremlinAdapterError:
            concurrent = self.read_edge(expected.id)
            if (
                concurrent is not None
                and concurrent.get("content_digest") == expected.content_digest
            ):
                return "existing"
            raise
        return "created"

    def _iter_pages(
        self,
        traversal: str,
        bindings: dict[str, Any],
        *,
        page_size: int,
        operation_name: str,
    ) -> Iterator[Mapping[str, Any]]:
        offset = 0
        while True:
            page_bindings = {**bindings, "page_start": offset, "page_end": offset + page_size}
            values = self._submit(
                traversal + ".range(page_start,page_end)",
                page_bindings,
                operation_name,
            )
            for value in values:
                yield self._normalize(value)
            if len(values) < page_size:
                return
            offset += page_size

    @staticmethod
    def _edge_projection(base: str) -> str:
        return (
            base + ".order().by(id()).project('id','label','properties')"
            ".by(id()).by(label()).by(valueMap())"
        )

    def iter_transaction_edges(
        self, batch_id: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        traversal = self._edge_projection("g.E().has('kind','money_flow').has('batch_id',batch_id)")
        return self._iter_pages(
            traversal,
            {"batch_id": batch_id},
            page_size=page_size,
            operation_name="transaction edge enumeration",
        )

    def iter_batch_edges(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        traversal = self._edge_projection("g.E().has('batch_id',batch_id)")
        return self._iter_pages(
            traversal,
            {"batch_id": batch_id},
            page_size=page_size,
            operation_name="batch edge enumeration",
        )

    def iter_vertices_by_ids(
        self,
        vertex_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        if chunk_size <= 0:
            raise ValueError("Gremlin vertex-ID chunk size must be positive")
        unique_ids = list(dict.fromkeys(vertex_ids))
        for start in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[start : start + chunk_size]
            traversal = (
                "g.V().hasId(within(vertex_ids)).order().by(id())"
                ".project('id','label','properties')"
                ".by(id()).by(label()).by(valueMap())"
            )
            yield from self._iter_pages(
                traversal,
                {"vertex_ids": chunk},
                page_size=page_size,
                operation_name="vertex enumeration",
            )

    def iter_elements_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        if chunk_size <= 0:
            raise ValueError("Gremlin transaction-ID chunk size must be positive")
        unique_ids = list(dict.fromkeys(transaction_ids))
        for start in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[start : start + chunk_size]
            for element_type, traversal in [
                (
                    "vertex",
                    "g.V().has('transaction_id',within(transaction_ids))"
                    ".order().by(id()).project('id','transaction_id')"
                    ".by(id()).by(values('transaction_id'))",
                ),
                (
                    "edge",
                    "g.E().has('transaction_id',within(transaction_ids))"
                    ".order().by(id()).project('id','transaction_id')"
                    ".by(id()).by(values('transaction_id'))",
                ),
            ]:
                for item in self._iter_pages(
                    traversal,
                    {"transaction_ids": chunk},
                    page_size=page_size,
                    operation_name="transaction-ID evidence lookup",
                ):
                    yield {**item, "element_type": element_type}

    def iter_evidence_before(
        self, cutoff_step: int, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        traversal = self._edge_projection("g.E().has('step',lt(cutoff_step))")
        return self._iter_pages(
            traversal,
            {"cutoff_step": cutoff_step},
            page_size=page_size,
            operation_name="historical evidence enumeration",
        )

    def transaction_evidence(self, transaction_id: str, *, limit: int) -> list[Mapping[str, Any]]:
        traversal = (
            "g.E().has('transaction_id',transaction_id).order().by(id()).limit(result_limit)"
            ".project('id','label','out_id','out_pk','in_id','in_pk','properties')"
            ".by(id()).by(label()).by(outV().id()).by(outV().values('pk'))"
            ".by(inV().id()).by(inV().values('pk')).by(valueMap())"
        )
        return [
            self._normalize(value)
            for value in self._submit(
                traversal,
                {"transaction_id": transaction_id, "result_limit": limit},
                "transaction investigation evidence",
            )
        ]

    def related_accounts(
        self,
        account_id: str,
        *,
        relationship_type: str,
        cutoff_step: int,
        minimum_weight: float,
        offset: int,
        limit: int,
    ) -> list[Mapping[str, Any]]:
        traversal = (
            "g.V(account_id).union("
            "bothE().has('kind','money_flow').has('step',lt(cutoff_step)).as('e')"
            ".otherV().project('related_account_id','evidence_type','artifact_id')"
            ".by(id()).by(constant('money_flow')).by(select('e').id()),"
            "outE('USES_DEVICE').has('step',lt(cutoff_step)).inV().as('artifact')"
            ".inE('USES_DEVICE').has('step',lt(cutoff_step)).outV()"
            ".project('related_account_id','evidence_type','artifact_id')"
            ".by(id()).by(constant('shared_device')).by(select('artifact').id()),"
            "outE('USES_IP').has('step',lt(cutoff_step)).inV().as('artifact')"
            ".inE('USES_IP').has('step',lt(cutoff_step)).outV()"
            ".project('related_account_id','evidence_type','artifact_id')"
            ".by(id()).by(constant('shared_ip')).by(select('artifact').id()))"
            ".limit(scan_limit)"
        )
        rows = [
            self._normalize(value)
            for value in self._submit(
                traversal,
                {
                    "account_id": account_id,
                    "cutoff_step": cutoff_step,
                    "scan_limit": min(max((offset + limit) * 100, 100), 5_000),
                },
                "bounded related-account traversal",
            )
        ]
        evidence: dict[str, dict[str, set[str]]] = {}
        for row in rows:
            related = str(row["related_account_id"])
            if related == account_id:
                continue
            kind = str(row["evidence_type"])
            if relationship_type != "all" and kind != relationship_type:
                continue
            values = evidence.setdefault(
                related,
                {"money_flow": set(), "shared_device": set(), "shared_ip": set()},
            )
            values[kind].add(str(row["artifact_id"]))
        results: list[Mapping[str, Any]] = []
        for related, values in sorted(evidence.items()):
            transactions = len(values["money_flow"])
            devices = len(values["shared_device"])
            ips = len(values["shared_ip"])
            weight = transactions + 0.5 * devices + 0.25 * ips
            if weight < minimum_weight:
                continue
            results.append(
                {
                    "related_account_id": related,
                    "relationship_type": relationship_type,
                    "transaction_count": transactions,
                    "shared_device_count": devices,
                    "shared_ip_count": ips,
                    "evidence_weight": weight,
                }
            )
        return results[offset : offset + limit]

    def close(self) -> None:
        self._client.close()
