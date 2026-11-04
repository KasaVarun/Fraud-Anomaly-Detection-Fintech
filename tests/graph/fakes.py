"""Test-only graph store and Gremlin client doubles."""

from collections.abc import Iterator, Mapping
from typing import Any

from fraud_detection.persistence.graph.models import GraphEdge, GraphVertex


class FakeGraphStore:
    def __init__(self) -> None:
        self.vertices: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.fail_after: int | None = None
        self.calls = 0

    async def readiness(self) -> Any:
        raise NotImplementedError

    def validate_graph(self) -> None:
        return None

    def _maybe_fail(self) -> None:
        if self.fail_after is not None and self.calls >= self.fail_after:
            raise RuntimeError("controlled graph failure")
        self.calls += 1

    def upsert_vertex(self, vertex: Mapping[str, Any]) -> str:
        self._maybe_fail()
        expected = GraphVertex.model_validate(dict(vertex))
        existing = self.read_vertex(expected.id, expected.pk)
        if existing is None:
            self.vertices.append(expected.payload())
            return "created"
        stored = GraphVertex.model_validate(dict(existing))
        if stored.payload() != expected.payload():
            raise RuntimeError("controlled vertex conflict")
        return "existing"

    def upsert_edge(self, edge: Mapping[str, Any]) -> str:
        self._maybe_fail()
        expected = GraphEdge.model_validate(dict(edge))
        existing = self.read_edge(expected.id)
        if existing is None:
            self.edges.append(expected.payload())
            return "created"
        stored = GraphEdge.model_validate(dict(existing))
        if stored.payload() != expected.payload():
            raise RuntimeError("controlled edge conflict")
        return "existing"

    def read_vertex(self, vertex_id: str, pk: str) -> Mapping[str, Any] | None:
        return next(
            (
                vertex
                for vertex in self.vertices
                if vertex.get("id") == vertex_id and vertex.get("pk") == pk
            ),
            None,
        )

    def read_edge(self, edge_id: str) -> Mapping[str, Any] | None:
        return next((edge for edge in self.edges if edge.get("id") == edge_id), None)

    def iter_transaction_edges(
        self, batch_id: str, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (
            edge
            for edge in self.edges
            if edge.get("batch_id") == batch_id and edge.get("kind") == "money_flow"
        )

    def iter_batch_edges(self, batch_id: str, *, page_size: int) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (edge for edge in self.edges if edge.get("batch_id") == batch_id)

    def iter_vertices_by_ids(
        self,
        vertex_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        for start in range(0, len(vertex_ids), chunk_size):
            chunk = set(vertex_ids[start : start + chunk_size])
            yield from (vertex for vertex in self.vertices if vertex.get("id") in chunk)

    def iter_elements_by_transaction_ids(
        self,
        transaction_ids: list[str],
        *,
        chunk_size: int,
        page_size: int,
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        for start in range(0, len(transaction_ids), chunk_size):
            chunk = set(transaction_ids[start : start + chunk_size])
            yield from (
                {
                    "id": edge.get("id"),
                    "transaction_id": edge.get("transaction_id"),
                    "element_type": "edge",
                }
                for edge in self.edges
                if edge.get("transaction_id") in chunk
            )
            yield from (
                {
                    "id": vertex.get("id"),
                    "transaction_id": vertex.get("transaction_id"),
                    "element_type": "vertex",
                }
                for vertex in self.vertices
                if vertex.get("transaction_id") in chunk
            )

    def iter_evidence_before(
        self, cutoff_step: int, *, page_size: int
    ) -> Iterator[Mapping[str, Any]]:
        del page_size
        yield from (edge for edge in self.edges if int(edge.get("step", cutoff_step)) < cutoff_step)


class FakeFuture:
    def __init__(self, value: list[Any] | Exception) -> None:
        self.value = value

    def result(self, timeout: float) -> list[Any]:
        del timeout
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FakeResultSet:
    def __init__(self, value: list[Any] | Exception) -> None:
        self.value = value

    def all(self) -> FakeFuture:
        return FakeFuture(self.value)


class ScriptedGremlinClient:
    def __init__(self, responses: list[list[Any] | Exception]) -> None:
        self.responses = responses
        self.submissions: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    def submit(
        self,
        query: str,
        *,
        bindings: dict[str, Any],
        request_options: dict[str, Any],
    ) -> FakeResultSet:
        del request_options
        self.submissions.append((query, bindings))
        if not self.responses:
            raise AssertionError("unexpected Gremlin submission")
        return FakeResultSet(self.responses.pop(0))

    def close(self) -> None:
        self.closed = True
