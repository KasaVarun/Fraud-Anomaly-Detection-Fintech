"""Leakage-safe historical NetworkX graph analysis and snapshot persistence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from decimal import Decimal
from itertools import combinations
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection, Engine, create_engine, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from fraud_detection.ingestion.hashing import canonical_json_bytes
from fraud_detection.persistence.graph.errors import GremlinEvidenceConflictError
from fraud_detection.persistence.graph.models import GRAPH_PROJECTION_VERSION, GraphEdge
from fraud_detection.persistence.relational import Base

TRANSACTION_WEIGHT = 1.0
SHARED_DEVICE_WEIGHT = 0.5
SHARED_IP_WEIGHT = 0.25
GRAPH_ANALYSIS_VERSION = "account-evidence-v1"

graph_snapshot = Base.metadata.tables["graph_snapshot"]
community_table = Base.metadata.tables["community"]
risk_profile = Base.metadata.tables["account_risk_profile"]


class AccountProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    account_id: str
    community_id: int
    degree_centrality: float
    weighted_degree: float


class HistoricalAnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    batch_id: str
    cutoff_step: int
    algorithm_version: str
    node_count: int
    edge_count: int
    community_count: int
    profiles: list[AccountProfile]


def algorithm_version(*, louvain_seed: int, resolution: float) -> str:
    return (
        f"{GRAPH_ANALYSIS_VERSION};projection={GRAPH_PROJECTION_VERSION};"
        f"strict_step_lt=true;transaction_weight={TRANSACTION_WEIGHT};"
        f"shared_device_weight={SHARED_DEVICE_WEIGHT};shared_ip_weight={SHARED_IP_WEIGHT};"
        f"louvain_seed={louvain_seed};resolution={resolution:g};"
        "community_numbering=min-node-lexicographic-v1"
    )


def deterministic_snapshot_id(
    batch_id: str,
    cutoff_step: int,
    analysis_version: str,
) -> str:
    import hashlib

    payload = {
        "batch_id": batch_id,
        "cutoff_step": cutoff_step,
        "graph_projection_version": GRAPH_PROJECTION_VERSION,
        "algorithm_version": analysis_version,
    }
    return "snapshot:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_historical_account_graph(
    evidence: Iterable[GraphEdge],
    *,
    cutoff_step: int,
) -> nx.Graph[str]:
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: dict[tuple[str, str], set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for edge in evidence:
        if edge.step >= cutoff_step:
            continue
        if edge.kind == "money_flow":
            nodes.update([edge.out_id, edge.in_id])
            if edge.out_id == edge.in_id:
                continue
            pair = tuple(sorted((edge.out_id, edge.in_id)))
            assert len(pair) == 2
            typed_pair = (pair[0], pair[1])
            stats = pair_stats.setdefault(
                typed_pair,
                {"transaction_count": 0, "devices": set(), "ips": set()},
            )
            stats["transaction_count"] += 1
        elif edge.kind == "device_observation":
            nodes.add(edge.out_id)
            artifacts[("device", edge.in_id)].add(edge.out_id)
        elif edge.kind == "ip_observation":
            nodes.add(edge.out_id)
            artifacts[("ip", edge.in_id)].add(edge.out_id)

    for (artifact_kind, artifact_id), accounts in sorted(artifacts.items()):
        for left, right in combinations(sorted(accounts), 2):
            if left == right:
                continue
            pair = (left, right)
            stats = pair_stats.setdefault(
                pair,
                {"transaction_count": 0, "devices": set(), "ips": set()},
            )
            target = "devices" if artifact_kind == "device" else "ips"
            stats[target].add(artifact_id)

    graph: nx.Graph[str] = nx.Graph()
    graph.add_nodes_from(sorted(nodes))
    for (left, right), stats in sorted(pair_stats.items()):
        transaction_count = int(stats["transaction_count"])
        device_count = len(stats["devices"])
        ip_count = len(stats["ips"])
        weight = (
            TRANSACTION_WEIGHT * transaction_count
            + SHARED_DEVICE_WEIGHT * device_count
            + SHARED_IP_WEIGHT * ip_count
        )
        graph.add_edge(
            left,
            right,
            transaction_count=transaction_count,
            shared_device_count=device_count,
            shared_ip_count=ip_count,
            weight=weight,
        )
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def analyze_historical_evidence(
    evidence: Iterable[GraphEdge],
    *,
    batch_id: str,
    cutoff_step: int,
    louvain_seed: int,
    resolution: float,
) -> HistoricalAnalysisResult:
    version = algorithm_version(louvain_seed=louvain_seed, resolution=resolution)
    graph = build_historical_account_graph(evidence, cutoff_step=cutoff_step)
    node_count = graph.number_of_nodes()
    if graph.number_of_edges():
        communities = nx.community.louvain_communities(
            graph,
            weight="weight",
            resolution=resolution,
            seed=louvain_seed,
        )
    else:
        communities = [{node} for node in sorted(graph.nodes)]
    ordered_communities = sorted(
        (sorted(community) for community in communities),
        key=lambda community: community[0] if community else "",
    )
    membership = {
        account_id: community_id
        for community_id, community in enumerate(ordered_communities, start=1)
        for account_id in community
    }
    weighted_degrees = dict(graph.degree(weight="weight"))
    profiles = [
        AccountProfile(
            account_id=account_id,
            community_id=membership[account_id],
            degree_centrality=(
                graph.degree(account_id) / (node_count - 1) if node_count > 1 else 0
            ),
            weighted_degree=float(weighted_degrees[account_id]),
        )
        for account_id in sorted(graph.nodes)
    ]
    return HistoricalAnalysisResult(
        snapshot_id=deterministic_snapshot_id(batch_id, cutoff_step, version),
        batch_id=batch_id,
        cutoff_step=cutoff_step,
        algorithm_version=version,
        node_count=node_count,
        edge_count=graph.number_of_edges(),
        community_count=len(ordered_communities),
        profiles=profiles,
    )


def _ensure_snapshot(connection: Connection, result: HistoricalAnalysisResult) -> None:
    record = {
        "snapshot_id": result.snapshot_id,
        "batch_id": result.batch_id,
        "cutoff_step": result.cutoff_step,
        "algorithm_version": result.algorithm_version,
        "computed_at": datetime.now(UTC),
    }
    connection.execute(
        postgres_insert(graph_snapshot)
        .values(record)
        .on_conflict_do_nothing(index_elements=["snapshot_id"])
    )
    stored = (
        connection.execute(
            select(graph_snapshot).where(graph_snapshot.c.snapshot_id == result.snapshot_id)
        )
        .mappings()
        .one()
    )
    for field in ("batch_id", "cutoff_step", "algorithm_version"):
        if stored[field] != record[field]:
            raise GremlinEvidenceConflictError(
                f"snapshot {result.snapshot_id} conflicts on {field}"
            )


def persist_historical_analysis(
    database_url: str,
    result: HistoricalAnalysisResult,
) -> HistoricalAnalysisResult:
    engine: Engine = create_engine(database_url, hide_parameters=True)
    try:
        with engine.begin() as connection:
            _ensure_snapshot(connection, result)
            for community_id in range(1, result.community_count + 1):
                connection.execute(
                    postgres_insert(community_table)
                    .values(snapshot_id=result.snapshot_id, community_id=community_id)
                    .on_conflict_do_nothing(index_elements=["snapshot_id", "community_id"])
                )
            for profile in result.profiles:
                record = {
                    "account_id": profile.account_id,
                    "snapshot_id": result.snapshot_id,
                    "community_id": profile.community_id,
                    "degree_centrality": profile.degree_centrality,
                    "weighted_degree": Decimal(str(profile.weighted_degree)),
                }
                connection.execute(
                    postgres_insert(risk_profile)
                    .values(record)
                    .on_conflict_do_nothing(index_elements=["account_id", "snapshot_id"])
                )
                stored = (
                    connection.execute(
                        select(risk_profile).where(
                            risk_profile.c.account_id == profile.account_id,
                            risk_profile.c.snapshot_id == result.snapshot_id,
                        )
                    )
                    .mappings()
                    .one()
                )
                for field in ("community_id", "degree_centrality", "weighted_degree"):
                    if Decimal(str(stored[field])) != Decimal(str(record[field])):
                        raise GremlinEvidenceConflictError(
                            f"snapshot profile {profile.account_id} conflicts on {field}"
                        )
        return result
    finally:
        engine.dispose()
