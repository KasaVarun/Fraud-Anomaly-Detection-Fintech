"""Strict historical graph analysis tests."""

from fraud_detection.persistence.graph.analysis import (
    analyze_historical_evidence,
    build_historical_account_graph,
)
from fraud_detection.persistence.graph.models import GraphEdge, graph_digest
from fraud_detection.persistence.graph.projection import build_graph_evidence
from tests.graph.helpers import raw_document


def evidence_edge(
    *,
    kind: str,
    label: str,
    out_id: str,
    in_id: str,
    transaction_id: str,
    step: int,
) -> GraphEdge:
    bundle = build_graph_evidence(
        raw_document(transaction_id=transaction_id, step=step, source_row=step),
        shard_count=64,
    )
    template = next(edge for edge in bundle.edges if edge.kind == kind)
    payload = template.payload()
    payload.update(
        {
            "id": f"edge:{label}:{transaction_id}",
            "label": label,
            "out_id": out_id,
            "in_id": in_id,
            "transaction_id": transaction_id,
            "step": step,
            "kind": kind,
            "amount": "10.00" if kind == "money_flow" else None,
            "synthetic": kind != "money_flow",
        }
    )
    payload["content_digest"] = graph_digest(payload)
    return GraphEdge.model_validate(payload)


def historical_evidence() -> list[GraphEdge]:
    return [
        evidence_edge(
            kind="money_flow",
            label="TRANSFER_TO",
            out_id="C1",
            in_id="C2",
            transaction_id="tx-1",
            step=1,
        ),
        evidence_edge(
            kind="money_flow",
            label="TRANSFER_TO",
            out_id="C1",
            in_id="C2",
            transaction_id="tx-2",
            step=2,
        ),
        evidence_edge(
            kind="money_flow",
            label="TRANSFER_TO",
            out_id="C1",
            in_id="C1",
            transaction_id="tx-self",
            step=1,
        ),
        evidence_edge(
            kind="money_flow",
            label="TRANSFER_TO",
            out_id="C2",
            in_id="C3",
            transaction_id="tx-future",
            step=3,
        ),
        evidence_edge(
            kind="device_observation",
            label="USES_DEVICE",
            out_id="C1",
            in_id="device-shared",
            transaction_id="tx-d1",
            step=1,
        ),
        evidence_edge(
            kind="device_observation",
            label="USES_DEVICE",
            out_id="C3",
            in_id="device-shared",
            transaction_id="tx-d2",
            step=2,
        ),
        evidence_edge(
            kind="ip_observation",
            label="USES_IP",
            out_id="C1",
            in_id="ip-shared",
            transaction_id="tx-i1",
            step=1,
        ),
        evidence_edge(
            kind="ip_observation",
            label="USES_IP",
            out_id="C3",
            in_id="ip-shared",
            transaction_id="tx-i2",
            step=2,
        ),
    ]


def test_strict_cutoff_self_loop_removal_pair_collapsing_and_weights() -> None:
    graph = build_historical_account_graph(historical_evidence(), cutoff_step=3)

    assert set(graph.nodes) == {"C1", "C2", "C3"}
    assert not graph.has_edge("C2", "C3")
    assert not graph.has_edge("C1", "C1")
    assert graph.number_of_edges() == 2
    assert graph["C1"]["C2"] == {
        "transaction_count": 2,
        "shared_device_count": 0,
        "shared_ip_count": 0,
        "weight": 2.0,
    }
    assert graph["C1"]["C3"] == {
        "transaction_count": 0,
        "shared_device_count": 1,
        "shared_ip_count": 1,
        "weight": 0.75,
    }


def test_centrality_louvain_and_community_ids_are_reproducible() -> None:
    first = analyze_historical_evidence(
        historical_evidence(),
        batch_id="batch-1",
        cutoff_step=3,
        louvain_seed=73751,
        resolution=1.0,
    )
    second = analyze_historical_evidence(
        list(reversed(historical_evidence())),
        batch_id="batch-1",
        cutoff_step=3,
        louvain_seed=73751,
        resolution=1.0,
    )

    assert first == second
    profiles = {profile.account_id: profile for profile in first.profiles}
    assert profiles["C1"].degree_centrality == 1.0
    assert profiles["C2"].degree_centrality == 0.5
    assert profiles["C3"].degree_centrality == 0.5
    assert profiles["C1"].weighted_degree == 2.75
    assert [profile.community_id for profile in first.profiles] == sorted(
        profile.community_id for profile in first.profiles
    )
    assert "strict_step_lt=true" in first.algorithm_version
