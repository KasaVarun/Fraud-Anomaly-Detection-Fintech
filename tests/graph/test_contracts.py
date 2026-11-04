"""Stable graph identities, mappings, and leakage safeguards."""

from fraud_detection.persistence.graph.models import (
    FORBIDDEN_GRAPH_PROPERTIES,
    MONEY_EDGE_BY_TRANSACTION_TYPE,
    deterministic_artifact_id,
    partition_key,
)
from fraud_detection.persistence.graph.projection import build_graph_evidence
from tests.graph.helpers import raw_document


def test_partition_vertex_and_edge_ids_are_stable() -> None:
    document = raw_document()
    first = build_graph_evidence(document, shard_count=64)
    second = build_graph_evidence(document, shard_count=64)

    assert first == second
    assert partition_key("C1", 64) == partition_key("C1", 64)
    assert first.vertices[0].id == "C1"
    assert first.edges[0].id == "edge:TRANSFER_TO:tx-1"
    assert all(vertex.pk.startswith("shard-") for vertex in first.vertices)


def test_all_five_transaction_types_have_explicit_money_edge_mapping() -> None:
    assert MONEY_EDGE_BY_TRANSACTION_TYPE == {
        "CASH_IN": "TRANSFER_TO",
        "CASH_OUT": "TRANSFER_TO",
        "DEBIT": "TRANSFER_TO",
        "PAYMENT": "PAYS",
        "TRANSFER": "TRANSFER_TO",
    }
    for index, transaction_type in enumerate(MONEY_EDGE_BY_TRANSACTION_TYPE):
        evidence = build_graph_evidence(
            raw_document(
                transaction_id=f"tx-{index}",
                transaction_type=transaction_type,
                source_row=index,
            ),
            shard_count=64,
        )
        assert evidence.edges[0].label == MONEY_EDGE_BY_TRANSACTION_TYPE[transaction_type]
        assert evidence.edges[0].transaction_type == transaction_type


def test_merchant_account_labels_and_synthetic_provenance() -> None:
    evidence = build_graph_evidence(raw_document(sender_id="C1", receiver_id="M1"), shard_count=64)

    assert evidence.vertices[0].label == "account"
    assert evidence.vertices[1].label == "merchant"
    assert evidence.vertices[1].account_id == "M1"
    assert evidence.vertices[2].synthetic is True
    assert evidence.vertices[2].enrichment_seed == 73751
    assert evidence.edges[1].label == "USES_DEVICE"
    assert evidence.edges[2].label == "USES_IP"
    assert evidence.edges[1].transaction_id == "tx-1"


def test_synthetic_enrichment_is_label_and_balance_independent() -> None:
    clean = raw_document(is_fraud=False, source_flag=False, old_balance="100.00")
    changed_protected_values = raw_document(
        is_fraud=True,
        source_flag=True,
        old_balance="999999.00",
    )

    clean_evidence = build_graph_evidence(clean, shard_count=64)
    changed_evidence = build_graph_evidence(changed_protected_values, shard_count=64)

    assert clean_evidence == changed_evidence
    assert (
        deterministic_artifact_id("device", transaction_id="tx-1", source_row=1, step=1, seed=73751)
        == clean_evidence.vertices[2].id
    )
    graph_keys = {
        key
        for item in [*clean_evidence.vertices, *clean_evidence.edges]
        for key in item.model_dump()
    }
    assert not (FORBIDDEN_GRAPH_PROPERTIES & graph_keys)
