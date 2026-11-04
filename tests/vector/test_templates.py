"""Deterministic strict prior-step behavioral template tests."""

from fraud_detection.features.behavioral import (
    AMOUNT_BAND_VERSION,
    BEHAVIORAL_TEMPLATE_VERSION,
    amount_band,
    build_behavioral_templates,
)
from tests.graph.helpers import raw_document


def test_fixed_amount_bands_are_versioned_and_deterministic() -> None:
    assert AMOUNT_BAND_VERSION == "fixed-paysim-amount-bands-v1"
    assert [
        amount_band(value)
        for value in ["0", "0.01", "10", "10.01", "100", "1000", "10000", "10000.01"]
    ] == [
        "zero",
        "gt0_le10",
        "gt0_le10",
        "gt10_le100",
        "gt10_le100",
        "gt100_le1000",
        "gt1000_le10000",
        "gt10000",
    ]


def test_same_step_isolation_strict_history_and_input_order_independence() -> None:
    documents = [
        raw_document(transaction_id="tx-a", sender_id="C1", receiver_id="C2", step=1, source_row=1),
        raw_document(transaction_id="tx-b", sender_id="C1", receiver_id="C3", step=1, source_row=2),
        raw_document(transaction_id="tx-c", sender_id="C1", receiver_id="C4", step=2, source_row=3),
        raw_document(transaction_id="tx-d", sender_id="C5", receiver_id="C2", step=2, source_row=4),
    ]
    first = build_behavioral_templates(documents)
    second = build_behavioral_templates(list(reversed(documents)))

    assert first == second
    by_id = {template.transaction_id: template for template in first}
    assert by_id["tx-a"].prior_sender_count == 0
    assert by_id["tx-b"].prior_sender_count == 0
    assert by_id["tx-c"].prior_sender_count == 2
    assert by_id["tx-c"].prior_sender_type_count == 2
    assert by_id["tx-c"].prior_sender_distinct_counterparties == 2
    assert by_id["tx-d"].prior_receiver_count == 1
    assert all(template.template_version == BEHAVIORAL_TEMPLATE_VERSION for template in first)


def test_protected_values_do_not_change_template_text_or_digest() -> None:
    clean = raw_document(
        transaction_id="tx-1",
        sender_id="C-SENDER",
        receiver_id="M-RECEIVER",
        is_fraud=False,
        source_flag=False,
        old_balance="100.00",
    )
    changed = raw_document(
        transaction_id="tx-1",
        sender_id="C-SENDER",
        receiver_id="M-RECEIVER",
        is_fraud=True,
        source_flag=True,
        old_balance="999999.00",
    )

    first = build_behavioral_templates([clean])[0]
    second = build_behavioral_templates([changed])[0]

    assert first.text == second.text
    assert first.template_digest == second.template_digest
    assert first.reference_eligible is False
    assert second.reference_eligible is True
    assert "C-SENDER" not in first.text
    assert "M-RECEIVER" not in first.text
    assert "isFraud" not in first.text
    assert "balance" not in first.text.lower()
