"""Graph test evidence builders."""

from datetime import UTC, datetime

from fraud_detection.persistence.document_vector.models import (
    FORBIDDEN_RAW_PREDICTIVE_FIELDS,
    RawTransactionDocument,
    content_digest,
)


def raw_document(
    *,
    transaction_id: str = "tx-1",
    sender_id: str = "C1",
    receiver_id: str = "C2",
    transaction_type: str = "TRANSFER",
    step: int = 1,
    source_row: int = 1,
    is_fraud: bool = False,
    source_flag: bool = False,
    old_balance: str = "100.00",
) -> RawTransactionDocument:
    payload: dict[str, object] = {
        "id": transaction_id,
        "sender_id": sender_id,
        "receiver_id": receiver_id,
        "batch_id": "batch-1",
        "source_row": source_row,
        "step": step,
        "data_split": "train",
        "transaction_type": transaction_type,
        "amount": "10.00",
        "forensic_balances": {
            "oldbalanceOrg": old_balance,
            "newbalanceOrig": "90.00",
            "oldbalanceDest": "0.00",
            "newbalanceDest": "10.00",
            "predictive_use": "forbidden",
        },
        "source_truth": {
            "is_fraud": is_fraud,
            "source_flag": source_flag,
            "provenance": "PaySim:ealaxi/paysim1",
            "predictive_use": "forbidden",
        },
        "source_provenance": {
            "source_filename": "paysim.csv",
            "source_sha256": "a" * 64,
            "sampling_seed": 73750,
            "enrichment_seed": 73751,
        },
        "forbidden_predictive_fields": FORBIDDEN_RAW_PREDICTIVE_FIELDS,
        "document_schema_version": "1.0.0",
        "projection_version": "raw-paysim-v1",
        "projected_at": datetime(2026, 10, 31, tzinfo=UTC),
    }
    payload["content_digest"] = content_digest(payload)
    return RawTransactionDocument.model_validate(payload)
