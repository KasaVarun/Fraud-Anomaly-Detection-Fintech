"""Deterministic leakage-safe behavioral template generation."""

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Iterator
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from fraud_detection.ingestion.hashing import canonical_json_bytes
from fraud_detection.persistence.document_vector.models import RawTransactionDocument

BEHAVIORAL_TEMPLATE_VERSION = "behavior-template-v1"
AMOUNT_BAND_VERSION = "fixed-paysim-amount-bands-v1"
AMOUNT_THRESHOLDS = (
    Decimal("0"),
    Decimal("10"),
    Decimal("100"),
    Decimal("1000"),
    Decimal("10000"),
)
FORBIDDEN_TEMPLATE_TOKENS = (
    "isfraud",
    "isflaggedfraud",
    "source_flag",
    "source_truth",
    "oldbalance",
    "newbalance",
    "fraud_probability",
    "model_version",
    "future",
)


class BehavioralTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    sender_id: str
    batch_id: str
    step: int
    data_split: str
    raw_id: str
    raw_pk: str
    transaction_type: str
    amount_band: str
    prior_sender_count: int
    prior_receiver_count: int
    prior_sender_type_count: int
    prior_sender_distinct_counterparties: int
    text: str
    template_version: str
    amount_band_version: str
    template_digest: str
    reference_eligible: bool
    source_version: str


def amount_band(amount: str | Decimal) -> str:
    value = Decimal(str(amount))
    if value == AMOUNT_THRESHOLDS[0]:
        return "zero"
    if value <= AMOUNT_THRESHOLDS[1]:
        return "gt0_le10"
    if value <= AMOUNT_THRESHOLDS[2]:
        return "gt10_le100"
    if value <= AMOUNT_THRESHOLDS[3]:
        return "gt100_le1000"
    if value <= AMOUNT_THRESHOLDS[4]:
        return "gt1000_le10000"
    return "gt10000"


def canonical_template_text(
    *,
    transaction_type: str,
    band: str,
    prior_sender_count: int,
    prior_receiver_count: int,
    prior_sender_type_count: int,
    prior_sender_distinct_counterparties: int,
) -> str:
    return (
        f"transaction_type={transaction_type};amount_band={band};"
        f"prior_sender_count={prior_sender_count};"
        f"prior_receiver_count={prior_receiver_count};"
        f"prior_sender_type_count={prior_sender_type_count};"
        f"prior_sender_distinct_counterparties={prior_sender_distinct_counterparties}"
    )


def _validate_template_text(text: str, document: RawTransactionDocument) -> None:
    lowered = text.lower()
    forbidden = [token for token in FORBIDDEN_TEMPLATE_TOKENS if token in lowered]
    if forbidden:
        raise ValueError("behavioral template contains forbidden fields")
    if document.sender_id in text or document.receiver_id in text or document.id in text:
        raise ValueError("behavioral template contains an account or transaction identifier")


def template_digest(
    text: str,
    template_version: str = BEHAVIORAL_TEMPLATE_VERSION,
    amount_band_version: str = AMOUNT_BAND_VERSION,
) -> str:
    """Return the canonical digest for versioned behavioral template content."""
    payload = {
        "template_version": template_version,
        "amount_band_version": amount_band_version,
        "text": text,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def iter_behavioral_templates_sorted(
    documents: Iterable[RawTransactionDocument],
) -> Iterator[BehavioralTemplate]:
    sender_counts: dict[str, int] = defaultdict(int)
    receiver_counts: dict[str, int] = defaultdict(int)
    sender_type_counts: dict[tuple[str, str], int] = defaultdict(int)
    sender_counterparties: dict[str, set[str]] = defaultdict(set)
    current_step: int | None = None
    step_documents: list[RawTransactionDocument] = []

    def emit_step(group: list[RawTransactionDocument]) -> Iterator[BehavioralTemplate]:
        for document in sorted(group, key=lambda item: item.id):
            band = amount_band(document.amount)
            text = canonical_template_text(
                transaction_type=document.transaction_type,
                band=band,
                prior_sender_count=sender_counts[document.sender_id],
                prior_receiver_count=receiver_counts[document.receiver_id],
                prior_sender_type_count=sender_type_counts[
                    (document.sender_id, document.transaction_type)
                ],
                prior_sender_distinct_counterparties=len(sender_counterparties[document.sender_id]),
            )
            _validate_template_text(text, document)
            yield BehavioralTemplate(
                transaction_id=document.id,
                sender_id=document.sender_id,
                batch_id=document.batch_id,
                step=document.step,
                data_split=document.data_split,
                raw_id=document.id,
                raw_pk=document.sender_id,
                transaction_type=document.transaction_type,
                amount_band=band,
                prior_sender_count=sender_counts[document.sender_id],
                prior_receiver_count=receiver_counts[document.receiver_id],
                prior_sender_type_count=sender_type_counts[
                    (document.sender_id, document.transaction_type)
                ],
                prior_sender_distinct_counterparties=len(sender_counterparties[document.sender_id]),
                text=text,
                template_version=BEHAVIORAL_TEMPLATE_VERSION,
                amount_band_version=AMOUNT_BAND_VERSION,
                template_digest=template_digest(text),
                reference_eligible=(
                    document.data_split == "train" and document.source_truth.is_fraud
                ),
                source_version=document.projection_version,
            )
        for document in group:
            sender_counts[document.sender_id] += 1
            receiver_counts[document.receiver_id] += 1
            sender_type_counts[(document.sender_id, document.transaction_type)] += 1
            sender_counterparties[document.sender_id].add(document.receiver_id)

    for document in documents:
        if current_step is not None and document.step < current_step:
            raise ValueError("behavioral template input is not ordered by step")
        if current_step is None:
            current_step = document.step
        if document.step != current_step:
            yield from emit_step(step_documents)
            step_documents = []
            current_step = document.step
        step_documents.append(document)
    if step_documents:
        yield from emit_step(step_documents)


def build_behavioral_templates(
    documents: Iterable[RawTransactionDocument],
) -> list[BehavioralTemplate]:
    ordered = sorted(documents, key=lambda document: (document.step, document.id))
    return list(iter_behavioral_templates_sorted(ordered))
