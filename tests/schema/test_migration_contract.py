"""Static checks for the atomic Phase 2 schema revision."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.schema

ROOT = Path(__file__).resolve().parents[2]
REVISION = ROOT / "alembic/versions/20261031_0001_complete_relational_schema.py"


def test_phase_two_has_one_atomic_schema_revision() -> None:
    revisions = sorted((ROOT / "alembic/versions").glob("*.py"))
    assert REVISION in revisions
    assert len(revisions) == 2
    text = REVISION.read_text(encoding="utf-8")
    assert 'revision: str = "20261031_0001"' in text
    assert "down_revision: str | None = None" in text
    assert text.count("op.create_table(") == 12


def test_cross_row_invariants_are_trigger_backed() -> None:
    text = REVISION.read_text(encoding="utf-8")
    assert "enforce_merchant_account_kind" in text
    assert "trg_merchant_account_kind" in text
    assert "trg_account_merchant_kind" in text
    assert "enforce_fraud_flag_threshold" in text
    assert "trg_fraud_flag_threshold" in text
