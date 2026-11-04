"""Canonical hashing and stable identity helpers."""

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def selected_rows_digest(source_rows: Iterable[int]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(source_rows))).hexdigest()


def transaction_id(source_sha256: str, source_row: int) -> str:
    """Return the canonical stable identity for one original CSV data row."""

    return f"paysim:{source_sha256}:{source_row}"


def deterministic_rank_salt(source_sha256: str, sampling_seed: int, split: str) -> int:
    material = f"{source_sha256}:{sampling_seed}:{split}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2_147_483_647
