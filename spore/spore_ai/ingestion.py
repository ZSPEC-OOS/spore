"""
SPORE AI Framework — Data Ingestion Layer (§2.2.1)

Accepts structured datasets (JSONL / CSV) of question-response groups,
validates them against the required schema (§3.1.1 / §3.1.2), and returns
normalised DataRecord or ScoredDataRecord lists.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Iterator, List, Union

from .models import DataRecord, ScoredDataRecord, ScoredResponse


class DataIngestionLayer:
    """
    Validates and ingests question-response datasets.

    Supported formats:
    - JSONL: one JSON object per line
    - CSV: headers ``question``, ``responses`` (pipe-separated), ``correct_index``

    Usage::

        layer = DataIngestionLayer()
        records = layer.load_jsonl("data/train.jsonl")
        validated = layer.validate(records)
    """

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_jsonl(self, source: Union[str, Path, io.StringIO]) -> List[DataRecord]:
        """Load the required schema from a JSONL source."""
        records: List[DataRecord] = []
        for obj in self._iter_jsonl(source):
            records.append(self._parse_required(obj))
        return records

    def load_jsonl_scored(self, source: Union[str, Path, io.StringIO]) -> List[ScoredDataRecord]:
        """Load the extended scored schema from a JSONL source."""
        records: List[ScoredDataRecord] = []
        for obj in self._iter_jsonl(source):
            records.append(self._parse_scored(obj))
        return records

    def load_csv(self, source: Union[str, Path, io.StringIO]) -> List[DataRecord]:
        """
        Load the required schema from CSV.

        Expected columns: ``question``, ``responses`` (pipe ``|`` separated),
        ``correct_index`` (0-based integer).
        """
        if isinstance(source, (str, Path)):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source.read()

        reader = csv.DictReader(io.StringIO(text))
        records: List[DataRecord] = []
        for row in reader:
            responses = [r.strip() for r in row["responses"].split("|") if r.strip()]
            records.append(DataRecord(
                question=row["question"].strip(),
                responses=responses,
                correct_index=int(row["correct_index"]),
            ))
        return records

    def load_raw(self, data: Union[str, bytes]) -> List[DataRecord]:
        """Load from raw JSONL string/bytes."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return self.load_jsonl(io.StringIO(data))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, records: List[DataRecord]) -> List[DataRecord]:
        """
        Validate a list of DataRecords against the required schema (§3.1.1).

        Raises ValueError on the first invalid record.
        Returns the validated list unchanged.
        """
        for i, rec in enumerate(records):
            if not rec.question.strip():
                raise ValueError(f"Record {i}: empty question.")
            if not rec.responses:
                raise ValueError(f"Record {i}: responses list is empty.")
            if not 0 <= rec.correct_index < len(rec.responses):
                raise ValueError(
                    f"Record {i}: correct_index {rec.correct_index} out of range."
                )
        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_jsonl(source: Union[str, Path, io.StringIO]) -> Iterator[dict]:
        if isinstance(source, (str, Path)):
            lines = Path(source).read_text(encoding="utf-8").splitlines()
        else:
            lines = source.read().splitlines()

        for line in lines:
            line = line.strip()
            if line:
                yield json.loads(line)

    @staticmethod
    def _parse_required(obj: dict) -> DataRecord:
        return DataRecord(
            question=str(obj["question"]),
            responses=[str(r) for r in obj["responses"]],
            correct_index=int(obj.get("correct_index", 0)),
        )

    @staticmethod
    def _parse_scored(obj: dict) -> ScoredDataRecord:
        responses = [
            ScoredResponse(text=str(r["text"]), score=float(r["score"]))
            for r in obj["responses"]
        ]
        return ScoredDataRecord(question=str(obj["question"]), responses=responses)
