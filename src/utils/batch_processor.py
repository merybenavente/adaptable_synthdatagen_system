"""Format-aware batch I/O utilities for dataset ingestion."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.models import Sample, Spec
from src.core.type_guards import is_batch_input_dict

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchReadResult:
    """Lightweight container describing a batch input dataset."""

    format: str
    source: Any
    input_field: str


class BatchProcessor:
    """Pure I/O helper for batch datasets (CSV and JSONL)."""

    _FORMAT_BY_EXTENSION = {
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
    }

    @classmethod
    def read_row_specs(
        cls, spec: Spec
    ) -> tuple[BatchReadResult, Iterator[tuple[int, Spec, int, dict[str, Any]]]]:
        """Detect format, load dataset, and yield per-row specs."""
        if not is_batch_input_dict(spec.task_input):
            raise ValueError(
                "BatchProcessor.read_row_specs requires task_input to be a dict with 'input_file'"
            )

        task_input = spec.task_input
        input_file = Path(task_input["input_file"])
        format_name = cls._detect_format(input_file)

        if format_name == "csv":
            return cls._read_csv(spec, input_file)

        if format_name == "jsonl":
            return cls._read_jsonl(spec, input_file)

        raise ValueError(f"Unsupported batch input format for {input_file}")

    @classmethod
    def write_results(
        cls,
        batch_info: BatchReadResult,
        samples: list[Sample],
        output_path: str,
    ) -> None:
        """Write generated samples using the same format as the input."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if batch_info.format == "csv":
            cls._write_csv(batch_info, samples, destination)
            return

        if batch_info.format == "jsonl":
            cls._write_jsonl(batch_info, samples, destination)
            return

        raise ValueError(f"Unsupported batch output format: {batch_info.format}")

    @classmethod
    def _read_csv(
        cls,
        spec: Spec,
        input_file: Path,
    ) -> tuple[BatchReadResult, Iterator[tuple[int, Spec, int, dict[str, Any]]]]:
        task_input = spec.task_input
        input_column = task_input["input_column"]
        output_column = task_input["output_column"]

        logger.info(f"Reading CSV from {input_file}")
        df = pd.read_csv(input_file)

        if input_column not in df.columns:
            raise ValueError(f"Column '{input_column}' not found in CSV")
        if output_column not in df.columns:
            raise ValueError(f"Column '{output_column}' not found in CSV")

        rows = df.to_dict(orient="records")
        iterator = cls._build_row_iterator(rows, spec, input_column, output_column)
        batch_info = BatchReadResult(format="csv", source=df, input_field=input_column)
        return batch_info, iterator

    @classmethod
    def _read_jsonl(
        cls,
        spec: Spec,
        input_file: Path,
    ) -> tuple[BatchReadResult, Iterator[tuple[int, Spec, int, dict[str, Any]]]]:
        task_input = spec.task_input
        input_field = task_input["input_column"]
        output_field = task_input["output_column"]

        logger.info(f"Reading JSONL from {input_file}")
        rows: list[dict[str, Any]] = []
        with open(input_file) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} in {input_file}") from exc
                rows.append(record)

        if not rows:
            logger.warning("JSONL input is empty")

        for field in (input_field, output_field):
            if any(field not in row for row in rows):
                raise ValueError(f"Field '{field}' not found in every JSONL row")

        iterator = cls._build_row_iterator(rows, spec, input_field, output_field)
        batch_info = BatchReadResult(format="jsonl", source=rows, input_field=input_field)
        return batch_info, iterator

    @staticmethod
    def _build_row_iterator(
        rows: list[dict[str, Any]],
        spec: Spec,
        input_field: str,
        output_field: str,
    ) -> Iterator[tuple[int, Spec, int, dict[str, Any]]]:
        task_input = spec.task_input
        task_description = task_input.get("task_description", "")
        context = task_input.get("context")
        examples = task_input.get("examples")

        num_rows = len(rows)
        if num_rows == 0:
            raise ValueError("Batch input must contain at least one row")

        samples_per_row = spec.num_samples // num_rows
        remainder = spec.num_samples % num_rows

        logger.info(f"Batch has {num_rows} rows, {samples_per_row} samples per row")

        def _generate():
            for idx, row in enumerate(rows):
                row_samples = samples_per_row + (1 if idx < remainder else 0)
                if row_samples == 0:
                    continue

                row_task_input = {
                    "original_input": row[input_field],
                    "expected_output": row[output_field],
                    "task_description": task_description,
                }
                if context:
                    row_task_input["context"] = context
                if examples:
                    row_task_input["examples"] = examples

                row_spec = Spec(
                    domain=spec.domain,
                    task_input=row_task_input,
                    num_samples=row_samples,
                    constraints=spec.constraints,
                    output_format="text",
                )

                yield idx, row_spec, row_samples, dict(row)

        return _generate()

    @staticmethod
    def _build_metadata(sample: Sample) -> dict[str, Any]:
        lineage = sample.lineage
        lineage_payload = (
            {
                "generator": str(lineage.generator) if lineage and lineage.generator else None,
                "generator_parameters": lineage.generator_parameters if lineage else {},
                "original_sample": (
                    str(lineage.original_sample) if lineage and lineage.original_sample else None
                ),
                "parent_id": str(lineage.parent_id) if lineage and lineage.parent_id else None,
                "num_of_evolutions": lineage.num_of_evolutions if lineage else 0,
            }
        )

        return {
            "original_input": sample.metadata.get("original_input"),
            "expected_output": sample.metadata.get("expected_output"),
            "lineage": lineage_payload,
            "quality_scores": sample.quality_scores,
            "timestamp": sample.metadata.get("timestamp"),
        }

    @classmethod
    def _write_csv(
        cls, batch_info: BatchReadResult, samples: list[Sample], output_path: Path
    ) -> None:
        logger.info(f"Writing {len(samples)} samples to CSV {output_path}")

        output_rows = []
        for sample in samples:
            row_data = dict(sample.metadata.get("batch_row", {}))
            output_row = {
                "uuid": str(sample.id),
                **row_data,
                "metadata": json.dumps(cls._build_metadata(sample)),
            }
            output_row[batch_info.input_field] = sample.content
            output_rows.append(output_row)

        original_columns = (
            batch_info.source.columns.tolist()
            if isinstance(batch_info.source, pd.DataFrame)
            else []
        )
        column_order = ["uuid"] + original_columns + ["metadata"]
        if output_rows:
            output_df = pd.DataFrame(output_rows)[column_order]
        else:
            output_df = pd.DataFrame(columns=column_order)

        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote {len(output_df)} rows to {output_path}")

    @classmethod
    def _write_jsonl(
        cls, batch_info: BatchReadResult, samples: list[Sample], output_path: Path
    ) -> None:
        logger.info(f"Writing {len(samples)} samples to JSONL {output_path}")

        with open(output_path, "w") as f:
            for sample in samples:
                row_data = dict(sample.metadata.get("batch_row", {}))
                row_data[batch_info.input_field] = sample.content
                record = {
                    "uuid": str(sample.id),
                    **row_data,
                    "metadata": cls._build_metadata(sample),
                }
                f.write(json.dumps(record) + "\n")

        logger.info(f"Successfully wrote {len(samples)} rows to {output_path}")

    @classmethod
    def _detect_format(cls, input_file: Path) -> str:
        extension = input_file.suffix.lower()
        format_name = cls._FORMAT_BY_EXTENSION.get(extension)
        if not format_name:
            raise ValueError(f"Unsupported batch file extension '{extension}'")
        return format_name

