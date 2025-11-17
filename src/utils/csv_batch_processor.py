"""CSV I/O utility for batch data augmentation."""

import json
import logging
from collections.abc import Iterator

import pandas as pd

from src.core.spec import Sample, Spec

logger = logging.getLogger(__name__)


# TODO: Refactor to generic BatchProcessor supporting CSV and JSONL - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/12
class CSVBatchProcessor:
    """Pure I/O utility for reading and writing CSV files."""

    @staticmethod
    def read_row_specs(spec: Spec) -> tuple[pd.DataFrame, Iterator[tuple[int, Spec, int]]]:
        """Read CSV and yield (original_df, iterator of (row_idx, row_spec, row_samples, row))."""
        task_input = spec.task_input
        input_file = task_input["input_file"]
        input_column = task_input["input_column"]
        output_column = task_input["output_column"]
        task_description = task_input.get("task_description", "")
        context = task_input.get("context")
        examples = task_input.get("examples")

        # Read CSV
        logger.info(f"Reading CSV from {input_file}")
        df = pd.read_csv(input_file)

        if input_column not in df.columns:
            raise ValueError(f"Column '{input_column}' not found in CSV")
        if output_column not in df.columns:
            raise ValueError(f"Column '{output_column}' not found in CSV")

        # Calculate samples per row (uniform distribution)
        num_rows = len(df)
        samples_per_row = spec.num_samples // num_rows
        remainder = spec.num_samples % num_rows

        logger.info(f"CSV has {num_rows} rows, {samples_per_row} samples per row")

        def _generate_specs():
            for idx, row in df.iterrows():
                original_input = row[input_column]
                expected_output = row[output_column]

                # Calculate samples for this row
                row_samples = samples_per_row + (1 if idx < remainder else 0)

                # Create spec for this row
                row_task_input = {
                    "original_input": original_input,
                    "expected_output": expected_output,
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

                yield idx, row_spec, row_samples, row

        return df, _generate_specs()

    @staticmethod
    def write_results(
        original_df: pd.DataFrame,
        samples: list[Sample],
        output_path: str,
        input_column: str,
    ):
        """Write augmented samples to CSV with uuid, original columns, and metadata."""
        logger.info(f"Writing {len(samples)} samples to {output_path}")

        output_rows = []
        for sample in samples:
            csv_row = sample.metadata.get("csv_row", {})
            output_row = {
                "uuid": str(sample.id),
                **csv_row,
                "metadata": json.dumps(
                    {
                        "original_input": sample.metadata.get("original_input"),
                        "expected_output": sample.metadata.get("expected_output"),
                        "lineage": {
                            "generator": str(sample.lineage.generator),
                            "generator_parameters": sample.lineage.generator_parameters,
                            "original_sample": (
                                str(sample.lineage.original_sample)
                                if sample.lineage.original_sample
                                else None
                            ),
                            "parent_id": (
                                str(sample.lineage.parent_id) if sample.lineage.parent_id else None
                            ),
                            "num_of_evolutions": sample.lineage.num_of_evolutions,
                        },
                        "quality_scores": sample.quality_scores,
                        "timestamp": sample.metadata.get("timestamp"),
                    }
                ),
            }
            # Replace input column with generated variant
            output_row[input_column] = sample.content
            output_rows.append(output_row)

        # Create DataFrame
        output_df = pd.DataFrame(output_rows)

        # Reorder columns
        original_columns = original_df.columns.tolist()
        column_order = ["uuid"] + original_columns + ["metadata"]
        output_df = output_df[column_order]

        output_df.to_csv(output_path, index=False)
        logger.info(f"Successfully wrote {len(output_df)} rows to {output_path}")
