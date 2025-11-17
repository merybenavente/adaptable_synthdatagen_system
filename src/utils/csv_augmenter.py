"""CSV data augmentation utility for batch processing."""

import json

import pandas as pd

from src.core.spec import Spec
from src.generators.naive_generator import NaiveGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CSVDataAugmenter:
    """Processes CSV files for ML data augmentation."""

    def __init__(self, spec: Spec):
        self.spec = spec
        self._validate_spec()

    def _validate_spec(self):
        """Validate that spec has required fields for CSV augmentation."""
        if not isinstance(self.spec.task_input, dict):
            raise ValueError("task_input must be a dict for CSV augmentation")

        required_fields = ["input_file", "input_column", "output_column"]
        for field in required_fields:
            if field not in self.spec.task_input:
                raise ValueError(f"task_input missing required field: {field}")

        if not self.spec.output_path:
            raise ValueError("output_path is required for CSV augmentation")

    def process(self) -> str:
        """Process CSV and generate augmented data."""
        logger.info("Starting CSV augmentation process")

        # Extract config from task_input
        input_file = self.spec.task_input["input_file"]
        input_column = self.spec.task_input["input_column"]
        output_column = self.spec.task_input["output_column"]
        task_description = self.spec.task_input.get("task_description", "")
        context = self.spec.task_input.get("context")
        examples = self.spec.task_input.get("examples")

        # Read input CSV
        logger.info(f"Reading CSV from {input_file}")
        df = pd.read_csv(input_file)

        if input_column not in df.columns:
            raise ValueError(f"Column '{input_column}' not found in CSV")
        if output_column not in df.columns:
            raise ValueError(f"Column '{output_column}' not found in CSV")

        # Calculate samples per row (uniform distribution)
        num_rows = len(df)
        samples_per_row = self.spec.num_samples // num_rows
        remainder = self.spec.num_samples % num_rows

        logger.info(f"Processing {num_rows} rows, generating {samples_per_row} samples per row")

        # Generate augmented samples
        all_samples = []
        for idx, row in df.iterrows():
            original_input = row[input_column]
            expected_output = row[output_column]

            # Calculate samples for this row (distribute remainder)
            row_samples = samples_per_row + (1 if idx < remainder else 0)

            logger.info(f"Row {idx + 1}/{num_rows}: Generating {row_samples} variants")

            # Create task_input for this row in ML augmentation format
            row_task_input = {
                "original_input": original_input,
                "expected_output": expected_output,
                "task_description": task_description,
            }
            if context:
                row_task_input["context"] = context
            if examples:
                row_task_input["examples"] = examples

            # Create spec for this row
            row_spec = Spec(
                domain=self.spec.domain,
                task_input=row_task_input,
                num_samples=row_samples,
                constraints=self.spec.constraints,
                output_format=self.spec.output_format,
            )

            # Generate samples using NaiveGenerator
            generator = NaiveGenerator(row_spec)
            samples = generator.generate()

            # Convert samples to dict format with row data
            for sample in samples:
                sample_dict = {
                    "uuid": str(sample.id),
                    **row.to_dict(),  # Preserve all original columns
                    "metadata": json.dumps({
                        "original_input": original_input,
                        "expected_output": expected_output,
                        "lineage": {
                            "generator": sample.lineage.generator,
                            "generator_parameters": sample.lineage.generator_parameters,
                            "original_sample": (
                                str(sample.lineage.original_sample)
                                if sample.lineage.original_sample
                                else None
                            ),
                            "parent_id": (
                                str(sample.lineage.parent_id)
                                if sample.lineage.parent_id
                                else None
                            ),
                            "num_of_evolutions": sample.lineage.num_of_evolutions,
                        },
                        "quality_scores": sample.quality_scores,
                        "timestamp": sample.metadata.get("timestamp"),
                    })
                }
                # Replace input column with the generated variant
                sample_dict[input_column] = sample.content
                all_samples.append(sample_dict)

        # Create output DataFrame
        output_df = pd.DataFrame(all_samples)

        # Reorder columns: uuid first, then original columns, then metadata
        original_columns = df.columns.tolist()
        column_order = ["uuid"] + original_columns + ["metadata"]
        output_df = output_df[column_order]

        # Write output CSV
        logger.info(f"Writing {len(output_df)} samples to {self.spec.output_path}")
        output_df.to_csv(self.spec.output_path, index=False)

        logger.info("CSV augmentation complete")
        return self.spec.output_path
