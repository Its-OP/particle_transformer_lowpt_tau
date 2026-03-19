"""Split a single Parquet file into N equal parts for multi-worker DataLoader.

Uses DuckDB for out-of-core streaming: it reads and writes parquet data with
controlled memory usage, avoiding the PyArrow issue where single-row-group
files require decompressing entire column chunks into memory.

Usage:
    python split_parquet.py input.parquet --num-parts 10 --output-dir output/
    python split_parquet.py input.parquet --num-parts 10 --memory-limit 4GB
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Split a Parquet file into N equal parts (memory-safe via DuckDB)."
    )
    parser.add_argument("input", type=str, help="Path to the input Parquet file")
    parser.add_argument(
        "--num-parts",
        type=int,
        default=10,
        help="Number of output parts (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input file)",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help="DuckDB memory limit (default: 4GB). "
        "Controls peak RAM usage during splitting.",
    )
    return parser.parse_args()


def split_parquet(input_path, output_paths, num_parts, memory_limit):
    """Split a parquet file into N parts using DuckDB's out-of-core engine.

    DuckDB streams through the parquet file with bounded memory, using
    LIMIT/OFFSET to select row ranges for each output part. Each part is
    written directly to a compressed parquet file via COPY ... TO.

    Args:
        input_path: Path to the source parquet file.
        output_paths: List of Path objects for each output part.
        num_parts: Number of parts to split into.
        memory_limit: DuckDB memory limit string (e.g., '4GB').
    """
    # Read metadata without loading data (lightweight PyArrow call)
    parquet_file = pq.ParquetFile(input_path)
    metadata = parquet_file.metadata
    total_rows = metadata.num_rows
    input_size_megabytes = os.path.getsize(input_path) / (1024 * 1024)

    print(f"Input:      {input_path}")
    print(f"  Rows:       {total_rows:,}")
    print(f"  Row groups: {metadata.num_row_groups}")
    print(f"  Size:       {input_size_megabytes:,.1f} MB")
    print(f"  Schema:     {len(parquet_file.schema_arrow)} columns")
    print(f"Splitting into {num_parts} parts (memory_limit={memory_limit})...")
    print()

    # Configure DuckDB for low-memory streaming
    connection = duckdb.connect()
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute("SET threads = 1")

    # Compute row ranges: first (N-1) parts get floor(total/N) rows,
    # last part gets the remainder.
    rows_per_part = total_rows // num_parts
    total_rows_written = 0

    input_path_str = str(input_path)

    for part_index in range(num_parts):
        offset = part_index * rows_per_part
        if part_index < num_parts - 1:
            limit = rows_per_part
        else:
            # Last part gets all remaining rows
            limit = total_rows - offset

        output_path_str = str(output_paths[part_index])

        # COPY with LIMIT/OFFSET streams rows directly from input to output
        # without materializing the entire dataset in memory.
        connection.execute(
            f"COPY (SELECT * FROM read_parquet('{input_path_str}') "
            f"LIMIT {limit} OFFSET {offset}) "
            f"TO '{output_path_str}' (FORMAT PARQUET, COMPRESSION LZ4)"
        )

        total_rows_written += limit
        file_size_megabytes = os.path.getsize(output_paths[part_index]) / (1024 * 1024)
        print(
            f"  Part {part_index:02d}: {limit:>8,} rows  "
            f"({file_size_megabytes:>8.1f} MB)  "
            f"[{output_paths[part_index].name}]"
        )

    connection.close()

    # Summary
    print()
    total_output_megabytes = sum(
        os.path.getsize(path) / (1024 * 1024) for path in output_paths
    )
    print(f"Total rows written: {total_rows_written:,} (original: {total_rows:,})")
    if total_rows_written != total_rows:
        print("WARNING: Row count mismatch!")
        sys.exit(1)
    print(
        f"Total output size:  {total_output_megabytes:,.1f} MB "
        f"(input: {input_size_megabytes:,.1f} MB)"
    )
    print("Done.")


def main():
    arguments = parse_arguments()

    input_path = Path(arguments.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    num_parts = arguments.num_parts
    memory_limit = arguments.memory_limit
    output_directory = (
        Path(arguments.output_dir) if arguments.output_dir else input_path.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    # Derive output filenames from input stem: train.parquet → train_00.parquet
    stem = input_path.stem
    output_paths = [
        output_directory / f"{stem}_{part_index:02d}.parquet"
        for part_index in range(num_parts)
    ]

    split_parquet(input_path, output_paths, num_parts, memory_limit)


if __name__ == "__main__":
    main()
