"""Convert a Chicago CSV to a typed, date-partitioned Parquet dataset.

Example: uv run python scripts/convert_to_parquet.py data/chicago-speeds.csv --kind speeds
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


def _timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%m/%d/%Y %I:%M:%S %p", errors="coerce").astype(
        "datetime64[us]"
    )
    missing = parsed.isna()
    if missing.any():
        fallback = pd.to_datetime(values.loc[missing], format="ISO8601", errors="coerce")
        if fallback.dt.tz is not None:
            raise ValueError("Expected the portal's timezone-naive timestamps")
        parsed.loc[missing] = fallback.astype("datetime64[us]")
    if parsed.isna().any():
        raise ValueError(f"Found {int(parsed.isna().sum())} missing or invalid timestamps")
    if parsed.dt.tz is not None:
        raise ValueError("Expected the portal's timezone-naive timestamps")
    return parsed


def csv_to_parquet(
    csv_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    kind: str,
    chunksize: int = 500_000,
    overwrite: bool = False,
) -> dict[str, object]:
    """Stream speeds/crashes into date=YYYY-MM-DD partitions and return a manifest.

    Columns are lowercased and time/crash_date becomes timestamp (timezone-naive).
    Speeds become nullable float32; negative/nonfinite/nonnumeric values become
    null with speed_available=False. Raw CSVs remain untouched. Replacement uses
    a complete staged dataset, so failures preserve an existing conversion.
    """
    if kind not in {"speeds", "crashes"}:
        raise ValueError("kind must be 'speeds' or 'crashes'")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    csv_path = Path(csv_path).resolve()
    destination = (
        Path(output_dir).resolve() if output_dir else csv_path.parent / "parquet" / csv_path.stem
    )
    if csv_path.is_relative_to(destination):
        raise ValueError("The output directory cannot contain the input CSV")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    def check_destination() -> None:
        if destination.exists():
            if not overwrite:
                raise FileExistsError(
                    f"{destination} already exists; pass overwrite=True or --overwrite"
                )
            if not destination.is_dir() or not (destination / "_conversion.json").is_file():
                raise ValueError("Will only replace a dataset previously created by this converter")

    check_destination()
    destination.parent.mkdir(parents=True, exist_ok=True)
    time_column = "time" if kind == "speeds" else "crash_date"
    required = {time_column, "segment_id", "speed"} if kind == "speeds" else {time_column}
    total_rows = unavailable = 0
    first_time = last_time = None
    parquet_format = ds.ParquetFileFormat()
    options = parquet_format.make_write_options(compression="zstd")
    partitioning = ds.partitioning(pa.schema([("date", pa.date32())]), flavor="hive")
    with (
        tempfile.TemporaryDirectory(
            prefix=f".{destination.name}-", dir=destination.parent
        ) as temporary,
        pd.read_csv(csv_path, dtype=str, chunksize=chunksize) as reader,
    ):
        staged = Path(temporary) / "dataset"
        staged.mkdir()
        for batch, frame in enumerate(reader):
            frame.columns = frame.columns.str.strip().str.lower()
            if not frame.columns.is_unique or "timestamp" in frame or "date" in frame:
                raise ValueError("Duplicate or reserved timestamp/date columns in input CSV")
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
            if frame.empty:
                continue
            frame = frame.rename(columns={time_column: "timestamp"})
            frame["timestamp"] = _timestamps(frame["timestamp"])
            column_types = {column: pa.string() for column in frame.columns}
            column_types["timestamp"] = pa.timestamp("us")
            if kind == "speeds":
                ids = pd.to_numeric(frame["segment_id"], errors="raise")
                if not (np.isfinite(ids) & (ids >= 0) & (ids % 1 == 0)).all():
                    raise ValueError("segment_id must contain nonnegative integers")
                frame["segment_id"] = ids.astype("int64")
                values = pd.to_numeric(frame["speed"], errors="coerce")
                frame["speed_available"] = np.isfinite(values) & (values >= 0)
                frame["speed"] = values.where(frame["speed_available"]).astype("float32")
                unavailable += int((~frame["speed_available"]).sum())
                column_types.update(
                    segment_id=pa.int64(), speed=pa.float32(), speed_available=pa.bool_()
                )
            else:
                for column in ("latitude", "longitude"):
                    if column in frame:
                        values = pd.to_numeric(frame[column], errors="coerce")
                        frame[column] = values.where(np.isfinite(values))
                        column_types[column] = pa.float64()
            schema = pa.schema([(column, column_types[column]) for column in frame.columns])
            table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
            table = table.append_column("date", pc.cast(table["timestamp"], pa.date32()))
            ds.write_dataset(
                table,
                base_dir=staged,
                format=parquet_format,
                file_options=options,
                partitioning=partitioning,
                basename_template=f"batch-{batch:05d}-{{i}}.parquet",
                existing_data_behavior="overwrite_or_ignore",
                max_rows_per_file=250_000,
                max_rows_per_group=64_000,
            )
            total_rows += len(frame)
            lower, upper = frame.timestamp.min(), frame.timestamp.max()
            first_time = min(first_time, lower) if first_time is not None else lower
            last_time = max(last_time, upper) if last_time is not None else upper
            if (batch + 1) % 20 == 0:
                print(f"Converted {total_rows:,} {kind} rows...", flush=True)
        if total_rows == 0:
            raise ValueError("Input CSV has no records")
        manifest = {
            "kind": kind,
            "source_csv": str(csv_path),
            "output_dir": str(destination),
            "rows": total_rows,
            "first_timestamp": first_time.isoformat(),
            "last_timestamp": last_time.isoformat(),
            "unavailable_speeds": unavailable if kind == "speeds" else None,
            "timestamp_timezone": None,
            "partition_column": "date",
            "compression": "zstd",
            "parquet_files": len(list(staged.rglob("*.parquet"))),
        }
        (staged / "_conversion.json").write_text(json.dumps(manifest, indent=2) + "\n")
        check_destination()
        backup = Path(temporary) / "previous"
        if destination.exists():
            destination.rename(backup)
        try:
            staged.rename(destination)
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
    return manifest
