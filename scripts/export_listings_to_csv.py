"""Export property listings from a JSON catalogue to a flat CSV file."""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger("export_listings_to_csv")


def _cell_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def flatten_listing(listing: dict[str, Any]) -> dict[str, str]:
    """Flatten nested objects into dotted columns, keeping arrays as JSON text."""
    flattened: dict[str, str] = {}

    def add_value(value: Any, column: str) -> None:
        if isinstance(value, dict) and value:
            for key, nested_value in value.items():
                nested_column = f"{column}.{key}" if column else key
                add_value(nested_value, nested_column)
            return

        if column in flattened:
            raise ValueError(f"multiple values map to column {column!r}")
        flattened[column] = _cell_value(value)

    for key, value in listing.items():
        add_value(value, key)
    return flattened


def export_catalog(input_path: Path, output_path: Path) -> tuple[int, int, int]:
    """Export catalog listings and return (input count, exported count, malformed count)."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must be different")

    with input_path.open("r", encoding="utf-8") as input_file:
        catalog = json.load(input_file)

    if not isinstance(catalog, dict):
        raise ValueError("catalogue JSON must contain a top-level object")
    listings = catalog.get("listings")
    if not isinstance(listings, list):
        raise ValueError("catalogue must contain a top-level 'listings' array")

    rows: list[dict[str, str]] = []
    malformed_count = 0
    for index, listing in enumerate(listings, start=1):
        if not isinstance(listing, dict):
            malformed_count += 1
            LOGGER.warning(
                "Skipping malformed listing at index %d: expected an object, got %s",
                index,
                type(listing).__name__,
            )
            continue

        try:
            rows.append(flatten_listing(listing))
        except (TypeError, ValueError) as error:
            malformed_count += 1
            LOGGER.warning("Could not export listing at index %d: %s", index, error)

    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    for row in rows:
        for fieldname in row:
            if fieldname not in seen_fields:
                seen_fields.add(fieldname)
                fieldnames.append(fieldname)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    return len(listings), len(rows), malformed_count


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export listing records from a JSON catalogue to CSV."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSON catalogue")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _argument_parser().parse_args(argv)

    try:
        input_count, exported_count, malformed_count = export_catalog(
            args.input, args.output
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1

    unprocessed_count = input_count - exported_count
    LOGGER.info("Input listings: %d", input_count)
    LOGGER.info("Successfully exported rows: %d", exported_count)
    LOGGER.info("Malformed records: %d", malformed_count)
    LOGGER.info("Records not processed: %d", unprocessed_count)
    LOGGER.info("CSV written to %s", args.output)
    return 1 if unprocessed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
