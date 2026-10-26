"""Profile a raw property-listings CSV without changing its contents."""

import argparse
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


LOGGER = logging.getLogger("profile_property_listings")

PLACEHOLDERS = {"<br>", "n/a", "na", "none", "unknown", "not available", "-"}
JSON_COLUMNS = (
    "images",
    "special",
    "building",
    "accommodation_summary",
    "agent_languages",
    "video_tour",
    "virtual_tour",
)
CATEGORY_COLUMNS = (
    "listing_type",
    "department",
    "search_type",
    "status",
    "property_type",
    "community",
    "building",
)
NUMERIC_COLUMNS = {
    "price": {"min_positive": 0, "max": 1_000_000_000},
    "beds": {"min_positive": 0, "max": 20},
    "bathrooms": {"min_positive": 0, "max": 30},
    "floorarea_min": {"min_positive": 0, "max": 100_000},
    "plot_area_min": {"min_positive": 0, "max": 1_000_000},
}
COORDINATE_BOUNDS = {
    "latitude": (-90, 90),
    "longitude": (-180, 180),
}
DUBAI_BOUNDS = {"latitude": (24.5, 25.8), "longitude": (54.8, 56.5)}
PROPERTY_TERMS = (
    "apartment",
    "villa",
    "townhouse",
    "penthouse",
    "duplex",
    "office",
    "warehouse",
    "hotel apartment",
    "residential plot",
    "commercial plot",
    "land",
)
HTML_TAG_RE = re.compile(r"<\s*/?\s*[a-zA-Z][^>]*>")
BREAK_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
ESCAPING_PATTERNS = {
    "HTML entities": re.compile(r"&(?:lt|gt|amp|quot|#\d+|#x[0-9a-f]+);", re.I),
    "backslash escapes": re.compile(r"\\(?:[\"'\\/bfnrt]|u[0-9a-fA-F]{4})"),
    "replacement character": re.compile("\ufffd"),
}


def _is_blank(value: str) -> bool:
    return not value.strip()


def _is_null(value: str) -> bool:
    return value.strip().casefold() in {"null", "none"}


def _type_of(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "blank"
    if stripped.casefold() in {"null", "none"}:
        return "null"
    if stripped.casefold() in {"true", "false"}:
        return "boolean"
    if stripped.startswith(("[", "{")):
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return "text"
        if isinstance(decoded, list):
            return "array"
        if isinstance(decoded, dict):
            return "object"
    try:
        number = Decimal(stripped)
    except InvalidOperation:
        return "text"
    if number.is_finite():
        return "integer" if number == number.to_integral_value() else "float"
    return "text"


def _numeric(value: str) -> float | None:
    if _is_blank(value) or _is_null(value):
        return None
    try:
        number = float(value.strip())
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _format_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _examples(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    return items[:limit]


def _column_labels(header: list[str]) -> list[str]:
    totals = Counter(header)
    seen: Counter[str] = Counter()
    labels = []
    for name in header:
        seen[name] += 1
        labels.append(f"{name}#{seen[name]}" if totals[name] > 1 else name)
    return labels


def _load_csv(path: Path) -> tuple[list[str], list[list[str]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.reader(csv_file, strict=True)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError("CSV file is empty") from error
        if not header:
            raise ValueError("CSV file has an empty header")

        rows: list[list[str]] = []
        short_rows = 0
        long_rows = 0
        for record_number, row in enumerate(reader, start=1):
            if len(row) < len(header):
                short_rows += 1
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                long_rows += 1
                row = row[: len(header)]
            rows.append(row)

    duplicates = [
        {"name": name, "count": count}
        for name, count in Counter(header).items()
        if count > 1
    ]
    return header, rows, {
        "record_count": len(rows),
        "column_count": len(header),
        "duplicate_column_names": duplicates,
        "rows_with_too_few_fields": short_rows,
        "rows_with_too_many_fields": long_rows,
        "column_labels": _column_labels(header),
    }


def _column_profile(
    header: list[str], rows: list[list[str]], column_labels: list[str]
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for index, name in enumerate(header):
        values = [row[index] for row in rows]
        type_counts = Counter(_type_of(value) for value in values)
        placeholder_counts = Counter(
            value.strip()
            for value in values
            if value.strip().casefold() in PLACEHOLDERS
        )
        nonmissing = [
            value for value in values if not _is_blank(value) and not _is_null(value)
        ]
        value_counts = Counter(nonmissing)
        profiles[column_labels[index]] = {
            "name": name,
            "index": index,
            "types": dict(sorted(type_counts.items())),
            "blank_count": type_counts["blank"],
            "null_count": type_counts["null"],
            "missing_count": type_counts["blank"] + type_counts["null"],
            "placeholder_counts": dict(placeholder_counts),
            "distinct_nonmissing_count": len(value_counts),
            "top_values": [
                {"value": value, "count": count}
                for value, count in value_counts.most_common(10)
            ],
            "html_break_tag_cells": sum(bool(BREAK_TAG_RE.search(v)) for v in values),
        }
    return profiles


def _first_column(header: list[str], name: str) -> int | None:
    try:
        return header.index(name)
    except ValueError:
        return None


def _get(row: list[str], header: list[str], name: str) -> str:
    index = _first_column(header, name)
    return row[index] if index is not None else ""


def _duplicate_report(
    rows: list[list[str]], header: list[str], field: str
) -> dict[str, Any]:
    index = _first_column(header, field)
    if index is None:
        return {"available": False, "duplicate_value_count": 0, "duplicate_row_count": 0}

    grouped: dict[str, list[int]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        value = row[index].strip()
        if value and not _is_null(value):
            grouped[value].append(row_number)
    duplicates = [
        {"value": value, "count": len(row_numbers), "row_numbers": row_numbers[:10]}
        for value, row_numbers in grouped.items()
        if len(row_numbers) > 1
    ]
    duplicates.sort(key=lambda item: (-item["count"], item["value"]))
    return {
        "available": True,
        "duplicate_value_count": len(duplicates),
        "duplicate_row_count": sum(item["count"] for item in duplicates),
        "extra_occurrence_count": sum(item["count"] - 1 for item in duplicates),
        "duplicate_groups": duplicates[:25],
    }


def _distribution(
    rows: list[list[str]], header: list[str], field: str
) -> dict[str, Any]:
    index = _first_column(header, field)
    if index is None:
        return {"available": False, "counts": []}

    counts: Counter[str] = Counter()
    missing = 0
    for row in rows:
        value = row[index]
        if _is_blank(value) or _is_null(value):
            missing += 1
            continue
        if field == "building":
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                decoded = value
            values = decoded if isinstance(decoded, list) else [decoded]
            for item in values:
                counts[str(item)] += 1
        else:
            counts[value.strip()] += 1
    return {
        "available": True,
        "missing_count": missing,
        "distinct_value_count": len(counts),
        "counts": [{"value": key, "count": count} for key, count in counts.most_common()],
    }


def _numeric_report(
    rows: list[list[str]], header: list[str], field: str
) -> dict[str, Any]:
    index = _first_column(header, field)
    if index is None:
        return {"available": False}

    values: list[float] = []
    missing_count = 0
    invalid_count = 0
    suspicious: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        raw = row[index]
        if _is_blank(raw) or _is_null(raw):
            missing_count += 1
            continue
        value = _numeric(raw)
        if value is None:
            invalid_count += 1
            continue
        values.append(value)
        if value < 0:
            suspicious["negative_value"].append(
                {"row": row_number, "value": raw, "id": _get(row, header, "id")}
            )
        elif value == 0:
            suspicious["zero_value"].append(
                {"row": row_number, "value": raw, "id": _get(row, header, "id")}
            )
        if value > NUMERIC_COLUMNS[field]["max"]:
            suspicious["above_review_threshold"].append(
                {"row": row_number, "value": raw, "id": _get(row, header, "id")}
            )

    sorted_values = sorted(values)
    median: float | None = None
    if sorted_values:
        middle = len(sorted_values) // 2
        median = (
            sorted_values[middle]
            if len(sorted_values) % 2
            else (sorted_values[middle - 1] + sorted_values[middle]) / 2
        )
    return {
        "available": True,
        "valid_numeric_count": len(values),
        "missing_count": missing_count,
        "invalid_numeric_count": invalid_count,
        "minimum": _format_number(min(values)) if values else None,
        "maximum": _format_number(max(values)) if values else None,
        "median": _format_number(median) if median is not None else None,
        "suspicious_candidates": {
            reason: {"count": len(items), "examples": _examples(items)}
            for reason, items in sorted(suspicious.items())
        },
        "review_thresholds": NUMERIC_COLUMNS[field],
    }


def _coordinate_report(
    rows: list[list[str]], header: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, bounds in COORDINATE_BOUNDS.items():
        index = _first_column(header, field)
        if index is None:
            result[field] = {"available": False}
            continue
        missing = invalid = out_of_range = outside_dubai = 0
        examples: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=1):
            raw = row[index]
            if _is_blank(raw) or _is_null(raw):
                missing += 1
                continue
            value = _numeric(raw)
            if value is None:
                invalid += 1
                examples.append({"row": row_number, "value": raw})
            elif not bounds[0] <= value <= bounds[1]:
                out_of_range += 1
                examples.append({"row": row_number, "value": raw})
            elif not DUBAI_BOUNDS[field][0] <= value <= DUBAI_BOUNDS[field][1]:
                outside_dubai += 1
                examples.append({"row": row_number, "value": raw})
        result[field] = {
            "available": True,
            "missing_count": missing,
            "invalid_numeric_count": invalid,
            "outside_valid_geographic_range_count": out_of_range,
            "outside_approximate_dubai_bounds_count": outside_dubai,
            "review_examples": _examples(examples),
            "approximate_dubai_bounds": DUBAI_BOUNDS[field],
        }

    lat_index = _first_column(header, "latitude")
    lon_index = _first_column(header, "longitude")
    result["coordinate_pairs"] = {"available": lat_index is not None and lon_index is not None}
    if lat_index is not None and lon_index is not None:
        result["coordinate_pairs"]["incomplete_pair_count"] = sum(
            _is_blank(row[lat_index])
            or _is_null(row[lat_index])
            or _is_blank(row[lon_index])
            or _is_null(row[lon_index])
            for row in rows
        )
    return result


def _json_report(
    rows: list[list[str]], header: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in JSON_COLUMNS:
        index = _first_column(header, field)
        if index is None:
            result[field] = {"available": False}
            continue
        checked = parsed = invalid = wrong_shape = 0
        expected_type = "array"
        examples: list[dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=1):
            raw = row[index]
            if _is_blank(raw):
                continue
            checked += 1
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as error:
                invalid += 1
                if len(examples) < 10:
                    examples.append(
                        {"row": row_number, "error": error.msg, "value": raw[:160]}
                    )
                continue
            parsed += 1
            if not isinstance(decoded, list):
                wrong_shape += 1
        result[field] = {
            "available": True,
            "nonblank_cells_checked": checked,
            "parsed_successfully": parsed,
            "parse_failure_count": invalid,
            "not_array_count": wrong_shape,
            "expected_shape": expected_type,
            "failure_examples": examples,
        }
    return result


def _cross_field_report(
    rows: list[list[str]], header: list[str]
) -> dict[str, Any]:
    title_type_mismatch: list[dict[str, Any]] = []
    listing_title_mismatch: list[dict[str, Any]] = []
    missing_text_fields: dict[str, int] = {}
    qualifier_by_listing_type: dict[str, Counter[str]] = defaultdict(Counter)
    listing_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for field in ("title", "description", "long_description", "property_type"):
        missing_text_fields[field] = sum(
            _is_blank(_get(row, header, field)) or _is_null(_get(row, header, field))
            for row in rows
        )

    for row_number, row in enumerate(rows, start=1):
        title = _get(row, header, "title").casefold()
        property_type = _get(row, header, "property_type").strip().casefold()
        listing_type = _get(row, header, "listing_type").strip()
        qualifier = _get(row, header, "price_qualifier").strip()
        status = _get(row, header, "status").strip()
        if listing_type and qualifier:
            qualifier_by_listing_type[listing_type][qualifier] += 1
        if listing_type and status:
            listing_status_counts[listing_type][status] += 1

        title_terms = [
            term for term in PROPERTY_TERMS if re.search(rf"\b{re.escape(term)}\b", title)
        ]
        if property_type and title_terms:
            normalized_type = property_type.replace("_", " ").replace("-", " ")
            matches_type = any(
                term in normalized_type or normalized_type in term
                for term in title_terms
            )
            if not matches_type:
                title_type_mismatch.append(
                    {
                        "row": row_number,
                        "id": _get(row, header, "id"),
                        "title": _get(row, header, "title")[:160],
                        "property_type": _get(row, header, "property_type"),
                        "detected_title_terms": title_terms,
                    }
                )

        normalized_listing = listing_type.casefold()
        sale_words = bool(re.search(r"\b(for sale|sale|buy|purchase)\b", title))
        rent_words = bool(re.search(r"\b(for rent|to rent|rental|rent)\b", title))
        expects_sale = any(token in normalized_listing for token in ("purchase", "sale"))
        expects_rent = any(token in normalized_listing for token in ("rent", "let"))
        if (expects_sale and rent_words) or (expects_rent and sale_words):
            listing_title_mismatch.append(
                {
                    "row": row_number,
                    "id": _get(row, header, "id"),
                    "listing_type": listing_type,
                    "title": _get(row, header, "title")[:160],
                }
            )

    return {
        "missing_text_field_counts": missing_text_fields,
        "property_type_title_term_candidates": {
            "count": len(title_type_mismatch),
            "examples": _examples(title_type_mismatch),
            "interpretation": "A recognized title term differs from property_type; review context before treating it as an error.",
        },
        "listing_type_title_candidates": {
            "count": len(listing_title_mismatch),
            "examples": _examples(listing_title_mismatch),
            "interpretation": "Listing type suggests sale/rent while title wording suggests the opposite.",
        },
        "price_qualifier_by_listing_type": {
            key: dict(value.most_common()) for key, value in sorted(qualifier_by_listing_type.items())
        },
        "status_by_listing_type": {
            key: dict(value.most_common()) for key, value in sorted(listing_status_counts.items())
        },
    }


def _text_report(
    rows: list[list[str]], header: list[str]
) -> dict[str, Any]:
    descriptions = [
        field
        for field in ("description", "long_description", "introtext")
        if _first_column(header, field) is not None
    ]
    result: dict[str, Any] = {"html_and_escaping_by_column": {}}
    for field in descriptions:
        index = _first_column(header, field)
        values = [(number, row[index]) for number, row in enumerate(rows, start=1)]
        html_rows = [
            {"row": number, "value": value[:160]}
            for number, value in values
            if HTML_TAG_RE.search(value)
        ]
        escaping = {}
        for label, pattern in ESCAPING_PATTERNS.items():
            samples = [
                {"row": number, "value": value[:160]}
                for number, value in values
                if pattern.search(value)
            ]
            escaping[label] = {"count": len(samples), "examples": _examples(samples)}
        result["html_and_escaping_by_column"][field] = {
            "rows_containing_html_tags": len(html_rows),
            "html_tag_examples": _examples(html_rows),
            "escaping_indicators": escaping,
        }
    return result


def _url_report(rows: list[list[str]], header: list[str]) -> dict[str, Any]:
    inspected = 0
    malformed_count = 0
    malformed: list[dict[str, Any]] = []
    url_columns = [
        name
        for name in header
        if name == "link" or name.casefold().endswith("_url")
    ]

    def inspect(value: Any, row_number: int, location: str) -> None:
        nonlocal inspected, malformed_count
        if not isinstance(value, str) or _is_blank(value) or _is_null(value):
            return
        inspected += 1
        candidate = value.strip()
        reason = None
        if any(character.isspace() or ord(character) < 32 for character in candidate):
            reason = "contains whitespace or control characters"
        else:
            try:
                parsed = urlsplit(candidate)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    reason = "missing valid http(s) scheme or host"
                elif parsed.hostname is None:
                    reason = "host could not be parsed"
            except ValueError:
                reason = "URL parsing failed"
        if reason:
            malformed_count += 1
            if len(malformed) < 100:
                malformed.append(
                    {
                        "row": row_number,
                        "column": location,
                        "reason": reason,
                        "value": candidate[:200],
                    }
                )

    for row_number, row in enumerate(rows, start=1):
        for column in url_columns:
            inspect(_get(row, header, column), row_number, column)
        for field in JSON_COLUMNS:
            raw = _get(row, header, field)
            if _is_blank(raw):
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError:
                continue

            def walk(value: Any, path: str) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        child_path = f"{path}.{key}"
                        if key.casefold() in {"url", "src_url", "href"} or key.casefold().endswith("_url"):
                            inspect(child, row_number, child_path)
                        else:
                            walk(child, child_path)
                elif isinstance(value, list):
                    for child_index, child in enumerate(value):
                        walk(child, f"{path}[{child_index}]")

            walk(decoded, field)
    return {
        "url_values_inspected": inspected,
        "malformed_candidate_count": malformed_count,
        "examples_limited_to": 100,
        "examples": malformed,
    }


def _additional_checks(
    rows: list[list[str]], header: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    timestamp_fields = [
        field
        for field in ("scraped_at", "updated_at", "enriched_at")
        if _first_column(header, field) is not None
    ]
    timestamp_results = {}
    for field in timestamp_fields:
        index = _first_column(header, field)
        invalid = []
        missing = 0
        for row_number, row in enumerate(rows, start=1):
            value = row[index].strip()
            if not value or _is_null(value):
                missing += 1
                continue
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                invalid.append({"row": row_number, "value": value})
        timestamp_results[field] = {
            "missing_count": missing,
            "invalid_timestamp_count": len(invalid),
            "examples": _examples(invalid),
        }
    result["timestamps"] = timestamp_results

    image_index = _first_column(header, "images")
    count_index = _first_column(header, "image_count")
    if image_index is not None and count_index is not None:
        mismatches = []
        unavailable = 0
        for row_number, row in enumerate(rows, start=1):
            try:
                images = json.loads(row[image_index])
                declared_count = _numeric(row[count_index])
            except json.JSONDecodeError:
                unavailable += 1
                continue
            if not isinstance(images, list) or declared_count is None:
                unavailable += 1
                continue
            if len(images) != declared_count:
                mismatches.append(
                    {
                        "row": row_number,
                        "id": _get(row, header, "id"),
                        "image_count": declared_count,
                        "images_array_length": len(images),
                    }
                )
        result["image_count_vs_images_length"] = {
            "comparable_rows": len(rows) - unavailable,
            "unavailable_rows": unavailable,
            "mismatch_count": len(mismatches),
            "examples": _examples(mismatches),
        }
    return result


def profile_csv(input_path: Path) -> dict[str, Any]:
    header, rows, structure = _load_csv(input_path)
    column_labels = structure.pop("column_labels")
    return {
        "input_file": str(input_path),
        "structure": structure,
        "column_names": header,
        "columns": _column_profile(header, rows, column_labels),
        "duplicate_listings": {
            field: _duplicate_report(rows, header, field)
            for field in ("id", "crm_id", "strapi_id")
        },
        "distributions": {
            field: _distribution(rows, header, field) for field in CATEGORY_COLUMNS
        },
        "numeric_fields": {
            field: _numeric_report(rows, header, field) for field in NUMERIC_COLUMNS
        },
        "coordinates": _coordinate_report(rows, header),
        "json_columns": _json_report(rows, header),
        "cross_field_candidates": _cross_field_report(rows, header),
        "description_text": _text_report(rows, header),
        "urls": _url_report(rows, header),
        "additional_checks": _additional_checks(rows, header),
    }


def _markdown_report(profile: dict[str, Any]) -> str:
    structure = profile["structure"]
    lines = [
        "# Raw property listings data profile",
        "",
        f"- Input: `{profile['input_file']}`",
        f"- Rows: **{structure['record_count']:,}**",
        f"- Columns: **{structure['column_count']}**",
        f"- Duplicate header names: **{len(structure['duplicate_column_names'])}**",
        f"- Rows with too few/many fields: **{structure['rows_with_too_few_fields']} / {structure['rows_with_too_many_fields']}**",
        "",
        "> This report describes the CSV as read. It does not clean, infer, or modify source values.",
        "",
        "## Column inventory and completeness",
        "",
        "| Column | Observed types | Blank | Null | Missing | Placeholder values | Distinct nonmissing |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for label, column in profile["columns"].items():
        types = ", ".join(f"{key} ({count})" for key, count in column["types"].items())
        placeholders = ", ".join(
            f"{key} ({count})" for key, count in column["placeholder_counts"].items()
        ) or "—"
        lines.append(
            f"| `{label}` | {types or '—'} | {column['blank_count']} | "
            f"{column['null_count']} | {column['missing_count']} | {placeholders} | "
            f"{column['distinct_nonmissing_count']} |"
        )

    lines.extend(["", "## Duplicate identifiers", ""])
    for field, details in profile["duplicate_listings"].items():
        lines.append(
            f"- **{field}:** {details.get('duplicate_value_count', 0)} repeated values; "
            f"{details.get('duplicate_row_count', 0)} rows in repeated groups "
            f"({details.get('extra_occurrence_count', 0)} extra occurrences)."
        )
        if details.get("duplicate_groups"):
            lines.append(f"  - Examples: `{details['duplicate_groups'][:5]}`")

    lines.extend(["", "## Categorical distributions", ""])
    for field, distribution in profile["distributions"].items():
        lines.append(f"### {field}")
        if not distribution["available"]:
            lines.append("Column not present.")
            continue
        lines.append(
            f"Missing: {distribution['missing_count']}; "
            f"distinct values: {distribution['distinct_value_count']}."
        )
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|---|---:|")
        for item in distribution["counts"][:50]:
            value = item["value"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {value} | {item['count']} |")
        if len(distribution["counts"]) > 50:
            lines.append(f"| … ({len(distribution['counts']) - 50} more values in JSON report) | … |")
        lines.append("")

    lines.extend(["## Numeric fields", ""])
    lines.extend(
        [
            "| Field | Valid numeric | Missing | Invalid | Min | Median | Max | Suspicious candidates |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for field, details in profile["numeric_fields"].items():
        if not details["available"]:
            lines.append(f"| `{field}` | — | — | — | — | — | — | column not present |")
            continue
        candidates = ", ".join(
            f"{name}: {item['count']}" for name, item in details["suspicious_candidates"].items()
        ) or "none"
        lines.append(
            f"| `{field}` | {details['valid_numeric_count']} | {details['missing_count']} | "
            f"{details['invalid_numeric_count']} | {details['minimum']} | "
            f"{details['median']} | {details['maximum']} | {candidates} |"
        )
    lines.append("")
    lines.append(
        "Suspicious-value limits are broad review thresholds, not validation rules: "
        f"`{NUMERIC_COLUMNS}`."
    )

    lines.extend(["", "## Coordinates", ""])
    for field in ("latitude", "longitude"):
        details = profile["coordinates"].get(field, {})
        if details.get("available"):
            lines.append(
                f"- **{field}:** missing {details['missing_count']}, invalid "
                f"{details['invalid_numeric_count']}, outside global bounds "
                f"{details['outside_valid_geographic_range_count']}, outside approximate "
                f"Dubai bounds {details['outside_approximate_dubai_bounds_count']}."
            )
    pair_details = profile["coordinates"]["coordinate_pairs"]
    if pair_details.get("available"):
        lines.append(f"- Incomplete latitude/longitude pairs: {pair_details['incomplete_pair_count']}.")

    lines.extend(["", "## JSON-encoded columns", ""])
    for field, details in profile["json_columns"].items():
        if details.get("available"):
            lines.append(
                f"- **{field}:** parsed {details['parsed_successfully']}/"
                f"{details['nonblank_cells_checked']} nonblank cells; "
                f"{details['parse_failure_count']} parse failures; "
                f"{details['not_array_count']} parsed values were not arrays."
            )
            if details["failure_examples"]:
                lines.append(f"  - Failure examples: `{details['failure_examples'][:3]}`")
        else:
            lines.append(f"- **{field}:** column not present.")

    lines.extend(["", "## Cross-field review candidates", ""])
    cross = profile["cross_field_candidates"]
    lines.append(f"- Missing text fields: `{cross['missing_text_field_counts']}`")
    for field in ("property_type_title_term_candidates", "listing_type_title_candidates"):
        item = cross[field]
        lines.append(f"- **{field}:** {item['count']}; examples: `{item['examples'][:5]}`")
        lines.append(f"  - {item['interpretation']}")
    lines.append(f"- Qualifier distribution by listing type: `{cross['price_qualifier_by_listing_type']}`")
    lines.append(f"- Status distribution by listing type: `{cross['status_by_listing_type']}`")

    lines.extend(["", "## Description markup and escaping", ""])
    for field, details in profile["description_text"]["html_and_escaping_by_column"].items():
        lines.append(
            f"- **{field}:** HTML tags in {details['rows_containing_html_tags']} rows; "
            f"escaping indicators: "
            f"`{ {key: value['count'] for key, value in details['escaping_indicators'].items()} }`."
        )

    lines.extend(["", "## URL and related consistency checks", ""])
    lines.append(
        f"- URL values inspected: {profile['urls']['url_values_inspected']}; "
        f"malformed candidates found: {profile['urls']['malformed_candidate_count']}."
    )
    if profile["urls"]["examples"]:
        lines.append(f"- Examples: `{profile['urls']['examples'][:10]}`")
    checks = profile["additional_checks"]
    for field, details in checks.get("timestamps", {}).items():
        lines.append(
            f"- **{field}:** {details['missing_count']} missing; "
            f"{details['invalid_timestamp_count']} invalid timestamp strings."
        )
    image_check = checks.get("image_count_vs_images_length")
    if image_check:
        lines.append(
            f"- **image_count vs images array:** {image_check['mismatch_count']} mismatches "
            f"across {image_check['comparable_rows']} comparable rows; "
            f"{image_check['unavailable_rows']} could not be compared."
        )
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- The file is CSV text: observed types are inferred from cell representations, not original JSON types.",
            "- Placeholder, outlier, title/type, and URL findings are candidates for review, not automatic errors.",
            "- Approximate Dubai coordinate bounds are broad screening bounds; confirm the intended geographic scope before using them as a validation rule.",
            "- Blank CSV cells and the literal text `null` are reported separately.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile a raw property-listings CSV without changing it."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input listings CSV")
    parser.add_argument(
        "--report",
        type=Path,
        help="Human-readable Markdown report (default: <input>_profile.md)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Machine-readable JSON summary (default: <input>_profile.json)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parser().parse_args(argv)
    report_path = args.report or args.input.with_name(f"{args.input.stem}_profile.md")
    json_path = args.json_output or args.input.with_name(f"{args.input.stem}_profile.json")

    try:
        input_resolved = args.input.resolve()
        if input_resolved in {report_path.resolve(), json_path.resolve()}:
            raise ValueError("report output paths must not overwrite the input CSV")
        if report_path.resolve() == json_path.resolve():
            raise ValueError("Markdown and JSON report paths must be different")
        profile = profile_csv(args.input)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(_markdown_report(profile), encoding="utf-8")
        json_path.write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, csv.Error, ValueError) as error:
        LOGGER.error("Profiling failed: %s", error)
        return 1

    structure = profile["structure"]
    LOGGER.info("Rows: %d; columns: %d", structure["record_count"], structure["column_count"])
    LOGGER.info(
        "Duplicate headers: %d; malformed-width rows: %d",
        len(structure["duplicate_column_names"]),
        structure["rows_with_too_few_fields"] + structure["rows_with_too_many_fields"],
    )
    LOGGER.info("Markdown report: %s", report_path)
    LOGGER.info("JSON summary: %s", json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
