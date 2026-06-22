"""Parse CloudAI benchmark reports/logs into a normalized metrics dict.

CloudAI scenarios can emit either a structured JSON report or plain-text
stdout/log output depending on the backend. This parser supports both,
normalizing everything into the metric set Autotune tracks:

    latency_ms, throughput_tokens_per_sec, runtime_sec, failure_rate, ttft_ms
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

METRIC_KEYS = (
    "latency_ms",
    "throughput_tokens_per_sec",
    "runtime_sec",
    "failure_rate",
    "ttft_ms",
)

# Aliases seen across CloudAI backend report formats, mapped to our normalized keys.
_JSON_ALIASES: dict[str, tuple[str, ...]] = {
    "latency_ms": (
        "latency_ms",
        "latency",
        "p50_latency_ms",
        "mean_latency_ms",
        "e2e_latency_ms",
        "mean_e2e_latency_ms",
        "request_latency_ms",
        "mean_request_latency_ms",
    ),
    "throughput_tokens_per_sec": (
        "throughput_tokens_per_sec",
        "throughput",
        "tokens_per_sec",
        "tokens_per_second",
        "request_throughput",
        "output_throughput",
    ),
    "runtime_sec": ("runtime_sec", "runtime", "duration_sec", "elapsed_sec"),
    "failure_rate": ("failure_rate", "error_rate", "failed_ratio"),
    "ttft_ms": (
        "ttft_ms",
        "time_to_first_token_ms",
        "time_to_first_token",
        "mean_ttft_ms",
        "p50_ttft_ms",
    ),
}

# Regex fallbacks for free-text logs, e.g. "Throughput: 330.5 tokens/sec".
_TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "latency_ms": re.compile(r"latency[^0-9\-]*([\d.]+)\s*ms", re.IGNORECASE),
    "throughput_tokens_per_sec": re.compile(
        r"throughput[^0-9\-]*([\d.]+)\s*tokens?\s*/\s*s(ec)?", re.IGNORECASE
    ),
    "runtime_sec": re.compile(r"runtime[^0-9\-]*([\d.]+)\s*s(ec)?", re.IGNORECASE),
    "failure_rate": re.compile(r"failure[ _-]?rate[^0-9\-]*([\d.]+)", re.IGNORECASE),
    "ttft_ms": re.compile(
        r"(ttft|time[ -]?to[ -]?first[ -]?token)[^0-9\-]*([\d.]+)\s*ms",
        re.IGNORECASE,
    ),
}


def parse_report(path: Path | str) -> dict[str, Optional[float]]:
    """Parse a CloudAI report file (JSON or text) into normalized metrics.

    Unknown/missing metrics are returned as None rather than omitted, so
    downstream consumers can rely on a stable key set.
    """
    text = Path(path).read_text()

    metrics: dict[str, Optional[float]] = {key: None for key in METRIC_KEYS}

    data = _try_parse_json(text)
    if data is None:
        data = _try_parse_jsonl(text)
    if data is not None:
        metrics.update(_extract_from_json(data))
    else:
        metrics.update(_extract_from_text(text))

    return metrics


def _try_parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _try_parse_jsonl(text: str) -> Optional[Any]:
    rows: list[Any] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = _try_parse_json(line)
        if parsed is None:
            continue
        rows.append(parsed)
    return _best_metric_row(rows)


def _extract_from_json(data: Any) -> dict[str, Optional[float]]:
    if isinstance(data, list):
        best = _best_metric_row(data)
        return _extract_from_json(best) if best is not None else {}
    return _extract_from_json_object(data)


def _best_metric_row(rows: list[Any]) -> Optional[Any]:
    best: Optional[Any] = None
    best_score = 0
    for row in rows:
        extracted = _extract_from_json_object(row)
        score = sum(value is not None for value in extracted.values())
        if score >= best_score:
            best = row
            best_score = score
    return best if best_score > 0 else None


def _extract_from_json_object(data: Any) -> dict[str, Optional[float]]:
    flat = _flatten(data)
    result: dict[str, Optional[float]] = {}
    for norm_key, aliases in _JSON_ALIASES.items():
        for alias in aliases:
            if alias in flat:
                result[norm_key] = _to_float(flat[alias])
                break
    if "failure_rate" not in result:
        result["failure_rate"] = _failure_rate_from_counts(flat)
    return result


def _extract_from_text(text: str) -> dict[str, Optional[float]]:
    result: dict[str, Optional[float]] = {}
    for norm_key, pattern in _TEXT_PATTERNS.items():
        match = pattern.search(text)
        if match:
            result[norm_key] = _to_float(_first_numeric_group(match))
    return result


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts so 'metrics.latency' is also reachable as 'latency'."""
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            flat[full_key] = value
            flat.setdefault(key, value)
            flat.update(_flatten(value, full_key))
    return flat


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_numeric_group(match: re.Match[str]) -> Optional[str]:
    for group in match.groups():
        if group is not None and _to_float(group) is not None:
            return group
    return None


def _failure_rate_from_counts(flat: dict[str, Any]) -> Optional[float]:
    completed = _to_float(flat.get("completed"))
    total = _to_float(flat.get("num_prompts"))
    if completed is None or total in (None, 0):
        return None
    return max(0.0, min(1.0, 1.0 - completed / total))
