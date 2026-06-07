import json

from autotune.parser import parse_report


def test_parse_json_report(tmp_path):
    report = {
        "throughput_tokens_per_sec": 330.5,
        "latency_ms": 160.0,
        "runtime_sec": 42.0,
        "failure_rate": 0.0,
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))

    metrics = parse_report(path)

    assert metrics["throughput_tokens_per_sec"] == 330.5
    assert metrics["latency_ms"] == 160.0
    assert metrics["runtime_sec"] == 42.0
    assert metrics["failure_rate"] == 0.0


def test_parse_json_report_with_aliases_and_nesting(tmp_path):
    report = {"results": {"tokens_per_second": 120, "mean_latency_ms": 90}}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))

    metrics = parse_report(path)

    assert metrics["throughput_tokens_per_sec"] == 120.0
    assert metrics["latency_ms"] == 90.0


def test_parse_text_log_via_regex(tmp_path):
    log = (
        "Starting benchmark...\n"
        "Throughput: 330.5 tokens/sec\n"
        "Latency: 160.2 ms\n"
        "Runtime: 42 sec\n"
        "Failure rate: 0.01\n"
    )
    path = tmp_path / "stdout.log"
    path.write_text(log)

    metrics = parse_report(path)

    assert metrics["throughput_tokens_per_sec"] == 330.5
    assert metrics["latency_ms"] == 160.2
    assert metrics["runtime_sec"] == 42.0
    assert metrics["failure_rate"] == 0.01


def test_parse_missing_metrics_are_none(tmp_path):
    path = tmp_path / "stdout.log"
    path.write_text("nothing useful here")

    metrics = parse_report(path)

    assert metrics == {
        "latency_ms": None,
        "throughput_tokens_per_sec": None,
        "runtime_sec": None,
        "failure_rate": None,
    }
