#!/usr/bin/env python3
"""Qualify Locus traffic control against two live engine processes.

This script is intentionally opt-in. It never calls a fixture and only reports
an engine as exercised when that engine's own prompt and generation token
counters increase during the run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from live_engine_conformance import METRICS, metric_samples, metric_value


LOCUS_SAMPLE = re.compile(
    r'^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\d+)?$'
)
LABEL = re.compile(r'(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')
ALLOWED_LOCUS_LABELS = {"class", "tenant", "outcome", "reason"}


@dataclass(frozen=True)
class Tenant:
    name: str
    api_key: str


@dataclass(frozen=True)
class Engine:
    runtime: str
    base_url: str
    model: str


def parse_tenant(value: str) -> Tenant:
    name, separator, variable = value.partition("=")
    if not separator or not name or not variable:
        raise argparse.ArgumentTypeError("tenant must be NAME=API_KEY_ENV")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
        raise argparse.ArgumentTypeError(
            "tenant name must match the configured bounded Prometheus label"
        )
    api_key = os.environ.get(variable)
    if not api_key:
        raise argparse.ArgumentTypeError(
            f"tenant credential environment variable is missing or empty: {variable}"
        )
    return Tenant(name=name, api_key=api_key)


def parse_engine(value: str) -> Engine:
    parts = value.split(",", 2)
    if len(parts) != 3 or parts[0] not in METRICS or not parts[1] or not parts[2]:
        raise argparse.ArgumentTypeError(
            "engine must be RUNTIME,BASE_URL,UPSTREAM_MODEL with runtime sglang or vllm"
        )
    return Engine(runtime=parts[0], base_url=parts[1].rstrip("/"), model=parts[2])


def request(
    method: str,
    url: str,
    *,
    api_key: str | None = None,
    body: dict[str, Any] | None = None,
    timeout: float,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    encoded = None if body is None else json.dumps(body).encode()
    call = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def scrape_text(url: str, timeout: float, max_bytes: int) -> str:
    call = urllib.request.Request(url, headers={"Accept": "text/plain"}, method="GET")
    with urllib.request.urlopen(call, timeout=timeout) as response:
        body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise AssertionError(f"metrics response exceeded {max_bytes} bytes: {url}")
    return body.decode(errors="strict")


def engine_counters(
    engine: Engine, timeout: float, max_bytes: int, max_samples: int
) -> dict[str, Any]:
    text = scrape_text(f"{engine.base_url}/metrics", timeout, max_bytes)
    samples = metric_samples(text, engine.model, max_samples)
    result: dict[str, Any] = {}
    for logical in ("prompt_tokens", "generation_tokens"):
        metric, value = metric_value(samples, METRICS[engine.runtime][logical])
        if metric is None or value is None:
            raise AssertionError(
                f"{engine.runtime} engine omitted {logical} for model {engine.model}"
            )
        result[logical] = {"metric": metric, "value": value}
    return result


def engine_counter_deltas(
    engine: Engine,
    before: dict[str, Any],
    after: dict[str, Any],
    phase: str,
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for logical in ("prompt_tokens", "generation_tokens"):
        before_value = before[logical]["value"]
        after_value = after[logical]["value"]
        if after_value < before_value:
            raise AssertionError(
                f"{engine.runtime} {logical} counter reset during {phase}"
            )
        deltas[logical] = after_value - before_value
    return deltas


def parse_locus_metrics(text: str, max_samples: int) -> dict[str, Any]:
    samples = 0
    series: set[tuple[str, tuple[tuple[str, str], ...]]] = set()
    forbidden_labels: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        samples += 1
        if samples > max_samples:
            raise AssertionError(f"Locus metrics exceeded {max_samples} samples")
        match = LOCUS_SAMPLE.match(line)
        if not match:
            raise AssertionError(f"invalid Locus metric on line {line_number}")
        name, labels_text, value_text = match.groups()
        try:
            value = float(value_text)
        except ValueError as error:
            raise AssertionError(f"invalid metric value on line {line_number}") from error
        if not math.isfinite(value):
            raise AssertionError(f"non-finite metric value on line {line_number}")
        labels = tuple(sorted(LABEL.findall(labels_text or "")))
        if name.startswith("locus_"):
            forbidden_labels.update(key for key, _ in labels if key not in ALLOWED_LOCUS_LABELS)
        series.add((name, labels))
    if forbidden_labels:
        raise AssertionError(
            f"Locus export contains unbounded or undocumented labels: {sorted(forbidden_labels)}"
        )
    return {"samples": samples, "series": len(series)}


def locus_metric_value(
    text: str, metric_name: str, expected_labels: dict[str, str]
) -> float:
    total = 0.0
    for raw in text.splitlines():
        match = LOCUS_SAMPLE.match(raw.strip())
        if not match or match.group(1) != metric_name:
            continue
        labels = dict(LABEL.findall(match.group(2) or ""))
        if all(labels.get(key) == value for key, value in expected_labels.items()):
            total += float(match.group(3))
    return total


def one_completion(
    base_url: str,
    model: str,
    tenant: Tenant,
    index: int,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    status, raw = request(
        "POST",
        f"{base_url}/v1/responses",
        api_key=tenant.api_key,
        body={
            "model": model,
            "input": f"Return the integer {index} and one short sentence.",
            "max_output_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"raw": raw.decode(errors="replace")[:512]}
    usage = payload.get("usage") if isinstance(payload, dict) else None
    return {
        "tenant": tenant.name,
        "status": status,
        "latency_seconds": elapsed,
        "input_tokens": (usage or {}).get("input_tokens", 0),
        "output_tokens": (usage or {}).get("output_tokens", 0),
        "error_code": ((payload.get("error") or {}).get("code") if isinstance(payload, dict) else None),
    }


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def deadline_probe(args: argparse.Namespace, tenant: Tenant) -> dict[str, Any]:
    status, raw = request(
        "POST",
        f"{args.base_url}/v1/responses",
        api_key=tenant.api_key,
        body={
            "model": args.model,
            "input": "Generate a response that exercises the request deadline.",
            "max_output_tokens": args.max_tokens,
        },
        timeout=args.timeout,
        extra_headers={"x-request-timeout-ms": str(args.deadline_probe_millis)},
    )
    payload = json.loads(raw)
    code = (payload.get("error") or {}).get("code")
    if status != 408 or code != "deadline_exceeded":
        raise AssertionError(
            f"deadline probe expected HTTP 408/deadline_exceeded, got {status}/{code}"
        )
    return {"status": status, "error_code": code}


def cancellation_probe(
    args: argparse.Namespace, tenant: Tenant, metrics_before: str
) -> dict[str, Any]:
    labels = {"tenant": tenant.name, "reason": "client_cancelled"}
    before = locus_metric_value(
        metrics_before, "locus_request_terminations_total", labels
    )
    body = json.dumps(
        {
            "model": args.model,
            "input": "Generate a long response for client cancellation qualification.",
            "max_output_tokens": max(args.max_tokens, 256),
            "stream": True,
        }
    ).encode()
    call = urllib.request.Request(
        f"{args.base_url}/v1/responses",
        data=body,
        headers={
            "Authorization": f"Bearer {tenant.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    response = urllib.request.urlopen(call, timeout=args.timeout)
    saw_event = False
    try:
        while True:
            line = response.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                saw_event = True
                break
    finally:
        response.close()
    if not saw_event:
        raise AssertionError("cancellation probe received no stream event")

    deadline = time.monotonic() + min(args.timeout, 10.0)
    after = before
    while time.monotonic() < deadline:
        text = scrape_text(
            f"{args.base_url}/metrics", args.timeout, args.max_metrics_bytes
        )
        after = locus_metric_value(text, "locus_request_terminations_total", labels)
        if after > before:
            break
        time.sleep(0.1)
    if after <= before:
        raise AssertionError(
            "client stream close did not advance the bounded cancellation counter"
        )
    return {"stream_event_observed": True, "counter_delta": after - before}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="live Locus base URL")
    parser.add_argument("--model", required=True, help="Locus public model alias")
    parser.add_argument(
        "--tenant",
        action="append",
        type=parse_tenant,
        required=True,
        help="repeat NAME=API_KEY_ENV for each configured tenant",
    )
    parser.add_argument(
        "--engine",
        action="append",
        type=parse_engine,
        required=True,
        help="repeat exactly twice: RUNTIME,BASE_URL,UPSTREAM_MODEL",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-p95-seconds", type=float, default=30.0)
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    parser.add_argument(
        "--overload-requests",
        type=int,
        default=0,
        help="optional burst size that must produce bounded HTTP 429 overload shedding",
    )
    parser.add_argument("--overload-concurrency", type=int, default=64)
    parser.add_argument(
        "--overload-tenant",
        help="tenant to use for the overload burst (defaults to the last --tenant)",
    )
    parser.add_argument("--overload-max-tokens", type=int, default=1024)
    parser.add_argument("--deadline-probe-millis", type=int, default=1)
    parser.add_argument("--metrics-settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--background-settle-seconds",
        type=float,
        default=1.0,
        help="quiet window used to reject engine counters moving without Locus load",
    )
    parser.add_argument(
        "--max-background-token-delta",
        type=float,
        default=0.0,
        help="maximum prompt or generation counter movement allowed in the quiet window",
    )
    parser.add_argument("--max-metrics-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-metric-samples", type=int, default=20_000)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if len(args.engine) != 2:
        parser.error("exactly two --engine values are required")
    if len({engine.base_url for engine in args.engine}) != 2:
        parser.error("the two --engine values must reference distinct runtime endpoints")
    if len(args.tenant) < 2:
        parser.error("at least two --tenant values are required")
    if len({tenant.name for tenant in args.tenant}) != len(args.tenant):
        parser.error("each --tenant name must be unique")
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        parser.error("request, concurrency, and token limits must be greater than zero")
    if (
        args.overload_requests < 0
        or args.overload_concurrency <= 0
        or args.overload_max_tokens <= 0
        or args.background_settle_seconds <= 0
        or args.max_background_token_delta < 0
        or args.timeout <= 0
        or args.max_p95_seconds <= 0
        or not 0 <= args.max_error_ratio <= 1
        or args.deadline_probe_millis <= 0
        or args.metrics_settle_seconds < 0
        or args.max_metrics_bytes <= 0
        or args.max_metric_samples <= 0
    ):
        parser.error("one or more timeout, metric, error-ratio, or overload limits are invalid")

    ready_status, ready_raw = request(
        "GET", f"{args.base_url}/readyz", timeout=args.timeout
    )
    if ready_status != 200:
        raise AssertionError(
            f"Locus readiness returned {ready_status}: {ready_raw.decode(errors='replace')}"
        )
    readiness = json.loads(ready_raw)
    if readiness.get("ready_targets", 0) < 2:
        raise AssertionError("Locus readiness did not report at least two ready targets")

    locus_before_text = scrape_text(
        f"{args.base_url}/metrics", args.timeout, args.max_metrics_bytes
    )
    locus_before = parse_locus_metrics(locus_before_text, args.max_metric_samples)
    engine_quiet_start = [
        engine_counters(
            engine, args.timeout, args.max_metrics_bytes, args.max_metric_samples
        )
        for engine in args.engine
    ]
    time.sleep(args.background_settle_seconds)
    engine_before = [
        engine_counters(
            engine, args.timeout, args.max_metrics_bytes, args.max_metric_samples
        )
        for engine in args.engine
    ]
    background_evidence = []
    for engine, before, after in zip(
        args.engine, engine_quiet_start, engine_before, strict=True
    ):
        deltas = engine_counter_deltas(engine, before, after, "quiet window")
        if any(
            delta > args.max_background_token_delta for delta in deltas.values()
        ):
            raise AssertionError(
                f"{engine.runtime} counters moved during the quiet attribution window"
            )
        background_evidence.append(
            {"runtime": engine.runtime, "model": engine.model, "counter_deltas": deltas}
        )

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                one_completion,
                args.base_url,
                args.model,
                args.tenant[index % len(args.tenant)],
                index,
                args.max_tokens,
                args.timeout,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    elapsed = time.perf_counter() - started
    time.sleep(args.metrics_settle_seconds)
    engine_after = [
        engine_counters(
            engine, args.timeout, args.max_metrics_bytes, args.max_metric_samples
        )
        for engine in args.engine
    ]
    engine_evidence = []
    for engine, before, after in zip(args.engine, engine_before, engine_after, strict=True):
        deltas = engine_counter_deltas(engine, before, after, "normal load")
        if any(delta <= 0 for delta in deltas.values()):
            raise AssertionError(
                f"{engine.runtime} at {engine.base_url} did not execute normal live traffic"
            )
        engine_evidence.append(
            {
                "runtime": engine.runtime,
                "model": engine.model,
                "counter_deltas": deltas,
            }
        )

    overload_results: list[dict[str, Any]] = []
    if args.overload_requests > 0:
        overload_tenant = next(
            (
                tenant
                for tenant in args.tenant
                if tenant.name == (args.overload_tenant or args.tenant[-1].name)
            ),
            None,
        )
        if overload_tenant is None:
            parser.error("--overload-tenant must name one of the configured --tenant values")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.overload_concurrency
        ) as executor:
            futures = [
                executor.submit(
                    one_completion,
                    args.base_url,
                    args.model,
                    overload_tenant,
                    args.requests + index,
                    args.overload_max_tokens,
                    args.timeout,
                )
                for index in range(args.overload_requests)
            ]
            overload_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        shed = [
            result
            for result in overload_results
            if result["status"] == 429 and result["error_code"] == "overloaded"
        ]
        if not shed:
            raise AssertionError(
                "overload burst did not exercise HTTP 429/overloaded degradation"
            )
    locus_after_text = scrape_text(
        f"{args.base_url}/metrics", args.timeout, args.max_metrics_bytes
    )
    locus_after = parse_locus_metrics(locus_after_text, args.max_metric_samples)
    cancellation = cancellation_probe(args, args.tenant[0], locus_after_text)
    deadline = deadline_probe(args, args.tenant[0])

    successful = [result for result in results if result["status"] == 200]
    failures = [result for result in results if result["status"] != 200]
    error_ratio = len(failures) / len(results)
    p95 = percentile([result["latency_seconds"] for result in successful], 0.95)
    if not successful:
        raise AssertionError("load produced no successful inference requests")
    if error_ratio > args.max_error_ratio:
        raise AssertionError(
            f"error ratio {error_ratio:.4f} exceeded {args.max_error_ratio:.4f}"
        )
    if p95 > args.max_p95_seconds:
        raise AssertionError(
            f"p95 latency {p95:.3f}s exceeded {args.max_p95_seconds:.3f}s"
        )
    per_tenant = {}
    for tenant in args.tenant:
        tenant_results = [result for result in results if result["tenant"] == tenant.name]
        tenant_success = sum(result["status"] == 200 for result in tenant_results)
        if tenant_success == 0:
            raise AssertionError(f"tenant {tenant.name} had no successful request")
        per_tenant[tenant.name] = {
            "requests": len(tenant_results),
            "successful": tenant_success,
            "p95_seconds": percentile(
                [
                    result["latency_seconds"]
                    for result in tenant_results
                    if result["status"] == 200
                ],
                0.95,
            ),
        }

    output = {
        "evidence_level": "live_dual_engine",
        "locus": {
            "model": args.model,
            "ready_targets": readiness["ready_targets"],
            "metrics_before": locus_before,
            "metrics_after": locus_after,
        },
        "load": {
            "requests": len(results),
            "successful": len(successful),
            "failures": len(failures),
            "error_ratio": error_ratio,
            "elapsed_seconds": elapsed,
            "requests_per_second": len(results) / elapsed,
            "p50_seconds": statistics.median(
                result["latency_seconds"] for result in successful
            ),
            "p95_seconds": p95,
            "input_tokens": sum(result["input_tokens"] for result in successful),
            "output_tokens": sum(result["output_tokens"] for result in successful),
            "per_tenant": per_tenant,
            "failure_codes": sorted(
                {
                    f"{result['status']}:{result['error_code']}"
                    for result in failures
                }
            ),
        },
        "overload": {
            "enabled": bool(overload_results),
            "requests": len(overload_results),
            "successful": sum(
                result["status"] == 200 for result in overload_results
            ),
            "shed": sum(
                result["status"] == 429 and result["error_code"] == "overloaded"
                for result in overload_results
            ),
        },
        "deadline_probe": deadline,
        "cancellation_probe": cancellation,
        "background_attribution": {
            "settle_seconds": args.background_settle_seconds,
            "max_token_delta": args.max_background_token_delta,
            "engines": background_evidence,
        },
        "engines": engine_evidence,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, urllib.error.URLError) as error:
        print(f"traffic-control-load: {error}", file=sys.stderr)
        raise SystemExit(1) from error
