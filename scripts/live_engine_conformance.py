#!/usr/bin/env python3
"""Run opt-in conformance checks against a live SGLang or vLLM server."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


METRIC_LINE = re.compile(
    r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)(?:\s+\d+)?$"
)
MODEL_LABEL = re.compile(
    r'(?:^|,)\s*(model_name|model|served_model_name)="((?:\\.|[^"\\])*)"'
)

METRICS = {
    "sglang": {
        "running": ("sglang:num_running_reqs", "sglang_num_running_reqs"),
        "waiting": ("sglang:num_queue_reqs", "sglang_num_queue_reqs"),
        "kv_usage": (
            "sglang:token_usage",
            "sglang_token_usage",
            "sglang:full_token_usage",
            "sglang_full_token_usage",
        ),
        "decode_rate": ("sglang:gen_throughput", "sglang_gen_throughput"),
        "prompt_tokens": (
            "sglang:prompt_tokens_total",
            "sglang_prompt_tokens_total",
            "sglang:input_tokens_total",
            "sglang_input_tokens_total",
        ),
        "generation_tokens": (
            "sglang:generation_tokens_total",
            "sglang_generation_tokens_total",
            "sglang:output_tokens_total",
            "sglang_output_tokens_total",
        ),
    },
    "vllm": {
        "running": ("vllm:num_requests_running", "vllm_num_requests_running"),
        "waiting": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
        "kv_usage": (
            "vllm:kv_cache_usage_perc",
            "vllm_kv_cache_usage_perc",
            "vllm:gpu_cache_usage_perc",
            "vllm_gpu_cache_usage_perc",
        ),
        "decode_rate": (),
        "prompt_tokens": (
            "vllm:prompt_tokens_total",
            "vllm_prompt_tokens_total",
            "vllm:prompt_tokens",
            "vllm_prompt_tokens",
        ),
        "generation_tokens": (
            "vllm:generation_tokens_total",
            "vllm_generation_tokens_total",
            "vllm:generation_tokens",
            "vllm_generation_tokens",
        ),
    },
}


def headers(api_key: str | None) -> dict[str, str]:
    result = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def open_request(
    method: str,
    url: str,
    api_key: str | None,
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
):
    encoded = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=headers(api_key),
        method=method,
    )
    try:
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{method} {url} returned {error.code}: {detail}") from error


def sse_data(response) -> Iterator[str]:
    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode(errors="strict").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def completion_body(args: argparse.Namespace, request_id: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt_token_ids,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
    }
    if args.runtime == "sglang":
        body["rid"] = request_id
    else:
        body["request_id"] = request_id
        body["add_special_tokens"] = False
    if args.json_schema:
        schema = json.loads(Path(args.json_schema).read_text())
        if args.runtime == "sglang":
            body["json_schema"] = json.dumps(schema, separators=(",", ":"))
        else:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "locus_live_conformance",
                    "schema": schema,
                    "strict": True,
                },
            }
    return body


def metric_samples(text: str, model: str, max_samples: int) -> dict[str, list[float]]:
    samples: dict[str, list[float]] = {}
    observed = 0
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        observed += 1
        if observed > max_samples:
            raise AssertionError(f"metrics exceeded {max_samples} samples")
        match = METRIC_LINE.match(line)
        if not match:
            raise AssertionError(f"invalid Prometheus sample on line {line_number}")
        name, labels, value_text = match.groups()
        model_labels = MODEL_LABEL.findall(labels or "")
        if model_labels and not any(value == model for _, value in model_labels):
            continue
        try:
            value = float(value_text)
        except ValueError as error:
            raise AssertionError(
                f"invalid Prometheus value on line {line_number}"
            ) from error
        if not math.isfinite(value) or value < 0:
            continue
        samples.setdefault(name, []).append(value)
    return samples


def metric_value(
    samples: dict[str, list[float]], aliases: tuple[str, ...]
) -> tuple[str | None, float | None]:
    for name in aliases:
        values = samples.get(name)
        if values:
            return name, sum(values)
    return None, None


def scrape_metrics(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.metrics_path
    if not endpoint.startswith(("http://", "https://")):
        endpoint = args.base_url.rstrip("/") + "/" + endpoint.lstrip("/")
    request_headers = {"Accept": "text/plain"}
    if args.api_key:
        request_headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(endpoint, headers=request_headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = response.read(args.max_metrics_bytes + 1)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(
            f"GET metrics endpoint returned {error.code}: {detail}"
        ) from error
    if len(raw) > args.max_metrics_bytes:
        raise AssertionError(
            f"metrics response exceeded {args.max_metrics_bytes} bytes"
        )
    samples = metric_samples(raw.decode(errors="strict"), args.model, args.max_metric_samples)
    result: dict[str, Any] = {}
    for logical_name, aliases in METRICS[args.runtime].items():
        metric_name, value = metric_value(samples, aliases)
        result[logical_name] = {"metric": metric_name, "value": value}
    for required in ("running", "waiting", "kv_usage", "prompt_tokens", "generation_tokens"):
        if result[required]["value"] is None:
            raise AssertionError(
                f"metrics endpoint omitted supported {args.runtime} {required} metric"
            )
    if args.runtime == "sglang" and result["decode_rate"]["value"] is None:
        raise AssertionError("metrics endpoint omitted SGLang decode throughput")
    return result


def telemetry_probe(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    before = scrape_metrics(args)
    stream = normal_stream(args)
    after = scrape_metrics(args)
    deltas = {}
    for name in ("prompt_tokens", "generation_tokens"):
        before_value = before[name]["value"]
        after_value = after[name]["value"]
        if before_value is None or after_value is None or after_value < before_value:
            raise AssertionError(f"{name} counter reset or disappeared during probe")
        delta = after_value - before_value
        if delta <= 0:
            raise AssertionError(f"{name} counter did not increase during completion")
        deltas[name] = delta
    return stream, {
        "metrics_endpoint": (
            "absolute_override"
            if args.metrics_path.startswith(("http://", "https://"))
            else "base_url_relative"
        ),
        "before": before,
        "after": after,
        "counter_deltas": deltas,
    }


def normal_stream(args: argparse.Namespace) -> dict[str, Any]:
    request_id = f"locus-live-{uuid.uuid4().hex}"
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    response = open_request(
        "POST",
        endpoint,
        args.api_key,
        completion_body(args, request_id),
        args.timeout,
    )
    chunks = 0
    text_deltas = 0
    finish_reasons: list[str] = []
    usage: dict[str, Any] | None = None
    done = False
    with response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            raise AssertionError(f"expected text/event-stream, got {content_type!r}")
        for data in sse_data(response):
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            if "error" in chunk and chunk["error"]:
                raise AssertionError(f"runtime streamed an error: {chunk['error']}")
            chunks += 1
            for choice in chunk.get("choices", []):
                if choice.get("text"):
                    text_deltas += 1
                if choice.get("finish_reason") is not None:
                    finish_reasons.append(str(choice["finish_reason"]))
            if chunk.get("usage") is not None:
                usage = chunk["usage"]
    if chunks == 0:
        raise AssertionError("completion stream contained no JSON chunks")
    if not done:
        raise AssertionError("completion stream omitted terminal [DONE]")
    if not finish_reasons:
        raise AssertionError("completion stream omitted finish_reason")
    if usage is None:
        raise AssertionError("completion stream omitted requested usage")
    if usage.get("prompt_tokens", 0) <= 0:
        raise AssertionError(f"invalid prompt token usage: {usage}")
    return {
        "request_id": request_id,
        "json_chunks": chunks,
        "text_delta_chunks": text_deltas,
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def cancellation_probe(args: argparse.Namespace) -> dict[str, Any]:
    request_id = f"locus-cancel-{uuid.uuid4().hex}"
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    response = open_request(
        "POST",
        endpoint,
        args.api_key,
        completion_body(args, request_id),
        args.timeout,
    )
    first_event = None
    try:
        for data in sse_data(response):
            if data != "[DONE]":
                first_event = json.loads(data)
                break
    finally:
        response.close()
    if first_event is None:
        raise AssertionError("cancellation probe received no stream event")
    if args.runtime == "sglang":
        abort_endpoint = args.base_url.rstrip("/") + "/abort_request"
        with open_request(
            "POST",
            abort_endpoint,
            args.api_key,
            {"rid": request_id, "abort_all": False},
            args.timeout,
        ) as abort_response:
            abort_response.read()
        evidence = "client_disconnect_and_abort_endpoint_acknowledged"
    else:
        evidence = "client_disconnect_issued_no_public_abort_ack"
    return {"request_id": request_id, "evidence": evidence}


def parse_token_ids(value: str) -> list[int]:
    try:
        token_ids = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("token IDs must be comma-separated integers") from error
    if not token_ids or any(token_id < 0 for token_id in token_ids):
        raise argparse.ArgumentTypeError("at least one non-negative token ID is required")
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--prompt-token-ids", type=parse_token_ids, required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--health-path", default="/health")
    parser.add_argument("--metrics-path", default="/metrics")
    parser.add_argument("--max-metrics-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--max-metric-samples", type=int, default=20_000)
    parser.add_argument("--json-schema")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    if args.max_metrics_bytes <= 0 or args.max_metric_samples <= 0:
        parser.error("metric limits must be greater than zero")

    health_url = args.base_url.rstrip("/") + "/" + args.health_path.lstrip("/")
    with open_request("GET", health_url, args.api_key, timeout=args.timeout) as response:
        response.read()
    stream, telemetry = telemetry_probe(args)
    result = {
        "schema_version": "locus.live-engine-conformance.v2",
        "status": "passed",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": args.runtime,
        "model": args.model,
        "pretokenized_prompt_tokens": len(args.prompt_token_ids),
        "health": "passed",
        "stream": stream,
        "telemetry": telemetry,
        "cancellation": cancellation_probe(args),
        "structured_output_requested": bool(args.json_schema),
        "claim_boundary": (
            "This is live HTTP/runtime and Prometheus-shape evidence for the configured "
            "endpoint; it does not establish calibrated-placement accuracy, GPU performance, "
            "state transfer, or cross-runtime semantic equality."
        ),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 - CLI must emit one clear failure
        print(f"live engine conformance failed: {error}", file=sys.stderr)
        raise
