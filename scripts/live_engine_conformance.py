#!/usr/bin/env python3
"""Run opt-in conformance checks against a live SGLang or vLLM server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


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
    parser.add_argument("--json-schema")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")

    health_url = args.base_url.rstrip("/") + "/" + args.health_path.lstrip("/")
    with open_request("GET", health_url, args.api_key, timeout=args.timeout) as response:
        response.read()
    result = {
        "schema_version": "locus.live-engine-conformance.v1",
        "status": "passed",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "runtime": args.runtime,
        "model": args.model,
        "pretokenized_prompt_tokens": len(args.prompt_token_ids),
        "health": "passed",
        "stream": normal_stream(args),
        "cancellation": cancellation_probe(args),
        "structured_output_requested": bool(args.json_schema),
        "claim_boundary": (
            "This is live HTTP/runtime evidence for the configured endpoint; it does not "
            "establish GPU performance, state transfer, or cross-runtime semantic equality."
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
