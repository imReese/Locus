#!/usr/bin/env python3
"""Exercise Locus through the official OpenAI Python SDK.

This script is transport-level conformance evidence. It can target the bundled
mock fixture or a configured Locus deployment; it never calls OpenAI-hosted
models.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import openai


def assert_text(value: str | None) -> None:
    assert value == '{"answer":"ok"}', value


def call_counts(base_url: str) -> dict[str, int]:
    endpoint = base_url.removesuffix("/v1") + "/test/call-counts"
    with urllib.request.urlopen(endpoint, timeout=5) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--api-key", default="locus-test-key")
    parser.add_argument("--model", default="locus-test")
    parser.add_argument(
        "--fixture-counts",
        action="store_true",
        help="verify downstream cancellation with the bundled test fixture",
    )
    args = parser.parse_args()
    client = openai.OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=10.0,
        max_retries=0,
    )
    rejected_client = openai.OpenAI(
        api_key="deliberately-invalid-locus-key",
        base_url=args.base_url,
        timeout=10.0,
        max_retries=0,
    )
    try:
        rejected_client.models.list()
    except openai.AuthenticationError as error:
        assert error.status_code == 401
        assert error.code == "invalid_api_key"
    else:
        raise AssertionError("invalid bearer token did not raise AuthenticationError")

    response = client.responses.create(model=args.model, input="respond with JSON")
    assert_text(response.output_text)

    response_events: list[str] = []
    response_text = ""
    stream = client.responses.create(model=args.model, input="stream JSON", stream=True)
    try:
        for event in stream:
            response_events.append(event.type)
            if event.type == "response.output_text.delta":
                response_text += event.delta
    finally:
        stream.close()
    assert_text(response_text)
    assert "response.created" in response_events, response_events
    assert "response.completed" in response_events, response_events

    completion = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "respond with JSON"}],
    )
    assert_text(completion.choices[0].message.content)

    chat_text = ""
    chat_chunks = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": "stream JSON"}],
        stream=True,
    )
    try:
        for chunk in chat_chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                chat_text += chunk.choices[0].delta.content
    finally:
        chat_chunks.close()
    assert_text(chat_text)

    structured = client.responses.create(
        model=args.model,
        input="return an answer object",
        text={
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        },
    )
    assert json.loads(structured.output_text) == {"answer": "ok"}

    try:
        client.responses.create(model="missing-locus-model", input="fail")
    except openai.NotFoundError as error:
        assert error.status_code == 404
        assert error.code == "model_not_found"
    else:
        raise AssertionError("unknown model did not raise openai.NotFoundError")

    if args.fixture_counts:
        before = call_counts(args.base_url)["cancel"]
        cancelled = client.responses.create(
            model=args.model,
            input="cancel after the first text delta",
            stream=True,
        )
        for event in cancelled:
            if event.type == "response.output_text.delta":
                break
        cancelled.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if call_counts(args.base_url)["cancel"] > before:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("SDK stream close did not reach EngineAdapter.cancel")

    print(
        json.dumps(
            {
                "status": "passed",
                "sdk": "openai-python",
                "sdk_version": openai.__version__,
                "authentication": "invalid_api_key_rejected",
                "responses": ["json", "sse", "structured_output", "error", "cancel"],
                "chat_completions": ["json", "sse"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
