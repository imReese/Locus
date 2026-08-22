#!/usr/bin/env python3
"""Exercise Locus through the official Anthropic Python SDK.

This is protocol and transport conformance against a caller-provided Locus
endpoint. It does not call Anthropic-hosted models.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import anthropic


def call_counts(base_url: str) -> dict[str, int]:
    with urllib.request.urlopen(
        base_url.removesuffix("/") + "/test/call-counts", timeout=5
    ) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--api-key", default="locus-test-key")
    parser.add_argument("--model", default="locus-test")
    parser.add_argument("--parser-model", default="locus-parser-test")
    parser.add_argument(
        "--fixture-counts",
        action="store_true",
        help="verify downstream cancellation with the bundled fixture",
    )
    args = parser.parse_args()
    client = anthropic.Anthropic(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=10.0,
        max_retries=0,
    )
    rejected = anthropic.Anthropic(
        api_key="deliberately-invalid-locus-key",
        base_url=args.base_url,
        timeout=10.0,
        max_retries=0,
    )

    try:
        rejected.messages.create(
            model=args.model,
            max_tokens=8,
            messages=[{"role": "user", "content": "reject"}],
        )
    except anthropic.AuthenticationError as error:
        assert error.status_code == 401
    else:
        raise AssertionError("invalid API key did not raise AuthenticationError")

    message = client.messages.create(
        model=args.model,
        max_tokens=32,
        system="respond concisely",
        messages=[{"role": "user", "content": "respond with JSON"}],
        stop_sequences=["DONE"],
        temperature=0.25,
        top_p=0.9,
        metadata={"user_id": "sdk-e2e"},
    )
    assert message.type == "message"
    assert message.role == "assistant"
    assert message.content[0].type == "text"
    assert message.content[0].text == '{"answer":"ok"}'
    assert message.stop_reason == "end_turn"
    assert message.usage.input_tokens > 0
    assert message.usage.output_tokens > 0
    assert message._request_id

    event_types: list[str] = []
    text = ""
    with client.messages.stream(
        model=args.model,
        max_tokens=32,
        messages=[{"role": "user", "content": "stream JSON"}],
    ) as stream:
        for event in stream:
            event_types.append(event.type)
            if event.type == "content_block_delta" and event.delta.type == "text_delta":
                text += event.delta.text
        final_message = stream.get_final_message()
    assert text == '{"answer":"ok"}'
    assert final_message.content[0].text == text
    assert event_types[0] == "message_start", event_types
    assert event_types[-1] == "message_stop", event_types
    assert "message_delta" in event_types

    tool_message = client.messages.create(
        model=args.parser_model,
        max_tokens=64,
        messages=[{"role": "user", "content": "call the weather tool"}],
        tools=[
            {
                "name": "weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            }
        ],
        tool_choice={"type": "any"},
    )
    tool_blocks = [block for block in tool_message.content if block.type == "tool_use"]
    assert len(tool_blocks) == 1, tool_message.content
    assert tool_blocks[0].name == "weather"
    assert tool_blocks[0].input == {"city": "Beijing"}
    assert tool_message.stop_reason == "tool_use"

    try:
        client.messages.create(
            model="missing-locus-model",
            max_tokens=8,
            messages=[{"role": "user", "content": "fail"}],
        )
    except anthropic.NotFoundError as error:
        assert error.status_code == 404
    else:
        raise AssertionError("unknown model did not raise NotFoundError")

    if args.fixture_counts:
        before = call_counts(args.base_url)["cancel"]
        cancelled = client.messages.create(
            model=args.model,
            max_tokens=32,
            messages=[{"role": "user", "content": "cancel after first delta"}],
            stream=True,
        )
        try:
            for event in cancelled:
                if event.type == "content_block_delta":
                    break
        finally:
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
                "sdk": "anthropic-python",
                "sdk_version": anthropic.__version__,
                "messages": ["json", "sse", "tools", "error", "cancel"],
                "authentication": "invalid_api_key_rejected",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
