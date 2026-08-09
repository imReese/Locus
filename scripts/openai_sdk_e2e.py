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
    parser.add_argument("--parser-model", default="locus-parser-test")
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

    parsed = client.responses.create(
        model=args.parser_model,
        input="reason and call the weather tool",
        reasoning={"effort": "high"},
        tools=[
            {
                "type": "function",
                "name": "weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        tool_choice="required",
    )
    reasoning_items = [item for item in parsed.output if item.type == "reasoning"]
    assert len(reasoning_items) == 1, parsed.output
    assert reasoning_items[0].summary[0].text == "checked constraints"
    function_calls = [item for item in parsed.output if item.type == "function_call"]
    assert len(function_calls) == 1, parsed.output
    assert function_calls[0].name == "weather"
    assert json.loads(function_calls[0].arguments) == {"city": "Beijing"}

    parser_reasoning = ""
    parser_tool_name = None
    parser_arguments = ""
    parser_tool_indices: list[int] = []
    parser_finish_reason = None
    parser_chunks = client.chat.completions.create(
        model=args.parser_model,
        messages=[{"role": "user", "content": "call the weather tool"}],
        reasoning_effort="high",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        tool_choice="required",
        stream=True,
    )
    try:
        for chunk in parser_chunks:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            reasoning_delta = getattr(choice.delta, "reasoning_content", None)
            if reasoning_delta:
                parser_reasoning += reasoning_delta
            for tool_call in choice.delta.tool_calls or []:
                parser_tool_indices.append(tool_call.index)
                if tool_call.function and tool_call.function.name:
                    parser_tool_name = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    parser_arguments += tool_call.function.arguments
            if choice.finish_reason:
                parser_finish_reason = choice.finish_reason
    finally:
        parser_chunks.close()
    assert parser_reasoning == "checked constraints", parser_reasoning
    assert parser_tool_name == "weather", parser_tool_name
    assert json.loads(parser_arguments) == {"city": "Beijing"}, parser_arguments
    assert parser_tool_indices and set(parser_tool_indices) == {0}, parser_tool_indices
    assert parser_finish_reason == "tool_calls", parser_finish_reason

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
                "responses": [
                    "json",
                    "sse",
                    "structured_output",
                    "reasoning_tool_parser",
                    "error",
                    "cancel",
                ],
                "chat_completions": ["json", "sse", "reasoning_tool_parser_sse"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
