"""Wire-level SSE checks using the installed SDKs; no external model calls."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import httpx
from openai import AsyncOpenAI

spec = importlib.util.spec_from_file_location("scicode_streaming", Path(__file__).parents[1] / "src/scicode/gen/streaming.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sse(rows):
    return "".join("data: " + json.dumps(row) + "\n\n" for row in rows) + "data: [DONE]\n\n"


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_preserves_payload_content_usage_and_single_request(self):
        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse([
                {"id": "chatcmpl-one", "object": "chat.completion.chunk", "created": 1, "model": "candidate",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]},
                {"id": "chatcmpl-one", "object": "chat.completion.chunk", "created": 1, "model": "candidate",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}},
            ]))
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            model = SimpleNamespace(api=SimpleNamespace(client=client))
            module.enable_streaming(model)
            module.enable_streaming(model)
            result = await client.chat.completions.create(model="candidate", messages=[{"role": "user", "content": "question"}],
                max_completion_tokens=123, extra_body={"reasoning_effort": "high"})
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["stream"], True)
        self.assertEqual(requests[0]["reasoning_effort"], "high")
        self.assertEqual(requests[0]["max_completion_tokens"], 123)
        self.assertEqual(result.choices[0].message.content, "hello")
        self.assertEqual(result.usage.total_tokens, 16)

    async def test_timeout_is_not_replaced_by_partial_results(self):
        from openai import APITimeoutError
        async def handle(request):
            raise httpx.ReadTimeout("fixture timeout", request=request)
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            module.enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            with self.assertRaises(APITimeoutError):
                await client.chat.completions.create(model="candidate", messages=[])

    async def test_cancellation_propagates(self):
        async def handle(request):
            raise asyncio.CancelledError()
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            module.enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            with self.assertRaises(asyncio.CancelledError):
                await client.chat.completions.create(model="candidate", messages=[])

    async def test_responses_preserves_reasoning_budget_output_and_usage(self):
        requests = []
        response = {"id": "resp-one", "object": "response", "created_at": 1, "model": "candidate",
                    "status": "completed", "output": [{"id": "msg-one", "type": "message",
                        "role": "assistant", "status": "completed", "content": [
                            {"type": "output_text", "text": "hello", "annotations": []}]}],
                    "usage": {"input_tokens": 11, "output_tokens": 5, "total_tokens": 16,
                              "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens_details": {"reasoning_tokens": 2}}}
        async def handle(request):
            requests.append(json.loads(request.content))
            rows = [{"type": "response.created", "sequence_number": 0,
                     "response": {**response, "status": "in_progress", "output": []}},
                    {"type": "response.completed", "sequence_number": 1, "response": response}]
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(rows))
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            module.enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            result = await client.responses.create(model="candidate", input="question",
                                                   max_output_tokens=123, reasoning={"effort": "high"})
        self.assertIs(requests[0]["stream"], True)
        self.assertEqual(requests[0]["max_output_tokens"], 123)
        self.assertEqual(requests[0]["reasoning"], {"effort": "high"})
        self.assertEqual(result.output_text, "hello")
        self.assertEqual(result.usage.total_tokens, 16)

    async def test_anthropic_preserves_thinking_budget_output_and_usage(self):
        from anthropic import AsyncAnthropic
        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            rows = [
                {"type": "message_start", "message": {"id": "msg-one", "type": "message",
                    "role": "assistant", "content": [], "model": "candidate", "stop_reason": None,
                    "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 0}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                 "usage": {"output_tokens": 5}},
                {"type": "message_stop"},
            ]
            payload = "".join(f"event: {row['type']}\ndata: {json.dumps(row)}\n\n" for row in rows)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=payload)
        async with AsyncAnthropic(api_key="fixture", base_url="https://fixture.test", max_retries=0,
                                  http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            module.enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            result = await client.messages.create(model="candidate", messages=[{"role": "user", "content": "question"}],
                max_tokens=123, thinking={"type": "enabled", "budget_tokens": 100})
        self.assertIs(requests[0]["stream"], True)
        self.assertEqual(requests[0]["max_tokens"], 123)
        self.assertEqual(requests[0]["thinking"]["budget_tokens"], 100)
        self.assertEqual(result.content[0].text, "hello")
        self.assertEqual(result.usage.output_tokens, 5)

    async def test_unfinished_stream_is_an_error(self):
        async def handle(request):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse([
                {"id": "chatcmpl-one", "object": "chat.completion.chunk", "created": 1, "model": "candidate",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "partial"}, "finish_reason": None}]},
            ]))
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            module.enable_streaming(SimpleNamespace(api=SimpleNamespace(client=client)))
            with self.assertRaisesRegex(RuntimeError, "completed choice"):
                await client.chat.completions.create(model="candidate", messages=[])

    async def test_inspect_generate_keeps_provider_conversion_and_usage(self):
        from inspect_ai.model import GenerateConfig, get_model

        requests = []
        async def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse([
                {"id": "chatcmpl-one", "object": "chat.completion.chunk", "created": 1, "model": "candidate",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": None}]},
                {"id": "chatcmpl-one", "object": "chat.completion.chunk", "created": 1, "model": "candidate",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}},
            ]))
        model = get_model("openai/candidate", base_url="https://fixture.test/v1", api_key="fixture",
                          responses_api=False)
        await model.api.client.close()
        async with AsyncOpenAI(api_key="fixture", base_url="https://fixture.test/v1", max_retries=0,
                               http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle))) as client:
            model.api.client = client
            module.enable_streaming(model)
            output = await model.generate("question", config=GenerateConfig(max_tokens=123, max_retries=0))
        self.assertEqual(output.completion, "hello")
        self.assertEqual(output.usage.total_tokens, 16)
        self.assertEqual(output.stop_reason, "stop")
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["stream"], True)


if __name__ == "__main__":
    unittest.main()
