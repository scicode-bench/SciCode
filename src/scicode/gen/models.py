from __future__ import annotations

from functools import partial
import re

from openai import OpenAI
from scicode.utils.log import get_logger
from scicode.gen.token_records import record_response

logger = get_logger("models")


def generate_openai_compatible_response(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    timeout: float = 3600.0,
    repetition_penalty: float | None = None,
    stream: bool = False,
    api_type: str = "chat_completions",
    extra_body: dict | None = None,
    sample_id: str | None = None,
) -> str:
    """Call any OpenAI-compatible API (OpenAI, Bailian, local vLLM, etc.)"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    api_type = (api_type or "chat_completions").lower()
    request_extra_body = dict(extra_body or {})
    if 'gpt' in model and not request_extra_body:
        request_extra_body["reasoning_effort"] = "xhigh"
    if repetition_penalty is not None:
        request_extra_body["repetition_penalty"] = repetition_penalty

    create_kwargs: dict = {
        "model": model,
        "temperature": temperature,
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    if api_type == "responses":
        create_kwargs["input"] = messages
        create_kwargs["max_output_tokens"] = max_tokens
    else:
        create_kwargs["messages"] = messages
        create_kwargs["max_tokens"] = max_tokens
    if request_extra_body:
        create_kwargs["extra_body"] = request_extra_body

    if stream:
        create_kwargs["stream"] = True
        pieces: list[str] = []
        reasoning_pieces: list[str] = []
        stream_usage = None
        if api_type == "responses":
            completion = client.responses.create(**create_kwargs)
            for event in completion:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    pieces.append(getattr(event, "delta", "") or "")
                elif "reasoning" in event_type and event_type.endswith(".delta"):
                    reasoning_pieces.append(getattr(event, "delta", "") or "")
                elif event_type in {"response.completed", "response.incomplete"}:
                    response = getattr(event, "response", None)
                    stream_usage = getattr(response, "usage", None)
        else:
            create_kwargs["stream_options"] = {"include_usage": True}
            completion = client.chat.completions.create(**create_kwargs)
            for chunk in completion:
                if getattr(chunk, "usage", None):
                    stream_usage = chunk.usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if piece:
                    pieces.append(piece)
                r_piece = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if r_piece:
                    reasoning_pieces.append(r_piece)
        text = "".join(pieces)
        reasoning_text = "".join(reasoning_pieces) or None
        record_response(stream_usage, text=text, reasoning_text=reasoning_text, sample_id=sample_id)
        return text

    if api_type == "responses":
        completion = client.responses.create(**create_kwargs)
        text = getattr(completion, "output_text", None)
        reasoning_parts: list[str] = []
        if not text:
            answer_parts: list[str] = []
            for item in getattr(completion, "output", []) or []:
                item_type = getattr(item, "type", "")
                if item_type == "message":
                    for content in getattr(item, "content", []) or []:
                        content_type = getattr(content, "type", "")
                        if content_type in {"output_text", "text"}:
                            content_text = getattr(content, "text", None)
                            if content_text:
                                answer_parts.append(content_text)
                elif item_type == "reasoning":
                    for summary in getattr(item, "summary", []) or []:
                        summary_text = getattr(summary, "text", None)
                        if summary_text:
                            reasoning_parts.append(summary_text)
            text = "".join(answer_parts)
        reasoning_text = "\n".join(reasoning_parts) or None
        text = text or ""
        record_response(getattr(completion, "usage", None), text=text, reasoning_text=reasoning_text, sample_id=sample_id)
        return text

    completion = client.chat.completions.create(**create_kwargs)
    msg = completion.choices[0].message
    text = msg.content or ""
    reasoning_text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    record_response(getattr(completion, "usage", None), text=text, reasoning_text=reasoning_text, sample_id=sample_id)
    return text


def generate_dummy_response(prompt: str, **kwargs) -> str:
    """Used for testing as a substitute for actual models"""
    return "Blah blah\n```python\nprint('Hello, World!')\n```\n"


def get_model_function(
    model: str,
    *,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0,
    timeout: float = 3600.0,
    repetition_penalty: float | None = None,
    stream: bool = False,
    api_type: str = "chat_completions",
    extra_body: dict | None = None,
    **kwargs,
):
    """Return a callable (prompt: str) -> str for the given model."""
    if model == "dummy":
        return partial(generate_dummy_response, model=model)

    return partial(
        generate_openai_compatible_response,
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        repetition_penalty=repetition_penalty,
        stream=stream,
        api_type=api_type,
        extra_body=extra_body,
        **kwargs,
    )


def extract_python_script(response: str) -> str:
    if '```' in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if '```python' in response
            else response.split('```')[1].split('```')[0]
        )
    else:
        print("Fail to extract python code from specific format.")
        python_script = response
    python_script = re.sub(
        r'^\s*(import .*|from .*\s+import\s+.*)', '', python_script, flags=re.MULTILINE
    )
    return python_script
