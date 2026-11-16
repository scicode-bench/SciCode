"""将 SDK 流式事件合成为原生响应对象。"""
from __future__ import annotations


def _install(resource, final_method, *, usage=False):
    if getattr(resource, "_scicode_streaming", False):
        return
    original = resource.create

    async def create(*args, **kwargs):
        # SDK 流管理器可能再次调用 create(stream=True)。
        if kwargs.get("stream") is True:
            return await original(*args, **kwargs)
        request = dict(kwargs)
        request.pop("stream", None)
        if usage:
            options = dict(request.get("stream_options") or {})
            options.setdefault("include_usage", True)
            request["stream_options"] = options
        async with resource.stream(*args, **request) as stream:
            result = await getattr(stream, final_method)()
        if final_method == "get_final_completion" and (
            not result.choices or any(choice.finish_reason is None for choice in result.choices)
        ):
            raise RuntimeError("Model stream ended without a completed choice")
        if final_method == "get_final_message" and result.stop_reason is None:
            raise RuntimeError("Model stream ended without a stop reason")
        return result

    resource.create = create
    resource._scicode_streaming = True


def enable_streaming(model):
    """保留提供方的响应转换、重试、思考及用量统计。"""
    client = getattr(model.api, "client", None)
    if client is None:
        raise ValueError("The selected provider has no supported streaming client")
    if hasattr(client, "chat"):
        _install(client.chat.completions, "get_final_completion", usage=True)
        if hasattr(client, "responses"):
            _install(client.responses, "get_final_response")
    elif hasattr(client, "messages"):
        _install(client.messages, "get_final_message")
    else:
        raise ValueError("The selected provider has no supported streaming client")
