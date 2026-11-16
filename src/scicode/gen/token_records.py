"""逐条 token 用量落盘工具（用于统计 token_stats / error_stats）。

各生成线程通过 append_token_record() 向 SCICODE_TOKEN_RECORDS 指定的 jsonl 文件
逐条追加记录。路径为空时静默跳过，不影响正常生成流程。

token 计数口径与 sensebench 的 OpenAI_vLLM.token_count 对齐：
优先用 usage 里的 reasoning_tokens；若 API 未提供（如 QwQ 只在 content 里用
<think></think> 包裹思考），则把响应拆成 think / prediction 两段，分别用
tiktoken 计数，避免把 reasoning 算进 prediction。
"""

from __future__ import annotations

import json
import os
import threading

_lock = threading.Lock()

_written_sample_ids: set[str] = set()

# 复用一个 cl100k_base 编码器做兜底计数
try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENC = None


def _records_path() -> str:
    return os.getenv("SCICODE_TOKEN_RECORDS", "")


def _get(obj, key):
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _tiktoken_len(text) -> int:
    if not text or _ENC is None:
        return 0
    try:
        return len(_ENC.encode(text))
    except Exception:  # noqa: BLE001
        return 0


def _split_think(text):
    """把响应文本拆成 (think_text, prediction_text)。

    兼容 QwQ 等把思考写进 content 的场景：
      - 含 </think>：前半为 think，后半为 prediction
    无 think 标签时 think_text 为空、prediction 为整段。
    """
    if not text:
        return "", ""
    if "</think>" in text:
        head, _, tail = text.partition("</think>")
        head = head.replace("<think>", "")
        return head, tail
    return "", text


def compute_tokens(usage, text=None, reasoning_text=None):
    """计算 (input_tokens, prediction_tokens, think_tokens)。

    Args:
        usage: response.usage（可为 None）
        text: 模型最终返回的完整文本（用于兜底拆分/计数）
        reasoning_text: 若 API 单独提供了 reasoning_content，则传入
    """
    input_tokens = None
    completion_tokens = None
    reasoning_tokens = None

    if usage is not None:
        input_tokens = _get(usage, "prompt_tokens")
        if input_tokens is None:
            input_tokens = _get(usage, "input_tokens")
        completion_tokens = _get(usage, "completion_tokens")
        if completion_tokens is None:
            completion_tokens = _get(usage, "output_tokens")
        reasoning_tokens = _get(usage, "reasoning_tokens")
        if reasoning_tokens is None:
            details = _get(usage, "completion_tokens_details")
            if details is not None:
                reasoning_tokens = _get(details, "reasoning_tokens")
        if reasoning_tokens is None:
            details = _get(usage, "output_tokens_details")
            if details is not None:
                reasoning_tokens = _get(details, "reasoning_tokens")

    # 拆分 think / prediction 文本
    if reasoning_text:
        think_text = reasoning_text
        pred_text = text or ""
    else:
        think_text, pred_text = _split_think(text)

    # input：usage 没有则不兜底（保持 None），避免误算
    # reasoning：usage 有则用；否则用 think 文本 tiktoken 兜底
    if reasoning_tokens is None or reasoning_tokens == 0:
        if think_text:
            think_tokens = _tiktoken_len(think_text)
            # usage 的 completion 含 reasoning，弃用，改由 prediction 文本兜底
            completion_tokens = None
        else:
            think_tokens = 0
    else:
        think_tokens = reasoning_tokens
        # usage 提供了 reasoning，从 completion 中扣除得到纯 prediction
        if completion_tokens is not None and completion_tokens >= reasoning_tokens:
            completion_tokens = completion_tokens - reasoning_tokens

    # prediction：completion 可用则用；否则用 prediction 文本 tiktoken 兜底
    if completion_tokens is None:
        prediction_tokens = _tiktoken_len(pred_text) if pred_text else 0
    else:
        prediction_tokens = completion_tokens

    return input_tokens, prediction_tokens, think_tokens


def append_token_record(
    input_tokens=None,
    prediction_tokens=None,
    think_tokens=None,
    is_error: bool = False,
    is_empty: bool = False,
    sample_id: str | None = None,
) -> None:
    """线程安全地向 token_records.jsonl 追加一条记录。任何异常都被吞掉。"""
    path = _records_path()
    if not path:
        return
    sid_str = str(sample_id) if (sample_id is not None and sample_id != "") else None
    rec = {
        "input_tokens": input_tokens,
        "prediction_tokens": prediction_tokens,
        "think_tokens": think_tokens,
        "is_error": bool(is_error),
        "is_empty": bool(is_empty),
    }
    if sid_str is not None:
        rec["sample_id"] = sid_str
    try:
        # 加锁一次性检查+写入，保证并发下 set 与文件严格一致
        with _lock:
            if sid_str is not None and sid_str in _written_sample_ids:
                return
            line = json.dumps(rec, ensure_ascii=False)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if sid_str is not None:
                _written_sample_ids.add(sid_str)
    except Exception:
        pass


def mark_written(sample_id) -> None:
    """把某个 sample_id 标记为已写入（不写文件），供预加载使用。"""
    if sample_id is None or sample_id == "":
        return
    with _lock:
        _written_sample_ids.add(str(sample_id))


def preload_written_sample_ids() -> int:
    """启动时读取现有 token_records.jsonl，把已有 sample_id 载入 _written_sample_ids。

    返回加载条数。文件不存在或为空返回 0。任何异常都被吞掉，不影响主流程。
    ThreadPoolExecutor 场景全线程共享同一份 set，只需在 main() 里调用一次。
    """
    path = _records_path()
    if not path or not os.path.exists(path):
        return 0
    n = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("sample_id")
                if sid is not None and sid != "":
                    _written_sample_ids.add(str(sid))
                    n += 1
    except Exception:
        pass
    return n


def record_response(usage, text=None, reasoning_text=None, sample_id: str | None = None):
    """便捷入口：从 usage + 文本算 token 并落盘（含 error/empty 判定）。"""
    try:
        input_tokens, prediction_tokens, think_tokens = compute_tokens(
            usage, text=text, reasoning_text=reasoning_text
        )
        stripped = (text or "").strip()
        is_empty = stripped == ""
        is_error = stripped.upper().startswith("ERROR")
        append_token_record(
            input_tokens=input_tokens,
            prediction_tokens=prediction_tokens,
            think_tokens=think_tokens,
            is_error=is_error,
            is_empty=is_empty,
            sample_id=sample_id,
        )
    except Exception:
        pass


# 向后兼容旧名字
def extract_usage_tokens(usage):
    return compute_tokens(usage, text=None, reasoning_text=None)
