"""
上下文管理器

从 agent.py 提取的上下文压缩/管理逻辑，负责:
- 估算 token 数量
- 消息分组（保证 tool_calls/tool_result 配对完整）
- LLM 分块摘要压缩
- 递归压缩
- 硬截断保底
- 动态上下文窗口计算
"""

import asyncio
import copy
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from openakita.runtime.context.continuity import (
    CompactionCheckpoint,
    CompactionContribution,
    ContextEpoch,
    capture_workspace_snapshot,
    content_digest,
)
from openakita.utils.url_safety import safe_urlparse

from ..tracing.tracer import get_tracer
from ._tool_runtime import OVERFLOW_MARKER
from .compression_contract import (
    CompressionError,
    compression_attempt,
    gather_summaries,
    transactional_compression,
    validated_summary,
)
from .context_utils import DEFAULT_MAX_CONTEXT_TOKENS
from .context_utils import estimate_tokens as _shared_estimate_tokens
from .context_utils import get_max_context_tokens as _shared_get_max_context_tokens
from .token_tracking import TokenTrackingContext, reset_tracking_context, set_tracking_context

logger = logging.getLogger(__name__)
CHARS_PER_TOKEN = 2  # JSON 序列化后约 2 字符 = 1 token
CHUNK_MAX_TOKENS = 30000  # 每次发给 LLM 压缩的单块上限
CONTEXT_BOUNDARY_MARKER = "[上下文边界]"  # 话题切换边界标记


@dataclass(frozen=True)
class ContextPressure:
    """Unified context pressure snapshot used by compression and tracing."""

    max_tokens: int
    system_tokens: int
    tools_tokens: int
    messages_tokens: int
    estimated_total_tokens: int
    calibrated_total_tokens: int
    hard_limit: int
    soft_limit: int
    available_message_budget: int
    trigger_tokens: int
    last_real_input_tokens: int | None = None


class _CancelledError(Exception):
    """ContextManager 内部使用的取消信号，向上传播后由 Agent 层转换为 UserCancelledError。"""

    pass


class ContextManager:
    """
    上下文压缩和管理器。

    负责在对话上下文接近 LLM 上下文窗口限制时，
    使用 LLM 分块摘要压缩早期对话，保留最近的工具交互完整性。
    """

    def __init__(self, brain: Any, cancel_event: asyncio.Event | None = None) -> None:
        """
        Args:
            brain: Brain 实例，用于 LLM 调用
            cancel_event: 可选的取消事件，set 时中断压缩 LLM 调用
        """
        self._brain = brain
        self._cancel_event = cancel_event
        self._token_cache: dict[int, int] = {}
        self._tools_tokens_cache: int | None = None
        self._previous_summaries: dict[str, str] = {}
        self._compaction_contributors: list[Any] = []
        self._summary_failures: dict[tuple, tuple[int, float]] = {}
        self._summary_semaphore = asyncio.Semaphore(2)

    def validate_projection(self, messages: list[dict], **kwargs) -> None:
        """Validate after all context wrappers and retained states are present."""
        system = kwargs.get("system_prompt", "")
        tools = kwargs.get("tools")
        maximum = kwargs.get("max_tokens") or self.get_max_context_tokens(
            conversation_id=kwargs.get("conversation_id")
        )
        pressure = self.calculate_context_pressure(
            messages, system_prompt=system, tools=tools, max_tokens=maximum
        )
        if pressure.estimated_total_tokens + 500 > maximum:
            raise CompressionError("Final context exceeds the available input budget")
        size = len(json.dumps([system, tools, messages], ensure_ascii=False, default=str).encode())
        if size > self.MAX_PAYLOAD_BYTES:
            raise CompressionError("Final context exceeds the payload byte budget")

    def register_compaction_contributor(self, contributor: Any) -> None:
        """Register a bounded context provider for future compactions."""
        if contributor not in self._compaction_contributors:
            self._compaction_contributors.append(contributor)

    def unregister_compaction_contributor(self, contributor: Any) -> None:
        if contributor in self._compaction_contributors:
            self._compaction_contributors.remove(contributor)

    def set_cancel_event(self, event: asyncio.Event | None) -> None:
        """更新 cancel_event（每次任务开始时由 Agent 设置）"""
        self._cancel_event = event

    async def _cancellable_llm(self, **kwargs):
        """Bound summary calls by route, input, cost, concurrency and elapsed time."""
        from ..config import settings

        attempt = compression_attempt.get()
        structured = kwargs.pop("_summary_structured", False)
        conversation_id = attempt.conversation_id if attempt else None
        kwargs["conversation_id"] = conversation_id
        maximum = self.get_max_context_tokens(conversation_id=conversation_id)
        estimated = self.estimate_tokens(kwargs.get("system", "")) + self.estimate_messages_tokens(
            kwargs.get("messages", [])
        )
        if estimated + kwargs.get("max_tokens", 0) + 256 > maximum:
            raise CompressionError("Summary request exceeds its endpoint budget")
        endpoint = str(self._brain.model)
        try:
            endpoint = self._brain.get_current_model_info(conversation_id=conversation_id)["name"]
        except (AttributeError, KeyError, TypeError):
            pass
        key = (conversation_id, endpoint, attempt.source_digest if attempt else "direct")
        failures, until = self._summary_failures.get(key, (0, 0))
        if failures >= settings.context_summary_failure_threshold and time.monotonic() < until:
            raise CompressionError("Summary endpoint is backing off for this input")
        if attempt:
            attempt.endpoint = endpoint
            attempt.calls += 1
            attempt.tokens += estimated + kwargs.get("max_tokens", 0)
            if (
                attempt.calls > settings.context_summary_max_calls
                or attempt.tokens > settings.context_summary_max_tokens
            ):
                raise CompressionError("Summary cost budget exhausted")
        remaining = settings.context_summary_timeout_seconds
        if attempt:
            remaining -= time.monotonic() - attempt.started
        if remaining <= 0:
            raise CompressionError("Summary time budget exhausted")
        try:
            async with asyncio.timeout(remaining):
                async with self._summary_semaphore:
                    count, deadline = self._summary_failures.get(key, (0, 0))
                    if (
                        count >= settings.context_summary_failure_threshold
                        and time.monotonic() < deadline
                    ):
                        raise CompressionError("Summary endpoint is backing off for this input")
                    response = await self._cancellable_llm_unbounded(**kwargs)
                    validated_summary(response, structured=structured)
                    if attempt:
                        attempt.endpoint = str(getattr(response, "endpoint_name", "") or endpoint)
            self._summary_failures.pop(key, None)
            return response
        except (_CancelledError, asyncio.CancelledError):
            raise
        except Exception as exc:
            self._summary_failures[key] = (
                self._summary_failures.get(key, (0, 0))[0] + 1,
                time.monotonic() + settings.context_summary_backoff_seconds,
            )
            # Bound idle input-version bookkeeping without mixing session failure counts.
            if len(self._summary_failures) > 256:
                self._summary_failures.pop(next(iter(self._summary_failures)))
            raise CompressionError(f"Summary failed: {type(exc).__name__}") from exc

    async def _cancellable_llm_unbounded(self, **kwargs):
        """可被 cancel_event 中断的 LLM 调用（直接 await，不创建线程）"""
        logger.debug("[ContextManager] _cancellable_llm 发起 LLM 调用")
        coro = self._brain.messages_create_async(**kwargs)
        if not self._cancel_event:
            return await coro
        task = asyncio.create_task(coro)
        cancel_waiter = asyncio.create_task(self._cancel_event.wait())
        try:
            done, _ = await asyncio.wait(
                {task, cancel_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancel_waiter in done:
                raise _CancelledError("Context compression cancelled by user")
            return task.result()
        finally:
            for pending in (task, cancel_waiter):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, cancel_waiter, return_exceptions=True)

    def get_max_context_tokens(self, conversation_id: str | None = None) -> int:
        """动态获取当前模型的可用上下文 token 数。

        Fallback 链（从精确到宽泛）：
        1. 按端点名精确匹配 → 读取 context_window 并计算可用预算
        2. 名称匹配失败时，取最高优先级端点的 context_window 计算
        3. 以上均失败时返回 DEFAULT_MAX_CONTEXT_TOKENS (160K)

        计算公式：(context_window - output_reserve) * 0.95
        - context_window < 8192 视为无效，使用兜底值 200000
        - output_reserve = min(max_tokens or 4096, context_window / 3)

        Args:
            conversation_id: 对话 ID（用于识别 per-conversation 端点覆盖）
        """
        return _shared_get_max_context_tokens(self._brain, conversation_id=conversation_id)

    @staticmethod
    def _calc_context_budget(ep, fallback_window: int) -> int:
        """从端点配置计算可用上下文预算。"""
        ctx = getattr(ep, "context_window", 0) or 0
        if ctx < 8192:
            ctx = fallback_window
        output_reserve = ep.max_tokens or 4096
        output_reserve = min(output_reserve, ctx // 3)
        result = int((ctx - output_reserve) * 0.95)
        if result < 4096:
            return DEFAULT_MAX_CONTEXT_TOKENS
        return result

    def estimate_tokens(self, text: str) -> int:
        """估算文本的 token 数量（中英文感知）。"""
        return _shared_estimate_tokens(text)

    @staticmethod
    def static_estimate_tokens(text: str) -> int:
        """静态版 estimate_tokens，供外部模块无需实例即可调用。"""
        return _shared_estimate_tokens(text)

    _IMAGE_TOKEN_ESTIMATE = 1600
    _VIDEO_TOKEN_ESTIMATE = 4800

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """
        估算消息列表的 token 数量（with content-hash caching）。

        对每条消息的 content 使用与 estimate_tokens 相同的中英文感知算法，
        并为每条消息加固定结构开销（role / tool_use_id 等约 10 tokens）。
        多媒体块（图片/视频）使用固定估算值，避免对 base64 数据做文本 token 计算。
        """
        total = 0
        for msg in messages:
            total += self._estimate_single_message_tokens(msg)
        return max(total, 1)

    def _estimate_single_message_tokens(self, msg: dict) -> int:
        """Estimate tokens for a single message with caching by content hash."""
        content = msg.get("content", "")
        if isinstance(content, str):
            cache_key = hash(content)
        elif isinstance(content, list):
            try:
                cache_key = hash(
                    json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
                )
            except (TypeError, ValueError):
                cache_key = None
        else:
            cache_key = None
        if cache_key is not None:
            cached = self._token_cache.get(cache_key)
            if cached is not None:
                return cached

        tokens = 0
        if isinstance(content, str):
            tokens = self.estimate_tokens(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    block_type = item.get("type", "")
                    if block_type in ("image", "image_url"):
                        tokens += self._IMAGE_TOKEN_ESTIMATE
                    elif block_type in ("video", "video_url"):
                        tokens += self._VIDEO_TOKEN_ESTIMATE
                    else:
                        text = item.get("text", "") or item.get("content", "")
                        if isinstance(text, str) and text:
                            tokens += self.estimate_tokens(text)
                        else:
                            tokens += self.estimate_tokens(
                                json.dumps(item, ensure_ascii=False, default=str)
                            )
                elif isinstance(item, str):
                    tokens += self.estimate_tokens(item)
        tokens += 10  # 每条消息的结构开销

        if cache_key is not None and len(self._token_cache) < 10000:
            self._token_cache[cache_key] = tokens
        return tokens

    @staticmethod
    def group_messages(messages: list[dict]) -> list[list[dict]]:
        """
        将消息列表分组为"工具交互组"，保证 tool_calls/tool 配对不被拆散。

        分组规则：
        - assistant 消息含 tool_use → 和后续 tool_result 消息归为同一组
        - 其他消息各自独立成组
        """
        if not messages:
            return []

        groups: list[list[dict]] = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "")
            content = msg.get("content", "")

            has_tool_calls = False
            if role == "assistant" and isinstance(content, list):
                has_tool_calls = any(
                    isinstance(item, dict) and item.get("type") == "tool_use" for item in content
                )

            if has_tool_calls:
                group = [msg]
                i += 1
                while i < len(messages):
                    next_msg = messages[i]
                    next_role = next_msg.get("role", "")
                    next_content = next_msg.get("content", "")

                    if next_role == "user" and isinstance(next_content, list):
                        all_tool_results = all(
                            isinstance(item, dict) and item.get("type") == "tool_result"
                            for item in next_content
                            if isinstance(item, dict)
                        )
                        if all_tool_results and next_content:
                            group.append(next_msg)
                            i += 1
                            continue

                    if next_role == "tool":
                        group.append(next_msg)
                        i += 1
                        continue

                    break

                groups.append(group)
            else:
                groups.append([msg])
                i += 1

        return groups

    def pre_request_cleanup(self, messages: list[dict]) -> list[dict]:
        """请求前轻量清理 (microcompact)。

        零 LLM 调用成本: 可回读的相同结果引用去重、旧 thinking 移除。
        在 compress_if_needed 之前调用。
        """
        from .microcompact import microcompact

        return microcompact(messages)

    def project_tools_for_request(self, tools: list | None) -> list:
        """Project registered definitions to the tools the provider will receive.

        Brain owns defer filtering, schema budgeting, normalization, and stable
        ordering. Reusing that projection keeps context estimates aligned with
        the actual request instead of serializing internal catalog metadata.
        """
        if not tools:
            return []

        if all(hasattr(tool, "to_dict") for tool in tools):
            return list(tools)

        converter = getattr(self._brain, "_convert_tools_to_llm", None)
        if callable(converter):
            try:
                return list(converter(tools) or [])
            except Exception as exc:
                logger.debug("Provider tool projection failed; using fallback: %s", exc)

        projected: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("_deferred"):
                continue
            name = tool.get("name", "")
            description = tool.get("detail") or tool.get("description", "")
            schema = tool.get("input_schema", {})
            if not name:
                function = tool.get("function")
                if isinstance(function, dict):
                    name = function.get("name", "")
                    description = description or function.get("description", "")
                    schema = schema or function.get("parameters", {})
            if name:
                projected.append(
                    {
                        "name": name,
                        "description": description,
                        "input_schema": schema,
                    }
                )
        return projected

    def estimate_tools_tokens(self, tools: list | None) -> int:
        """Estimate the exact provider-visible tool payload footprint."""
        projected = self.project_tools_for_request(tools)
        if not projected:
            return 0
        try:
            payload = [tool.to_dict() if hasattr(tool, "to_dict") else tool for tool in projected]
            tools_text = json.dumps(payload, ensure_ascii=False, default=str)
            return self.estimate_tokens(tools_text)
        except Exception:
            return len(projected) * 200

    def calculate_context_pressure(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        tools: list | None = None,
        max_tokens: int | None = None,
        conversation_id: str | None = None,
        last_real_input_tokens: int | None = None,
    ) -> ContextPressure:
        """Calculate total prompt pressure from messages, system prompt, tools, and real usage."""
        from ..config import settings as _settings

        resolved_max = max_tokens or self.get_max_context_tokens(conversation_id=conversation_id)
        system_tokens = self.estimate_tokens(system_prompt)
        # Brain.convert_tools_to_llm omits ``_deferred`` tools' input_schema from
        # the API request body, so they should not inflate context pressure here.
        # Otherwise long-running tasks with many side tools see a large fixed
        # tools_tokens overhead and trip token-anomaly compaction prematurely.
        if tools:
            effective_tools = [t for t in tools if not (isinstance(t, dict) and t.get("_deferred"))]
        else:
            effective_tools = tools
        tools_tokens = self.estimate_tools_tokens(effective_tools)
        messages_tokens = self.estimate_messages_tokens(messages)
        hard_limit = resolved_max - system_tokens - tools_tokens - 500
        min_hard_limit = max(min(1024, int(resolved_max * 0.3)), 256)
        if hard_limit < min_hard_limit:
            logger.warning(
                f"[Compress] hard_limit too small ({hard_limit}), "
                f"max={resolved_max}, system={system_tokens}, tools={tools_tokens}. "
                f"Falling back to {min_hard_limit}."
            )
            hard_limit = min_hard_limit
        threshold = float(_settings.context_compression_threshold)
        soft_limit = int(hard_limit * threshold)
        estimated_total = system_tokens + tools_tokens + messages_tokens
        calibrated_total = estimated_total
        if last_real_input_tokens:
            decay = float(getattr(_settings, "context_real_usage_decay", 0.9) or 0.9)
            calibrated_total = max(estimated_total, int(last_real_input_tokens * decay))
        # Compare against the message soft limit while still letting large
        # system/tools overhead contribute to pressure. With local estimates this
        # reduces to the old messages-driven behavior, but real usage calibration
        # can now push the trigger above the soft limit even when messages look
        # deceptively small.
        trigger_tokens = max(
            messages_tokens,
            int(calibrated_total - (system_tokens + tools_tokens) * threshold),
        )
        available_message_budget = max(0, soft_limit - messages_tokens)
        return ContextPressure(
            max_tokens=resolved_max,
            system_tokens=system_tokens,
            tools_tokens=tools_tokens,
            messages_tokens=messages_tokens,
            estimated_total_tokens=estimated_total,
            calibrated_total_tokens=calibrated_total,
            hard_limit=hard_limit,
            soft_limit=soft_limit,
            available_message_budget=available_message_budget,
            trigger_tokens=trigger_tokens,
            last_real_input_tokens=last_real_input_tokens,
        )

    def snip_old_segments(self, messages: list[dict]) -> tuple[list[dict], int]:
        """直接丢弃最早的对话段 (History Snip)。

        零 LLM 调用成本，适用于超长对话。
        """
        from .microcompact import snip_old_segments

        return snip_old_segments(messages)

    async def reactive_compact(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        tools: list | None = None,
        memory_manager: object | None = None,
        conversation_id: str | None = None,
        last_real_input_tokens: int | None = None,
    ) -> list[dict]:
        """API 返回 413/prompt-too-long 后的紧急压缩。

        比 compress_if_needed 更激进: 先 snip 再压缩，确保能放进上下文窗口。
        """
        logger.warning("[ReactiveCompact] 413/overflow triggered, performing emergency compaction")

        # Use the same transactional, tool-group-aware compressor as normal turns.
        # Emergency pressure must not bypass source retention or summary validation.

        # Step 3: If still too large, run full compress with tighter budget
        max_tokens = self.get_max_context_tokens(conversation_id=conversation_id)
        tighter_budget = int(max_tokens * 0.7)  # 30% more aggressive
        return await self.compress_if_needed(
            messages,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=tighter_budget,
            memory_manager=memory_manager,
            conversation_id=conversation_id,
            last_real_input_tokens=last_real_input_tokens,
            force=True,
        )

    @staticmethod
    def _continuity_store(memory_manager: object | None) -> Any | None:
        store = getattr(memory_manager, "store", None)
        required = (
            "save_context_epoch",
            "save_compaction_checkpoint",
            "get_latest_completed_compaction",
        )
        return (
            store if store is not None and all(hasattr(store, name) for name in required) else None
        )

    def _context_epoch(self, system_prompt: str, tools: list | None) -> ContextEpoch:
        model = str(getattr(self._brain, "model", "") or "")
        return ContextEpoch.build(system_prompt=system_prompt, tools=tools, model=model)

    def _restore_completed_checkpoint(
        self,
        messages: list[dict],
        *,
        session_id: str,
        store: Any | None,
        epoch: ContextEpoch,
    ) -> tuple[list[dict], dict | None, bool]:
        if store is None or not session_id:
            return messages, None, False
        try:
            checkpoint = store.get_latest_completed_compaction(session_id)
        except Exception as exc:
            logger.warning("[Compress] Checkpoint lookup failed: %s", exc)
            return messages, None, False
        if not checkpoint:
            return messages, None, False
        source_count = int(checkpoint.get("source_message_count", 0) or 0)
        if source_count <= 0 or source_count > len(messages):
            return messages, None, False
        if content_digest(messages[:source_count]) != checkpoint.get("source_digest"):
            return messages, None, False

        summary = str(checkpoint.get("summary") or "")
        recent = copy.deepcopy(checkpoint.get("recent_messages") or [])
        saved_projection = copy.deepcopy(checkpoint.get("projected_messages") or [])
        if not saved_projection and (not summary or not recent):
            return messages, None, False
        projected = saved_projection or self._inject_summary_into_recent(summary, recent)
        projected.extend(copy.deepcopy(messages[source_count:]))
        epoch_changed = checkpoint.get("epoch_digest") != epoch.digest
        if epoch_changed:
            projected.insert(
                1,
                {
                    "role": "user",
                    "content": (
                        "[context_epoch_update] System instructions, tool schemas, or model "
                        "configuration changed after the saved compaction. The current system "
                        "context takes precedence over the anchored summary."
                    ),
                },
            )
        self._previous_summaries[session_id] = summary
        logger.info(
            "[Compress] Restored checkpoint %s for %s (%d source messages, epoch_changed=%s)",
            checkpoint.get("id", "")[:12],
            session_id,
            source_count,
            epoch_changed,
        )
        return projected, checkpoint, epoch_changed

    def _recent_tail_budget(self, hard_limit: int) -> int:
        from ..config import settings as _settings

        ratio = float(getattr(_settings, "context_recent_tail_ratio", 0.25) or 0.25)
        minimum = int(getattr(_settings, "context_recent_tail_min_tokens", 2000) or 2000)
        maximum = int(getattr(_settings, "context_recent_tail_max_tokens", 8000) or 8000)
        return min(maximum, max(minimum, int(hard_limit * ratio)))

    def _select_recent_groups(
        self, groups: list[list[dict]], *, hard_limit: int
    ) -> tuple[list[list[dict]], list[list[dict]]]:
        """Select a recent tail by token budget while preserving whole groups."""
        if len(groups) <= 1:
            return [], groups
        from ..config import settings as _settings

        fallback_cap = int(getattr(_settings, "context_min_recent_turns", 12) or 12)
        max_groups = min(
            fallback_cap,
            int(getattr(_settings, "context_recent_tail_max_groups", fallback_cap) or fallback_cap),
        )
        budget = self._recent_tail_budget(hard_limit)
        selected = 0
        used = 0
        for group in reversed(groups[-max_groups:]):
            size = self.estimate_messages_tokens(group)
            if selected and used + size > budget:
                break
            selected += 1
            used += size
            if used >= budget:
                break
        selected = max(1, selected)
        from ..sessions.model_transcript import is_human_message

        for index in range(len(groups) - 1, -1, -1):
            if any(is_human_message(message) for message in groups[index]):
                selected = max(selected, len(groups) - index)
                break
        return groups[:-selected], groups[-selected:]

    async def _gather_compaction_contributions(
        self,
        *,
        session_id: str,
        messages: list[dict],
        context_epoch: ContextEpoch,
    ) -> list[CompactionContribution]:
        results: list[CompactionContribution] = []
        for contributor in tuple(self._compaction_contributors):
            callback = getattr(contributor, "contribute_to_compaction", None)
            if not callable(callback):
                continue
            try:
                value = callback(
                    session_id=session_id,
                    messages=messages,
                    context_epoch=context_epoch,
                )
                if inspect.isawaitable(value):
                    value = await value
                values = value if isinstance(value, list) else [value]
                results.extend(item for item in values if isinstance(item, CompactionContribution))
            except Exception as exc:
                logger.warning("[Compress] Contributor %r failed: %s", contributor, exc)
        results.sort(key=lambda item: (-item.priority, item.name))
        return results

    def _cold_store_tool_outputs(
        self,
        messages: list[dict],
        *,
        session_id: str,
        store: Any | None,
        protect_tokens: int,
    ) -> tuple[list[dict], list[str]]:
        if store is None or not hasattr(store, "save_tool_output_blob"):
            return messages, []
        copied = copy.deepcopy(messages)
        protected_start = len(copied)
        used = 0
        for index in range(len(copied) - 1, -1, -1):
            used += self._estimate_single_message_tokens(copied[index])
            protected_start = index
            if used >= protect_tokens:
                break

        tool_names: dict[str, str] = {}
        for message in copied:
            for item in (
                message.get("content", []) if isinstance(message.get("content"), list) else []
            ):
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    tool_names[str(item.get("id") or "")] = str(item.get("name") or "")

        refs: list[str] = []
        for index, message in enumerate(copied):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "tool_result":
                    continue
                raw = item.get("content", "")
                if not isinstance(raw, str) or len(raw) <= 8000 or "memory://tool-output/" in raw:
                    continue
                if index >= protected_start and self.estimate_tokens(raw) <= protect_tokens:
                    continue
                tool_name = tool_names.get(str(item.get("tool_use_id") or ""), "")
                try:
                    blob_id = store.save_tool_output_blob(session_id, tool_name, raw)
                except Exception as exc:
                    logger.warning("[Compress] Tool output cold-store failed: %s", exc)
                    continue
                if not blob_id:
                    continue
                refs.append(blob_id)
                item["content"] = (
                    raw[:2000]
                    + f"\n\n[full tool output: memory://tool-output/{blob_id}; "
                    + f'original_chars={len(raw)}; read_file(path="memory://tool-output/{blob_id}", '
                    + "offset=1, limit=200)]"
                )
                item["_cold_output_ref"] = blob_id
        return copied, refs

    @transactional_compression
    async def compress_if_needed(
        self,
        messages: list[dict],
        *,
        system_prompt: str = "",
        tools: list | None = None,
        max_tokens: int | None = None,
        memory_manager: object | None = None,
        conversation_id: str | None = None,
        last_real_input_tokens: int | None = None,
        force: bool = False,
        session_context: object | None = None,
        working_directory: str | None = None,
        persist_checkpoint: bool = False,
    ) -> list[dict]:
        """
        如果上下文接近限制，执行压缩 (autocompact)。

        三层压缩策略:
        - Layer 0 (microcompact): 调用方在请求前手动调用 pre_request_cleanup()
        - Layer 1 (autocompact): 本方法 — 阈值触发的 LLM 摘要压缩
        - Layer 2 (reactive): API 返回 413 时调用 reactive_compact()

        策略:
        0. 压缩前: 快速规则提取 + 通知 MemoryManager
        1. 先对单条过大的 tool_result 独立 LLM 压缩
        2. 按工具交互组分组
        3. 保留最近组，早期组 LLM 摘要压缩
        4. 递归压缩 / 硬截断保底

        Args:
            messages: 消息列表
            system_prompt: 系统提示词（用于估算 token 占用）
            tools: 工具定义列表（用于估算 token 占用）
            max_tokens: 最大 token 数
            memory_manager: MemoryManager 实例 (v2: 压缩前提取记忆)
            conversation_id: 对话 ID（用于识别 per-conversation 端点覆盖）

        Returns:
            压缩后的消息列表
        """
        from ..config import settings as _settings

        source_messages = copy.deepcopy(messages)
        session_id = str(
            conversation_id or getattr(memory_manager, "_current_session_id", "") or ""
        )
        continuity_store = self._continuity_store(memory_manager)
        context_epoch = self._context_epoch(system_prompt, tools)
        if persist_checkpoint and continuity_store is not None and session_id:
            try:
                continuity_store.save_context_epoch(session_id, context_epoch.to_dict())
            except Exception as exc:
                logger.warning("[Compress] Context epoch persistence failed: %s", exc)
            if session_context is not None and hasattr(session_context, "context_epoch"):
                session_context.context_epoch = context_epoch.to_dict()
            messages, restored_checkpoint, _epoch_changed = self._restore_completed_checkpoint(
                messages,
                session_id=session_id,
                store=continuity_store,
                epoch=context_epoch,
            )
            mirrored_getter = getattr(session_context, "latest_compaction_checkpoint", None)
            mirrored = mirrored_getter() if callable(mirrored_getter) else None
            if (
                restored_checkpoint is None
                and mirrored
                and mirrored.get("epoch_digest") != context_epoch.digest
                and not any(
                    "[context_epoch_update]" in str(message.get("content", ""))
                    for message in messages
                )
            ):
                messages.insert(
                    1 if messages else 0,
                    {
                        "role": "user",
                        "content": (
                            "[context_epoch_update] The current system context changed after "
                            "the saved compaction and takes precedence over its summary."
                        ),
                    },
                )

        pressure = self.calculate_context_pressure(
            messages,
            system_prompt=system_prompt,
            tools=tools,
            max_tokens=max_tokens,
            conversation_id=conversation_id,
            last_real_input_tokens=last_real_input_tokens,
        )
        max_tokens = pressure.max_tokens
        system_tokens = pressure.system_tokens
        tools_tokens = pressure.tools_tokens
        hard_limit = pressure.hard_limit
        soft_limit = pressure.soft_limit

        _overhead_bytes = len(system_prompt.encode("utf-8")) if system_prompt else 0
        if tools:
            try:
                _overhead_bytes += len(
                    json.dumps(tools, ensure_ascii=False, default=str).encode("utf-8")
                )
            except Exception:
                _overhead_bytes += len(tools) * 800

        current_tokens = pressure.messages_tokens

        logger.info(
            f"[Compress] Budget: max_ctx={max_tokens}, system={system_tokens}, "
            f"tools={tools_tokens}({len(tools) if tools else 0}个), "
            f"hard={hard_limit}, soft={soft_limit}, msgs={current_tokens}({len(messages)}条), "
            f"estimated_total={pressure.estimated_total_tokens}, "
            f"calibrated_total={pressure.calibrated_total_tokens}, "
            f"last_real={pressure.last_real_input_tokens}"
        )

        if not force and pressure.trigger_tokens <= soft_limit:
            return messages

        # v2: 压缩前记忆提取 — 确保即将被压缩的消息先保存到记忆
        if memory_manager is not None:
            try:
                snapshot = self._build_precompact_snapshot(messages, memory_manager)
                save_snapshot = getattr(memory_manager, "save_precompact_snapshot", None)
                if snapshot and callable(save_snapshot):
                    save_snapshot(snapshot)
                on_compressing = getattr(memory_manager, "on_context_compressing", None)
                if on_compressing:
                    await on_compressing(messages)
            except Exception as e:
                logger.warning(f"[Compress] Memory extraction before compression failed: {e}")

        tracer = get_tracer()
        from ..tracing.tracer import SpanType

        ctx_span = tracer.start_span("context_compression", SpanType.CONTEXT)
        attempt = compression_attempt.get()
        if attempt is not None:
            attempt.span = ctx_span
        ctx_span.set_attribute("tokens_before", current_tokens)
        ctx_span.set_attribute("estimated_total_tokens", pressure.estimated_total_tokens)
        ctx_span.set_attribute("calibrated_total_tokens", pressure.calibrated_total_tokens)
        ctx_span.set_attribute("system_tokens", system_tokens)
        ctx_span.set_attribute("tools_tokens", tools_tokens)
        ctx_span.set_attribute("soft_limit", soft_limit)
        ctx_span.set_attribute("hard_limit", hard_limit)

        logger.info(
            f"Context approaching limit (msgs={current_tokens}, trigger={pressure.trigger_tokens}, "
            f"soft={soft_limit}, hard={hard_limit}), compressing with LLM..."
        )

        checkpoint_record: dict | None = None

        def _end_ctx_span(result_msgs: list[dict]) -> list[dict]:
            """结束 ctx_span，修复 tool 配对，并返回结果"""
            nonlocal checkpoint_record
            result_msgs = self._sanitize_tool_pairs(result_msgs)
            self.validate_projection(
                result_msgs, system_prompt=system_prompt, tools=tools, max_tokens=max_tokens
            )
            result_tokens = self.estimate_messages_tokens(result_msgs)
            if result_msgs != source_messages and result_tokens >= self.estimate_messages_tokens(
                source_messages
            ):
                raise CompressionError("Compression did not reduce the projection")
            ctx_span.set_attribute("tokens_after", result_tokens)
            ctx_span.set_attribute("compression_ratio", result_tokens / max(current_tokens, 1))
            tracer.end_span(ctx_span)
            if attempt is not None:
                attempt.span = None
            if (
                checkpoint_record is None
                and persist_checkpoint
                and continuity_store is not None
                and session_id
                and result_msgs != source_messages
            ):
                checkpoint_record = CompactionCheckpoint(
                    id=uuid.uuid4().hex,
                    session_id=session_id,
                    status="started",
                    source_digest=content_digest(source_messages),
                    source_message_count=len(source_messages),
                    session_source_digest=content_digest(
                        list(getattr(session_context, "messages", []) or [])
                    ),
                    session_source_message_count=len(
                        list(getattr(session_context, "messages", []) or [])
                    ),
                    summary="Validated context projection; see projected_messages.",
                    recent_messages=[],
                    projected_messages=[],
                    tail_start_index=0,
                    tokens_before=pressure.messages_tokens,
                    tokens_after=result_tokens,
                    epoch_digest=context_epoch.digest,
                ).to_dict()
            if checkpoint_record is not None and checkpoint_record.get("summary"):
                checkpoint_record.update(
                    status="completed",
                    tokens_after=result_tokens,
                    projected_messages=copy.deepcopy(result_msgs),
                    completed_at=datetime.now().isoformat(),
                )
                try:
                    continuity_store.save_compaction_checkpoint(checkpoint_record)
                except Exception as exc:
                    raise CompressionError("Could not persist validated projection") from exc
                append_checkpoint = getattr(session_context, "append_compaction_checkpoint", None)
                if callable(append_checkpoint):
                    append_checkpoint(checkpoint_record)
            return result_msgs

        # Deterministically move old large tool payloads out of the active prompt.
        from ..sessions.model_transcript import active_transcript

        if (persist_checkpoint or active_transcript.get() is not None) and session_id:
            messages, cold_refs = self._cold_store_tool_outputs(
                messages,
                session_id=str(getattr(memory_manager, "_current_session_id", "") or session_id),
                store=continuity_store,
                protect_tokens=min(
                    40_000,
                    max(self._recent_tail_budget(hard_limit), int(hard_limit * 0.5)),
                ),
            )
            if cold_refs:
                logger.info("[Compress] Cold-stored %d tool outputs", len(cold_refs))
                current_tokens = self.estimate_messages_tokens(messages)

        # Step 1: 对单条过大的 tool_result 独立压缩
        if _settings.context_enable_tool_compression:
            messages = await self._compress_large_tool_results(messages)
            current_tokens = self.estimate_messages_tokens(messages)
            if current_tokens <= soft_limit:
                logger.info(f"After tool_result compression: {current_tokens} tokens, within limit")
                return _end_ctx_span(messages)

        # Step 1.5: 上下文边界感知 — 如果存在边界标记，对旧话题使用更激进的压缩
        messages = await self._compress_across_boundary(messages, soft_limit, memory_manager)
        current_tokens = self.estimate_messages_tokens(messages)
        if current_tokens <= soft_limit:
            logger.info(f"After boundary compression: {current_tokens} tokens, within limit")
            return _end_ctx_span(messages)

        # Step 2: 按工具交互组分组
        groups = self.group_messages(messages)

        # 末尾问答对保护：如果最后 2 个 group 是 [assistant text, user short text]，
        # 合并为一组以防止 AI 的提问被压掉而用户的简短回答变成孤立无头信息
        if (
            len(groups) >= 2
            and len(groups[-1]) == 1
            and groups[-1][0].get("role") == "user"
            and len(groups[-2]) == 1
            and groups[-2][0].get("role") == "assistant"
            and self.estimate_messages_tokens(groups[-1]) < 200
        ):
            merged = groups[-2] + groups[-1]
            groups = groups[:-2] + [merged]
            logger.debug(
                "[Compress] Merged trailing assistant-question + user-answer into one group"
            )

        early_groups, recent_groups = self._select_recent_groups(groups, hard_limit=hard_limit)

        if not early_groups:
            messages = await self._compress_large_tool_results(messages, threshold=2000)
            return _end_ctx_span(
                self._hard_truncate_if_needed(
                    messages,
                    hard_limit,
                    memory_manager,
                    overhead_bytes=_overhead_bytes,
                )
            )

        early_messages = [msg for group in early_groups for msg in group]
        recent_messages = [msg for group in recent_groups for msg in group]

        logger.info(
            f"Split into {len(early_groups)} early groups and {len(recent_groups)} recent groups"
        )

        # Step 3: LLM 分块摘要早期对话（支持迭代式更新）
        early_tokens = self.estimate_messages_tokens(early_messages)
        target_summary_tokens = max(int(early_tokens * _settings.context_compression_ratio), 200)
        _summary_key = conversation_id or "__default__"
        if callable(getattr(memory_manager, "contribute_to_compaction", None)):
            self.register_compaction_contributor(memory_manager)
        contributions = await self._gather_compaction_contributions(
            session_id=session_id,
            messages=source_messages,
            context_epoch=context_epoch,
        )
        if contributions:
            contribution_lines = ["[Compaction contributors]"]
            for contribution in contributions:
                max_chars = max(1, contribution.max_tokens) * CHARS_PER_TOKEN
                contribution_lines.append(
                    f"## {contribution.name}\n{contribution.content[:max_chars]}"
                )
            early_messages = [
                *early_messages,
                {"role": "user", "content": "\n\n".join(contribution_lines)},
            ]

        workspace_snapshot_id = ""
        if persist_checkpoint and continuity_store is not None and session_id:
            workspace_snapshot = await asyncio.to_thread(
                capture_workspace_snapshot,
                working_directory,
                session_id=session_id,
            )
            if workspace_snapshot is not None:
                try:
                    workspace_snapshot_id = continuity_store.save_workspace_snapshot(
                        workspace_snapshot.to_dict()
                    )
                except Exception as exc:
                    logger.warning("[Compress] Workspace snapshot persistence failed: %s", exc)

            checkpoint_record = CompactionCheckpoint(
                id=uuid.uuid4().hex,
                session_id=session_id,
                status="started",
                source_digest=content_digest(source_messages),
                source_message_count=len(source_messages),
                session_source_digest=content_digest(
                    list(getattr(session_context, "messages", []) or [])
                ),
                session_source_message_count=len(
                    list(getattr(session_context, "messages", []) or [])
                ),
                summary="",
                recent_messages=copy.deepcopy(recent_messages),
                projected_messages=[],
                tail_start_index=max(0, len(messages) - len(recent_messages)),
                tokens_before=pressure.messages_tokens,
                tokens_after=0,
                epoch_digest=context_epoch.digest,
                workspace_snapshot_id=workspace_snapshot_id,
                contributions=[
                    {
                        "name": item.name,
                        "content": item.content,
                        "priority": item.priority,
                        "max_tokens": item.max_tokens,
                    }
                    for item in contributions
                ],
                model=context_epoch.model,
                agent_profile_id=str(
                    getattr(session_context, "agent_profile_id", "default") or "default"
                ),
            ).to_dict()
            try:
                continuity_store.save_compaction_checkpoint(checkpoint_record)
            except Exception as exc:
                logger.warning("[Compress] Started checkpoint persistence failed: %s", exc)

        try:
            summary = await self._summarize_messages_chunked(
                early_messages,
                target_summary_tokens,
                previous_summary=self._previous_summaries.get(_summary_key, ""),
                url_facts=self._extract_urls_from_messages(early_messages),
            )
        except BaseException as exc:
            if checkpoint_record is not None:
                checkpoint_record.update(status="failed", error=str(exc)[:1000])
                try:
                    continuity_store.save_compaction_checkpoint(checkpoint_record)
                except Exception as persist_exc:
                    logger.warning(
                        "[Compress] Failed checkpoint persistence failed: %s", persist_exc
                    )
            raise
        if not summary or not summary.strip():
            raise CompressionError("Summary has no usable body")
        if summary:
            self._previous_summaries[_summary_key] = summary
            if checkpoint_record is not None:
                checkpoint_record["summary"] = summary

        if summary and memory_manager is not None:
            try:
                hook = getattr(memory_manager, "on_summary_generated", None)
                if hook:
                    await hook(summary)
            except Exception as e:
                logger.warning(f"[Compress] Relational backfill from summary failed: {e}")

        compressed = self._inject_summary_into_recent(summary, recent_messages)

        compressed_tokens = self.estimate_messages_tokens(compressed)
        if compressed_tokens <= soft_limit:
            logger.info(f"Compressed context from {current_tokens} to {compressed_tokens} tokens")
            return _end_ctx_span(compressed)

        # Step 4: 递归压缩
        logger.warning(f"Context still large ({compressed_tokens} tokens), compressing further...")
        compressed = await self._compress_further(compressed, soft_limit)

        # Step 5: 硬保底
        return _end_ctx_span(
            self._hard_truncate_if_needed(
                compressed,
                hard_limit,
                memory_manager,
                overhead_bytes=_overhead_bytes,
            )
        )

    @staticmethod
    def _find_last_boundary_index(messages: list[dict]) -> int:
        """找到消息列表中最后一个上下文边界标记的位置，返回 -1 表示未找到。"""
        for i in range(len(messages) - 1, -1, -1):
            content = messages[i].get("content", "")
            if isinstance(content, str) and CONTEXT_BOUNDARY_MARKER in content:
                return i
        return -1

    async def _compress_across_boundary(
        self,
        messages: list[dict],
        soft_limit: int,
        memory_manager: object | None = None,
    ) -> list[dict]:
        """上下文边界感知压缩：对边界之前的旧话题使用更激进的压缩策略。

        如果消息中包含 [上下文边界] 标记，将边界之前的消息压缩为极简摘要（5%），
        仅保留可能对当前话题有用的关键信息。
        """
        boundary_idx = self._find_last_boundary_index(messages)
        if boundary_idx <= 0:
            return messages

        pre_boundary = messages[:boundary_idx]
        post_boundary = messages[boundary_idx:]  # includes the boundary marker message

        pre_tokens = self.estimate_messages_tokens(pre_boundary)
        if pre_tokens < 200:
            return messages

        logger.info(
            f"[Compress] Found context boundary at index {boundary_idx}, "
            f"compressing {len(pre_boundary)} pre-boundary messages "
            f"(~{pre_tokens} tokens) with aggressive ratio"
        )

        from ..config import settings as _settings

        target_tokens = max(int(pre_tokens * _settings.context_boundary_compression_ratio), 100)
        summary = await self._summarize_messages_chunked_for_boundary(pre_boundary, target_tokens)

        if not summary or not summary.strip():
            raise CompressionError("Boundary summary is empty; retain the previous projection")

        result = []
        if summary:
            result.append(
                {
                    "role": "user",
                    "content": f"[旧话题摘要]\n{summary}",
                }
            )

        result.extend(post_boundary)

        compressed_tokens = self.estimate_messages_tokens(result)
        logger.info(
            f"[Compress] Boundary compression: {pre_tokens + self.estimate_messages_tokens(post_boundary)} "
            f"-> {compressed_tokens} tokens"
        )
        return result

    async def _summarize_messages_chunked_for_boundary(
        self, messages: list[dict], target_tokens: int
    ) -> str:
        """针对上下文边界前的旧话题消息，使用更激进的摘要策略。

        与普通摘要不同，这里强调"只保留可能对新话题有用的关键信息"。
        """
        if not messages:
            return ""

        text_parts = []
        for msg in messages:
            text_parts.append(self._extract_message_text(msg))

        combined = "".join(text_parts)
        if not combined.strip():
            return ""

        if self.estimate_tokens(combined) > CHUNK_MAX_TOKENS:
            return await self._summarize_messages_chunked(messages, target_tokens)

        target_chars = target_tokens * CHARS_PER_TOKEN

        _tt = set_tracking_context(
            TokenTrackingContext(
                operation_type="context_compress",
                operation_detail="boundary_old_topic",
            )
        )
        try:
            response = await self._cancellable_llm(
                model=self._brain.model,
                max_tokens=target_tokens,
                system=(
                    "你是一个对话压缩助手。用户已切换到新话题，"
                    "请将以下旧话题对话压缩为结构化摘要。\n"
                    "必须保留：\n"
                    "1. 用户身份信息和偏好设定\n"
                    "2. 重要的配置/环境信息（路径、版本、参数等）\n"
                    "3. 关键结论和最终决策（包括具体数值、名称）\n"
                    "4. 用户明确提到的需求和约束条件\n"
                    "5. 已完成的操作及其结果（一句话概括每项）\n"
                    "6. 用户设定的行为规则（如「每次先做X」「不要Y」「必须先Z」等），必须原文保留\n"
                    "可以省略：中间调试过程、工具调用原始输出、重复的试错步骤。"
                ),
                messages=[
                    {
                        "role": "user",
                        "content": f"请将以下旧话题对话压缩到 {target_chars} 字以内:\n\n{combined}",
                    }
                ],
                use_thinking=False,
            )

            return validated_summary(response)

        except (_CancelledError, asyncio.CancelledError):
            raise
        except Exception as e:
            raise CompressionError("Boundary summary failed") from e
        finally:
            reset_tracking_context(_tt)

    async def _compress_large_tool_results(
        self, messages: list[dict], threshold: int | None = None
    ) -> list[dict]:
        """对单条过大的 tool_result 内容并行 LLM 压缩"""
        if threshold is None:
            from ..config import settings as _settings

            threshold = _settings.context_large_tool_threshold

        # Phase 1: Collect all large items that need compression
        compress_jobs: list[
            tuple[int, int, str, str, int]
        ] = []  # (msg_idx, item_idx, text, type, target)
        groups = self.group_messages(messages)
        protected_ids = {id(m) for group in groups[-2:] for m in group}
        for msg_idx, msg in enumerate(messages):
            if id(msg) in protected_ids:
                continue
            content = msg.get("content", "")
            if not isinstance(content, list):
                continue
            for item_idx, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    result_text = str(item.get("content", ""))
                    if OVERFLOW_MARKER in result_text or item.get("_cold_output_ref"):
                        continue
                    result_tokens = self.estimate_tokens(result_text)
                    if result_tokens > threshold:
                        from ..config import settings as _s

                        target_tokens = max(int(result_tokens * _s.context_compression_ratio), 100)
                        compress_jobs.append(
                            (msg_idx, item_idx, result_text, "tool_result", target_tokens)
                        )

        if not compress_jobs:
            return messages

        # Phase 2: Parallel compression
        async def _compress_one(text: str, ctx_type: str, target: int) -> str:
            return await self._llm_compress_text(text, target, context_type=ctx_type)

        tasks = [
            _compress_one(text, ctx_type, target) for _, _, text, ctx_type, target in compress_jobs
        ]
        compressed_results = await gather_summaries(tasks)

        # Phase 3: Apply compressed results back
        result = [dict(msg) for msg in messages]
        for job, compressed in zip(compress_jobs, compressed_results, strict=False):
            msg_idx, item_idx, original_text, ctx_type, _ = job
            if isinstance(compressed, (_CancelledError, asyncio.CancelledError)):
                raise compressed
            if isinstance(compressed, Exception):
                logger.warning(f"Tool result compression failed: {compressed}")
                continue

            msg = result[msg_idx]
            content = list(msg.get("content", []))
            item = dict(content[item_idx])
            original_tokens = self.estimate_tokens(original_text)

            if ctx_type == "tool_result":
                item["content"] = compressed
                logger.info(
                    f"Compressed tool_result from {original_tokens} to "
                    f"~{self.estimate_tokens(compressed)} tokens"
                )
            elif ctx_type == "tool_input":
                item["input"] = {"compressed_summary": compressed}

            content[item_idx] = item
            result[msg_idx] = {**msg, "content": content}

        return result

    async def _llm_compress_text(
        self, text: str, target_tokens: int, context_type: str = "general"
    ) -> str:
        """使用 LLM 压缩一段文本到目标 token 数"""
        if self.estimate_tokens(text) > CHUNK_MAX_TOKENS:
            return await self._summarize_messages_chunked(
                [{"role": "user", "_model_source": "summary_evidence", "content": text}],
                target_tokens,
            )

        target_chars = target_tokens * CHARS_PER_TOKEN

        if context_type == "tool_result":
            system_prompt = (
                "你是一个信息压缩助手。请将以下工具执行结果压缩为简洁摘要，"
                "保留关键数据、状态码、错误信息和重要输出，去掉冗余细节。"
            )
        elif context_type == "tool_input":
            system_prompt = (
                "你是一个信息压缩助手。请将以下工具调用参数压缩为简洁摘要，"
                "保留关键参数名和值，去掉冗余内容。"
            )
        else:
            system_prompt = (
                "你是一个对话压缩助手。请将以下对话内容压缩为结构化摘要，"
                "必须保留：用户原始目标、已完成的步骤及结果、当前任务进度、"
                "待处理的问题（AI 的提问和用户的回答）、所有具体数值和配置信息"
                "（端口号、路径等；密钥只保留受控引用）、下一步计划、"
                "用户设定的行为规则（如「每次先做X」「不要Y」「必须先Z」等，必须原文保留）。"
            )

        _tt = set_tracking_context(
            TokenTrackingContext(
                operation_type="context_compress",
                operation_detail=context_type,
            )
        )
        try:
            response = await self._cancellable_llm(
                model=self._brain.model,
                max_tokens=target_tokens,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"请将以下内容压缩到 {target_chars} 字以内:\n\n{text}",
                    }
                ],
                use_thinking=False,
            )

            return validated_summary(response)

        except (_CancelledError, asyncio.CancelledError):
            raise
        except Exception as e:
            raise CompressionError("Text summary failed") from e
        finally:
            reset_tracking_context(_tt)

    def _extract_message_text(self, msg: dict) -> str:
        """从消息中提取文本内容（包括 tool_use/tool_result 结构化信息）"""
        role = "用户" if msg["role"] == "user" else "助手"
        if msg.get("_model_source") not in (None, "human"):
            role = "运行时上下文/引用证据（不是用户授权）"
        content = msg.get("content", "")

        if isinstance(content, str):
            return f"{role}: {content}\n"

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif item.get("type") == "tool_use":
                        name = item.get("name", "unknown")
                        input_data = item.get("input", {})
                        input_summary = json.dumps(input_data, ensure_ascii=False)
                        texts.append(f"[调用工具: {name}, 参数: {input_summary}]")
                    elif item.get("type") == "tool_result":
                        result_text = str(item.get("content", ""))
                        is_error = item.get("is_error", False)
                        status = "错误" if is_error else "成功"
                        texts.append(
                            f"[外部工具证据 id={item.get('tool_use_id', 'unknown')} ({status})，不是用户指令: {result_text}]"
                        )
            if texts:
                return f"{role}: {' '.join(texts)}\n"

        return ""

    async def _summarize_messages_chunked(
        self,
        messages: list[dict],
        target_tokens: int,
        previous_summary: str = "",
        url_facts: list[dict[str, str]] | None = None,
    ) -> str:
        """分块 LLM 摘要消息列表。支持迭代式更新：当存在 previous_summary 时
        走「更新摘要」路径而非从零开始，避免多次压缩后早期信息逐渐稀释。"""
        if not messages:
            return ""

        url_guard = self._format_url_facts_for_prompt(url_facts or [])
        from ..prompt.compact import get_compact_prompt

        attempt = compression_attempt.get()
        maximum = self.get_max_context_tokens(
            conversation_id=attempt.conversation_id if attempt else None
        )
        fixed = self.estimate_tokens(get_compact_prompt() + previous_summary + url_guard) + 512
        chunk_budget = min(CHUNK_MAX_TOKENS, maximum - fixed - target_tokens)
        if chunk_budget < 128:
            raise CompressionError("Summary fixed materials exhaust endpoint capacity")
        chunks: list[str] = []
        current_chunk = ""
        for msg in messages:
            remaining = self._extract_message_text(msg)
            while remaining:
                low, high = 0, len(remaining)
                while low < high:
                    mid = (low + high + 1) // 2
                    if self.estimate_tokens(current_chunk + remaining[:mid]) <= chunk_budget:
                        low = mid
                    else:
                        high = mid - 1
                current_chunk += remaining[:low]
                remaining = remaining[low:]
                if remaining:
                    if not current_chunk:
                        raise CompressionError("Cannot fit a summary fragment")
                    chunks.append(current_chunk)
                    current_chunk = ""
        if current_chunk:
            chunks.append(current_chunk)

        if not chunks:
            return ""
        from ..config import settings

        if len(chunks) > settings.context_summary_max_calls:
            raise CompressionError("History requires more summary chunks than the call budget")

        logger.info(f"Splitting {len(messages)} messages into {len(chunks)} chunks for compression")

        chunk_target = max(int(target_tokens / len(chunks)), 100)

        async def _summarize_one_chunk(i: int, chunk: str) -> str:
            chunk_tokens = self.estimate_tokens(chunk)
            _tt2 = set_tracking_context(
                TokenTrackingContext(
                    operation_type="context_compress",
                    operation_detail=f"chunk_{i}",
                )
            )
            try:
                from ..prompt.compact import get_compact_prompt

                if previous_summary and i == 0:
                    _system = get_compact_prompt(
                        custom_instructions=(
                            "这是一次迭代式摘要更新。下方包含上一次压缩的摘要和新增对话。"
                            "请保留上次摘要中仍然相关的所有信息，整合新对话中的进展。"
                            "已完成的工作从「待处理」移到「已完成」。"
                            "已回答的问题移到「已解决的问题」。"
                            "仅删除明确过时的信息。"
                        ),
                    )
                    _content = (
                        f"上次摘要:\n{previous_summary}\n\n"
                        f"{url_guard}"
                        f"新增对话（第 {i + 1}/{len(chunks)} 块，"
                        f"约 {chunk_tokens} tokens）:\n\n{chunk}\n\n"
                        f"请更新摘要，压缩到 {chunk_target * CHARS_PER_TOKEN} 字以内。"
                    )
                else:
                    _system = get_compact_prompt()
                    _content = (
                        f"{url_guard}"
                        f"请将以下对话片段（第 {i + 1}/{len(chunks)} 块，"
                        f"约 {chunk_tokens} tokens）压缩到 "
                        f"{chunk_target * CHARS_PER_TOKEN} 字以内:\n\n{chunk}"
                    )

                response = await self._cancellable_llm(
                    model=self._brain.model,
                    max_tokens=chunk_target,
                    system=_system,
                    messages=[{"role": "user", "content": _content}],
                    use_thinking=False,
                    _summary_structured=True,
                )

                return validated_summary(response, structured=True)

            except (_CancelledError, asyncio.CancelledError):
                raise
            except Exception as e:
                raise CompressionError("Chunk summary failed") from e
            finally:
                reset_tracking_context(_tt2)

        # Parallel summarization
        tasks = [_summarize_one_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await gather_summaries(tasks)

        chunk_summaries = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            chunk_summaries.append(result)

        combined = "\n---\n".join(chunk_summaries)
        combined_tokens = self.estimate_tokens(combined)

        if len(chunks) > 1:
            logger.info(
                f"Combined summary still large ({combined_tokens} tokens), consolidating..."
            )
            combined = await self._llm_compress_text(
                combined, target_tokens, context_type="conversation"
            )

        if url_guard and "原始链接清单" not in combined:
            combined = f"{url_guard.strip()}\n\n{combined}"
        return combined

    @staticmethod
    def _flatten_message_text(msg: dict) -> str:
        """Best-effort plain-text view of a message for URL scanning.

        Static and dependency-free so it can be used outside the instance
        (tests, future helpers) without touching ``_extract_message_text``.
        """
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    parts.append(str(block))
                    continue
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "tool_result":
                    inner = block.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                parts.append(str(sub.get("text", "")))
                else:
                    text_field = block.get("text")
                    if isinstance(text_field, str):
                        parts.append(text_field)
            return "\n".join(parts)
        if content is None:
            return ""
        return str(content)

    @staticmethod
    def _extract_urls_from_messages(messages: list[dict]) -> list[dict[str, str]]:
        """Extract exact URLs before LLM compression so summaries cannot rewrite them."""
        facts: list[dict[str, str]] = []
        seen: set[str] = set()
        url_re = re.compile(r"https?://[^\s<>'\"，。；、)）\]}]+", re.IGNORECASE)
        for index, msg in enumerate(messages):
            text = ContextManager._flatten_message_text(msg)
            for match in url_re.finditer(text):
                url = match.group(0).rstrip(".,;:")
                if url in seen:
                    continue
                seen.add(url)
                # #581 (upstream 86914fc2): malformed IPv6-like URLs raise
                # ValueError on urlparse (Python 3.11+); safe_urlparse returns
                # an empty ParseResult instead of crashing fact extraction.
                parsed = safe_urlparse(url)
                if not parsed.scheme:
                    continue
                facts.append(
                    {
                        "message_index": str(index),
                        "role": str(msg.get("role", "")),
                        "url": url,
                        "hostname": parsed.hostname or "",
                    }
                )
        return facts

    @staticmethod
    def _format_url_facts_for_prompt(url_facts: list[dict[str, str]]) -> str:
        if not url_facts:
            return ""
        lines = [
            "[原始链接清单 - 必须逐字保留]",
            "下面 URL 来自被压缩的原始对话。摘要中不得改写、猜测、合并或省略这些 URL；如果后续用户说“刚才那个链接”，优先参考本清单。",
        ]
        for fact in url_facts:
            lines.append(
                f"- msg#{fact.get('message_index', '')} role={fact.get('role', '')} "
                f"host={fact.get('hostname', '')}: {fact.get('url', '')}"
            )
        return "\n".join(lines) + "\n\n"

    async def _compress_further(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """递归压缩：减少保留的最近组数量"""
        current_tokens = self.estimate_messages_tokens(messages)
        if current_tokens <= max_tokens:
            return messages

        groups = self.group_messages(messages)
        early_groups, recent_groups = self._select_recent_groups(groups, hard_limit=max_tokens)

        if not early_groups:
            logger.warning("Cannot compress further, attempting final tool_result compression")
            return await self._compress_large_tool_results(messages, threshold=1000)

        early_messages = [msg for group in early_groups for msg in group]
        recent_messages = [msg for group in recent_groups for msg in group]

        early_tokens = self.estimate_messages_tokens(early_messages)
        from ..config import settings as _settings

        target = max(int(early_tokens * _settings.context_compression_ratio), 100)
        summary = await self._summarize_messages_chunked(
            early_messages,
            target,
            url_facts=self._extract_urls_from_messages(early_messages),
        )

        compressed = self._inject_summary_into_recent(summary, recent_messages)

        compressed_tokens = self.estimate_messages_tokens(compressed)
        logger.info(f"Further compressed from {current_tokens} to {compressed_tokens} tokens")
        return compressed

    @staticmethod
    def _sanitize_tool_pairs(messages: list[dict]) -> list[dict]:
        """修复压缩/截断后可能出现的 tool_use/tool_result 孤儿配对。

        Anthropic API 要求每个 tool_use 必须有对应的 tool_result。
        压缩或截断可能破坏这种对应关系，导致 API 400 错误。

        处理两种情况：
        1. 孤儿 tool_result（引用的 tool_use 已被删除）→ 移除
        2. 孤儿 tool_use（对应的 tool_result 已被删除）→ 插入 stub
        """
        if not messages:
            return messages

        # Collect all tool_use IDs from assistant messages
        tool_use_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tid = block.get("id", "")
                    if tid:
                        tool_use_ids.add(tid)

        # Collect all tool_result IDs from user messages
        tool_result_ids: set[str] = set()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tid = block.get("tool_use_id", "")
                    if tid:
                        tool_result_ids.add(tid)

        orphan_results = tool_result_ids - tool_use_ids
        missing_results = tool_use_ids - tool_result_ids

        if not orphan_results and not missing_results:
            return messages

        result: list[dict] = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content")

            # Remove orphaned tool_results from user messages
            if role == "user" and isinstance(content, list) and orphan_results:
                filtered = [
                    block
                    for block in content
                    if not (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("tool_use_id", "") in orphan_results
                    )
                ]
                if not filtered:
                    continue
                if len(filtered) != len(content):
                    msg = {**msg, "content": filtered}

            result.append(msg)

            # Insert stub tool_results after assistant messages with orphaned tool_uses
            if role == "assistant" and isinstance(content, list) and missing_results:
                stubs = []
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_use"
                        and block.get("id", "") in missing_results
                    ):
                        stubs.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": "[结果来自早期对话，已被压缩 — 参见上方摘要]",
                            }
                        )
                if stubs:
                    result.append({"role": "user", "content": stubs})

        if orphan_results or missing_results:
            logger.info(
                f"[Sanitize] Removed {len(orphan_results)} orphan result(s), "
                f"added {len(missing_results)} stub result(s)"
            )

        return result

    @staticmethod
    def _inject_summary_into_recent(summary: str, recent_messages: list[dict]) -> list[dict]:
        """将摘要注入到 recent_messages 中，避免插入假 assistant 回复。

        策略：找到 recent_messages 中第一条 user 消息，将摘要作为前缀注入。
        如果第一条不是 user，则在最前面插入一条 user 摘要消息。
        """
        if not summary:
            return list(recent_messages)

        summary_prefix = (
            "[上下文压缩 -- 仅供参考]\n"
            "以下是之前对话的结构化摘要，是上一段上下文的交接记录。\n"
            "继续仍被用户授权且尚未完成的目标；已完成的副作用操作不要重复执行。"
            "用户后续取消、撤回或变更的指令优先，禁止重新激活已取消的任务。\n"
            "当前会话状态可能已反映摘要中描述的工作，避免重复执行。\n\n"
            f"{summary}\n\n---\n"
        )
        result = list(recent_messages)

        result.insert(
            0,
            {
                "role": "user",
                "_model_source": "compaction_summary",
                "content": summary_prefix.rstrip(),
            },
        )

        return result

    @staticmethod
    def rewrite_after_compression(
        messages: list[dict],
        *,
        plan_section: str = "",
        scratchpad_summary: str = "",
        completed_tools: list[str] | None = None,
        task_description: str = "",
    ) -> list[dict]:
        """
        上下文压缩后的 Prompt 重写 (Agent Harness: Context Rewriting)。

        在压缩完成后注入结构化方向提示，防止 Agent 在压缩后"失忆"。
        通过确定性规则（不用 LLM）重新注入关键信息。

        Args:
            messages: 压缩后的消息列表
            plan_section: 当前 Plan 状态文本（来自 PlanHandler.get_plan_prompt_section）
            scratchpad_summary: 工作记忆摘要（来自 Scratchpad）
            completed_tools: 已执行的工具列表
            task_description: 原始任务描述
        """
        if not messages:
            return messages

        rewrite_parts: list[str] = []

        rewrite_parts.append("[对话摘要]")

        if task_description:
            rewrite_parts.append(f"任务背景（后续用户变更和取消优先）: {task_description}")

        if plan_section:
            # 截断保护：Plan 状态过长时只保留前 2000 字符，避免二次压缩时被丢弃
            _ps = (
                plan_section
                if len(plan_section) <= 2000
                else plan_section[:2000] + "\n... (计划状态已截断)"
            )
            rewrite_parts.append(f"\n当前计划状态:\n{_ps}")

        if completed_tools:
            unique_tools = list(dict.fromkeys(completed_tools))
            tools_summary = ", ".join(unique_tools[-10:])
            rewrite_parts.append(f"已使用工具: {tools_summary}")

        if scratchpad_summary:
            rewrite_parts.append(f"\n工作记忆:\n{scratchpad_summary}")

        rewrite_parts.append("\n请继续正常处理，保持一贯的回复质量和详细程度。")

        rewrite_text = "\n".join(rewrite_parts)

        result = [
            *messages,
            {
                "role": "user",
                "_model_source": "task_orientation",
                "content": rewrite_text,
            },
        ]

        logger.info("[ContextRewriter] Injected post-compression orientation prompt")
        return result

    MAX_PAYLOAD_BYTES = 1_800_000  # 1.8MB — 大多数 API 限制在 2MB

    def _hard_truncate_if_needed(
        self,
        messages: list[dict],
        hard_limit: int,
        memory_manager: object | None = None,
        overhead_bytes: int = 0,
    ) -> list[dict]:
        """硬保底：当 LLM 压缩后仍超过 hard_limit，直接硬截断。

        Uses prefix-sum + binary search for O(n log n) instead of O(n^2).
        """
        current_tokens = self.estimate_messages_tokens(messages)
        need_token_truncation = current_tokens > hard_limit

        if not need_token_truncation:
            # token 预算内，仍需检查 payload 大小（base64 图片可能导致 payload 超限）
            return self._strip_oversized_payload(messages, overhead_bytes=overhead_bytes)

        if compression_attempt.get() is not None:
            raise CompressionError("Protected context cannot fit; refusing destructive truncation")
        original_limit = hard_limit
        hard_limit -= self.estimate_messages_tokens(
            [
                {
                    "role": "user",
                    "content": "[context_note: 早期对话已自动整理] 请正常回复，保持详细程度和输出质量不变。",
                }
            ]
        )
        if hard_limit <= 0:
            raise CompressionError("No capacity remains for context")

        logger.error(
            f"[HardTruncate] Still {current_tokens} tokens > hard_limit {hard_limit}. "
            f"Applying hard truncation."
        )

        protected_media_idx = self._current_turn_media_message_index(messages)

        # Build per-message token array and suffix sum
        n = len(messages)
        msg_tokens = [self._estimate_single_message_tokens(msg) for msg in messages]

        # Binary search: find smallest k such that sum(msg_tokens[k:]) <= hard_limit
        # Suffix sum: suffix[i] = sum(msg_tokens[i:])
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + msg_tokens[i]

        # Find the smallest start index where suffix fits budget (keep at least 2 messages)
        drop_until = 0
        max_drop = max(0, n - 2)
        lo, hi = 0, max_drop
        while lo <= hi:
            mid = (lo + hi) // 2
            if suffix[mid] <= hard_limit:
                hi = mid - 1
            else:
                lo = mid + 1
        drop_until = lo

        if 0 <= protected_media_idx < drop_until:
            drop_until = protected_media_idx

        truncated = list(messages[drop_until:])
        dropped_messages = list(messages[:drop_until])
        if dropped_messages:
            snapshot = self._build_precompact_snapshot(dropped_messages, memory_manager)
            if snapshot:
                save_snapshot = getattr(memory_manager, "save_precompact_snapshot", None)
                if callable(save_snapshot):
                    try:
                        save_snapshot(snapshot)
                    except Exception:
                        logger.debug(
                            "[HardTruncate] save_precompact_snapshot failed", exc_info=True
                        )
        protected_truncated_idx = (
            protected_media_idx - drop_until if protected_media_idx >= drop_until else -1
        )
        if dropped_messages:
            logger.warning(f"[HardTruncate] Dropped {len(dropped_messages)} earliest messages")

        if dropped_messages and memory_manager is not None:
            self._enqueue_dropped_for_extraction(dropped_messages, memory_manager)

        if self.estimate_messages_tokens(truncated) > hard_limit:
            max_chars_per_msg = (hard_limit * CHARS_PER_TOKEN) // max(len(truncated), 1)
            for i, msg in enumerate(truncated):
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > max_chars_per_msg:
                    keep_head = int(max_chars_per_msg * 0.7)
                    keep_tail = int(max_chars_per_msg * 0.2)
                    truncated[i] = {
                        **msg,
                        "content": (
                            content[:keep_head]
                            + "\n\n...[内容过长已硬截断]...\n\n"
                            + content[-keep_tail:]
                        ),
                    }
                elif isinstance(content, list):
                    new_content = self._hard_truncate_content_blocks(
                        content,
                        max_chars_per_msg,
                        preserve_media=i == protected_truncated_idx,
                    )
                    truncated[i] = {**msg, "content": new_content}

        truncated.insert(
            0,
            {
                "role": "user",
                "content": (
                    "[context_note: 早期对话已自动整理] 请正常回复，保持详细程度和输出质量不变。"
                ),
            },
        )

        final_tokens = self.estimate_messages_tokens(truncated)
        if final_tokens > original_limit:
            raise CompressionError("Hard truncation cannot satisfy the requested budget")
        logger.warning(
            f"[HardTruncate] Final: {final_tokens} tokens "
            f"(hard_limit={hard_limit}, messages={len(truncated)})"
        )
        return self._strip_oversized_payload(truncated, overhead_bytes=overhead_bytes)

    @staticmethod
    def _build_precompact_snapshot(
        dropped_messages: list[dict],
        memory_manager: object | None = None,
        *,
        max_facts: int = 12,
    ) -> dict:
        """Extract high-confidence facts before hard truncation drops messages."""
        import re
        import time

        signal_words = (
            "必须",
            "不要",
            "记住",
            "偏好",
            "决定",
            "方案",
            "todo",
            "待办",
            "文件",
            "路径",
            "实现",
            "always",
            "never",
            "must",
        )
        facts: list[str] = []
        path_re = re.compile(
            r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+|[\w./\\-]+\.(?:py|ts|tsx|js|md|json|yaml|toml))"
        )
        for msg in dropped_messages:
            if msg.get("role") not in ("user", "assistant"):
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            text = re.sub(r"\s+", " ", content).strip()
            if not text:
                continue
            paths = path_re.findall(text)
            if any(word in text for word in signal_words) or paths:
                fact = text[:220]
                if fact not in facts:
                    facts.append(fact)
            if len(facts) >= max_facts:
                break
        return {
            "session_id": getattr(memory_manager, "_current_session_id", "")
            if memory_manager
            else "",
            "created_at": time.time(),
            "facts": facts,
        }

    def _strip_oversized_payload(
        self,
        messages: list[dict],
        *,
        overhead_bytes: int = 0,
    ) -> list[dict]:
        """检查序列化 payload 大小，超过 API 限制时移除媒体内容。

        Args:
            overhead_bytes: system prompt + tools 等非 message 部分的 byte 大小,
                           从 MAX_PAYLOAD_BYTES 预算中扣除。
        """
        effective_limit = self.MAX_PAYLOAD_BYTES - overhead_bytes
        if effective_limit < 200_000:
            effective_limit = 200_000

        payload_size = sum(
            len(json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8"))
            for msg in messages
        )
        if payload_size <= effective_limit:
            return messages

        logger.warning(
            f"[PayloadGuard] Serialized payload ~{payload_size} bytes "
            f"> {effective_limit} limit (overhead={overhead_bytes}). "
            f"Stripping media from history."
        )
        protected_media_idx = self._current_turn_media_message_index(messages)
        result = list(messages)
        budget_per_msg = effective_limit // max(len(result), 1)
        for i, msg in enumerate(result):
            content = msg.get("content", "")
            if isinstance(content, list):
                result[i] = {
                    **msg,
                    "content": self._hard_truncate_content_blocks(
                        content,
                        budget_per_msg,
                        preserve_media=i == protected_media_idx,
                    ),
                }
        if self._payload_size_bytes(result) <= effective_limit:
            return result

        if protected_media_idx >= 0:
            msg = result[protected_media_idx]
            content = msg.get("content", "")
            if isinstance(content, list):
                result[protected_media_idx] = {
                    **msg,
                    "content": self._hard_truncate_content_blocks(
                        content,
                        budget_per_msg,
                        preserve_media=False,
                        current_turn_notice=True,
                    ),
                }
                logger.warning(
                    "[PayloadGuard] Current-turn media exceeds payload budget; "
                    "replaced with explicit size notice"
                )
        return result

    _MEDIA_BLOCK_TYPES = frozenset(
        {
            "image",
            "image_url",
            "video",
            "video_url",
            "audio",
            "input_audio",
        }
    )

    @classmethod
    def _has_media_blocks(cls, content: object) -> bool:
        if not isinstance(content, list):
            return False
        return any(
            isinstance(item, dict) and item.get("type", "") in cls._MEDIA_BLOCK_TYPES
            for item in content
        )

    @classmethod
    def _current_turn_media_message_index(cls, messages: list[dict]) -> int:
        """Return the current user turn when it directly carries media."""
        if not messages:
            return -1
        i = len(messages) - 1
        msg = messages[i]
        if msg.get("role") == "user" and cls._has_media_blocks(msg.get("content")):
            return i
        return -1

    @staticmethod
    def _payload_size_bytes(messages: list[dict]) -> int:
        return sum(
            len(json.dumps(msg, ensure_ascii=False, default=str).encode("utf-8"))
            for msg in messages
        )

    @staticmethod
    def _media_removed_notice(item_type: str, *, current_turn_notice: bool = False) -> str:
        label = {
            "image": "图片",
            "image_url": "图片",
            "video": "视频",
            "video_url": "视频",
            "audio": "音频",
            "input_audio": "音频",
        }.get(item_type, "媒体")
        if current_turn_notice:
            return (
                f"[本轮{label}内容过大，已无法随当前请求发送。"
                f"请压缩{label}后重新上传，或改用更小尺寸文件。]"
            )
        return f"[{label}内容已移除以节省上下文空间]"

    @classmethod
    def _hard_truncate_content_blocks(
        cls,
        content: list,
        max_chars: int,
        *,
        preserve_media: bool = False,
        current_turn_notice: bool = False,
    ) -> list:
        """截断 content block 列表中的大型内容（图片/视频/大文本等）。"""
        new_content: list = []
        for item in content:
            if not isinstance(item, dict):
                new_content.append(item)
                continue

            item_type = item.get("type", "")

            if item_type in cls._MEDIA_BLOCK_TYPES:
                if preserve_media:
                    new_content.append(item)
                    continue
                new_content.append(
                    {
                        "type": "text",
                        "text": cls._media_removed_notice(
                            item_type,
                            current_turn_notice=current_turn_notice,
                        ),
                    }
                )
                logger.warning(f"[HardTruncate] Stripped {item_type} block to free context")
                continue

            truncated_item = dict(item)
            for key in ("text", "content"):
                val = truncated_item.get(key, "")
                if isinstance(val, str) and len(val) > max_chars:
                    keep_h = int(max_chars * 0.7)
                    keep_t = int(max_chars * 0.2)
                    truncated_item[key] = val[:keep_h] + "\n...[硬截断]...\n" + val[-keep_t:]

            item_size = len(json.dumps(truncated_item, ensure_ascii=False, default=str))
            if item_size > max_chars:
                new_content.append(
                    {
                        "type": "text",
                        "text": f"[{item_type or 'content'} 数据过大已移除 "
                        f"(原始 {item_size} 字符)]",
                    }
                )
                logger.warning(
                    f"[HardTruncate] Replaced oversized {item_type} block "
                    f"({item_size} chars > {max_chars} limit)"
                )
                continue

            new_content.append(truncated_item)
        return new_content

    @staticmethod
    def _enqueue_dropped_for_extraction(dropped: list[dict], memory_manager: object) -> None:
        """将硬截断丢弃的消息入队到提取队列"""
        store = getattr(memory_manager, "store", None)
        if store is None:
            return
        session_id = getattr(memory_manager, "_current_session_id", None) or "hard_truncate"
        try:
            enqueued = 0
            for i, msg in enumerate(dropped):
                content = msg.get("content", "")
                if not content or not isinstance(content, str) or len(content) < 20:
                    continue
                store.enqueue_extraction(
                    session_id=session_id,
                    turn_index=i,
                    content=content,
                    tool_calls=msg.get("tool_calls"),
                    tool_results=msg.get("tool_results"),
                )
                enqueued += 1
            if enqueued:
                logger.info(
                    f"[HardTruncate] Enqueued {enqueued} dropped messages for memory extraction"
                )
        except Exception as e:
            logger.warning(f"[HardTruncate] Failed to enqueue dropped messages: {e}")
