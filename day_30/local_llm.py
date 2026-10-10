from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Awaitable, Callable

import protocol_log
from config import LOCAL_LLM_MAX_NEW_TOKENS, LOCAL_LLM_MODEL

logger = logging.getLogger(__name__)


class LocalLLM:
    """Chat completions from a model running in this process through Hugging Face
    transformers, over a conversation history the caller owns.

    The model is downloaded on first use and loaded once, on Apple's GPU (MPS)
    when there is one, else CUDA, else the CPU."""

    DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant chatting with a user over Telegram. Keep replies concise."

    # A chat reply is sampled, as the hosted API did by default; the values are
    # the ones Qwen2.5 recommends for its instruct models.
    TEMPERATURE = 0.7
    TOP_P = 0.8
    REPETITION_PENALTY = 1.05

    # A summary is a fidelity job, not a creative one, so it runs cold. The
    # prefix labels the summary inside the history it stands in for, so the
    # model reads it as notes on an earlier conversation rather than as a turn.
    SUMMARY_TEMPERATURE = 0.2
    SUMMARY_PREFIX = "Summary of the earlier part of our conversation:\n"
    SUMMARY_SYSTEM_PROMPT = (
        "You are compressing a chat conversation so it can be dropped from the context window. "
        "Write a compact set of notes that preserves the facts, decisions, names, numbers, "
        "preferences, and open questions a later reply would still need. Write notes, not "
        "dialogue, and never add anything that was not said."
    )

    # A run of tool calls is the model working, not a conversation, so it is
    # bounded: whatever it has after this many rounds is what the user gets.
    MAX_TOOL_ROUNDS = 4

    # Qwen2.5's chat template asks for each call as a JSON object inside these
    # tags; whatever the model writes outside them is its text.
    TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    _model = None
    _tokenizer = None
    # One model, one GPU: generations run one at a time, off the event loop so
    # the bot keeps answering Telegram while the model works.
    _lock = asyncio.Lock()

    @classmethod
    async def get_reply(
        cls,
        history: list[dict],
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        call_tool: Callable[[str, dict], Awaitable[str]] | None = None,
    ) -> str:
        """The model's answer to the conversation. Given `tools` and a `call_tool`
        to run them, the model may act before answering: each round its calls are
        run and their results handed back, until it replies with words.

        The rounds are local to this call - only the answer goes back to the
        caller - so a conversation stays what was said, not how it was served."""
        system_prompt = system_prompt or cls.DEFAULT_SYSTEM_PROMPT
        logger.info("System prompt (%d chars, %d messages): %s", len(system_prompt), len(history), system_prompt)

        messages = [{"role": "system", "content": system_prompt}, *history]
        tools = tools if tools and call_tool else None

        for _ in range(cls.MAX_TOOL_ROUNDS):
            message = await cls._complete(messages, tools=tools, temperature=cls.TEMPERATURE)
            if not message.get("tool_calls"):
                return message["content"]
            messages.append(message)
            for call in message["tool_calls"]:
                messages.append({"role": "tool", "name": call["function"]["name"],
                                 "content": await cls._run_tool(call, call_tool)})

        logger.warning("Gave up after %d rounds of tool calls", cls.MAX_TOOL_ROUNDS)
        return message["content"]

    @staticmethod
    async def _run_tool(call: dict, call_tool: Callable[[str, dict], Awaitable[str]]) -> str:
        """Run one call the model asked for. Whatever goes wrong - arguments that
        are not JSON, a tool that raises - is reported back to the model as the
        result, which is the one thing it can do something about."""
        if call["function"]["arguments"] is None:
            return "The call was not valid JSON. Send it again."
        try:
            return await call_tool(call["function"]["name"], call["function"]["arguments"])
        except Exception as error:
            logger.exception("Tool %s failed", call["function"]["name"])
            return f"The call failed: {error!r}"

    @classmethod
    def _parse(cls, text: str) -> dict:
        """The model's raw output as an assistant message: its words, and the
        tool calls it made, in the shape the chat template reads back. A call
        whose JSON does not parse is kept with no arguments, so the model is
        told so instead of the call vanishing."""
        tool_calls = []
        for block in cls.TOOL_CALL_PATTERN.findall(text):
            try:
                parsed = json.loads(block)
                name, arguments = str(parsed["name"]), parsed.get("arguments") or {}
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments are not an object")
            except (ValueError, KeyError, TypeError):
                name, arguments = "unknown", None
            tool_calls.append({"type": "function", "function": {"name": name, "arguments": arguments}})
        content = cls.TOOL_CALL_PATTERN.sub("", text).strip()
        return {"role": "assistant", "content": content, **({"tool_calls": tool_calls} if tool_calls else {})}

    @classmethod
    def _load(cls):
        if cls._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading %s on %s", LOCAL_LLM_MODEL, device)
            cls._tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
            cls._model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL, dtype=dtype).to(device).eval()
        return cls._tokenizer, cls._model

    @classmethod
    def _generate(cls, messages: list[dict], tools: list[dict] | None, temperature: float) -> str:
        import torch

        tokenizer, model = cls._load()
        text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS, do_sample=True,
                                 temperature=temperature, top_p=cls.TOP_P,
                                 repetition_penalty=cls.REPETITION_PENALTY, pad_token_id=tokenizer.eos_token_id)
        # <tool_call> is an added but not a special token in Qwen2.5, so it
        # survives skip_special_tokens while the end-of-turn marker does not.
        return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    @classmethod
    async def _complete(cls, messages: list[dict], tools: list[dict] | None = None,
                        temperature: float = TEMPERATURE) -> dict:
        """Every call to the model goes through here, so both halves of it reach
        the protocol log without either caller remembering to write them.

        The request is logged whole - the history being sent included, verbatim -
        so the log answers what was actually asked and not just roughly what. The
        reply is logged whole for the same reason: what the model did is in the
        tool calls it made, not only in the words it wrote."""
        request = {"model": LOCAL_LLM_MODEL, "messages": messages, "temperature": temperature,
                   **({"tools": tools} if tools else {})}
        # ensure_ascii=False: a log of the conversation should be readable in
        # whatever language it was held in.
        protocol_log.record("bot -> llm", json.dumps(request, ensure_ascii=False, indent=2))
        try:
            async with cls._lock:
                raw = await asyncio.to_thread(cls._generate, messages, tools, temperature)
        except Exception as error:
            # So a request in the log is never left without an outcome.
            protocol_log.record("llm -> bot", f"<call failed: {error!r}>")
            raise
        message = cls._parse(raw)
        protocol_log.record("llm -> bot", json.dumps(message, ensure_ascii=False, indent=2))
        return message

    @staticmethod
    def _transcript(messages: list[dict]) -> str:
        """A conversation as flat text, for a model being asked to read one
        rather than continue it."""
        return "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)

    @classmethod
    async def summarize(cls, history: list[dict]) -> str:
        """Notes standing in for a whole conversation, ready to be the history a
        later one resumes from. The transcript may already contain an earlier
        summary, which this one folds in - so the heading it may copy is stripped."""
        message = await cls._complete(
            [
                {"role": "system", "content": cls.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": cls._transcript(history)},
            ],
            temperature=cls.SUMMARY_TEMPERATURE,
        )
        summary = message["content"].strip()
        heading = cls.SUMMARY_PREFIX.strip()
        return summary[len(heading) :].lstrip() if summary.startswith(heading) else summary
