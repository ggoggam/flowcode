"""Prompt rendering, offline.

Every assertion here is about a *shape* that something downstream depends on: the message
list the chat template receives, the kwargs that select thinking mode, the stop list the
sampler is given, and the text the reward will try to extract code from. None of it needs
a real tokenizer — it needs a fake that records what it was asked to do.
"""

from __future__ import annotations

from typing import Any

import pytest

from flowcode.envs.base import Task
from flowcode.render import (
    RENDERERS,
    SYSTEM_PROMPT,
    available_renderers,
    completion_to_text,
    get_renderer,
    render_messages,
    render_prompt,
    stop_sequences,
)

TASK = Task(task_id="t0", prompt="Write a function that adds two numbers.")

QWEN_TOKENS = {"<|im_end|>": 151645, "<|endoftext|>": 151643}
HARMONY_TOKENS = {"<|return|>": 200002, "<|call|>": 200012, "<|endoftext|>": 199999}


class FakeTokenizer:
    """Records every call and returns deterministic, inspectable values."""

    def __init__(
        self,
        *,
        chat_template: str | None = "{{ messages }}",
        known_tokens: dict[str, int] | None = None,
        eos_token_id: int | None = 151645,
        unk_token_id: int | None = 0,
        decoded: str = "plain text",
        rendered: Any = None,
    ) -> None:
        self.chat_template = chat_template
        self.known_tokens = {} if known_tokens is None else dict(known_tokens)
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.decoded = decoded
        self.rendered = rendered
        self.template_calls: list[tuple[Any, dict[str, Any]]] = []
        self.encode_calls: list[str] = []
        self.decode_calls: list[tuple[list[int], dict[str, Any]]] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        self.template_calls.append((conversation, dict(kwargs)))
        if self.rendered is not None:
            return self.rendered
        # One token per message plus a generation-prompt marker, so the token count is a
        # readable function of the message list.
        return [1000 + i for i in range(len(conversation))] + [999]

    def encode(self, text: Any, **kwargs: Any) -> list[int]:
        self.encode_calls.append(str(text))
        return [7, 8, 9]

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        self.decode_calls.append((list(token_ids), dict(kwargs)))
        return self.decoded

    def convert_tokens_to_ids(self, tokens: Any) -> Any:
        return self.known_tokens.get(str(tokens), self.unk_token_id)


class TestRegistry:
    def test_conf_renderers_are_registered(self) -> None:
        # These two strings are what conf/model/*.yaml actually ship; an unregistered one
        # is a run that dies on its first prompt.
        assert "qwen3" in RENDERERS
        assert "gpt_oss_no_sysprompt" in RENDERERS

    def test_unknown_renderer_names_the_known_ones(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            get_renderer("llama3")
        message = str(excinfo.value)
        assert "llama3" in message
        assert "qwen3" in message

    def test_available_renderers_is_sorted(self) -> None:
        assert available_renderers() == sorted(RENDERERS)

    @pytest.mark.parametrize("name", sorted(RENDERERS))
    def test_every_renderer_renders(self, name: str) -> None:
        tokenizer = FakeTokenizer()
        assert render_prompt(tokenizer, TASK, name)


class TestSystemPrompt:
    def test_asks_for_a_fenced_python_block(self) -> None:
        # flowcode.envs.extract's first rule is "last complete ```python block wins".
        assert "```python" in SYSTEM_PROMPT
        assert "single" in SYSTEM_PROMPT.lower()

    def test_forbids_input_and_printing(self) -> None:
        # build_harness runs the candidate with __name__ != "__main__" for exactly this
        # reason; saying it in the prompt too is cheap.
        assert "input()" in SYSTEM_PROMPT


class TestMessageShape:
    def test_system_role_renderer_gets_two_messages(self) -> None:
        messages = render_messages(TASK, "qwen3")
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["content"] == TASK.prompt

    def test_no_sysprompt_renderer_folds_into_the_user_turn(self) -> None:
        messages = render_messages(TASK, "gpt_oss_no_sysprompt")
        assert [m["role"] for m in messages] == ["user"]
        assert SYSTEM_PROMPT in messages[0]["content"]
        assert TASK.prompt in messages[0]["content"]

    def test_template_is_called_for_generation_with_token_ids(self) -> None:
        tokenizer = FakeTokenizer()
        render_prompt(tokenizer, TASK, "qwen3")
        _conversation, kwargs = tokenizer.template_calls[0]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is True
        # transformers>=5 defaults return_dict to True, which hands back a BatchEncoding.
        assert kwargs["return_dict"] is False

    def test_qwen3_disables_thinking_and_qwen3_thinking_enables_it(self) -> None:
        off = FakeTokenizer()
        render_prompt(off, TASK, "qwen3")
        assert off.template_calls[0][1]["enable_thinking"] is False

        on = FakeTokenizer()
        render_prompt(on, TASK, "qwen3_thinking")
        assert on.template_calls[0][1]["enable_thinking"] is True

    def test_prompt_is_a_list_of_ints(self) -> None:
        tokens = render_prompt(FakeTokenizer(), TASK, "qwen3")
        assert tokens and all(isinstance(t, int) for t in tokens)

    def test_batch_encoding_is_rejected_with_an_actionable_message(self) -> None:
        tokenizer = FakeTokenizer(rendered={"input_ids": [1, 2, 3]})
        with pytest.raises(TypeError, match="return_dict"):
            render_prompt(tokenizer, TASK, "qwen3")

    def test_template_failure_names_the_renderer_and_task(self) -> None:
        class Exploding(FakeTokenizer):
            def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
                raise ValueError("no chat template for this model")

        with pytest.raises(ValueError) as excinfo:
            render_prompt(Exploding(), TASK, "qwen3")
        assert "qwen3" in str(excinfo.value)
        assert "t0" in str(excinfo.value)


class TestBaseModelFallback:
    def test_no_chat_template_falls_back_to_plain_encoding(self) -> None:
        tokenizer = FakeTokenizer(chat_template=None)
        tokens = render_prompt(tokenizer, TASK, "qwen3")
        assert tokens == [7, 8, 9]
        assert not tokenizer.template_calls
        text = tokenizer.encode_calls[0]
        assert TASK.prompt in text
        # Opening the fence is the closest thing to a generation prompt a base model has.
        assert text.rstrip().endswith("```python")


class TestStopSequences:
    def test_known_special_tokens_become_ids(self) -> None:
        tokenizer = FakeTokenizer(known_tokens=QWEN_TOKENS, eos_token_id=151645)
        stops = stop_sequences(tokenizer, "qwen3")
        assert stops == [151645, 151643]

    def test_eos_is_appended_when_it_is_not_already_there(self) -> None:
        tokenizer = FakeTokenizer(known_tokens=QWEN_TOKENS, eos_token_id=42)
        assert stop_sequences(tokenizer, "qwen3") == [151645, 151643, 42]

    def test_unknown_special_tokens_degrade_to_strings(self) -> None:
        # A renderer paired with the wrong model: dropping the stop condition silently
        # would let every sample run to max_tokens.
        tokenizer = FakeTokenizer(known_tokens={}, unk_token_id=0)
        assert stop_sequences(tokenizer, "qwen3") == ["<|im_end|>", "<|endoftext|>"]

    def test_harmony_stops(self) -> None:
        tokenizer = FakeTokenizer(known_tokens=HARMONY_TOKENS, eos_token_id=199999)
        assert stop_sequences(tokenizer, "gpt_oss_no_sysprompt") == [200002, 200012, 199999]

    def test_generic_chat_renderer_stops_on_eos_only(self) -> None:
        assert stop_sequences(FakeTokenizer(eos_token_id=5), "chat") == [5]

    def test_no_eos_and_no_stop_tokens_is_an_empty_list(self) -> None:
        assert stop_sequences(FakeTokenizer(eos_token_id=None), "chat") == []

    def test_unknown_renderer_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown renderer"):
            stop_sequences(FakeTokenizer(), "nope")


class TestCompletionToText:
    def test_decodes_without_special_tokens(self) -> None:
        tokenizer = FakeTokenizer(decoded="def f(): pass")
        assert completion_to_text(tokenizer, [1, 2, 3]) == "def f(): pass"
        assert tokenizer.decode_calls[0][1]["skip_special_tokens"] is True

    def test_empty_completion_is_empty_text(self) -> None:
        tokenizer = FakeTokenizer()
        assert completion_to_text(tokenizer, []) == ""
        assert not tokenizer.decode_calls

    def test_qwen_thinking_block_is_stripped(self) -> None:
        # The reasoning often contains its own fenced block; leaving it in means the
        # extractor executes the model's rejected first attempt.
        raw = "<think>maybe ```python\nwrong()\n```</think>\n```python\nright()\n```"
        tokenizer = FakeTokenizer(decoded=raw)
        text = completion_to_text(tokenizer, [1], "qwen3")
        assert "<think>" not in text
        assert "wrong()" not in text
        assert "right()" in text

    def test_implicitly_opened_thinking_block_is_stripped(self) -> None:
        # Qwen3's template can pre-open <think> in the generation prompt, so only the
        # closing tag appears in the completion.
        tokenizer = FakeTokenizer(decoded="reasoning...</think>\nanswer")
        assert completion_to_text(tokenizer, [1], "qwen3") == "answer"

    def test_harmony_final_channel_wins(self) -> None:
        raw = (
            "<|channel|>analysis<|message|>let me think ```python\nbad()\n```<|end|>"
            "<|start|>assistant<|channel|>final<|message|>```python\ngood()\n```<|return|>"
        )
        text = completion_to_text(FakeTokenizer(decoded=raw), [1], "gpt_oss_no_sysprompt")
        assert "good()" in text
        assert "bad()" not in text

    def test_harmony_without_a_final_channel_drops_the_analysis(self) -> None:
        raw = "<|channel|>analysis<|message|>thinking out loud"
        assert completion_to_text(FakeTokenizer(decoded=raw), [1], "gpt_oss") == ""

    def test_plain_text_is_untouched(self) -> None:
        raw = "```python\nx = 1\n```"
        assert completion_to_text(FakeTokenizer(decoded=raw), [1], "chat") == raw

    def test_unknown_renderer_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown renderer"):
            completion_to_text(FakeTokenizer(), [1], "nope")
