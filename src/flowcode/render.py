"""Turning a :class:`~flowcode.envs.base.Task` into prompt tokens, and tokens back into text.

``tinker_cookbook`` is **not** a dependency of this package, so its ``renderers`` module —
the usual way a Tinker job builds chat prompts — is unavailable. The substitute is the
tokenizer's own chat template: ``TrainingClient.get_tokenizer()`` hands back a
``transformers`` ``PreTrainedTokenizer`` (transformers arrives as a transitive dependency
of ``tinker``), and every instruct model on Tinker ships a Jinja chat template inside its
tokenizer config. Rendering with that template is not an approximation of what the model
was trained on — it *is* what the model was trained on.

Why a registry rather than one function
---------------------------------------
``model.renderer`` in ``conf/model/*.yaml`` names a family, and the families genuinely
differ in ways a single call cannot paper over:

* **qwen3** interleaves an optional thinking phase. Its template takes an
  ``enable_thinking`` flag, and with thinking on, the completion arrives as
  ``<think>...</think>`` followed by the answer. The reward runs
  :func:`flowcode.envs.extract.extract_code` over the *text* of a completion, so the
  reasoning block has to be stripped before scoring or a stray fenced block inside the
  model's own deliberation gets executed instead of its answer.
* **gpt_oss** uses the harmony format, where the reply is split into channels
  (``analysis`` for reasoning, ``final`` for the answer) and the useful part is the final
  channel. Its shipped config here is ``gpt_oss_no_sysprompt``, i.e. the system slot is
  reserved by the harmony template itself, so our instructions go into the user turn.
* **base models** (``Qwen/Qwen3.5-9B-Base``) have no chat template at all. Rendering falls
  back to plain text, which is the honest thing to do for a model that has never seen a
  chat turn.

Everything the registry decides is per-family data, not per-family code: which stop tokens
to send, whether a system role exists, what extra kwargs the template takes, and how to
strip a completion back down to the answer.

The system prompt
-----------------
:data:`SYSTEM_PROMPT` asks for one complete solution inside a fenced ``python`` block,
because that is exactly the shape :func:`flowcode.envs.extract.extract_code` scores best:
its first rule is "take the last complete block tagged ``python``". The dataset prompts
already ask for the same thing; saying it twice costs a few tokens and removes a whole
class of unscoreable samples.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from flowcode.envs.base import Task

__all__ = [
    "RENDERERS",
    "SYSTEM_PROMPT",
    "ChatTokenizer",
    "RendererSpec",
    "available_renderers",
    "completion_to_text",
    "get_renderer",
    "render_messages",
    "render_prompt",
    "stop_sequences",
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer.\n"
    "Answer with one complete, self-contained Python solution inside a single fenced code "
    "block tagged `python`, like this:\n"
    "```python\n"
    "def solution(...):\n"
    "    ...\n"
    "```\n"
    "Include every import and helper the solution needs, define the function the task asks "
    "for at module level, and do not call `input()` or print anything. Write no text after "
    "the closing fence."
)
"""Instructions prepended to every task. Shaped for :func:`flowcode.envs.extract.extract_code`."""

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
_OPEN_THINK = re.compile(r"^.*?</think>\s*", flags=re.DOTALL)
_HARMONY_FINAL = re.compile(
    r"<\|channel\|>final<\|message\|>(?P<body>.*?)(?:<\|return\|>|<\|end\|>|\Z)",
    flags=re.DOTALL,
)
_HARMONY_ANALYSIS = re.compile(
    r"<\|channel\|>analysis<\|message\|>.*?(?=<\|(?:start|channel|end|return)\|>|\Z)",
    flags=re.DOTALL,
)


class ChatTokenizer(Protocol):
    """The slice of ``PreTrainedTokenizer`` this module uses.

    A Protocol rather than the concrete class so the offline tests can pass a small fake
    and still type-check. Every method is declared with ``**kwargs`` so a
    real tokenizer — whose signatures are much wider — satisfies it structurally.
    """

    def apply_chat_template(self, conversation: Any, /, **kwargs: Any) -> Any:
        """Render a message list through the tokenizer's Jinja chat template."""
        ...

    def encode(self, text: Any, /, **kwargs: Any) -> Any:
        """Tokenise a plain string. Used only by the no-chat-template fallback."""
        ...

    def decode(self, token_ids: Any, /, **kwargs: Any) -> Any:
        """Turn token ids back into text."""
        ...

    def convert_tokens_to_ids(self, tokens: Any, /) -> Any:
        """Look up a special token's id, for building the stop list."""
        ...


def _strip_qwen_thinking(text: str) -> str:
    """Drop ``<think>...</think>``, including the case where the opening tag was implicit.

    Qwen3's template can *pre-open* the thinking block in the generation prompt, so the
    completion starts mid-reasoning and only the closing tag appears in what we decoded.
    Both shapes have to go, or the extractor scores the model's deliberation.
    """
    if "</think>" not in text:
        return text
    if "<think>" in text:
        return _THINK_BLOCK.sub("", text, count=1).lstrip()
    return _OPEN_THINK.sub("", text, count=1).lstrip()


def _strip_harmony_channels(text: str) -> str:
    """Keep the harmony ``final`` channel; failing that, drop the ``analysis`` channel."""
    match = _HARMONY_FINAL.search(text)
    if match is not None:
        return match.group("body").strip()
    return _HARMONY_ANALYSIS.sub("", text).strip()


@dataclass(frozen=True)
class RendererSpec:
    """Everything one renderer family decides, as data.

    Args:
        name: Registry key, matching ``model.renderer`` in ``conf/model/*.yaml``.
        system_role: Whether the family accepts a ``system`` message. When false,
            :data:`SYSTEM_PROMPT` is folded into the user turn instead — the harmony
            templates reserve the system slot for their own preamble, and a base model has
            no roles at all.
        stop_tokens: Special tokens that end an assistant turn, most specific first. Sent
            to the sampler as ids when the tokenizer knows them, as literal strings
            otherwise (``SamplingParams.stop`` accepts either).
        template_kwargs: Extra variables handed to the Jinja template, e.g.
            ``enable_thinking`` for qwen3. Unknown variables are ignored by Jinja, so a
            template that does not use one is unaffected.
        strip_completion: Post-processing applied by :func:`completion_to_text` to recover
            the answer from a reasoning-carrying completion. ``None`` means the decoded
            text is already the answer.
    """

    name: str
    system_role: bool = True
    stop_tokens: tuple[str, ...] = ()
    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    strip_completion: Any = None


RENDERERS: dict[str, RendererSpec] = {
    # Generic instruct model: system + user through whatever template ships with it.
    "chat": RendererSpec(name="chat"),
    # Qwen3 with thinking OFF. Reasoning triples the completion length, and every one of
    # those tokens is billed twice (sampling, then the training pass) while contributing
    # nothing the extractor can score.
    "qwen3": RendererSpec(
        name="qwen3",
        stop_tokens=("<|im_end|>", "<|endoftext|>"),
        template_kwargs={"enable_thinking": False},
        strip_completion=_strip_qwen_thinking,
    ),
    # Qwen3 with thinking ON, for ablations. Raise train.max_tokens if you use it.
    "qwen3_thinking": RendererSpec(
        name="qwen3_thinking",
        stop_tokens=("<|im_end|>", "<|endoftext|>"),
        template_kwargs={"enable_thinking": True},
        strip_completion=_strip_qwen_thinking,
    ),
    "gpt_oss": RendererSpec(
        name="gpt_oss",
        stop_tokens=("<|return|>", "<|call|>", "<|endoftext|>"),
        strip_completion=_strip_harmony_channels,
    ),
    # What conf/model/gpt-oss-20b.yaml selects: harmony owns the system slot.
    "gpt_oss_no_sysprompt": RendererSpec(
        name="gpt_oss_no_sysprompt",
        system_role=False,
        stop_tokens=("<|return|>", "<|call|>", "<|endoftext|>"),
        strip_completion=_strip_harmony_channels,
    ),
}
"""Renderer families, keyed by ``model.renderer``."""


def available_renderers() -> list[str]:
    """Registry keys, sorted. Used in error messages and by the tests."""
    return sorted(RENDERERS)


def get_renderer(renderer_name: str) -> RendererSpec:
    """Look up a renderer family.

    Args:
        renderer_name: Value of ``model.renderer``.

    Returns:
        The spec.

    Raises:
        ValueError: If the name is unknown, listing what is registered. Guessing a default
            here would silently render prompts in a format the model was never trained on,
            which shows up only as a mysteriously low reward.
    """
    try:
        return RENDERERS[renderer_name]
    except KeyError:
        raise ValueError(
            f"unknown renderer {renderer_name!r}; known renderers are "
            f"{available_renderers()}. `model.renderer` in conf/model/*.yaml selects one."
        ) from None


def render_messages(task: Task, renderer_name: str) -> list[dict[str, str]]:
    """Build the chat messages for one task.

    Args:
        task: The task whose ``prompt`` is the user turn.
        renderer_name: Renderer family.

    Returns:
        Either ``[system, user]`` or a single ``user`` message carrying both, depending on
        :attr:`RendererSpec.system_role`.

    Raises:
        ValueError: On an unknown renderer.
    """
    spec = get_renderer(renderer_name)
    if spec.system_role:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task.prompt},
        ]
    return [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{task.prompt}"}]


def _has_chat_template(tokenizer: ChatTokenizer) -> bool:
    """Whether this tokenizer can render chat turns at all (base models cannot)."""
    return getattr(tokenizer, "chat_template", None) is not None


def _as_token_ids(value: Any, context: str) -> list[int]:
    """Coerce a tokenizer return value to ``list[int]``, or explain why it cannot be."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tokens = list(value)
        if tokens and all(isinstance(t, int) for t in tokens):
            return [int(t) for t in tokens]
        if not tokens:
            raise ValueError(f"{context} produced an empty token list")
    raise TypeError(
        f"{context} returned {type(value).__name__}, not a list of token ids. Pass "
        "tokenize=True and return_dict=False; transformers>=5 defaults return_dict to True "
        "and hands back a BatchEncoding."
    )


def render_prompt(tokenizer: ChatTokenizer, task: Task, renderer_name: str) -> list[int]:
    """Render a task to the exact prompt token ids the sampler should condition on.

    Args:
        tokenizer: The model's own tokenizer, from ``TrainingClient.get_tokenizer()``.
        task: The task to render.
        renderer_name: Value of ``model.renderer``.

    Returns:
        Prompt token ids, ending in the template's generation prompt so the model's first
        sampled token is the first token of its reply.

    Raises:
        ValueError: On an unknown renderer, or if the template fails to render.
        TypeError: If the tokenizer returns something other than token ids.
    """
    spec = get_renderer(renderer_name)
    if not _has_chat_template(tokenizer):
        # A true base model. Concatenating the instructions and opening a fence is the
        # closest thing to a "generation prompt" that exists without a template.
        text = f"{SYSTEM_PROMPT}\n\n{task.prompt}\n\n```python\n"
        return _as_token_ids(
            tokenizer.encode(text, add_special_tokens=False), f"encode() for task {task.task_id!r}"
        )

    messages = render_messages(task, renderer_name)
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            **dict(spec.template_kwargs),
        )
    except Exception as exc:
        raise ValueError(
            f"renderer {renderer_name!r} failed to apply the tokenizer's chat template for "
            f"task {task.task_id!r}: {exc}"
        ) from exc
    return _as_token_ids(rendered, f"apply_chat_template() for task {task.task_id!r}")


def stop_sequences(tokenizer: ChatTokenizer, renderer_name: str) -> list[str] | list[int]:
    """Stop conditions to hand to ``SamplingParams.stop``.

    Token ids are preferred over strings: a string stop has to be matched against decoded
    text, which the server can only do at token boundaries, whereas an id match is exact.
    When the tokenizer does not know one of the family's special tokens — which is what
    happens if a renderer is paired with the wrong model — the whole list degrades to
    strings rather than silently dropping the stop condition.

    Args:
        tokenizer: The model's tokenizer.
        renderer_name: Value of ``model.renderer``.

    Returns:
        Token ids, or the literal stop strings. ``SamplingParams.stop`` accepts both.

    Raises:
        ValueError: On an unknown renderer.
    """
    spec = get_renderer(renderer_name)
    unknown = getattr(tokenizer, "unk_token_id", None)
    ids: list[int] = []
    for token in spec.stop_tokens:
        resolved = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(resolved, int) or resolved < 0 or resolved == unknown:
            return list(spec.stop_tokens)
        if resolved not in ids:
            ids.append(resolved)

    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int) and eos >= 0 and eos not in ids:
        ids.append(eos)
    if not ids:
        # Nothing to stop on: max_tokens is then the only bound, which is legal but worth
        # noticing, so return an empty list rather than inventing a stop token.
        return []
    return ids


def completion_to_text(
    tokenizer: ChatTokenizer, tokens: Sequence[int], renderer_name: str | None = None
) -> str:
    """Decode sampled tokens into the text the reward will score.

    Args:
        tokenizer: The model's tokenizer.
        tokens: Sampled completion token ids.
        renderer_name: Optional renderer family. Passing it applies that family's
            reasoning-stripping (qwen3 ``<think>`` blocks, harmony channels); omitting it
            returns the plain decoded text.

    Returns:
        The completion as text, special tokens removed.

    Raises:
        ValueError: If ``renderer_name`` is given and unknown.
    """
    if not tokens:
        return ""
    decoded = tokenizer.decode(list(tokens), skip_special_tokens=True)
    text = decoded if isinstance(decoded, str) else str(decoded)
    if renderer_name is None:
        return text
    spec = get_renderer(renderer_name)
    if spec.strip_completion is None:
        return text
    stripped = spec.strip_completion(text)
    return str(stripped)
