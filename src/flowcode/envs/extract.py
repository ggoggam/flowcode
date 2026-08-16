"""Pull the Python out of a model completion.

A chat model asked for code answers with prose, then a fence, then code, then more prose
— or four fences, or a fence it forgot to close because it hit the token limit, or the
prompt restated back with the solution glued underneath. What gets executed is the
reward, so extraction is not cosmetic: pointing the sandbox at the model's explanation
instead of its code turns a correct sample into a syntax error, and in a GFlowNet that
mis-scored sample is not noise to be averaged out, it is a wrong entry in the target
distribution.

Everything here is a pure function of the completion string. No config, no regex that
depends on the dataset, nothing that varies between processes.

The extraction ladder, in order:

1. Fenced blocks tagged ``python`` / ``py`` / ``python3``. Take the **last** complete
   one: models write "here's a first attempt ... actually, here's the fix", and the fix
   is what they mean.
2. Any complete fenced block, preferring the last one whose contents parse as Python.
   Untagged fences are extremely common, and a block tagged ``sh`` sitting after the
   real answer should not win.
3. An unterminated fence — the completion ran out of tokens mid-block. Everything after
   the opening fence is the best guess.
4. No fences at all: the whole completion, with leading prose stripped and trailing prose
   trimmed until it parses.
"""

from __future__ import annotations

import re
from typing import Final

__all__ = ["extract_code", "extract_code_blocks", "is_parseable"]

_PYTHON_TAGS: Final[frozenset[str]] = frozenset({"python", "py", "python3", "py3", "pycon"})

_FENCE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>```+|~~~+)(?P<info>.*)$"
)

_CODE_START_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:from\s+[\w.]+\s+import\b|import\s+\w|def\s+\w|async\s+def\s+\w|class\s+\w|@\w"
    r"|if\s+__name__\b|#!|#\s|\w+\s*=\s*\S|try:|with\s+\w|for\s+\w|while\s+\w)"
)

_MAX_TRIM_LINES: Final[int] = 200
"""Bound on the trailing-prose trim loop. A pathological completion must not make
extraction quadratic in its own length."""


def is_parseable(text: str) -> bool:
    """True if ``text`` compiles as a Python module.

    Uses :func:`compile` rather than :mod:`ast` so that the answer matches exactly what
    the scoring harness will decide about the same string.
    """
    if not text.strip():
        return False
    try:
        compile(text, "<extracted>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _strip_repl_prompts(text: str) -> str:
    """Turn a pasted ``>>>`` session into a script, if that is what this is.

    Only fires when every non-blank line is a prompt line, so ordinary code containing a
    doctest in a docstring is left alone.
    """
    lines = text.splitlines()
    meaningful = [ln for ln in lines if ln.strip()]
    if not meaningful or not all(
        ln.lstrip().startswith((">>> ", "... ", ">>>", "...")) for ln in meaningful
    ):
        return text
    out: list[str] = []
    for ln in lines:
        stripped = ln.lstrip()
        if stripped.startswith((">>> ", "... ")):
            out.append(stripped[4:])
        elif stripped in {">>>", "..."} or not stripped:
            out.append("")
    return "\n".join(out)


def extract_code_blocks(completion: str) -> list[tuple[str, str, bool]]:
    """Split out every fenced block.

    Args:
        completion: Raw model output.

    Returns:
        One ``(info, body, closed)`` triple per opening fence, in order of appearance.
        ``info`` is the lower-cased language tag (``""`` when absent), ``body`` is the
        block contents with the fence's own indentation removed, and ``closed`` says
        whether a matching closing fence was found. An unterminated final block appears
        with ``closed=False`` rather than being dropped — a completion truncated by the
        token limit still contains most of its answer.
    """
    blocks: list[tuple[str, str, bool]] = []
    lines = completion.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        match = _FENCE_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        indent = match.group("indent")
        fence = match.group("fence")
        info = match.group("info").strip().lower()
        # A tag like ```python {highlight=1} or ```{.python} — take the first word.
        info = re.sub(r"[^a-z0-9+#]+", " ", info).strip().split(" ")[0] if info else ""
        body: list[str] = []
        i += 1
        closed = False
        while i < n:
            candidate = lines[i].strip()
            if candidate.startswith(fence[0] * len(fence)) and set(candidate) <= {fence[0]}:
                closed = True
                i += 1
                break
            line = lines[i]
            if indent and line.startswith(indent):
                line = line[len(indent) :]
            body.append(line)
            i += 1
        blocks.append((info, "\n".join(body).strip("\n"), closed))
    return blocks


def _trim_to_parseable(text: str) -> str:
    """Drop leading prose and trailing prose until what is left parses.

    Two cheap passes rather than a search: first skip forward to the first line that
    looks like the start of Python, then, if it still does not parse, shave lines off the
    end one at a time. Trailing prose ("This solution runs in O(n) time.") is the common
    case, and it is always at the end.
    """
    text = text.strip("\n")
    if not text.strip():
        return ""
    if is_parseable(text):
        return text

    lines = text.splitlines()
    for start, line in enumerate(lines):
        if _CODE_START_RE.match(line.strip()):
            head_trimmed = "\n".join(lines[start:])
            if is_parseable(head_trimmed):
                return head_trimmed
            lines = lines[start:]
            break

    for _ in range(min(_MAX_TRIM_LINES, len(lines))):
        lines = lines[:-1]
        candidate = "\n".join(lines)
        if not candidate.strip():
            break
        if is_parseable(candidate):
            return candidate
    return text


def _looks_like_code(text: str) -> bool:
    """Whether text that failed to parse was nonetheless *meant* as code.

    Two signals. A line that starts like a statement (``def``, ``import``, an
    assignment) is the obvious one. An indented first line is the subtler one: a model
    answering a HumanEval prompt often continues the signature it was given instead of
    restating it, so the completion is a bare function body that cannot parse on its own
    until the task's stub is prepended back. Treating that as prose would score a
    perfectly good answer as an empty completion.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    if lines[0].startswith((" ", "\t")):
        return True
    return any(_CODE_START_RE.match(ln.strip()) for ln in lines)


def _pick_block(blocks: list[tuple[str, str, bool]]) -> str | None:
    """Choose one block out of several, or ``None`` if none is usable."""
    closed = [(info, body) for info, body, ok in blocks if ok and body.strip()]

    tagged = [body for info, body in closed if info in _PYTHON_TAGS]
    if tagged:
        for body in reversed(tagged):
            if is_parseable(body):
                return body
        return tagged[-1]

    untagged = [body for info, body in closed if not info]
    if untagged:
        for body in reversed(untagged):
            if is_parseable(body):
                return body
        return untagged[-1]

    if closed:
        # Only foreign-tagged blocks (```sh, ```text). Accept one only if it parses —
        # otherwise fall through to the unfenced path, which may do better.
        for _info, body in reversed(closed):
            if is_parseable(body):
                return body

    unterminated = [body for _info, body, ok in blocks if not ok and body.strip()]
    if unterminated:
        return unterminated[-1]
    return None


def extract_code(completion: str) -> str:
    """Extract the Python program from a model completion.

    Args:
        completion: Raw model output, fences and prose and all.

    Returns:
        The extracted source, or ``""`` when the completion contains nothing that could
        plausibly be code. An empty return is meaningful: the caller scores it as
        :data:`flowcode.envs.base.ERROR_EMPTY` rather than as a syntax error, which keeps
        "the model refused / wrote only prose" separable from "the model wrote broken
        Python" in the training metrics.

    Note:
        Text that is clearly *meant* to be code but does not parse is returned as-is, so
        that it lands in the syntax-error bucket instead of silently disappearing.
    """
    if not completion or not completion.strip():
        return ""

    blocks = extract_code_blocks(completion)
    chosen = _pick_block(blocks) if blocks else None
    if chosen is None:
        chosen = completion

    chosen = _strip_repl_prompts(chosen)
    result = _trim_to_parseable(chosen)
    if not result.strip():
        return ""
    # Pure prose with no code-looking line at all: refuse rather than hand the sandbox a
    # sentence to compile.
    if not is_parseable(result) and not _looks_like_code(result):
        return ""
    return result.strip("\n")
