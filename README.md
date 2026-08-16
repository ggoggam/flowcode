# flowcode

GFlowNet fine-tuning of coding LLMs, on hosted [Tinker](https://tinker-docs.thinkingmachines.ai)
or on your own GPUs/TPUs. Trains a policy to sample solutions *in proportion to* how many
tests they pass, `p(x) ∝ R(x)`, rather than to maximize the pass rate — the bet being that
this finds more distinct working solutions per problem than reward-maximizing RL, which
converges to one.

Tinker has no trajectory-balance loss and no way to ship one to the server. The whole thing
rests on a bridge: `cross_entropy` computes `sum(-logprobs * weights)` with arbitrary-signed
per-token weights, so sending `weights = -dC/dlogprobs` backpropagates *any* client-side
differentiable loss. See [docs/DESIGN.md](docs/DESIGN.md).

## Status

Objectives are implemented and verified against exactly-computable toy distributions. The
training loop runs end to end against the code-execution environment. **The hypothesis itself
is untested** — nothing here has yet compared GFlowNet fine-tuning to a PPO baseline at equal
spend.

The local backend's gradient path is verified against single-pass autograd
(`tests/test_local_backend.py`), and everything around the sampling engine is unit-tested.
**Neither vLLM engine path has been run on real hardware yet**; `mise run test:hardware` is
the gate for that.

## Objectives

| Objective | Needs a learned flow? | Notes |
|---|---|---|
| `vargrad` | no | **Default.** Estimates `log Z` in-batch from the group; no partition function to tune. |
| `tb` | `log Z(x)` only | Trajectory Balance (Malkin et al. 2022). `log Z` is an embedding table over tasks — exact for a fixed task set. |
| `subtb` | `log F(s)` | Sub-Trajectory Balance (Madan et al. 2023), λ-weighted. Denser credit assignment; leans on an approximate flow estimator. |
| `db` | `log F(s)` | Detailed Balance — the adjacent-pair case. Densest, most sensitive to flow-estimator error. |

All four converge to the target distribution on an enumerable toy problem (TV distance
0.014–0.018, KL 0.005–0.007 against the exact `p(x)`), sampled off-policy with no importance
correction. `tests/test_convergence.py` is the file that proves the mathematics independently of
any API call — calibration checks in the same file confirm an argmax-collapsed policy scores
TV 0.79, so the thresholds are not vacuous.

## Stack

Two backends behind one structural contract (`BackendLike` in `src/flowcode/train.py`), so
the loop, the objectives and the environment are identical either way:

| `backend=` | Policy lives | Sampling | Notes |
|---|---|---|---|
| `tinker` | Thinking Machines' servers | same service | The original path. Gradients reach it through the cross-entropy bridge in [docs/DESIGN.md](docs/DESIGN.md). |
| `local` | this process | vLLM, `sampler.mode=colocated\|remote` | transformers + peft + accelerate. Runs on GPU or TPU. |

- **torch** — every objective is plain autograd with no backend import anywhere in
  `objectives/`, so the maths is unit-testable offline at zero cost. Neither backend
  changed a line of it.
- **[vLLM](https://docs.vllm.ai)** — continuous batching and paged KV for the local path,
  with the same API on CUDA and TPU. `sampler.mode=colocated` shares devices with the
  trainer; `remote` reaches an engine on another host and does not need vllm installed
  locally at all.
- **Hydra + OmegaConf** — structured configs, so a typo in an override fails at composition
  rather than ten minutes into a paid run. `--multirun` sweeps objectives directly.
- **uv** — src-layout, absolute imports enforced by ruff (`TID252`).

## Local development

Requires [mise](https://mise.jdx.dev). Copy `.env.example` to `.env` and set `TINKER_API_KEY`;
mise loads it automatically. Nothing below the `test` line needs a key.

```sh
mise install                 # uv, prek
mise run sync                # resolve and install into .venv
mise run test                # unit tests; no network, no API spend
```

| Task | What it does |
|------|--------------|
| `mise run sync` | every extra except `vllm`, which has no macOS wheel |
| `mise run sync:vllm` | adds the engine; Linux + GPU/TPU only |
| `mise run test` | pytest, excluding network-, API-, slow- and hardware-marked tests |
| `mise run test:all` | adds the dataset-loader tests that hit HuggingFace |
| `mise run test:slow` | the convergence suite only; minutes, no API spend |
| `mise run test:hardware` | sampler/trainer logprob agreement; needs a real accelerator |
| `mise run lint` | ruff check + `ty` type check |
| `mise run fmt` | ruff format + safe autofixes |
| `mise run check` | lint + test |
| `mise run pre-commit` | every hook in `.pre-commit-config.yaml`, via prek |
| `mise run cost` | price a config before spending anything |
| `mise run train` | run a training job |
| `mise run sweep` | Hydra `--multirun` sweep |

## Running

Everything is a Hydra override. Price it first:

```sh
mise run cost                                    # what the default config would cost
mise run train -- train=smoke env=fixtures       # ~$0.02, no dataset download, proves the loop turns
mise run train -- objective=subtb env=mbpp       # a real run
mise run sweep -- objective=tb,subtb,vargrad,db  # the actual experiment
```

### On your own hardware

`backend=local` puts the policy in this process. Nothing else about the run changes.

```sh
# One GPU or chip: engine and trainer share it.
mise run train -- backend=local sampler=colocated env=mbpp

# Engine on other devices, reached over its OpenAI-compatible server. The server needs
# --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1, and backend.adapter_dir has to be
# readable from the server's host.
mise run train -- backend=local sampler=remote sampler.base_url=http://10.0.0.4:8000
```

**Run `mise run test:hardware` once on new hardware before trusting a run.** It checks that
the sampler and the trainer agree about what the policy is. When they do not,
`on_policy_only` trains against logprobs the trainer never produced and importance weights
are wrong — and nothing raises, nothing looks off in the metrics.

Turn on the decoupled sampler to stop the accelerator idling between steps:

```sh
mise run train -- backend=local train.producer.enabled=true train.producer.concurrency=32
```

This trains on trajectories a few policy updates old, which the GFlowNet objectives permit
by construction and PPO would not — see `src/flowcode/producer.py`. Watch
`producer/waits`: near zero after warmup means the sampler is keeping up. Note that
`producer.concurrency * env.workers` is the peak number of sandboxed subprocesses, so bring
`env.workers` down as concurrency goes up.

Checkpoints cover the policy, the flow parameters, the replay buffer and the step index
together — TRC quota is preemptible, and a resume missing any of those is a different run
wearing the same weights.

`flowcode-cost` needs no API key and touches no network — it exists to answer "how much will
this cost me" before you commit.

## Cost

Measured by the estimator, not guessed. Default config = 8 prompts × 8 samples/step, ~700
tokens/trajectory, Qwen3-8B at Aug 2026 prices:

| Config | $/step | 1000 steps |
|---|---|---|
| defaults (`vargrad`, replay on) | $0.057 | $57 |
| `train.on_policy_only=true` | $0.037 | $37 |
| `model=gpt-oss-20b` | $0.049 | $49 |
| `train=smoke env=fixtures` | $0.005 | $0.02 (5 steps) |

The gap between the first two rows is the gradient bridge: the loop normally makes **two**
training passes per step — a `forward()` logprob oracle plus the `forward_backward()` gradient
push — so train tokens are 2×. Strictly on-policy runs skip the oracle and reuse the sampler's
own logprobs. That is incompatible with replay, and the trainer refuses the combination rather
than training on stale logprobs.

Prices are a hardcoded Aug 2026 snapshot and will drift; override via `cost.price_overrides`.
After a real run, `cost.from_usage` reports what was actually spent from observed token counts —
worth comparing against the estimate, since the model assumes prefix-cache hits are not billed
at the sampling rate.

## Reward

`log R = β · log(max(pass_fraction, floor))` from running generated code against hidden tests in
a subprocess sandbox. Tests run individually so partial credit is graded rather than binary —
this is load-bearing, not a nicety: under all-or-nothing scoring every sample in a group lands
on the reward floor and VarGrad's group variance, which *is* the learning signal, goes to zero.

Datasets: MBPP and HumanEval via `datasets` (`uv sync --extra data`), plus 30 hand-written
offline fixtures so the whole test suite runs with no network. HumanEval's single `check()` is
split per-assertion by AST — otherwise its partial credit would be a fiction.

**The sandbox is a safety net against accidents, not a security boundary.** It stops runaway
loops, memory and fork bombs, and credential inheritance (a completion printing `os.environ`
cannot see `TINKER_API_KEY` — there is a test). It does not stop code that deliberately reads
your filesystem or opens a socket. See [docs/DESIGN.md](docs/DESIGN.md#4-what-the-sandbox-is-and-is-not).

## Not implemented

- **A multi-turn agentic environment.** Trajectories are segmented (`Segment` boundaries are
  where SubTB's intermediate terms attach) and `granularity: turn` is wired through, but the
  current environment is single-turn code generation, so turn-level collapses to one segment.
  A tool-calling environment is the natural next step and the setting where turn-level SubTB
  should actually earn its keep.
- **A PPO baseline.** Without one there is no measurement of the diversity claim, which is the
  entire point.
- **Flow heads on the model.** Tinker gives no access to hidden states, so `log F` is estimated
  client-side from coarse features. This is the honest weak point of `subtb` and `db`, and the
  reason `vargrad` is the default.
