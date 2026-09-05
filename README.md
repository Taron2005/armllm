# armllm — Armenian SFT of Qwen3-4B with QLoRA

Supervised fine-tuning of **`Qwen/Qwen3-4B-Instruct-2507`** for **Armenian instruction following**, using a native Hugging Face **QLoRA** pipeline (4-bit NF4 base + trainable LoRA adapters), a **3-stage learning-rate curriculum**, **assistant-only loss masking**, and a **custom bilingual benchmark evaluator** (Armenian/English Wikipedia LM loss + Tatoeba cross-lingual similarity) that runs **inside training** and logs every checkpoint to **Weights & Biases**.

Everything here is hand-built on top of `transformers` + `peft` + `bitsandbytes` — no Unsloth, no TRL — so every stage of the pipeline (tokenization, masking, collation, schedulers, evaluation) is explicit and inspectable in `src/arm_llm/`.

- **Base model:** [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) (151,936 vocab, stock Qwen3 tokenizer — this project does **not** modify the tokenizer)
- **Hardware:** single consumer GPU, fp16 (no bf16), batch size 1 × gradient accumulation 8, gradient checkpointing
- **Tracking:** [wandb.ai/taronbabayan4-yerevan-state-university-ysu/armenian-qwen3-sft](https://wandb.ai/taronbabayan4-yerevan-state-university-ysu/armenian-qwen3-sft)

---

## Results snapshot

Final custom-benchmark values from the **stage-2 (constant-LR) run** ([`y3tndhs9`](https://wandb.ai/taronbabayan4-yerevan-state-university-ysu/armenian-qwen3-sft/runs/y3tndhs9)), logged by the in-training evaluator at the last eval step:

| Metric | Value | What it measures |
|---|---|---|
| Armenian Wikipedia PPL | **2.54** (loss 0.933) | Causal-LM perplexity on held-out Armenian Wikipedia text (512-token blocks) |
| English Wikipedia PPL | **10.11** (loss 2.313) | Same protocol on English — the "don't forget English" watch-metric |
| Tatoeba hy–en cosine | **0.217** ± 0.058 | Mean cosine similarity between mean-pooled hidden states of 20 Armenian–English translation pairs (max 0.366) |
| SFT validation loss | **0.643** (PPL ≈ 1.90) | Trainer eval on the held-out Armenian SFT split, assistant tokens only |
| Train loss (stage avg) | 0.618 | Assistant-token loss over the stage-2 training split |
| Train runtime (stage 2) | ≈ 1.8 h (6,439 s, 241 optimizer steps) | Single consumer GPU, effective batch size 8 |

**Honest scope of these numbers.** The custom benchmarks are small, fixed, seeded **sanity evals** — 20 sampled documents per language (≤20 chunks × 512 tokens each; ≈7.9k Armenian / ≈7.0k English target tokens) and 20 Tatoeba pairs. They exist to compare checkpoints *within* a run and to watch for English regression while training on Armenian — they are **not** a claim of state-of-the-art Armenian performance. No base-model (pre-SFT) Wikipedia baseline is logged yet; the comparison baseline for the tokenizer-level work lives in a separate project.

**Stage-1 sanity check.** The [`stage1-warmup-wiki-checkpoints`](https://wandb.ai/taronbabayan4-yerevan-state-university-ysu/armenian-qwen3-sft/runs/6rye3mlg) run generates Wikipedia continuations greedily from the base model and from each warmup checkpoint and compares them to the real continuation (lexical SequenceMatcher ratio, side-by-side table logged to W&B). Result: mean similarity **0.1794 vs 0.1796 for the base** (Δ −0.0002 at step 246) — the warmup stage did not perturb base-model behavior before the constant-LR stage began, which is exactly what it was designed to verify.

---

## Why three stages

The learning-rate curriculum is split across stages, each continuing the **same LoRA adapter** (adapter chaining — not resume-from-checkpoint: the optimizer state is rebuilt, the adapter weights carry over as the trainable initialization):

| Stage | Scheduler | LR behavior | Status |
|---|---|---|---|
| 1 — warmup | `warmup_only` (whole stage = one long warmup, ratio 1.0) | 0 → 2e-4 | ✅ finished (246 steps) |
| 2 — stable | `constant` | 2e-4 held flat | ✅ finished (241 steps) |
| 3 — decay | `linear` | 2e-4 → 0 | 📋 config committed, not yet run |

This mirrors warmup–stable–decay (WSD) practice, but implemented as explicit stages so each phase gets its own run, config, and W&B dashboard. `schedulers.py` also ships reusable `cosine_min_lr`, `sine_decay`, and single-call `wsd` builders for ablations.

---

## Method details

### QLoRA configuration (all stages)

| Setting | Value |
|---|---|
| Quantization | 4-bit **NF4**, **double quant**, fp16 compute dtype (`bitsandbytes`) |
| LoRA | r=8, α=8, dropout 0.05, bias none, `CAUSAL_LM` |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all attention + MLP projections) |
| Optimizer | AdamW, lr 2e-4, weight decay 0.1, β = (0.9, 0.999) |
| Sequence | max_seq_length 1024, dynamic padding, no packing |
| Stability | gradient checkpointing, max_grad_norm 1.0, fp16 (bf16 off — consumer-GPU-safe), seed 3407 |

### Assistant-only loss masking

Training rows are pre-rendered Qwen chat text (`{"text": "<|im_start|>user … <|im_end|><|im_start|>assistant … <|im_end|>"}`). `data.py` tokenizes the **marker strings themselves** and finds them as token subsequences, then:

- labels only the assistant response tokens **plus the closing `<|im_end|>`** (so the model learns when to stop),
- masks everything else (system, user, role markers) to `-100`,
- **filters rows where masking left zero active labels** (those would produce NaN loss in `CrossEntropyLoss`), and prints a label-mask report (share of tokens actually trained on).

A `full_text` loss mode is available for plain LM training.

### In-training custom evaluation

A `TrainerCallback` runs `CustomBenchmarkEvaluator` every 10 optimizer steps, alongside the normal SFT eval loss:

- **Wikipedia LM eval** — deterministic (seeded) sample per language → 512-token blocks → token-weighted cross-entropy → per-language loss/PPL, plus token-weighted and macro averages across languages (the code logs both and documents why the token-weighted one is the mathematically correct global average).
- **Tatoeba cross-lingual alignment** — mean-pooled last-hidden-state embeddings for Armenian and English halves of translation pairs, L2-normalized, cosine similarity reported as mean/std/min/max.
- **Checkpoint generation audit** (`evaluate_wiki_checkpoints.py`) — greedy continuations from the base model vs every saved adapter checkpoint, scored by lexical similarity against the true continuation, with the full prompt/reference/generation table logged to W&B for qualitative review.

---

## Repository layout

```
configs/
  stage1_debug.yaml       # stage 1 — warmup (whole stage ramps 0 → 2e-4)
  stage2_train.yaml       # stage 2 — constant 2e-4, continues stage-1 adapter
  stage3_train.yaml       # stage 3 — linear decay to 0 (planned)
src/arm_llm/
  train.py                # end-to-end trainer: config → data → QLoRA → Trainer → adapter export
  data.py                 # JSONL loading, chat-marker validation, assistant-only labels, zero-label filtering
  collator.py             # dynamic padding (labels padded to -100), optional pad_to_multiple_of
  model.py                # 4-bit NF4 loading, LoRA attach/chain, Qwen <|im_end|> EOS/PAD setup
  optim.py                # AdamW over trainable (adapter) params only
  schedulers.py           # warmup_only / constant / linear / cosine_min_lr / sine_decay / WSD
  training_math.py        # effective batch size and step-count computations
  custom_eval.py          # Wikipedia PPL + Tatoeba similarity evaluator (callback-friendly)
  callbacks.py            # wires the custom evaluator into Trainer every N steps
  evaluate_wiki_checkpoints.py  # base-vs-checkpoint generation audit → W&B table
scripts/
  debug_*.py              # dataset/tokenization/batch/scheduler/model sanity scripts
```

## Getting started

```bash
pip install -r requirements.txt   # torch, transformers, accelerate, datasets, peft, bitsandbytes, wandb, pyyaml

export WANDB_API_KEY=...          # or wandb login

PYTHONPATH=src python -m arm_llm.train --config configs/stage2_train.yaml
```

1. Point `data.train_file` / `data.eval_file` at your JSONL with pre-rendered Qwen-chat text (`{"text": ...}`).
2. Point `custom_evaluation.wikipedia.path` at a parquet with `text` + `lang` columns, and `custom_evaluation.tatoeba.path` at a parquet with a `translation` dict column (`hy`/`en` keys).
3. Stage 2/3 configs load the previous stage's adapter via `lora.adapter_checkpoint` — set it to the exported `final_adapter` directory of the previous stage.

The SFT dataset and the adapter checkpoints are not redistributed in this repo.

## Roadmap

- [ ] Run stage 3 (linear decay) on the stage-2 adapter
- [ ] Log a pre-SFT base-model Wikipedia/Tatoeba baseline under the same protocol
- [ ] Scale the custom eval beyond the 20-doc/20-pair sanity sets
- [ ] Merge the final adapter into the base weights and publish a merged checkpoint
- [ ] Task-level Armenian evals (e.g., ArmBench via LightEval) to complement LM metrics
