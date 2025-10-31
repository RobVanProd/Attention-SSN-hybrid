# Attention-SSN-hybrid

This project provides a runnable reference implementation of a Transformer/spiking neural network hybrid with a Temporal Attention Projection (TAP) bridge and PPO training loop driven by verifiable math/code rewards.

## Getting started

```bash
pip install torch

# train with structure-constrained decoding and GRPO-style groups
python hybrid_system.py train \
  --total-steps 200 \
  --batch-size 4 \
  --d-model 64 --num-layers 2 --num-heads 4 \
  --n-in-spike 16 --n-lif 16 --d-tap 16 --delta 8 \
  --answer-tags --enforce-structure --math-integer-only \
  --math-format-bonus 0.2 --code-compile-bonus 0.2 \
  --group-k 6 --ppo-clip 0.15 --gae-lambda 0.95 --gamma 0.99 \
  --target-kl 0.08 --kl-adapt 1.5 --value-clip --grad-norm 1.0 \
  --critic-lr 4e-4 --cosine-lr --spike-gain 1.3 \
  --output-dir checkpoints --no-cuda

# evaluate with the same structured decoding constraints
python hybrid_system.py eval \
  --checkpoint checkpoints/hybrid_model.pt \
  --answer-tags --enforce-structure --math-integer-only \
  --math-format-bonus 0.2 --code-compile-bonus 0.2 \
  --group-k 6 --no-cuda --deterministic
```

### Notable flags

* `--answer-tags` + `--enforce-structure` – force generations to follow `<think>...</think><answer>...</answer>` with constrained digits/code inside the answer span via the `StructureController`.
* `--math-format-bonus`, `--code-compile-bonus` – provide format/compilation bonuses inside the verifiable reward calculation.
* `--group-k` – enable GRPO-style group baselines; advantages are computed as reward minus group mean before PPO updates.
* `--ppo-clip`, `--target-kl`, `--kl-adapt`, `--value-clip`, `--grad-norm`, `--critic-lr`, `--cosine-lr` – expose PPO/optimization hyper-parameters for stabilising training on CPU-sized runs.
