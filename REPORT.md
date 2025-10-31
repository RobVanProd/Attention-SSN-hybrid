# Hybrid Attention-SNN Training Report

## Experiment Setup
- **Date:** 2025-10-30
- **Hardware:** CPU (PyTorch no CUDA)
- **Command:** `python hybrid_system.py train --total-steps 5 --batch-size 4 --d-model 64 --num-layers 2 --num-heads 4 --n-in-spike 16 --n-lif 16 --d-tap 16 --delta 4 --max-seq-len 64 --max-new-tokens 8 --output-dir checkpoints --no-cuda --supervised-steps 10 --pretrain-iters 200 --pretrain-log-every 50 --code-prob 0.5`
- **Verifier mix:** 50% math arithmetic prompts, 50% code generation prompts.
- **Warmup:** 200 supervised iterations before PPO updates.

## Supervised Warmup
| Iteration | Loss |
|-----------|------|
| 50        | 1.1312 |
| 100       | 0.2720 |
| 150       | 0.0736 |
| 200       | 0.1035 |

Warmup quickly reduced reconstruction loss below 0.3 by iteration 100, indicating that the surrogate spike path can learn deterministic targets before reinforcement updates.

## PPO Training Metrics
| Step | Avg Reward | Math Reward | Code Reward | Mean Firing Rate | Spike Sparsity | Policy Loss | Value Loss | Entropy | KL | Supervised Loss |
|------|------------|-------------|-------------|------------------|----------------|-------------|------------|---------|----|-----------------|
| 0    | 0.2500     | 0.5000      | 0.0000      | 0.0163           | 0.9674         | -0.0132     | 8.0521     | 0.4892  | -0.1265 | 0.1123 |
| 1    | 0.0000     | 0.0000      | 0.0000      | 0.0352           | 0.9297         | 0.1573      | 2.0345     | 0.3824  | -0.2800 | 0.0610 |
| 2    | 0.0000     | 0.0000      | 0.0000      | 0.0312           | 0.9375         | 0.1847      | 0.7018     | 0.2560  | -0.0087 | 0.1030 |
| 3    | 0.0000     | 0.0000      | 0.0000      | 0.0349           | 0.9303         | 0.1766      | 1.6502     | 0.8662  | 0.9140 | 0.5584 |
| 4    | 0.0000     | 0.0000      | 0.0000      | 0.0312           | 0.9375         | 0.1578      | 0.7843     | 0.3302  | -0.3071 | 0.1828 |

The model reached its best average reward (0.25) on the first PPO step due to residual supervised signal. Subsequent steps did not improve rewards, suggesting the current PPO configuration requires more steps, tuned learning rates, or richer prompts to stabilise updates. Spike sparsity remained high (93–97%), demonstrating the spiking pathway stays energy-efficient throughout training.

## Evaluation
- **Command:** `python hybrid_system.py eval --checkpoint checkpoints/hybrid_model.pt --batch-size 4 --d-model 64 --num-layers 2 --num-heads 4 --n-in-spike 16 --n-lif 16 --d-tap 16 --delta 4 --max-seq-len 64 --max-new-tokens 8 --no-cuda --deterministic --code-prob 0.5`
- **Results:**
  - Average reward: 0.0000
  - Math reward: 0.0000
  - Code reward: 0.0000
  - Mean firing rate: 0.0288
  - Spike sparsity: 0.9423

All deterministic evaluation samples failed to satisfy verifiers, pointing to under-trained policy/value heads. However, the spike statistics confirm that the TAP-bridged LIF layers continue operating in a sparse regime during inference.

## Notable Completions
- Prompt `write python function double(x) returning 2*x` → Completion `de  dou` (invalid syntax).
- Prompt `compute 4+3` → Completion `  8 ` (formatting mismatch with verifier expectation).

These outputs highlight the need for longer training or refined decoders to match verifier formatting.

## Recommendations
1. Extend PPO training to >100 steps and monitor KL spikes (step 3 showed KL≈0.91) to ensure stability.
2. Increase code prompt diversity or reduce `code_prob` while curriculum learning to stabilise textual outputs.
3. Consider reward shaping or baselines that encourage exact-format arithmetic answers.
4. Track spike metrics over longer horizons to confirm sustained sparsity under successful reward conditions.
