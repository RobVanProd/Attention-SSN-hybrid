# Attention-SSN-hybrid

This project provides a runnable reference implementation of a Transformer/spiking neural network hybrid with a Temporal Attention Projection (TAP) bridge and PPO training loop driven by verifiable math/code rewards.

## Getting started

```bash
pip install torch
python hybrid_system.py train --total-steps 2
python hybrid_system.py eval --checkpoint checkpoints/hybrid_model.pt
```
