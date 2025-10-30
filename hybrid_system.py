"""Hybrid Attention-SNN Transformer with TAP bridge and PPO training.

This module implements a runnable toy example that combines a Transformer
encoder/decoder hybrid with spiking neural networks, temporal attention
projection (TAP), and a PPO-style reinforcement learning loop driven by
verifiable rewards (math and code unit-test tasks).
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# -----------------------
# Utility helpers
# -----------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# -----------------------
# Tokeniser
# -----------------------


class SimpleTokenizer:
    """Character-level tokenizer with a small, fixed vocabulary."""

    def __init__(self) -> None:
        base_tokens = list("abcdefghijklmnopqrstuvwxyz0123456789+-*/=() :\n")
        special_tokens = ["<pad>", "<bos>", "<eos>"]
        vocab = special_tokens + base_tokens
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token: Dict[int, str] = {i: tok for tok, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        for ch in text.lower():
            tokens.append(self.token_to_id.get(ch, self.pad_id))
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, tokens: Sequence[int], skip_special: bool = True) -> str:
        chars: List[str] = []
        for tid in tokens:
            tok = self.id_to_token.get(int(tid), "")
            if skip_special and tok in ("<pad>", "<bos>", "<eos>"):
                continue
            chars.append(tok)
        return "".join(chars)


# -----------------------
# Core modules: Spike encoder, LIF, TAP, attention
# -----------------------


class SpikeEncoder(nn.Module):
    def __init__(self, d_model: int, n_in: int, mode: str = "poisson", gain: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, n_in)
        self.mode = mode
        self.gain = gain
        self.stochastic = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(self.proj(x)) * self.gain
        if not self.stochastic:
            return prob
        if self.mode == "threshold":
            thresh = torch.rand_like(prob)
            return (prob > thresh).float()
        return torch.bernoulli(prob.clamp(0.0, 1.0))


class LIF(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        beta: float = 0.95,
        theta: float = 1.0,
        v_reset: float = 0.0,
        surrogate_scale: float = 10.0,
    ):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_in, n_out) * 0.02)
        self.beta = beta
        self.theta = theta
        self.v_reset = v_reset
        self.surrogate_scale = surrogate_scale

    def forward(self, S_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = S_in.shape
        device = S_in.device
        n_out = self.W.shape[1]
        V = torch.zeros(B, n_out, device=device)
        V_trace = []
        S_out_list = []
        for t in range(T):
            I_t = S_in[:, t, :] @ self.W
            V = self.beta * V + I_t
            fire_binary = (V >= self.theta).float()
            surrogate = torch.sigmoid((V - self.theta) * self.surrogate_scale)
            fire = fire_binary + surrogate - surrogate.detach()
            V_trace.append(V)
            S_out_list.append(fire)
            V = torch.where(fire.bool(), torch.full_like(V, self.v_reset), V)
        S_out = torch.stack(S_out_list, dim=1)
        V_tr = torch.stack(V_trace, dim=1)
        return S_out, V_tr


class TAP(nn.Module):
    def __init__(self, n_spike: int, d_model: int, d_tap: int, delta: int):
        super().__init__()
        self.delta = delta
        self.conv = nn.Conv1d(n_spike, d_tap, kernel_size=delta, stride=1, padding=0, bias=False)
        nn.init.xavier_uniform_(self.conv.weight)
        self.WQ = nn.Linear(d_model + d_tap, d_model)
        self.WK = nn.Linear(d_model + d_tap, d_model)
        self.WV = nn.Linear(d_model, d_model)

    def forward(self, H: torch.Tensor, S_hist: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = H.shape
        x = S_hist.transpose(1, 2)
        x = F.pad(x, (self.delta - 1, 0))
        SH = self.conv(x)
        SH = SH.transpose(1, 2)
        if SH.shape[1] > T:
            SH = SH[:, -T:, :]
        elif SH.shape[1] < T:
            pad = torch.zeros(B, T - SH.shape[1], SH.shape[2], device=H.device, dtype=H.dtype)
            SH = torch.cat([pad, SH], dim=1)
        enriched = torch.cat([H, SH], dim=-1)
        Q = self.WQ(enriched)
        K = self.WK(enriched)
        V = self.WV(H)
        return Q, K, V


class TAPSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_spike: int, d_tap: int, delta: int):
        super().__init__()
        self.tap = TAP(n_spike, d_model, d_tap, delta)
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.transpose(1, 2)  # [B, num_heads, T, d_head]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, num_heads, T, d_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(B, T, num_heads * d_head)

    def forward(self, H: torch.Tensor, S_hist: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Q, K, V = self.tap(H, S_hist)
        q = self._split_heads(Q)
        k = self._split_heads(K)
        v = self._split_heads(V)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        merged = self._merge_heads(context)
        return self.out_proj(merged)


class HybridBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, n_in_spike: int, n_lif: int, d_tap: int, delta: int):
        super().__init__()
        self.tap_window = delta
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = TAPSelfAttention(d_model, num_heads, n_lif, d_tap, delta)
        self.spike_encoder = SpikeEncoder(d_model, n_in_spike)
        self.lif = LIF(n_in_spike, n_lif)
        self.proj_back = nn.Linear(n_lif, d_model)

    def forward(
        self,
        H: torch.Tensor,
        spike_hist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm_spike = self.norm2(H)
        S_in = self.spike_encoder(norm_spike)
        S_out, V_tr = self.lif(S_in)
        updated_hist = torch.cat([spike_hist, S_out], dim=1)
        if updated_hist.shape[1] > self.tap_window:
            updated_hist = updated_hist[:, -self.tap_window :, :]
        attn_out = self.attn(self.norm1(H), updated_hist)
        H1 = H + attn_out
        H2 = H1 + self.proj_back(S_out.float())
        return H2, updated_hist, S_out, V_tr


class ActorCriticHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.policy = nn.Linear(d_model, vocab_size)
        self.value = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.policy(hidden)
        values = self.value(hidden).squeeze(-1)
        return logits, values


class HybridTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        n_in_spike: int = 64,
        n_lif: int = 64,
        d_tap: int = 64,
        delta: int = 8,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                HybridBlock(d_model, num_heads, n_in_spike, n_lif, d_tap, delta)
                for _ in range(num_layers)
            ]
        )
        self.head = ActorCriticHead(d_model, vocab_size)
        self.max_seq_len = max_seq_len
        self.delta = delta
        self.n_lif = n_lif

    def init_spike_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [torch.zeros(batch_size, self.delta, self.n_lif, device=device) for _ in self.layers]

    def set_spike_sampling(self, stochastic: bool) -> None:
        for layer in self.layers:
            if hasattr(layer, "spike_encoder"):
                layer.spike_encoder.stochastic = stochastic

    def forward(
        self,
        input_ids: torch.Tensor,
        spike_state: Optional[List[torch.Tensor]] = None,
        collect_spikes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B, T = input_ids.shape
        device = input_ids.device
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        if spike_state is None:
            spike_state = self.init_spike_state(B, device)
        else:
            spike_state = [state.clone() for state in spike_state]
        pos = self.pos_emb[:T, :]
        hidden = self.token_emb(input_ids) + pos.unsqueeze(0)
        spike_records: List[torch.Tensor] = []
        for idx, (layer, state) in enumerate(zip(self.layers, spike_state)):
            hidden, new_state, S_out, _ = layer(hidden, state)
            spike_state[idx] = new_state
            if collect_spikes:
                spike_records.append(S_out.detach())
        logits, values = self.head(hidden)
        return logits, values, spike_records


# -----------------------
# Verifier tasks and environment
# -----------------------


@dataclass
class PromptSample:
    prompt: str
    payload: Dict[str, str]
    task_type: str


class VerifierEnv:
    def __init__(self, tokenizer: SimpleTokenizer, batch_size: int = 4, code_prob: float = 0.4) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.code_prob = code_prob

    def sample_batch(self) -> List[PromptSample]:
        samples: List[PromptSample] = []
        for _ in range(self.batch_size):
            if random.random() < 1.0 - self.code_prob:
                a, b = random.randint(0, 9), random.randint(0, 9)
                prompt = f"compute {a}+{b}"
                payload = {"answer": str(a + b)}
                samples.append(PromptSample(prompt, payload, "math"))
            else:
                prompt = "write python function double(x) returning 2*x"
                payload = {"tests": "assert double(3)==6\nassert double(-2)==-4"}
                samples.append(PromptSample(prompt, payload, "code"))
        return samples

    def target_completion(self, sample: PromptSample) -> str:
        if sample.task_type == "math":
            answer = sample.payload["answer"]
            return f"\n{answer}\n"
        return "\ndef double(x):\n    return 2*x\n"


def extract_final_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        start = text.index("<answer>") + len("<answer>")
        end = text.index("</answer>")
        return text[start:end].strip()
    if "boxed{" in text:
        start = text.index("boxed{") + len("boxed{")
        end = text.find("}", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip().splitlines()[-1].strip()


def math_verifier(output_text: str, ground_truth: str) -> float:
    try:
        pred = extract_final_answer(output_text)
        return 1.0 if pred == ground_truth else 0.0
    except Exception:
        return 0.0


def safe_exec(code: str, tests: str) -> float:
    loc: Dict[str, object] = {}
    try:
        compiled = compile(code, "generated", "exec")
        exec(compiled, {"__builtins__": {"range": range, "len": len}}, loc)
        compiled_tests = compile(tests, "tests", "exec")
        exec(compiled_tests, {**loc}, {})
        return 1.0
    except Exception:
        return 0.0


def code_verifier(output_text: str, tests: str) -> float:
    return safe_exec(output_text, tests)


# -----------------------
# PPO utilities
# -----------------------


@dataclass
class Trajectory:
    tokens: List[int]
    log_probs: List[float]
    values: List[float]
    reward: float
    completion: str
    spike_records: List[torch.Tensor]
    prompt_len: int


def compute_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
        returns[t] = advantages[t] + values[t]
    return returns, advantages


@dataclass
class PPOBatch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    mask: torch.Tensor


def collate_trajectories(
    trajectories: List[Trajectory],
    tokenizer: SimpleTokenizer,
    device: torch.device,
) -> PPOBatch:
    max_len = max(len(traj.tokens) for traj in trajectories)
    input_ids = []
    target_ids = []
    old_log_probs = []
    advantages = []
    returns = []
    mask = []
    for traj in trajectories:
        tokens = traj.tokens
        inp = tokens[:-1]
        tgt = tokens[1:]
        seq_len = len(inp)
        pad = max_len - seq_len
        action_start = max(traj.prompt_len - 1, 0)
        action_len = len(traj.values)
        action_end = min(action_start + action_len, seq_len)
        logp_full = torch.zeros(seq_len, dtype=torch.float32)
        adv_full = torch.zeros(seq_len, dtype=torch.float32)
        ret_full = torch.zeros(seq_len, dtype=torch.float32)
        mask_full = torch.zeros(seq_len, dtype=torch.float32)
        if action_len > 0 and action_end > action_start:
            values = torch.tensor(traj.values, dtype=torch.float32)
            rewards = torch.zeros(action_len, dtype=torch.float32)
            rewards[-1] = traj.reward
            ret, adv = compute_returns_advantages(rewards, values)
            logp_full[action_start:action_end] = torch.tensor(traj.log_probs, dtype=torch.float32)
            adv_full[action_start:action_end] = adv
            ret_full[action_start:action_end] = ret
            mask_full[action_start:action_end] = 1.0
        inp_tensor = torch.tensor(inp + [tokenizer.pad_id] * pad, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt + [tokenizer.pad_id] * pad, dtype=torch.long)
        logp_tensor = torch.cat([logp_full, torch.zeros(pad, dtype=torch.float32)])
        adv_tensor = torch.cat([adv_full, torch.zeros(pad, dtype=torch.float32)])
        ret_tensor = torch.cat([ret_full, torch.zeros(pad, dtype=torch.float32)])
        mask_tensor = torch.cat([mask_full, torch.zeros(pad, dtype=torch.float32)])
        input_ids.append(inp_tensor)
        target_ids.append(tgt_tensor)
        old_log_probs.append(logp_tensor)
        advantages.append(adv_tensor)
        returns.append(ret_tensor)
        mask.append(mask_tensor)
    return PPOBatch(
        input_ids=torch.stack(input_ids).to(device),
        target_ids=torch.stack(target_ids).to(device),
        old_log_probs=torch.stack(old_log_probs).to(device),
        advantages=torch.stack(advantages).to(device),
        returns=torch.stack(returns).to(device),
        mask=torch.stack(mask).to(device),
    )


def supervised_step(
    model: HybridTransformer,
    optimizer: torch.optim.Optimizer,
    tokenizer: SimpleTokenizer,
    samples: List[PromptSample],
    env: VerifierEnv,
    device: torch.device,
    max_seq_len: int,
    weight: float = 1.0,
) -> float:
    sequences: List[List[int]] = []
    targets: List[List[int]] = []
    for sample in samples:
        prompt_ids = tokenizer.encode(sample.prompt, add_bos=True, add_eos=False)
        completion_text = env.target_completion(sample)
        completion_ids = tokenizer.encode(completion_text, add_bos=False, add_eos=True)
        full = prompt_ids + completion_ids
        if len(full) < 2:
            continue
        full = full[:max_seq_len]
        inp = full[:-1]
        tgt = full[1:]
        sequences.append(inp)
        targets.append(tgt)
    if not sequences:
        return 0.0
    max_len = max(len(seq) for seq in sequences)
    pad_id = tokenizer.pad_id
    input_batch = []
    target_batch = []
    for seq, tgt in zip(sequences, targets):
        pad = max_len - len(seq)
        input_batch.append(seq + [pad_id] * pad)
        target_batch.append(tgt + [pad_id] * pad)
    inputs = torch.tensor(input_batch, device=device, dtype=torch.long)
    target_tensor = torch.tensor(target_batch, device=device, dtype=torch.long)
    model.train()
    encoder_states: List[Tuple[SpikeEncoder, bool]] = []
    for layer in model.layers:
        if hasattr(layer, "spike_encoder"):
            encoder_states.append((layer.spike_encoder, layer.spike_encoder.stochastic))
            layer.spike_encoder.stochastic = False
    optimizer.zero_grad()
    logits, _, _ = model(inputs)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_tensor.reshape(-1),
        ignore_index=pad_id,
    )
    loss = loss * weight
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    for encoder, prev in encoder_states:
        encoder.stochastic = prev
    return float(loss.detach())


def ppo_update(
    model: HybridTransformer,
    batch: PPOBatch,
    optimizer: torch.optim.Optimizer,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    value_coef: float = 0.5,
    kl_coef: float = 0.02,
    ref_model: Optional[HybridTransformer] = None,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad()
    logits, values, _ = model(batch.input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, -1, batch.target_ids.unsqueeze(-1)).squeeze(-1)
    ratios = torch.exp(gathered_log_probs - batch.old_log_probs)
    advantages = batch.advantages
    if batch.mask.sum() > 0:
        masked_adv = advantages[batch.mask > 0]
        mean_adv = masked_adv.mean()
        std_adv = masked_adv.std(unbiased=False).clamp_min(1e-8)
        advantages = torch.where(
            batch.mask > 0,
            (advantages - mean_adv) / std_adv,
            torch.zeros_like(advantages),
        )
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -(torch.min(surr1, surr2) * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    value_loss = F.mse_loss(values.squeeze(-1), batch.returns, reduction="none")
    value_loss = (value_loss * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    entropy = -(log_probs * torch.exp(log_probs)).sum(-1)
    entropy = (entropy * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    kl_term = torch.tensor(0.0, device=batch.input_ids.device)
    if ref_model is not None:
        with torch.no_grad():
            ref_logits, _, _ = ref_model(batch.input_ids)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_selected = torch.gather(ref_log_probs, -1, batch.target_ids.unsqueeze(-1)).squeeze(-1)
        kl_term = ((gathered_log_probs - ref_selected) * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    loss = policy_loss + value_coef * value_loss - ent_coef * entropy + kl_coef * kl_term
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "kl": float(kl_term.detach()),
    }


# -----------------------
# Sampling and rollout
# -----------------------


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 0.7,
    deterministic: bool = False,
) -> Tuple[int, float]:
    log_probs = F.log_softmax(logits, dim=-1)
    if deterministic:
        token = int(log_probs.argmax().item())
        return token, float(log_probs[token].item())
    scaled_logits = logits / max(temperature, 1e-4)
    probs = torch.softmax(scaled_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative <= top_p
    mask[..., 0] = True
    filtered_probs = sorted_probs * mask
    denom = filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    filtered_probs = filtered_probs / denom
    dist = Categorical(filtered_probs)
    sampled_index = dist.sample()
    token = sorted_indices[..., sampled_index]
    selected_prob = filtered_probs.gather(-1, sampled_index.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    log_prob = torch.log(selected_prob)
    return token.item(), float(log_prob.item())


def rollout(
    env: VerifierEnv,
    model: HybridTransformer,
    tokenizer: SimpleTokenizer,
    device: torch.device,
    max_new_tokens: int = 8,
    top_p: float = 0.9,
    temperature: float = 0.7,
    samples: Optional[List[PromptSample]] = None,
    deterministic: bool = False,
) -> Tuple[List[Trajectory], Dict[str, float]]:
    if samples is None:
        samples = env.sample_batch()
    trajectories: List[Trajectory] = []
    rewards = []
    spike_logs: List[torch.Tensor] = []
    model.eval()
    encoder_states: List[Tuple[SpikeEncoder, bool]] = []
    if deterministic:
        for layer in model.layers:
            if hasattr(layer, "spike_encoder"):
                encoder_states.append((layer.spike_encoder, layer.spike_encoder.stochastic))
                layer.spike_encoder.stochastic = False
    for sample in samples:
        prompt_ids = tokenizer.encode(sample.prompt, add_bos=True, add_eos=False)
        generated: List[int] = prompt_ids.copy()
        log_probs: List[float] = []
        values: List[float] = []
        spike_record_steps: List[torch.Tensor] = []
        prompt_len = len(prompt_ids)
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], device=device, dtype=torch.long)
            logits, vals, spikes = model(input_tensor, collect_spikes=True)
            last_logits = logits[0, -1, :]
            last_value = vals[0, -1].item()
            token, logp = top_p_sampling(
                last_logits,
                top_p=top_p,
                temperature=temperature,
                deterministic=deterministic,
            )
            generated.append(token)
            log_probs.append(logp)
            values.append(last_value)
            if spikes:
                layer_last_spikes = [layer_spikes[:, -1, :] for layer_spikes in spikes]
                step_spikes = torch.stack(layer_last_spikes, dim=0).mean(0)
                spike_record_steps.append(step_spikes.cpu())
            if token == tokenizer.eos_id:
                break
        completion_tokens = generated[prompt_len:]
        completion_text = tokenizer.decode(completion_tokens)
        if sample.task_type == "math":
            reward = math_verifier(completion_text, sample.payload["answer"])
        else:
            reward = code_verifier(completion_text, sample.payload["tests"])
        rewards.append(reward)
        trajectories.append(
            Trajectory(
                tokens=generated,
                log_probs=log_probs,
                values=values,
                reward=reward,
                completion=completion_text,
                spike_records=spike_record_steps,
                prompt_len=prompt_len,
            )
        )
        if spike_record_steps:
            spike_logs.append(torch.stack(spike_record_steps, dim=0))
    metrics = {
        "avg_reward": float(sum(rewards) / max(len(rewards), 1)),
        "reward_math": float(sum(r for r, s in zip(rewards, samples) if s.task_type == "math") / max(sum(1 for s in samples if s.task_type == "math"), 1)),
        "reward_code": float(sum(r for r, s in zip(rewards, samples) if s.task_type == "code") / max(sum(1 for s in samples if s.task_type == "code"), 1)),
    }
    if spike_logs:
        stacked = torch.cat(spike_logs, dim=0)
        metrics.update(compute_spike_stats(stacked))
    else:
        metrics.update({"mean_firing_rate": 0.0, "spike_sparsity": 1.0})
    for encoder, prev in encoder_states:
        encoder.stochastic = prev
    return trajectories, metrics


def compute_spike_stats(spike_tensor: torch.Tensor) -> Dict[str, float]:
    if spike_tensor.numel() == 0:
        return {"mean_firing_rate": 0.0, "spike_sparsity": 1.0}
    mean_rate = float(spike_tensor.mean().item())
    sparsity = float((spike_tensor == 0).float().mean().item())
    return {"mean_firing_rate": mean_rate, "spike_sparsity": sparsity}


# -----------------------
# Training and evaluation entry points
# -----------------------


def train(args: argparse.Namespace) -> None:
    logging.info("Starting training")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer = SimpleTokenizer()
    model = HybridTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        n_in_spike=args.n_in_spike,
        n_lif=args.n_lif,
        d_tap=args.d_tap,
        delta=args.delta,
        max_seq_len=args.max_seq_len,
    ).to(device)
    ref_model = HybridTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        n_in_spike=args.n_in_spike,
        n_lif=args.n_lif,
        d_tap=args.d_tap,
        delta=args.delta,
        max_seq_len=args.max_seq_len,
    ).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    env = VerifierEnv(tokenizer, batch_size=args.batch_size, code_prob=args.code_prob)
    if args.pretrain_iters > 0:
        logging.info("Running supervised warmup for %d iterations", args.pretrain_iters)
        pretrain_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for i in range(args.pretrain_iters):
            sup_samples = env.sample_batch()
            sup_loss = supervised_step(
                model,
                pretrain_opt,
                tokenizer,
                sup_samples,
                env,
                device,
                args.max_seq_len,
                weight=args.supervised_weight,
            )
            if (i + 1) % max(args.pretrain_log_every, 1) == 0 or i + 1 == args.pretrain_iters:
                logging.info(
                    "pretrain %d/%d supervised_loss=%.4f",
                    i + 1,
                    args.pretrain_iters,
                    sup_loss,
                )
        ref_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for step in range(args.total_steps):
        samples = env.sample_batch()
        sup_losses: List[float] = []
        for _ in range(args.supervised_steps):
            sup_loss = supervised_step(
                model,
                optimizer,
                tokenizer,
                samples,
                env,
                device,
                args.max_seq_len,
                weight=args.supervised_weight,
            )
            sup_losses.append(sup_loss)
        trajectories, rollout_metrics = rollout(
            env,
            model,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            samples=samples,
            deterministic=args.deterministic_rollout,
        )
        batch = collate_trajectories(trajectories, tokenizer, device)
        stats = ppo_update(
            model,
            batch,
            optimizer,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coef,
            value_coef=args.value_coef,
            kl_coef=args.kl_coef,
            ref_model=ref_model,
        )
        log_data = {**rollout_metrics, **stats}
        if sup_losses:
            log_data["supervised_loss"] = float(sum(sup_losses) / len(sup_losses))
        logging.info("step %d metrics: %s", step, {k: f"{v:.4f}" for k, v in log_data.items()})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "hybrid_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    logging.info("Saved checkpoint to %s", ckpt_path)


def evaluate(args: argparse.Namespace) -> None:
    logging.info("Starting evaluation")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    tokenizer = SimpleTokenizer()
    model = HybridTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        n_in_spike=args.n_in_spike,
        n_lif=args.n_lif,
        d_tap=args.d_tap,
        delta=args.delta,
        max_seq_len=args.max_seq_len,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    env = VerifierEnv(tokenizer, batch_size=args.batch_size, code_prob=args.code_prob)
    samples = env.sample_batch()
    trajectories, metrics = rollout(
        env,
        model,
        tokenizer,
        device,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        samples=samples,
        deterministic=args.deterministic,
    )
    for sample, traj in zip(samples, trajectories):
        logging.info(
            "Prompt: %s | completion: %s | reward=%.3f",
            sample.prompt,
            traj.completion.replace("\n", " "),
            traj.reward,
        )
    logging.info("Evaluation metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid Attention-SNN Transformer trainer/evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(p: argparse.ArgumentParser) -> None:
        p.add_argument("--d-model", type=int, default=128)
        p.add_argument("--num-layers", type=int, default=3)
        p.add_argument("--num-heads", type=int, default=4)
        p.add_argument("--n-in-spike", type=int, default=32)
        p.add_argument("--n-lif", type=int, default=32)
        p.add_argument("--d-tap", type=int, default=32)
        p.add_argument("--delta", type=int, default=6)
        p.add_argument("--max-seq-len", type=int, default=64)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--max-new-tokens", type=int, default=6)
        p.add_argument("--top-p", type=float, default=0.9)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--no-cuda", action="store_true")
        p.add_argument("--code-prob", type=float, default=0.4)

    train_parser = subparsers.add_parser("train", help="Train the hybrid model with PPO")
    add_shared_arguments(train_parser)
    train_parser.add_argument("--total-steps", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--clip-range", type=float, default=0.2)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--kl-coef", type=float, default=0.02)
    train_parser.add_argument("--output-dir", type=str, default="checkpoints")
    train_parser.add_argument("--supervised-steps", type=int, default=1)
    train_parser.add_argument("--supervised-weight", type=float, default=1.0)
    train_parser.add_argument("--deterministic-rollout", action="store_true")
    train_parser.add_argument("--pretrain-iters", type=int, default=0)
    train_parser.add_argument("--pretrain-log-every", type=int, default=50)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    add_shared_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    eval_parser.add_argument("--deterministic", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(42)
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
