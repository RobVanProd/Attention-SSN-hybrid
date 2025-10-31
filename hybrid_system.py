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
import re
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
        base_tokens = list("abcdefghijklmnopqrstuvwxyz0123456789+-*/=() :\n<>`")
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
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        n_in_spike: int,
        n_lif: int,
        d_tap: int,
        delta: int,
        spike_gain: float = 1.0,
    ):
        super().__init__()
        self.tap_window = delta
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = TAPSelfAttention(d_model, num_heads, n_lif, d_tap, delta)
        self.spike_encoder = SpikeEncoder(d_model, n_in_spike, gain=spike_gain)
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
        spike_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.layers = nn.ModuleList(
            [
                HybridBlock(d_model, num_heads, n_in_spike, n_lif, d_tap, delta, spike_gain=spike_gain)
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


@dataclass
class VerifierConfig:
    answer_tags: bool = False
    enforce_structure: bool = False
    math_integer_only: bool = True
    math_format_bonus: float = 0.0
    code_compile_bonus: float = 0.0


@dataclass
class StructureConfig:
    answer_tags: bool = False
    enforce_structure: bool = False
    math_integer_only: bool = True
    think_max_tokens: int = 24
    math_answer_max_tokens: int = 8
    code_answer_max_tokens: int = 64


class VerifierEnv:
    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        batch_size: int = 4,
        code_prob: float = 0.4,
        verifier_config: Optional[VerifierConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.code_prob = code_prob
        self.config = verifier_config or VerifierConfig()

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
        if self.config.answer_tags:
            if sample.task_type == "math":
                answer = sample.payload["answer"]
                return (
                    "<think>\n"
                    "add the two digits carefully\n"
                    "</think>\n"
                    "<answer>\n"
                    f"{answer}\n"
                    "</answer>\n"
                )
            return (
                "<think>\n"
                "write the function skeleton\n"
                "</think>\n"
                "<answer>\n"
                "```python\n"
                "def double(x):\n    return 2*x\n"
                "```\n"
                "</answer>\n"
            )
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


def math_verifier(output_text: str, ground_truth: str, config: VerifierConfig) -> float:
    try:
        pred = extract_final_answer(output_text)
    except Exception:
        return 0.0
    bonus = 0.0
    if config.math_format_bonus > 0.0:
        if "<answer>" in output_text and "</answer>" in output_text:
            bonus += min(0.5 * config.math_format_bonus, config.math_format_bonus)
        pattern = r"-?\d+" if config.math_integer_only else r"-?\d+(?:\.\d+)?"
        if re.fullmatch(pattern, pred.strip()):
            bonus += min(0.5 * config.math_format_bonus, config.math_format_bonus - bonus)
    target = ground_truth.strip()
    correct = float(pred.strip() == target)
    return float(min(1.0, bonus + correct))


def safe_exec(code: str, tests: str) -> Tuple[float, bool]:
    loc: Dict[str, object] = {}
    try:
        compiled = compile(code, "generated", "exec")
        exec(compiled, {"__builtins__": {"range": range, "len": len}}, loc)
        compiled_tests = compile(tests, "tests", "exec")
        exec(compiled_tests, {**loc}, {})
        return 1.0, True
    except SyntaxError:
        return 0.0, False
    except Exception:
        return 0.0, True


def code_verifier(output_text: str, tests: str, config: VerifierConfig) -> float:
    pass_rate, compiled = safe_exec(output_text, tests)
    bonus = config.code_compile_bonus if compiled else 0.0
    return float(min(1.0, pass_rate + bonus))


# -----------------------
# Structure controller for constrained decoding
# -----------------------


class StructureController:
    def __init__(
        self,
        tokenizer: SimpleTokenizer,
        config: StructureConfig,
        task_type: str,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.task_type = task_type
        self.force_queue: List[List[int]] = []
        self.force_states: List[Optional[str]] = []
        self.force_index = 0
        self._step_forced = False
        self.generated_text = ""
        self.answer_buffer = ""
        self.think_tokens = 0
        self.code_tokens = 0
        self.closing_enforced = False
        self.state = "free"
        if self.config.answer_tags:
            if self.config.enforce_structure:
                self.state = "forcing"
                self._enqueue_force("<think>\n", next_state="think")
            else:
                self.state = "think"
        self.math_allowed: List[int] = []
        if self.config.answer_tags:
            allowed_chars = set("0123456789-\n")
            if not self.config.math_integer_only:
                allowed_chars.add(".")
            for ch in allowed_chars:
                token_id = self.tokenizer.token_to_id.get(ch)
                if token_id is not None:
                    self.math_allowed.append(token_id)

    def _encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in text.lower():
            token_id = self.tokenizer.token_to_id.get(ch)
            if token_id is not None:
                ids.append(token_id)
        return ids

    def _enqueue_force(self, text: str, next_state: Optional[str] = None) -> None:
        ids = self._encode(text)
        if not ids:
            return
        self.force_queue.append(ids)
        self.force_states.append(next_state)

    def _answer_state(self) -> str:
        if self.task_type == "code":
            self.answer_buffer = ""
            self.code_tokens = 0
            self.closing_enforced = False
            if self.config.enforce_structure:
                self.state = "code_intro"
                self._enqueue_force("```python\n", next_state="code_body")
                return "code_intro"
            return "code_body"
        self.answer_buffer = ""
        self.closing_enforced = False
        return "math_answer"

    def allowed_ids(self) -> Optional[List[int]]:
        if not self.config.answer_tags:
            self._step_forced = False
            return None
        if self.force_queue:
            self._step_forced = True
            current = self.force_queue[0]
            return [current[self.force_index]]
        self._step_forced = False
        if self.state == "math_answer":
            return self.math_allowed or None
        return None

    def force_deterministic(self) -> bool:
        return self._step_forced

    def observe(self, token_id: int) -> None:
        char = self.tokenizer.id_to_token.get(int(token_id), "")
        self.generated_text += char
        if self.force_queue:
            self.force_index += 1
            current = self.force_queue[0]
            if self.force_index >= len(current):
                self.force_queue.pop(0)
                next_state = self.force_states.pop(0)
                self.force_index = 0
                if next_state is not None:
                    if next_state == "math_answer":
                        self.answer_buffer = ""
                        self.state = "math_answer"
                    elif next_state == "code_body":
                        self.answer_buffer = ""
                        self.code_tokens = 0
                        self.state = "code_body"
                    else:
                        self.state = next_state
                elif not self.force_queue and self.state == "forcing":
                    self.state = "think"
            return
        if not self.config.answer_tags:
            return
        if self.state == "think":
            self.think_tokens += 1
            if self.generated_text.endswith("</think>"):
                answer_state = self._answer_state()
                if self.config.enforce_structure:
                    self.state = "forcing"
                    self._enqueue_force("\n<answer>\n", next_state=answer_state)
                else:
                    self.state = answer_state
            elif self.generated_text.endswith("<answer>"):
                self.state = self._answer_state()
            elif (
                self.config.enforce_structure
                and self.think_tokens >= self.config.think_max_tokens
            ):
                answer_state = self._answer_state()
                self.state = "forcing"
                self._enqueue_force("</think>\n<answer>\n", next_state=answer_state)
        elif self.state == "math_answer":
            if char.strip():
                self.answer_buffer += char
            if (
                self.config.enforce_structure
                and not self.closing_enforced
                and (
                    char == "\n"
                    or len(self.answer_buffer.strip()) >= self.config.math_answer_max_tokens
                )
            ):
                self.closing_enforced = True
                self.state = "forcing"
                self._enqueue_force("</answer>\n", next_state="done")
        elif self.state == "code_intro":
            # Waiting for forced intro to finish
            pass
        elif self.state == "code_body":
            self.code_tokens += 1
            self.answer_buffer += char
            if (
                not self.closing_enforced
                and (
                    self.answer_buffer.endswith("```\n")
                    or self.code_tokens >= self.config.code_answer_max_tokens
                )
            ):
                self.closing_enforced = True
                self.state = "forcing"
                if self.answer_buffer.endswith("```\n"):
                    self._enqueue_force("</answer>\n", next_state="done")
                else:
                    self._enqueue_force("\n```\n</answer>\n", next_state="done")
        if self.generated_text.endswith("</answer>"):
            self.state = "done"


# -----------------------
# PPO utilities
# -----------------------


@dataclass
class Trajectory:
    tokens: List[int]
    log_probs: List[float]
    values: List[float]
    reward: float
    raw_reward: float
    completion: str
    spike_records: List[torch.Tensor]
    prompt_len: int
    group_index: int = 0


def compute_returns_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
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
    old_values: torch.Tensor


def collate_trajectories(
    trajectories: List[Trajectory],
    tokenizer: SimpleTokenizer,
    device: torch.device,
    gamma: float,
    gae_lambda: float,
) -> PPOBatch:
    max_len = max(len(traj.tokens) for traj in trajectories)
    input_ids = []
    target_ids = []
    old_log_probs = []
    advantages = []
    returns = []
    mask = []
    old_values = []
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
        value_full = torch.zeros(seq_len, dtype=torch.float32)
        if action_len > 0 and action_end > action_start:
            values = torch.tensor(traj.values, dtype=torch.float32)
            rewards = torch.zeros(action_len, dtype=torch.float32)
            rewards[-1] = traj.reward
            ret, adv = compute_returns_advantages(rewards, values, gamma, gae_lambda)
            logp_full[action_start:action_end] = torch.tensor(traj.log_probs, dtype=torch.float32)
            adv_full[action_start:action_end] = adv
            ret_full[action_start:action_end] = ret
            mask_full[action_start:action_end] = 1.0
            value_full[action_start:action_end] = values
        inp_tensor = torch.tensor(inp + [tokenizer.pad_id] * pad, dtype=torch.long)
        tgt_tensor = torch.tensor(tgt + [tokenizer.pad_id] * pad, dtype=torch.long)
        logp_tensor = torch.cat([logp_full, torch.zeros(pad, dtype=torch.float32)])
        adv_tensor = torch.cat([adv_full, torch.zeros(pad, dtype=torch.float32)])
        ret_tensor = torch.cat([ret_full, torch.zeros(pad, dtype=torch.float32)])
        mask_tensor = torch.cat([mask_full, torch.zeros(pad, dtype=torch.float32)])
        value_tensor = torch.cat([value_full, torch.zeros(pad, dtype=torch.float32)])
        input_ids.append(inp_tensor)
        target_ids.append(tgt_tensor)
        old_log_probs.append(logp_tensor)
        advantages.append(adv_tensor)
        returns.append(ret_tensor)
        mask.append(mask_tensor)
        old_values.append(value_tensor)
    return PPOBatch(
        input_ids=torch.stack(input_ids).to(device),
        target_ids=torch.stack(target_ids).to(device),
        old_log_probs=torch.stack(old_log_probs).to(device),
        advantages=torch.stack(advantages).to(device),
        returns=torch.stack(returns).to(device),
        mask=torch.stack(mask).to(device),
        old_values=torch.stack(old_values).to(device),
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
    grad_norm: float = 1.0,
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
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
    value_clip: bool = False,
    grad_norm: float = 1.0,
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
    values_pred = values
    if value_clip:
        values_clipped = batch.old_values + (values_pred - batch.old_values).clamp(-clip_range, clip_range)
        value_loss_unclipped = (values_pred - batch.returns) ** 2
        value_loss_clipped = (values_clipped - batch.returns) ** 2
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    else:
        value_loss = (values_pred - batch.returns) ** 2
    value_loss = (value_loss * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    entropy = -(log_probs * torch.exp(log_probs)).sum(-1)
    entropy = (entropy * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    approx_kl = ((batch.old_log_probs - gathered_log_probs) * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    kl_term = torch.tensor(0.0, device=batch.input_ids.device)
    if ref_model is not None:
        with torch.no_grad():
            ref_logits, _, _ = ref_model(batch.input_ids)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_selected = torch.gather(ref_log_probs, -1, batch.target_ids.unsqueeze(-1)).squeeze(-1)
        kl_term = ((gathered_log_probs - ref_selected) * batch.mask).sum() / batch.mask.sum().clamp_min(1.0)
    loss = policy_loss + value_coef * value_loss - ent_coef * entropy + kl_coef * kl_term
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
    optimizer.step()
    return {
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "kl": float(kl_term.detach()),
        "approx_kl": float(approx_kl.detach()),
    }


# -----------------------
# Sampling and rollout
# -----------------------


def top_p_sampling(
    logits: torch.Tensor,
    top_p: float = 0.9,
    temperature: float = 0.7,
    deterministic: bool = False,
    allowed_ids: Optional[Sequence[int]] = None,
) -> Tuple[int, float]:
    logits_mod = logits.clone()
    if allowed_ids is not None and len(allowed_ids) > 0:
        mask = torch.full_like(logits_mod, float("-inf"))
        mask[list(allowed_ids)] = 0.0
        logits_mod = logits_mod + mask
    log_probs = F.log_softmax(logits_mod, dim=-1)
    if allowed_ids is not None and len(allowed_ids) == 1:
        deterministic = True
    if deterministic:
        token = int(log_probs.argmax().item())
        return token, float(log_probs[token].item())
    scaled_logits = logits_mod / max(temperature, 1e-4)
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
    verifier_config: Optional[VerifierConfig] = None,
    structure_config: Optional[StructureConfig] = None,
    group_k: int = 1,
) -> Tuple[List[Trajectory], Dict[str, float]]:
    if samples is None:
        samples = env.sample_batch()
    verifier_conf = verifier_config or VerifierConfig()
    structure_conf = structure_config or StructureConfig()
    expanded: List[Tuple[PromptSample, int]] = []
    for idx, sample in enumerate(samples):
        repeats = max(group_k, 1)
        for _ in range(repeats):
            expanded.append((sample, idx))
    trajectories: List[Trajectory] = []
    rewards = []
    spike_logs: List[torch.Tensor] = []
    group_rewards: Dict[int, List[float]] = {}
    model.eval()
    encoder_states: List[Tuple[SpikeEncoder, bool]] = []
    if deterministic:
        for layer in model.layers:
            if hasattr(layer, "spike_encoder"):
                encoder_states.append((layer.spike_encoder, layer.spike_encoder.stochastic))
                layer.spike_encoder.stochastic = False
    for sample, group_idx in expanded:
        controller = StructureController(tokenizer, structure_conf, sample.task_type)
        prompt_ids = tokenizer.encode(sample.prompt, add_bos=True, add_eos=False)
        generated: List[int] = prompt_ids.copy()
        log_probs: List[float] = []
        values: List[float] = []
        spike_record_steps: List[torch.Tensor] = []
        prompt_len = len(prompt_ids)
        steps = 0
        while steps < max_new_tokens:
            input_tensor = torch.tensor([generated], device=device, dtype=torch.long)
            logits, vals, spikes = model(input_tensor, collect_spikes=True)
            last_logits = logits[0, -1, :]
            last_value = vals[0, -1].item()
            allowed_ids = controller.allowed_ids()
            forced = controller.force_deterministic()
            token, logp = top_p_sampling(
                last_logits,
                top_p=top_p,
                temperature=temperature,
                deterministic=deterministic or forced,
                allowed_ids=allowed_ids,
            )
            generated.append(token)
            log_probs.append(logp)
            values.append(last_value)
            if spikes:
                layer_last_spikes = [layer_spikes[:, -1, :] for layer_spikes in spikes]
                step_spikes = torch.stack(layer_last_spikes, dim=0).mean(0)
                spike_record_steps.append(step_spikes.cpu())
            controller.observe(token)
            steps += 1
            if token == tokenizer.eos_id or controller.state == "done":
                break
        completion_tokens = generated[prompt_len:]
        completion_text = tokenizer.decode(completion_tokens)
        if sample.task_type == "math":
            reward = math_verifier(completion_text, sample.payload["answer"], verifier_conf)
        else:
            reward = code_verifier(completion_text, sample.payload["tests"], verifier_conf)
        rewards.append(reward)
        group_rewards.setdefault(group_idx, []).append(reward)
        trajectories.append(
            Trajectory(
                tokens=generated,
                log_probs=log_probs,
                values=values,
                reward=reward,
                raw_reward=reward,
                completion=completion_text,
                spike_records=spike_record_steps,
                prompt_len=prompt_len,
                group_index=group_idx,
            )
        )
        if spike_record_steps:
            spike_logs.append(torch.stack(spike_record_steps, dim=0))
    if group_k > 1:
        group_means = {idx: sum(vals) / max(len(vals), 1) for idx, vals in group_rewards.items()}
        for traj in trajectories:
            mean = group_means.get(traj.group_index, 0.0)
            traj.reward = traj.raw_reward - mean
    metrics = {
        "avg_reward": float(sum(rewards) / max(len(rewards), 1)),
        "reward_math": float(
            sum(r for r, (s, _) in zip(rewards, expanded) if s.task_type == "math")
            / max(sum(1 for s, _ in expanded if s.task_type == "math"), 1)
        ),
        "reward_code": float(
            sum(r for r, (s, _) in zip(rewards, expanded) if s.task_type == "code")
            / max(sum(1 for s, _ in expanded if s.task_type == "code"), 1)
        ),
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
        spike_gain=args.spike_gain,
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
        spike_gain=args.spike_gain,
    ).to(device)
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    verifier_conf = VerifierConfig(
        answer_tags=args.answer_tags,
        enforce_structure=args.enforce_structure,
        math_integer_only=args.math_integer_only,
        math_format_bonus=args.math_format_bonus,
        code_compile_bonus=args.code_compile_bonus,
    )
    structure_conf = StructureConfig(
        answer_tags=args.answer_tags,
        enforce_structure=args.enforce_structure,
        math_integer_only=args.math_integer_only,
        think_max_tokens=args.think_max_tokens,
        math_answer_max_tokens=args.math_answer_max_tokens,
        code_answer_max_tokens=args.code_answer_max_tokens,
    )
    env = VerifierEnv(
        tokenizer,
        batch_size=args.batch_size,
        code_prob=args.code_prob,
        verifier_config=verifier_conf,
    )
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
                grad_norm=args.grad_norm,
            )
            if (i + 1) % max(args.pretrain_log_every, 1) == 0 or i + 1 == args.pretrain_iters:
                logging.info(
                    "pretrain %d/%d supervised_loss=%.4f",
                    i + 1,
                    args.pretrain_iters,
                    sup_loss,
                )
        ref_model.load_state_dict(model.state_dict())
    critic_lr = args.critic_lr if args.critic_lr > 0 else args.lr * 2.0
    critic_params = list(model.head.value.parameters())
    critic_ids = {id(p) for p in critic_params}
    actor_params = [p for p in model.parameters() if id(p) not in critic_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": args.lr},
            {"params": critic_params, "lr": critic_lr},
        ]
    )
    scheduler = None
    if args.cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.total_steps, 1))
    kl_coef = args.kl_coef
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
                grad_norm=args.grad_norm,
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
            verifier_config=verifier_conf,
            structure_config=structure_conf,
            group_k=args.group_k,
        )
        batch = collate_trajectories(
            trajectories,
            tokenizer,
            device,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        stats = ppo_update(
            model,
            batch,
            optimizer,
            clip_range=args.ppo_clip,
            ent_coef=args.entropy_coef,
            value_coef=args.value_coef,
            kl_coef=kl_coef,
            value_clip=args.value_clip,
            grad_norm=args.grad_norm,
            ref_model=ref_model,
        )
        measured_kl = stats.get("kl", 0.0) if ref_model is not None else stats.get("approx_kl", 0.0)
        if args.target_kl > 0:
            if measured_kl > args.target_kl:
                kl_coef *= args.kl_adapt
            elif measured_kl < args.target_kl / 2:
                kl_coef /= args.kl_adapt
            kl_coef = float(min(max(kl_coef, 0.005), 0.5))
        if scheduler is not None:
            scheduler.step()
        log_data = {**rollout_metrics, **stats}
        if sup_losses:
            log_data["supervised_loss"] = float(sum(sup_losses) / len(sup_losses))
        log_data["kl_coef"] = kl_coef
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
        spike_gain=args.spike_gain,
    ).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    verifier_conf = VerifierConfig(
        answer_tags=args.answer_tags,
        enforce_structure=args.enforce_structure,
        math_integer_only=args.math_integer_only,
        math_format_bonus=args.math_format_bonus,
        code_compile_bonus=args.code_compile_bonus,
    )
    structure_conf = StructureConfig(
        answer_tags=args.answer_tags,
        enforce_structure=args.enforce_structure,
        math_integer_only=args.math_integer_only,
        think_max_tokens=args.think_max_tokens,
        math_answer_max_tokens=args.math_answer_max_tokens,
        code_answer_max_tokens=args.code_answer_max_tokens,
    )
    env = VerifierEnv(
        tokenizer,
        batch_size=args.batch_size,
        code_prob=args.code_prob,
        verifier_config=verifier_conf,
    )
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
        verifier_config=verifier_conf,
        structure_config=structure_conf,
        group_k=args.group_k,
    )
    for sample, traj in zip(samples, trajectories):
        logging.info(
            "Prompt: %s | completion: %s | reward=%.3f",
            sample.prompt,
            traj.completion.replace("\n", " "),
            traj.raw_reward,
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
        p.add_argument("--answer-tags", action="store_true")
        p.add_argument("--enforce-structure", action="store_true")
        p.add_argument("--math-integer-only", action="store_true")
        p.add_argument("--math-format-bonus", type=float, default=0.0)
        p.add_argument("--code-compile-bonus", type=float, default=0.0)
        p.add_argument("--think-max-tokens", type=int, default=24)
        p.add_argument("--math-answer-max-tokens", type=int, default=8)
        p.add_argument("--code-answer-max-tokens", type=int, default=64)
        p.add_argument("--group-k", type=int, default=1)
        p.add_argument("--spike-gain", type=float, default=1.0)

    train_parser = subparsers.add_parser("train", help="Train the hybrid model with PPO")
    add_shared_arguments(train_parser)
    train_parser.add_argument("--total-steps", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--ppo-clip", type=float, default=0.2)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--kl-coef", type=float, default=0.02)
    train_parser.add_argument("--output-dir", type=str, default="checkpoints")
    train_parser.add_argument("--supervised-steps", type=int, default=1)
    train_parser.add_argument("--supervised-weight", type=float, default=1.0)
    train_parser.add_argument("--deterministic-rollout", action="store_true")
    train_parser.add_argument("--pretrain-iters", type=int, default=0)
    train_parser.add_argument("--pretrain-log-every", type=int, default=50)
    train_parser.add_argument("--gae-lambda", type=float, default=0.95)
    train_parser.add_argument("--gamma", type=float, default=0.99)
    train_parser.add_argument("--target-kl", type=float, default=0.0)
    train_parser.add_argument("--kl-adapt", type=float, default=1.5)
    train_parser.add_argument("--grad-norm", type=float, default=1.0)
    train_parser.add_argument("--value-clip", action="store_true")
    train_parser.add_argument("--critic-lr", type=float, default=0.0)
    train_parser.add_argument("--cosine-lr", action="store_true")

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
