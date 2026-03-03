# Objective: Take a GPT2 model that can complete sentences, and then train it to solve math problems from GSM8K.
# Currently the reward is the negative answer length, meaning we don't prioritize correct answers, just short answers.

from transformers import AutoTokenizer, AutoTokenizer
from huggingface_hub import snapshot_download
from ttml.common.model_factory import TransformerModelFactory
from ttml.common.utils import set_seed
import ttnn
import ttml
import os
import numpy as np
import datasets
import time
from typing import List

CONFIG = "training_gsm8k_rl_llama.yaml"
HF_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
LOAD_PRETRAINED = True

from ttml.common.config import (
    load_config,
    yaml_deep_update,
)

from ttml.common.utils import (
    create_optimizer,
    get_tt_metal_home,
    no_grad,
)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)


def _round_up_to_tile(x: int, tile: int = 32) -> int:
    return ((x + tile - 1) // tile) * tile


def _build_causal_mask_ttml(device, query_len: int, processed_tokens: int = 0):
    """
    query_len: number of tokens in current forward
    processed_tokens: number of tokens already in KV cache (prefix length)
    """
    whole_len = processed_tokens + query_len
    padded_q = _round_up_to_tile(query_len)
    padded_k = _round_up_to_tile(whole_len)

    mask_np = np.zeros((padded_q, padded_k), dtype=np.float32)
    for i in range(query_len):
        # query i can attend up to (processed_tokens + i)
        mask_np[i, : processed_tokens + i + 1] = 1.0

    return ttml.autograd.Tensor.from_numpy(
        mask_np.reshape(1, 1, padded_q, padded_k),
        layout=ttnn.Layout.TILE,
        new_type=ttnn.DataType.BFLOAT16,
    )


def _tokens_to_ttml_uint32(tokens, device=None):
    actual_len = len(tokens)
    padded_len = _round_up_to_tile(actual_len)
    arr = np.zeros((padded_len,), dtype=np.uint32)
    arr[:actual_len] = np.asarray(tokens, dtype=np.uint32)
    t = ttml.autograd.Tensor.from_numpy(
        arr.reshape(1, 1, 1, padded_len),
        layout=ttnn.Layout.ROW_MAJOR,
        new_type=ttnn.DataType.UINT32,
    )
    return t, actual_len


class InferenceOutput:
    prompt_ids: List[int]
    completion_ids: List[int]
    token_logprobs: List[float]

    def __init__(self, prompt_ids, completion_ids, token_logprobs):
        self.prompt_ids = prompt_ids
        self.completion_ids = completion_ids
        self.token_logprobs = token_logprobs


def _safe_deallocate(ttnn_tensor):
    if ttnn_tensor is None:
        return
    try:
        ttnn_tensor.deallocate(True)
    except Exception:
        pass


def _deallocate_list(ttml_tensors):
    for x in ttml_tensors:
        if x is None:
            continue

        _safe_deallocate(x.get_value())


def _extract_token_logprob_from_last_logits(last_logits_tt, token_id: int) -> float:
    """
    last_logits_tt: TT tensor shape [1,1,1,V]
    Convert once to numpy, compute log-softmax on CPU for scalar logprob.
    This is inference-only and then we free TT tensor.
    """
    logits_np = ttnn.to_torch(last_logits_tt).float().cpu().numpy().reshape(-1)
    m = np.max(logits_np)
    logsumexp = m + np.log(np.sum(np.exp(logits_np - m)))
    return float(logits_np[token_id] - logsumexp)


def _forward_last_logits_with_cache(
    tt_model,
    tokens: List[int],
    *,
    kv_cache,
    processed_tokens: int,
    vocab_size: int,
):
    """
    Returns last-position logits TT tensor of shape [1,1,1,V].
    Caller owns returned tensor and must deallocate it.
    Internals are deallocated here.
    """
    x = mask = logits = tt_logits = None
    try:
        x, new_tokens = _tokens_to_ttml_uint32(tokens)
        mask = _build_causal_mask_ttml(
            x.get_value().device(),
            query_len=new_tokens,
            processed_tokens=processed_tokens,
        )

        logits = tt_model(x, mask, kv_cache, new_tokens)
        tt_logits = logits.get_value()

        # last position in this chunk
        idx = new_tokens - 1
        last_logits = ttnn.slice(
            tt_logits,
            [0, 0, idx, 0],
            [1, 1, idx + 1, vocab_size],
        )
        return last_logits
    finally:
        if x is not None:
            _safe_deallocate(x.get_value())
        if mask is not None:
            _safe_deallocate(mask.get_value())
        if tt_logits is not None:
            _safe_deallocate(tt_logits)


def model_inference(
    tt_model,
    tokenizer,
    prompt_ids,
    *,
    mode: str,  # "sample" | "score"
    completion_ids=None,  # required when mode="score"
    max_new_tokens: int = 64,  # used when mode="sample"
    temperature: float = 0.8,
    max_t: int = 200,
    num_layers: int = 30,
    num_groups: int = 3,
    embedding_dim: int = 576,
    num_heads: int = 9,
):
    assert mode in {"sample", "score"}
    if mode == "score":
        assert completion_ids is not None and len(completion_ids) > 0

    ids = list(prompt_ids)
    print(f"model_inference, f{len(ids)=}, {max_t=}, {max_new_tokens=}")
    assert len(ids) < max_t, f"Prompt too long: {len(ids)} >= {max_t}"

    tt_model.eval()
    vocab_size = tokenizer.vocab_size
    head_dim = embedding_dim // num_heads

    kv_cfg = ttml.models.KvCacheConfig(num_layers, 1, num_groups, max_t, head_dim)
    kv_cache = ttml.models.KvCache(kv_cfg)
    kv_cache.reset()

    generated: List[int] = []
    token_logprobs: List[float] = []

    # ---- Prefill: score/sample first completion token from prompt ----
    last_logits = None
    try:
        last_logits = _forward_last_logits_with_cache(
            tt_model,
            ids,
            kv_cache=kv_cache,
            processed_tokens=0,
            vocab_size=vocab_size,
        )

        if mode == "sample":
            sampled = ttml.ops.sample.sample_op(
                ttml.autograd.Tensor(last_logits, False),
                temperature,
                np.random.randint(0, 2**32 - 1),
                None,
            )
            next_token = int(sampled.get_value().item())
            _safe_deallocate(sampled.get_value())

            if next_token != tokenizer.eos_token_id:
                lp = _extract_token_logprob_from_last_logits(last_logits, next_token)
                token_logprobs.append(lp)
                generated.append(next_token)
        else:
            tgt = completion_ids[0]
            lp = _extract_token_logprob_from_last_logits(last_logits, tgt)
            token_logprobs.append(lp)
            generated.append(tgt)
    finally:
        _safe_deallocate(last_logits)

    if mode == "sample" and len(generated) == 0:
        return InferenceOutput(prompt_ids=ids, completion_ids=[], token_logprobs=[])

    # ---- Decode loop ----
    steps = (max_new_tokens - 1) if mode == "sample" else (len(completion_ids) - 1)

    for i in range(max(0, steps)):
        if len(ids) + len(generated) >= max_t:
            break

        prev_token = generated[-1]
        processed_tokens = kv_cache.get_cache_position()

        last_logits = None
        try:
            last_logits = _forward_last_logits_with_cache(
                tt_model,
                [prev_token],
                kv_cache=kv_cache,
                processed_tokens=processed_tokens,
                vocab_size=vocab_size,
            )

            if mode == "sample":
                sampled = ttml.ops.sample.sample_op(
                    ttml.autograd.Tensor(last_logits, False),
                    temperature,
                    np.random.randint(0, 2**32 - 1),
                    None,
                )
                tok = int(sampled.get_value().item())
                _safe_deallocate(sampled.get_value())

                if tok == tokenizer.eos_token_id:
                    break

                lp = _extract_token_logprob_from_last_logits(last_logits, tok)
                token_logprobs.append(lp)
                generated.append(tok)
            else:
                tgt = completion_ids[i + 1]
                lp = _extract_token_logprob_from_last_logits(last_logits, tgt)
                token_logprobs.append(lp)
                generated.append(tgt)
        finally:
            _safe_deallocate(last_logits)

    # no explicit kv_cache tensor free API exposed here; cache will be released with object lifetime
    return InferenceOutput(
        prompt_ids=ids,
        completion_ids=generated,
        token_logprobs=token_logprobs,
    )


def tokenize_dataset(data, tokenizer: AutoTokenizer):
    """
    Tokenizes the questions and answers in the dataset using the provided tokenizer.

    data: dataset with "question" and "answer" fields
    tokenizer: HuggingFace tokenizer
    """
    X = [sample["question"] for sample in data]
    y = [sample["answer"] for sample in data]

    tok = lambda texts: tokenizer(texts, return_tensors="np", add_special_tokens=False)[
        "input_ids"
    ]
    return tok(X), tok(y)


def reward_fn_from_completion_ids(completion_ids):
    # Example reward: shorter completion is better
    return -float(len(completion_ids))


def train_gsm8k(tt_model, optimizer, max_steps=1000, group_size=2, max_new_tokens=8):
    print("Loading GSM8K dataset...")
    train_data = datasets.load_dataset("openai/gsm8k", "main", split="train")
    X, _ = tokenize_dataset(train_data, tokenizer)

    # You likely already have this from model config
    max_sequence_length = 128
    causal_mask_np = np.tril(
        np.ones((max_sequence_length, max_sequence_length), dtype=np.float32)
    )
    causal_mask = ttml.autograd.Tensor.from_numpy(
        causal_mask_np.reshape(1, 1, max_sequence_length, max_sequence_length),
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.BFLOAT16,
    )

    for step in range(min(max_steps, len(X))):
        prompt_ids = X[step].tolist()

        # -------------------------
        # PHASE 1: sample + rewards
        # -------------------------
        sampled_completions = []
        rewards = []

        with no_grad():
            for _ in range(group_size):
                out = model_inference(
                    tt_model,
                    tokenizer,
                    prompt_ids,
                    mode="sample",
                    max_t=max_sequence_length,
                    max_new_tokens=max_new_tokens,
                    temperature=0.8,
                )
                sampled_completions.append(out.completion_ids)
                rewards.append(reward_fn_from_completion_ids(out.completion_ids))

            rewards_np = np.asarray(rewards, dtype=np.float32)
            advantages_np = (
                rewards_np - rewards_np.mean()
            )  # no std division (as requested)
            # advantages are now constants (detached scalars)

        # ------------------------------------
        # PHASE 2: differentiable policy update
        # ------------------------------------
        optimizer.zero_grad()

        # only for scaling, so gradient matches mean over valid samples
        valid_count = sum(1 for c in sampled_completions if len(c) > 0)
        if valid_count == 0:
            continue

        for completion_ids, adv in zip(sampled_completions, advantages_np):
            if len(completion_ids) == 0:
                continue

            # Build one training sequence: prompt + sampled completion
            seq = prompt_ids + completion_ids

            # Teacher forcing setup:
            # input is seq[:-1], target is seq[1:]
            inp = seq[:-1]
            tgt = seq[1:]

            # Truncate to model max len
            if len(inp) > max_sequence_length:
                inp = inp[-max_sequence_length:]
                tgt = tgt[-max_sequence_length:]

            # Pad to fixed length
            x_np = np.zeros((1, 1, 1, max_sequence_length), dtype=np.uint32)
            y_np = np.zeros((1, max_sequence_length), dtype=np.uint32)
            T = len(inp)
            x_np[0, 0, 0, :T] = np.asarray(inp, dtype=np.uint32)
            y_np[0, :T] = np.asarray(tgt, dtype=np.uint32)

            # Mask only completion-token positions in the target
            # completion starts after prompt, but target is shifted by 1
            prompt_len = len(prompt_ids)
            completion_start_in_target = max(0, prompt_len - 1)

            loss_scaler_np = np.zeros((1, 1, max_sequence_length, 1), dtype=np.float32)
            active_end = min(T, completion_start_in_target + len(completion_ids))
            if active_end > completion_start_in_target:
                loss_scaler_np[0, 0, completion_start_in_target:active_end, 0] = 1.0
                active = float(active_end - completion_start_in_target)
                # normalize so mean() over all tokens becomes mean over active tokens
                loss_scaler_np *= max_sequence_length / active

            X_tt = ttml.autograd.Tensor.from_numpy(
                x_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
            )
            y_tt = ttml.autograd.Tensor.from_numpy(
                y_np, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
            )
            scaler_tt = ttml.autograd.Tensor.from_numpy(
                loss_scaler_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16
            )

            logits = tt_model(X_tt, causal_mask)
            per_tok_ce = ttml.ops.loss.cross_entropy_loss(
                logits, y_tt, ttml.ops.ReduceType.NONE
            )
            nll = ttml.ops.unary.mean(
                per_tok_ce * scaler_tt
            )  # mean NLL over completion tokens

            # GRPO policy loss: -A * logprob == A * NLL
            sample_loss = ttml.ops.binary.mul(nll, float(adv))
            sample_loss = ttml.ops.binary.mul(sample_loss, 1.0 / float(valid_count))

            # Backward per sample
            sample_loss.backward(False)
            ttml.autograd.AutoContext.get_instance().reset_graph()

            _deallocate_list(
                [
                    sample_loss,
                    nll,
                    per_tok_ce,
                    logits,
                    scaler_tt,
                    y_tt,
                    X_tt,
                ]
            )

            sample_loss = nll = per_tok_ce = logits = None
            scaler_tt = y_tt = X_tt = None

        optimizer.step()

        print(f"step={step} reward_mean={rewards_np.mean():.4f}")

    _safe_deallocate(causal_mask.get_value())


def load_training_config():
    yaml_config = load_config(
        CONFIG, f"{get_tt_metal_home()}/tt-train/configs/training_configs"
    )

    print(f"YAML config: {yaml_config}")
    model_config = load_config(yaml_config["training_config"]["model_config"])

    override_config_path = (
        f"{os.environ['TT_METAL_HOME']}/tt-train/configs/training_overrides.yaml"
    )

    if os.path.isfile(override_config_path):
        print("Applying training overrides...")

        override_config = load_config(override_config_path)

        yaml_config = yaml_deep_update(yaml_config, override_config)
        model_config = yaml_deep_update(model_config, override_config)

        # pretty output of yaml config
        import yaml

        print("Loaded YAML config:")
        print(yaml.dump(yaml_config, sort_keys=False, default_flow_style=False))
        print("*********************************\n\n")

    return model_config


def create_model(model_config):
    print("Setting up model...")
    orig_vocab_size = tokenizer.vocab_size

    tt_model_factory = TransformerModelFactory(model_config)
    tt_model_factory.transformer_config.vocab_size = orig_vocab_size
    print("Created Model Factory")

    print("Creating model...")
    tt_model = tt_model_factory.create_model()

    if LOAD_PRETRAINED:
        model_repo_path = snapshot_download(
            repo_id=HF_MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
        )
        print(f"Model snapshot path: {model_repo_path}")
        print("Loading from safetensors...")
        tt_model.load_from_safetensors(model_repo_path)

    return tt_model


if __name__ == "__main__":
    set_seed(42)
    training_config = load_training_config()
    print(training_config)

    tt_model = create_model(training_config)
    print(tt_model.__dir__())

    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt)

    inference_output = model_inference(
        tt_model, tokenizer=tokenizer, prompt_ids=input_ids, mode="sample"
    )

    generated_text = tokenizer.decode(
        inference_output.completion_ids, skip_special_tokens=False
    )
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")

    optim = create_optimizer(tt_model, training_config)
    train_gsm8k(tt_model, optimizer=optim)
