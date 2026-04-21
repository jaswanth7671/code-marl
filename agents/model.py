"""
model.py — Shared local model loader (4-bit QLoRA + SDPA attention).

For the debate loop (inference only — no training here), we load the model
in 4-bit so it fits comfortably on a Colab A100 alongside other things.

Uses SDPA (Scaled Dot Product Attention) — built into PyTorch 2.x,
no separate installation needed. Nearly as fast as Flash Attention 2.

Loaded once as a singleton → shared by CoderAgent and CriticAgent.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from typing import Optional

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

_pipeline = None
_tokenizer = None


def load_model(model_id: str = MODEL_ID) -> None:
    """Load the model in 4-bit (inference-only) with Flash Attention 2.

    Uses 4-bit quantization so the 7B model uses ~5GB VRAM instead of ~14GB.
    Flash Attention 2 makes the attention computation 2-4x faster.

    Call this once at the top of your Colab notebook.
    Subsequent calls are no-ops (model already in memory).

    Args:
        model_id: HuggingFace model ID or local path.
    """
    global _pipeline, _tokenizer

    if _pipeline is not None:
        print(f"[model] Already loaded: {model_id}")
        return

    print(f"[model] Loading {model_id} in 4-bit...")

    # T4 uses float16 (no bfloat16 support), A100 uses bfloat16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",  # built into PyTorch 2.x, no install needed
    )

    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=_tokenizer,
        device_map="auto",
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[model] Ready. {param_count/1e9:.1f}B parameters loaded in 4-bit.")


def generate(
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Generate a response using the loaded model.

    Applies Qwen's ChatML format automatically via the tokenizer's
    apply_chat_template — this is what the model was instruction-tuned with.

    Args:
        system_prompt: The system role instruction.
        user_message: The user's message / problem.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.

    Returns:
        The assistant's response text (new tokens only, not the prompt).

    Raises:
        RuntimeError: If load_model() has not been called.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Model not loaded. Call agents.model.load_model() first.\n"
            "In your Colab notebook:\n"
            "  from agents.model import load_model\n"
            "  load_model()"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    prompt_text = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = _pipeline(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=_tokenizer.eos_token_id,
        return_full_text=False,
    )

    return outputs[0]["generated_text"].strip()
