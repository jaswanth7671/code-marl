"""
grpo_trainer.py — GRPO + QLoRA training on Qwen2.5-Coder-1.5B (T4-optimised).

KEY FIX: Prompt ends with ```python\n + function signature so the model's
completion IS the code directly. No parsing ambiguity, reward always non-zero
when the model writes valid Python.

TECH HIGHLIGHTS:
  - GRPO (DeepSeek-R1 algorithm, 2025)
  - QLoRA 4-bit quantization (NeurIPS 2023)
  - SDPA attention (built into PyTorch 2.x)
  - Verifiable reward — no human labels
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

from environment.dataset import load_humaneval
from training.reward_fn import make_reward_fn_with_tests


MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # fits on T4 16GB


def get_qlora_config() -> BitsAndBytesConfig:
    """4-bit QLoRA quantization — T4 uses float16, A100 uses bfloat16."""
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """LoRA adapters — only 1.18% of weights are trained."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_model_qlora(model_id: str = MODEL_ID):
    """Load model in 4-bit QLoRA with LoRA adapters attached."""
    print(f"Loading {model_id} in 4-bit (QLoRA)...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=get_qlora_config(),
        device_map="auto",
        attn_implementation="sdpa",
    )

    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    model = get_peft_model(base_model, get_lora_config())
    model.print_trainable_parameters()
    return model, tokenizer


def format_for_grpo(problem: dict) -> str:
    """Format a HumanEval problem for GRPO training.

    WHY: The prompt must NOT include the function signature.
    If it did, the model would only write the body (no 'def'),
    and the sandbox can't run a bare body without the definition.

    Instead we show the spec and ask for the COMPLETE function.
    The model writes 'def ...' → parse.py strategy 4 finds it → reward works.
    """
    prompt_text = problem["prompt"].strip()
    return (
        f"Write a complete Python implementation for the function below.\n"
        f"Return ONLY the code starting with 'def'. No explanation.\n\n"
        f"Specification:\n{prompt_text}"
    )


def build_training_dataset(max_problems: int = 50):
    """Load HumanEval and format prompts for GRPO.

    Returns:
        Tuple of (Dataset with 'prompt' col, list of test strings).
    """
    problems = load_humaneval(max_problems=max_problems)
    prompts    = [format_for_grpo(p) for p in problems]
    test_cases = [p["test_code"] for p in problems]
    dataset = Dataset.from_dict({"prompt": prompts})
    print(f"      {len(dataset)} problems ready.")
    print(f"      Sample prompt ending:\n      ...{prompts[0][-120:]!r}")
    return dataset, test_cases


def build_grpo_config(
    output_dir: str = "./checkpoints",
    num_train_epochs: int = 1,
    num_generations: int = 4,
    learning_rate: float = 2e-5,
) -> GRPOConfig:
    """GRPOConfig tuned for T4 16GB."""
    is_bf16 = torch.cuda.is_bf16_supported()
    return GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        num_generations=num_generations,
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=is_bf16,
        fp16=not is_bf16,
        logging_steps=1,
        save_steps=50,
        report_to="wandb",
        max_completion_length=400,
        temperature=0.8,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
    )


def train(
    max_problems: int = 50,
    num_train_epochs: int = 1,
    num_generations: int = 4,
    output_dir: str = "./checkpoints/grpo-qlora",
) -> None:
    """Run the full GRPO + QLoRA training loop."""
    print("=" * 60)
    print("Code-MARL: GRPO + QLoRA on Qwen2.5-Coder-1.5B (T4)")
    print("=" * 60)

    print("\n[1/4] Loading HumanEval dataset...")
    dataset, test_cases = build_training_dataset(max_problems=max_problems)

    print("\n[2/4] Building reward function...")
    reward_function = make_reward_fn_with_tests(test_cases)

    print("\n[3/4] Loading model in 4-bit QLoRA...")
    model, tokenizer = load_model_qlora()

    print("\n[4/4] Starting GRPO training...")
    config = build_grpo_config(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        num_generations=num_generations,
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=reward_function,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nLoRA adapters saved to {output_dir}")


if __name__ == "__main__":
    import wandb
    wandb.init(project="code-marl", name="grpo-qlora-qwen1.5b-t4")
    train()
