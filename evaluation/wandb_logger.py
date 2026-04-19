"""
wandb_logger.py — Logs metrics to Weights & Biases.

WHY W&B: It gives you live dashboards, persistent experiment history,
and easy comparison across runs. You can see if the reward curve is going
up (model is learning) or flat (stuck) without digging through logs.

What we track:
  - Per-step: reward, debate_improvement
  - Per-episode: pass@1, pass@10, reward curves
  - Per-run: hyperparameters, model name, dataset size

Usage:
    logger = WandbLogger(project="code-marl", run_name="grpo-run-1")
    logger.log_step(step=1, reward=0.4, debate_improvement=0.1)
    logger.log_eval(step=100, pass_at_1=0.35, pass_at_10=0.78)
    logger.finish()
"""

import os
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[wandb_logger] WARNING: wandb not installed. Metrics will only print.")


class WandbLogger:
    """Logs training and evaluation metrics to Weights & Biases.

    Falls back to printing if W&B is not available or API key is missing.
    This makes development easier — you can run locally without W&B set up.

    Attributes:
        project: W&B project name.
        run_name: Name for this specific run.
        run: The active W&B run object (or None if unavailable).
    """

    def __init__(
        self,
        project: str = "code-marl",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
    ) -> None:
        """Initialize the W&B logger.

        Args:
            project: W&B project name. Groups related runs together.
            run_name: Human-readable name for this run.
            config: Hyperparameter dict to log. E.g. {"lr": 5e-6, "n_gens": 4}.
        """
        self.project = project
        self.run_name = run_name
        self.run = None

        if WANDB_AVAILABLE and os.environ.get("WANDB_API_KEY"):
            try:
                self.run = wandb.init(
                    project=project,
                    name=run_name,
                    config=config or {},
                )
                print(f"[WandbLogger] Run started: {self.run.url}")
            except Exception as e:
                print(f"[WandbLogger] Failed to init W&B: {e}. Will print metrics.")
        else:
            if not WANDB_AVAILABLE:
                print("[WandbLogger] wandb not installed — printing metrics only.")
            else:
                print("[WandbLogger] WANDB_API_KEY not set — printing metrics only.")

    def log_step(
        self,
        step: int,
        reward: float,
        debate_improvement: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> None:
        """Log per-training-step metrics.

        Args:
            step: Training step number.
            reward: Mean reward for this step's batch.
            debate_improvement: Mean reward gain from debate (optional).
            loss: Training loss (optional).
        """
        data = {"train/reward": reward, "step": step}
        if debate_improvement is not None:
            data["train/debate_improvement"] = debate_improvement
        if loss is not None:
            data["train/loss"] = loss

        self._log(data)

    def log_eval(
        self,
        step: int,
        pass_at_1: float,
        pass_at_10: Optional[float] = None,
        debate_improvement_mean: Optional[float] = None,
        debate_pct_improved: Optional[float] = None,
    ) -> None:
        """Log evaluation metrics (run periodically, not every step).

        Args:
            step: Training step at which evaluation was run.
            pass_at_1: pass@1 score on eval set.
            pass_at_10: pass@10 score (optional).
            debate_improvement_mean: Mean reward delta from debate (optional).
            debate_pct_improved: Fraction of episodes where debate helped.
        """
        data = {"eval/pass@1": pass_at_1, "step": step}
        if pass_at_10 is not None:
            data["eval/pass@10"] = pass_at_10
        if debate_improvement_mean is not None:
            data["eval/debate_improvement"] = debate_improvement_mean
        if debate_pct_improved is not None:
            data["eval/pct_improved_by_debate"] = debate_pct_improved

        self._log(data)

    def log_debate_episode(
        self,
        step: int,
        problem_id: str,
        reward_initial: float,
        reward_final: float,
        n_rounds: int,
    ) -> None:
        """Log a single debate episode's outcome.

        Args:
            step: Current training step.
            problem_id: Problem identifier (e.g. "HumanEval/42").
            reward_initial: Reward before debate.
            reward_final: Reward after debate.
            n_rounds: Number of debate rounds used.
        """
        data = {
            "debate/reward_initial": reward_initial,
            "debate/reward_final": reward_final,
            "debate/improvement": reward_final - reward_initial,
            "debate/n_rounds": n_rounds,
            "step": step,
        }
        self._log(data)

    def _log(self, data: dict) -> None:
        """Internal log dispatch: W&B or print.

        Args:
            data: Dict of metric name → value.
        """
        if self.run is not None:
            wandb.log(data)
        else:
            # Pretty-print for local development
            step = data.pop("step", "?")
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in data.items())
            print(f"[step {step}] {metrics_str}")

    def finish(self) -> None:
        """Close the W&B run. Call this at the end of training."""
        if self.run is not None:
            self.run.finish()
            print("[WandbLogger] Run finished.")


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use a fake run (no API key needed for this test)
    logger = WandbLogger(project="code-marl-test", run_name="smoke-test")

    logger.log_step(step=1, reward=0.3, debate_improvement=0.05)
    logger.log_step(step=2, reward=0.45, loss=1.23)
    logger.log_eval(step=10, pass_at_1=0.35, pass_at_10=0.78)
    logger.log_debate_episode(
        step=5,
        problem_id="HumanEval/0",
        reward_initial=0.5,
        reward_final=0.8,
        n_rounds=2,
    )
    logger.finish()
    print("WandbLogger smoke test complete.")
