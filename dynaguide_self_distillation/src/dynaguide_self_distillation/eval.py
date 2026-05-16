from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from dynaguide_self_distillation.calvin_labels import classify_behavior
from dynaguide_self_distillation.collect_traces import (
    _episode_seed,
    _load_policy_and_env,
    _load_guidance_function,
    _load_reset_poses,
    _prepare_imports,
    _set_seed,
    _state_and_proprio,
    _volume_path,
)


def evaluate_distilled(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    artifact_root: str | Path = "/artifacts",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    artifact_root = Path(artifact_root)
    _prepare_imports(repo_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    output_dir = _volume_path(artifact_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    policy, env = _load_policy_and_env(
        _volume_path(artifact_root, config["policy"]),
        sampler=config["sampler"],
        num_inference_timesteps=int(config["num_inference_timesteps"]),
    )
    guidance_function = None
    if config.get("guidance_mode", "none") == "dynaguide":
        guidance_function = _load_guidance_function(
            _volume_path(artifact_root, config["dynaguide_model"]),
            _volume_path(artifact_root, config["guidance_conditions"]),
            config,
        )
    elif config.get("guidance_mode", "none") != "none":
        raise ValueError(f"unsupported guidance_mode={config.get('guidance_mode')!r}")
    reset_poses = _load_reset_poses(repo_root, config.get("reset_poses"))

    from core.calvin_utils import check_state_difference, generate_reset_state

    per_seed = []
    behavior_counts: dict[str, int] = {}
    for seed in config["seeds"]:
        result = _evaluate_seed(
            policy=policy,
            env=env,
            config=config,
            seed=int(seed),
            reset_poses=reset_poses,
            generate_reset_state=generate_reset_state,
            check_state_difference=check_state_difference,
            guidance_function=guidance_function,
        )
        per_seed.append(result)
        for label, count in result["behavior_counts"].items():
            behavior_counts[label] = behavior_counts.get(label, 0) + int(count)
        partial = _build_metrics(config, artifact_root, per_seed, behavior_counts)
        (output_dir / "metrics.partial.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        print(
            f"[eval] seed={seed} successes={result['successes']}/{result['n_rollouts']} "
            f"rate={result['success_rate']:.3f}"
        )

    metrics = _build_metrics(config, artifact_root, per_seed, behavior_counts)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def _build_metrics(
    config: dict[str, Any],
    artifact_root: Path,
    per_seed: list[dict[str, Any]],
    behavior_counts: dict[str, int],
) -> dict[str, Any]:
    rates = np.asarray([item["success_rate"] for item in per_seed], dtype=np.float64)
    total_rollouts = int(sum(item["n_rollouts"] for item in per_seed))
    elapsed_seconds = float(sum(item["elapsed_seconds"] for item in per_seed))
    return {
        "task": config["task"],
        "policy": str(_volume_path(artifact_root, config["policy"])),
        "guidance_mode": config.get("guidance_mode", "none"),
        "n_rollouts_per_seed": int(config["n_rollouts_per_seed"]),
        "completed_seeds": [int(item["seed"]) for item in per_seed],
        "total_rollouts": total_rollouts,
        "success_rate_mean": float(rates.mean()) if len(rates) else None,
        "success_rate_se": _standard_error(rates),
        "per_seed": per_seed,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_rollout": elapsed_seconds / total_rollouts if total_rollouts else None,
        "config": config,
    }


def _evaluate_seed(
    *,
    policy: Any,
    env: Any,
    config: dict[str, Any],
    seed: int,
    reset_poses: list[np.ndarray] | None,
    generate_reset_state: Any,
    check_state_difference: Any,
    guidance_function: Any | None,
) -> dict[str, Any]:
    behavior_counts: dict[str, int] = {}
    successes = 0
    n_rollouts = int(config["n_rollouts_per_seed"])
    started = time.perf_counter()
    for rollout_index in range(n_rollouts):
        _set_seed(_episode_seed(seed, rollout_index))
        behavior = _run_rollout(
            policy=policy,
            env=env,
            config=config,
            reset_poses=reset_poses,
            generate_reset_state=generate_reset_state,
            check_state_difference=check_state_difference,
            guidance_function=guidance_function,
        )
        behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        successes += int(behavior == config["success_label"])
    return {
        "seed": seed,
        "n_rollouts": n_rollouts,
        "successes": successes,
        "success_rate": successes / n_rollouts,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _run_rollout(
    *,
    policy: Any,
    env: Any,
    config: dict[str, Any],
    reset_poses: list[np.ndarray] | None,
    generate_reset_state: Any,
    check_state_difference: Any,
    guidance_function: Any | None,
) -> str:
    policy.start_episode()
    env.reset()

    scene_state, articulated_binaries = generate_reset_state(sim_hold=config["env_setup"])
    reset_state: Any = scene_state
    if reset_poses:
        import random

        reset_state = {"scene": scene_state, "robot": random.choice(reset_poses)}

    obs = env.reset_to(reset_state)
    start_state, _ = _state_and_proprio(obs)
    states = []
    proprios = []
    state, proprio = _state_and_proprio(obs)
    states.append(state)
    proprios.append(proprio)

    for _ in range(int(config["horizon"])):
        if guidance_function is None:
            action = policy(ob=obs)
        else:
            action = policy(
                ob=obs,
                guidance_function=guidance_function,
                guidance_type="diffusion",
                ss=int(config["ss"]),
            )
        next_obs, _reward, _done, _info = env.step(action)
        state, proprio = _state_and_proprio(next_obs)
        states.append(state)
        proprios.append(proprio)
        done = check_state_difference(
            start_state,
            state,
            proprio[:3],
            articulated_binaries,
            for_display=False,
        )
        obs = next_obs
        if done:
            break

    return classify_behavior(np.asarray(states, dtype=np.float32), np.asarray(proprios, dtype=np.float32))


def _standard_error(values: np.ndarray) -> float | None:
    if values.size < 2:
        return None
    return float(values.std(ddof=1) / np.sqrt(values.size))
