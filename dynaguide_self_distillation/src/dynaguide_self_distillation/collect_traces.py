from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dynaguide_self_distillation.calvin_labels import classify_behavior
from dynaguide_self_distillation.trace_diffusion import (
    QueryTrace,
    QueryTraceSink,
    policy_will_query,
    trace_guided_denoising_at_obs,
)


def collect_traces(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    artifact_root: str | Path = "/artifacts",
    seed: int | None = None,
    rollout_start: int = 0,
    n_rollouts: int | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    """Collect rollout states with DynaGuide denoising targets.

    By default, the environment is stepped by the unguided student policy. The
    off-policy baseline uses the same trace format but sets action_source to
    dynaguide_guided, causing the guided policy to drive the simulator.
    """

    repo_root = Path(repo_root)
    artifact_root = Path(artifact_root)
    _prepare_imports(repo_root)
    config = _load_config(config_path)
    shard = _resolve_shard(config, seed=seed, rollout_start=rollout_start, n_rollouts=n_rollouts)

    output_dir = _volume_path(artifact_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_name or "trace"
    trace_path = output_dir / f"{output_stem}.hdf5"
    summary_path = output_dir / ("summary.json" if output_stem == "trace" else f"{output_stem}_summary.json")

    policy, env = _load_policy_and_env(
        _volume_path(artifact_root, config["base_policy"]),
        sampler=config["sampler"],
        num_inference_timesteps=int(config["num_inference_timesteps"]),
    )
    guidance_function = _load_guidance_function(
        _volume_path(artifact_root, config["dynaguide_model"]),
        _volume_path(artifact_root, config["guidance_conditions"]),
        config,
    )
    action_source = config.get("action_source", "student_unguided")
    if action_source not in {"student_unguided", "dynaguide_guided"}:
        raise ValueError(f"unsupported action_source={action_source!r}")

    from core.calvin_utils import check_state_difference, generate_reset_state

    reset_poses = _load_reset_poses(repo_root, config.get("reset_poses"))
    behavior_counts: dict[str, int] = {}
    successes = 0
    episode_index = 0

    import h5py

    with h5py.File(trace_path, "w") as h5:
        _write_attrs(
            h5,
            {
                "trace_schema": "trace.v2_obs_history",
                "task": config["task"],
                "success_label": config["success_label"],
                "action_source": action_source,
                "teacher": "dynaguide_side_computation",
                "horizon": config["horizon"],
                "sampler": config["sampler"],
                "num_inference_timesteps": config["num_inference_timesteps"],
                "scale": config["scale"],
                "alpha": config["alpha"],
                "ss": config["ss"],
                "seeds": config["seeds"],
                "n_rollouts_per_seed": config["n_rollouts_per_seed"],
                "selected_seeds": shard["seeds"],
                "rollout_start": shard["rollout_start"],
                "rollout_stop": shard["rollout_stop"],
            },
        )
        data_group = h5.create_group("data")

        for seed_value in shard["seeds"]:
            seed_index = [int(item) for item in config["seeds"]].index(seed_value)
            for rollout_index in range(shard["rollout_start"], shard["rollout_stop"]):
                _set_seed(_episode_seed(seed_value, rollout_index))
                episode = _run_episode(
                    policy=policy,
                    env=env,
                    config=config,
                    seed=seed_value,
                    rollout_index=rollout_index,
                    episode_index=seed_index * int(config["n_rollouts_per_seed"]) + rollout_index,
                    reset_poses=reset_poses,
                    generate_reset_state=generate_reset_state,
                    check_state_difference=check_state_difference,
                    guidance_function=guidance_function,
                    action_source=action_source,
                )
                _write_episode(data_group, episode)

                behavior = episode["behavior_label"]
                behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
                successes += int(episode["success"])
                episode_index += 1

    summary = {
        "trace_path": str(trace_path),
        "summary_path": str(summary_path),
        "n_episodes": episode_index,
        "successes": successes,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "shard": shard,
        "config": config,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _resolve_shard(
    config: dict[str, Any],
    *,
    seed: int | None,
    rollout_start: int,
    n_rollouts: int | None,
) -> dict[str, Any]:
    configured_seeds = [int(item) for item in config["seeds"]]
    if seed is None:
        selected_seeds = configured_seeds
    else:
        if seed not in configured_seeds:
            raise ValueError(f"seed {seed} is not present in configured seeds {configured_seeds}")
        selected_seeds = [int(seed)]

    total_rollouts = int(config["n_rollouts_per_seed"])
    rollout_count = total_rollouts if n_rollouts is None else int(n_rollouts)
    rollout_start = int(rollout_start)
    rollout_stop = rollout_start + rollout_count
    if rollout_start < 0 or rollout_count <= 0 or rollout_stop > total_rollouts:
        raise ValueError(
            f"invalid rollout slice [{rollout_start}, {rollout_stop}) "
            f"for n_rollouts_per_seed={total_rollouts}"
        )

    return {
        "seeds": selected_seeds,
        "rollout_start": rollout_start,
        "rollout_stop": rollout_stop,
        "n_rollouts": rollout_count,
    }


def _run_episode(
    *,
    policy: Any,
    env: Any,
    config: dict[str, Any],
    seed: int,
    rollout_index: int,
    episode_index: int,
    reset_poses: list[np.ndarray] | None,
    generate_reset_state: Any,
    check_state_difference: Any,
    guidance_function: Any,
    action_source: str,
) -> dict[str, Any]:
    policy.start_episode()
    env.reset()

    scene_state, articulated_binaries = generate_reset_state(sim_hold=config["env_setup"])
    reset_state: Any = scene_state
    if reset_poses:
        reset_state = {"scene": scene_state, "robot": random.choice(reset_poses)}

    obs = env.reset_to(reset_state)
    start_state, _ = _state_and_proprio(obs)

    sink = QueryTraceSink(
        rgb_keys=tuple(config["cameras"]),
        low_dim_keys=(config["proprio_key"],),
    )
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    proprios: list[np.ndarray] = []

    state, proprio = _state_and_proprio(obs)
    states.append(state)
    proprios.append(proprio)

    for _ in range(int(config["horizon"])):
        if policy_will_query(policy.policy):
            trace_guided_denoising_at_obs(
                policy.policy,
                obs,
                sink,
                guidance_function,
                ss=int(config["ss"]),
                num_inference_timesteps=int(config["num_inference_timesteps"]),
            )

        if action_source == "dynaguide_guided":
            action = policy(
                ob=obs,
                guidance_function=guidance_function,
                guidance_type="diffusion",
                ss=int(config["ss"]),
            )
        else:
            action = policy(ob=obs)
        next_obs, _reward, _done, _info = env.step(action)
        state, proprio = _state_and_proprio(next_obs)
        actions.append(np.asarray(action, dtype=np.float32))
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

    states_array = np.asarray(states, dtype=np.float32)
    proprios_array = np.asarray(proprios, dtype=np.float32)
    behavior_label = classify_behavior(states_array, proprios_array)
    return {
        "episode_index": episode_index,
        "seed": seed,
        "rollout_index": rollout_index,
        "success": behavior_label == config["success_label"],
        "behavior_label": behavior_label,
        "action_source": action_source,
        "queries": sink.queries,
        "actions": actions,
        "states": states,
        "proprios": proprios,
    }


def _load_policy_and_env(base_policy_path: Path, *, sampler: str, num_inference_timesteps: int):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=str(base_policy_path))
    if algo_name == "diffusion_policy":
        config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
        if sampler != "ddim":
            raise ValueError("DynaGuide CALVIN trace collection is configured for DDIM sampling")
        with config.values_unlocked():
            if not config.algo.ddim.enabled:
                raise RuntimeError("expected the base policy checkpoint to use DDIM sampling")
            config.algo.ddim.num_inference_timesteps = num_inference_timesteps
        ckpt_dict["config"] = config.dump()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_dict=ckpt_dict,
        device=device,
        verbose=True,
    )
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=False,
        verbose=True,
    )
    return policy, env


def _load_guidance_function(dynamics_model_path: Path, guidance_path: Path, config: dict[str, Any]):
    import torch
    from core.dynaguide import calculate_classifier_guidance
    from core.dynamics_models import FinalStatePredictionDino
    from core.embedder_datasets import MultiviewDataset

    model = FinalStatePredictionDino(
        action_dim=int(config["action_dim"]),
        action_horizon=int(config["action_chunk_length"]),
        cameras=[config["main_camera"]],
        proprio=config["proprio_key"],
        proprio_dim=15,
        reconstruction=True,
    )
    model.load_state_dict(torch.load(str(dynamics_model_path), map_location="cuda"))
    model.to("cuda")
    model.eval()

    dataset = MultiviewDataset(
        str(guidance_path),
        action_chunk_length=int(config["action_chunk_length"]),
        cameras=[config["main_camera"]],
        padding=True,
        pad_mode="repeat",
        proprio=config["proprio_key"],
    )
    guidance_function, _, _ = calculate_classifier_guidance(
        model,
        good_dataset=dataset,
        main_camera=config["main_camera"],
        scale=float(config["scale"]),
        alpha=float(config["alpha"]),
    )
    return guidance_function


def _write_episode(data_group: Any, episode: dict[str, Any]) -> None:
    demo = data_group.create_group(f"demo_{episode['episode_index']}")
    _write_attrs(
        demo,
        {
            "seed": episode["seed"],
            "rollout_index": episode["rollout_index"],
            "success": bool(episode["success"]),
            "behavior_label": episode["behavior_label"],
            "action_source": episode["action_source"],
        },
    )
    _write_queries(demo, episode["queries"])
    rollout = demo.create_group("rollout")
    rollout.create_dataset("actions", data=_stack(episode["actions"], np.float32))
    rollout.create_dataset("states", data=_stack(episode["states"], np.float32))
    rollout.create_dataset("proprios", data=_stack(episode["proprios"], np.float32))


def _write_queries(demo: Any, queries: list[QueryTrace]) -> None:
    query_group = demo.create_group("query")
    obs_group = query_group.create_group("obs")
    rgb_group = query_group.create_group("rgb")
    low_dim_group = query_group.create_group("low_dim")

    for key in sorted({key for query in queries for key in query.obs}):
        obs_group.create_dataset(
            key,
            data=_stack_preserve([query.obs[key] for query in queries]),
            compression="gzip",
        )
    for key in sorted({key for query in queries for key in query.obs_rgb}):
        rgb_group.create_dataset(
            key,
            data=_stack([query.obs_rgb[key] for query in queries], np.uint8),
            compression="gzip",
        )
    for key in sorted({key for query in queries for key in query.obs_low_dim}):
        low_dim_group.create_dataset(
            key,
            data=_stack([query.obs_low_dim[key] for query in queries], np.float32),
        )

    teacher = demo.create_group("teacher")
    teacher.create_dataset(
        "guided_action_chunk",
        data=_stack([query.teacher_guided_action_chunk for query in queries], np.float32),
        compression="gzip",
    )

    records = [record for query in queries for record in query.denoising_records]
    diffusion = demo.create_group("diffusion")
    diffusion.create_dataset("query_index", data=np.asarray([r.query_index for r in records], dtype=np.int64))
    diffusion.create_dataset("timestep", data=np.asarray([r.timestep for r in records], dtype=np.int64))
    for name in ("noisy_action", "unguided_noise_pred", "guided_noise_pred", "guidance_grad"):
        diffusion.create_dataset(
            name,
            data=_stack([getattr(record, name) for record in records], np.float32),
            compression="gzip",
        )


def _state_and_proprio(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    state = np.asarray(obs["states"])
    proprio = np.asarray(obs["proprio"])
    if state.ndim == 2:
        state = state[-1]
    if proprio.ndim == 2:
        proprio = proprio[-1]
    return state.astype(np.float32), proprio.astype(np.float32)


def _load_reset_poses(repo_root: Path, filename: str | None) -> list[np.ndarray] | None:
    if not filename:
        return None
    path = repo_root / "calvin_exp_configs_examples" / "reset_poses" / filename
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return [np.asarray(item, dtype=np.float32) for item in data["robot_states"]]


def _load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _volume_path(root: Path, path: str) -> Path:
    path = str(path)
    if path.startswith("/"):
        return root / path.lstrip("/")
    return root / path


def _prepare_imports(repo_root: Path) -> None:
    for path in (
        repo_root,
        repo_root / "robomimic",
        repo_root / "calvin" / "calvin_env",
        repo_root / "calvin" / "calvin_env" / "tacto",
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _episode_seed(seed: int, rollout_index: int) -> int:
    return int(seed) * 100_000 + int(rollout_index)


def _write_attrs(group: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        group.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value


def _stack(values: list[np.ndarray], dtype: Any) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    return np.stack(values, axis=0).astype(dtype)


def _stack_preserve(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.asarray([])
    return np.stack(values, axis=0)
