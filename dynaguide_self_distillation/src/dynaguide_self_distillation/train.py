from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from dynaguide_self_distillation.collect_traces import _prepare_imports, _volume_path


class TraceDataset(Dataset):
    """DynaGuide denoising targets from corrected trace shards."""

    def __init__(self, trace_dir: Path):
        self.query_obs: list[dict[str, np.ndarray]] = []
        self.query_refs: list[np.ndarray] = []
        noisy_actions = []
        timesteps = []
        targets = []

        paths = sorted(trace_dir.glob("*.hdf5"))
        if not paths:
            raise FileNotFoundError(f"no trace shards found in {trace_dir}")

        self.obs_keys: tuple[str, ...] | None = None
        for path in paths:
            with h5py.File(path, "r") as h5:
                if h5.attrs.get("trace_schema") != "trace.v2_obs_history":
                    raise RuntimeError(f"{path.name} is not a final training trace; recollect traces with current code")
                for demo_name in sorted(h5["data"].keys()):
                    demo = h5["data"][demo_name]
                    if "obs" not in demo["query"]:
                        raise RuntimeError(
                            f"{path.name}/{demo_name} was collected with the old trace schema; "
                            "recollect traces so query/obs contains full policy observation history"
                        )
                    obs_group = demo["query"]["obs"]
                    keys = tuple(sorted(obs_group.keys()))
                    if self.obs_keys is None:
                        self.obs_keys = keys
                    elif keys != self.obs_keys:
                        raise RuntimeError(f"inconsistent observation keys in {path.name}/{demo_name}: {keys}")

                    query_base = len(self.query_obs)
                    n_queries = obs_group[keys[0]].shape[0]
                    for query_index in range(n_queries):
                        self.query_obs.append({key: obs_group[key][query_index] for key in keys})

                    diffusion = demo["diffusion"]
                    query_index = diffusion["query_index"][:].astype(np.int64)
                    self.query_refs.append(query_base + query_index)
                    noisy_actions.append(diffusion["noisy_action"][:].astype(np.float32))
                    timesteps.append(diffusion["timestep"][:].astype(np.int64))
                    targets.append(diffusion["guided_noise_pred"][:].astype(np.float32))

        self.obs_keys = self.obs_keys or tuple()
        self.query_refs_array = np.concatenate(self.query_refs, axis=0)
        self.noisy_actions = np.concatenate(noisy_actions, axis=0)
        self.timesteps = np.concatenate(timesteps, axis=0)
        self.targets = np.concatenate(targets, axis=0)

    def __len__(self) -> int:
        return int(self.timesteps.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "obs": self.query_obs[int(self.query_refs_array[index])],
            "noisy_action": self.noisy_actions[index],
            "timestep": self.timesteps[index],
            "target": self.targets[index],
        }


def train_distilled(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    artifact_root: str | Path = "/artifacts",
    commit_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    artifact_root = Path(artifact_root)
    _prepare_imports(repo_root)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    _set_seed(int(config["seed"]))

    output_dir = _volume_path(artifact_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = _volume_path(artifact_root, config["trace_dir"])

    rollout_policy, ckpt_dict = _load_policy(
        _volume_path(artifact_root, config["base_policy"]),
        sampler=config["sampler"],
        num_inference_timesteps=int(config["num_inference_timesteps"]),
    )
    policy_algo = rollout_policy.policy
    _start_from_deployed_weights(policy_algo)

    dataset = TraceDataset(trace_dir)
    _validate_trace_shapes(dataset, policy_algo)
    train_loader, val_loader = _make_loaders(dataset, config)

    obs_encoder = policy_algo.nets["policy"]["obs_encoder"]
    noise_net = policy_algo.nets["policy"]["noise_pred_net"]
    for param in obs_encoder.parameters():
        param.requires_grad_(False)
    for param in noise_net.parameters():
        param.requires_grad_(True)
    policy_algo.nets.eval()
    noise_net.train()

    optimizer = torch.optim.AdamW(
        noise_net.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    ema_state = _clone_state(noise_net)
    train_state_path = output_dir / "training_state.pth"
    start_step, best_val_loss = _maybe_resume(train_state_path, policy_algo, optimizer, ema_state)

    writer = _summary_writer(output_dir)
    event_path = output_dir / "train_events.jsonl"
    train_iter = iter(train_loader)
    latest_train_loss = None
    latest_grad_norm = None

    for step in range(start_step + 1, int(config["max_steps"]) + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        latest_train_loss, latest_grad_norm = _train_step(
            batch,
            rollout_policy,
            noise_net,
            obs_encoder,
            optimizer,
            grad_clip=float(config["grad_clip"]),
            use_bfloat16=bool(config["use_bfloat16"]),
        )
        _update_ema(ema_state, noise_net, decay=float(config["ema_decay"]))

        if step % int(config["log_every"]) == 0:
            _log_event(event_path, {"step": step, "train_loss": latest_train_loss, "grad_norm": latest_grad_norm})
            if writer is not None:
                writer.add_scalar("train/loss", latest_train_loss, step)
                writer.add_scalar("train/grad_norm", latest_grad_norm, step)

        if step % int(config["save_every"]) == 0 or step == int(config["max_steps"]):
            val_loss = _validation_loss(
                val_loader,
                rollout_policy,
                noise_net,
                obs_encoder,
                use_bfloat16=bool(config["use_bfloat16"]),
            )
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, step)
            _log_event(
                event_path,
                {"step": step, "train_loss": latest_train_loss, "val_loss": val_loss, "grad_norm": latest_grad_norm},
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_policy_checkpoint(
                    output_dir / "best.pth",
                    ckpt_dict,
                    policy_algo,
                    noise_net,
                    ema_state,
                    config,
                    {"step": step, "val_loss": val_loss, "checkpoint": "best"},
                )

            _save_training_state(train_state_path, policy_algo, optimizer, ema_state, step, best_val_loss)
            _save_policy_checkpoint(
                output_dir / "final.pth",
                ckpt_dict,
                policy_algo,
                noise_net,
                ema_state,
                config,
                {"step": step, "val_loss": val_loss, "checkpoint": "final"},
            )
            if commit_callback is not None:
                commit_callback()

    if writer is not None:
        writer.close()

    summary = {
        "task": config["task"],
        "trace_dir": str(trace_dir),
        "output_dir": str(output_dir),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "total_samples": len(dataset),
        "best_val_loss": best_val_loss,
        "final_train_loss": latest_train_loss,
        "best_checkpoint": str(output_dir / "best.pth"),
        "final_checkpoint": str(output_dir / "final.pth"),
        "config": config,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _load_policy(base_policy_path: Path, *, sampler: str, num_inference_timesteps: int):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.torch_utils as TorchUtils

    algo_name, ckpt_dict = FileUtils.algo_name_from_checkpoint(ckpt_path=str(base_policy_path))
    if algo_name != "diffusion_policy":
        raise RuntimeError(f"expected diffusion_policy checkpoint, got {algo_name}")
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    if sampler != "ddim":
        raise ValueError("distillation is configured for DDIM sampling")
    with config.values_unlocked():
        if not config.algo.ddim.enabled:
            raise RuntimeError("expected the base policy checkpoint to use DDIM sampling")
        config.algo.ddim.num_inference_timesteps = num_inference_timesteps
    ckpt_dict["config"] = config.dump()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    return FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)


def _start_from_deployed_weights(policy_algo: Any) -> None:
    if policy_algo.ema is not None:
        policy_algo.ema.copy_to(parameters=policy_algo.nets.parameters())
        policy_algo.ema = None


def _validate_trace_shapes(dataset: TraceDataset, policy_algo: Any) -> None:
    sample = dataset[0]
    for key, shape in policy_algo.obs_shapes.items():
        if key not in sample["obs"]:
            raise RuntimeError(f"trace observations are missing policy key {key}")
        if sample["obs"][key].ndim - 1 != len(shape):
            raise RuntimeError(
                f"trace key {key} has shape {sample['obs'][key].shape}; expected time history plus {tuple(shape)}"
            )
    if int(policy_algo.algo_config.horizon.observation_horizon) != sample["obs"][next(iter(policy_algo.obs_shapes))].shape[0]:
        raise RuntimeError("trace observation history does not match policy observation_horizon")
    expected_action_shape = (int(policy_algo.algo_config.horizon.prediction_horizon), int(policy_algo.ac_dim))
    if sample["noisy_action"].shape != expected_action_shape or sample["target"].shape != expected_action_shape:
        raise RuntimeError("trace action targets do not match policy prediction horizon and action dimension")


def _make_loaders(dataset: TraceDataset, config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    val_count = max(1, int(round(len(dataset) * float(config["validation_fraction"]))))
    train_count = len(dataset) - val_count
    train_set, val_set = random_split(
        dataset,
        [train_count, val_count],
        generator=torch.Generator().manual_seed(int(config["seed"])),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _train_step(
    batch: dict[str, Any],
    rollout_policy: Any,
    noise_net: torch.nn.Module,
    obs_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    grad_clip: float,
    use_bfloat16: bool,
) -> tuple[float, float]:
    noise_net.train()
    optimizer.zero_grad(set_to_none=True)
    obs, noisy_action, timestep, target = _prepare_batch(batch, rollout_policy)
    with torch.no_grad():
        obs_cond = _encode_obs(obs_encoder, obs)
    with _autocast(rollout_policy.policy.device, use_bfloat16):
        pred = noise_net(sample=noisy_action, timestep=timestep, global_cond=obs_cond)
        loss = F.mse_loss(pred.float(), target.float())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(noise_net.parameters(), grad_clip)
    optimizer.step()
    return float(loss.detach().cpu()), float(grad_norm.detach().cpu())


@torch.no_grad()
def _validation_loss(
    loader: DataLoader,
    rollout_policy: Any,
    noise_net: torch.nn.Module,
    obs_encoder: torch.nn.Module,
    *,
    use_bfloat16: bool,
) -> float:
    noise_net.eval()
    losses = []
    for batch in loader:
        obs, noisy_action, timestep, target = _prepare_batch(batch, rollout_policy)
        obs_cond = _encode_obs(obs_encoder, obs)
        with _autocast(rollout_policy.policy.device, use_bfloat16):
            pred = noise_net(sample=noisy_action, timestep=timestep, global_cond=obs_cond)
            losses.append(float(F.mse_loss(pred.float(), target.float()).cpu()))
    noise_net.train()
    return float(np.mean(losses))


def _prepare_batch(batch: dict[str, Any], rollout_policy: Any):
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.tensor_utils as TensorUtils

    device = rollout_policy.policy.device
    obs = TensorUtils.to_device(TensorUtils.to_float(batch["obs"]), device)
    if rollout_policy.obs_normalization_stats is not None:
        stats = TensorUtils.to_device(TensorUtils.to_float(TensorUtils.to_tensor(rollout_policy.obs_normalization_stats)), device)
        obs = ObsUtils.normalize_dict(obs, normalization_stats=stats)
    noisy_action = batch["noisy_action"].to(device=device, dtype=torch.float32)
    timestep = batch["timestep"].to(device=device, dtype=torch.long)
    target = batch["target"].to(device=device, dtype=torch.float32)
    return obs, noisy_action, timestep, target


def _encode_obs(obs_encoder: torch.nn.Module, obs: dict[str, torch.Tensor]) -> torch.Tensor:
    import robomimic.utils.tensor_utils as TensorUtils

    obs_features = TensorUtils.time_distributed(
        {"obs": obs, "goal": None},
        obs_encoder,
        inputs_as_kwargs=True,
    )
    return obs_features.flatten(start_dim=1)


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


@torch.no_grad()
def _update_ema(ema_state: dict[str, torch.Tensor], module: torch.nn.Module, *, decay: float) -> None:
    for key, value in module.state_dict().items():
        ema_state[key].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def _save_policy_checkpoint(
    path: Path,
    ckpt_dict: dict[str, Any],
    policy_algo: Any,
    noise_net: torch.nn.Module,
    ema_state: dict[str, torch.Tensor],
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    current_state = _clone_state(noise_net)
    noise_net.load_state_dict(ema_state)
    try:
        out = copy.deepcopy(ckpt_dict)
        saved_config = json.loads(out["config"])
        saved_config["algo"]["ema"]["enabled"] = False
        out["config"] = json.dumps(saved_config)
        out["model"] = policy_algo.serialize()
        out["distillation"] = {"config": config, **metadata}
        torch.save(out, path)
    finally:
        noise_net.load_state_dict(current_state)


def _save_training_state(
    path: Path,
    policy_algo: Any,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, torch.Tensor],
    step: int,
    best_val_loss: float,
) -> None:
    torch.save(
        {
            "model": policy_algo.nets.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema_noise_pred": ema_state,
            "step": step,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def _maybe_resume(
    path: Path,
    policy_algo: Any,
    optimizer: torch.optim.Optimizer,
    ema_state: dict[str, torch.Tensor],
) -> tuple[int, float]:
    if not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location=policy_algo.device)
    policy_algo.nets.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    ema_state.clear()
    ema_state.update({key: value.to(policy_algo.device) for key, value in state["ema_noise_pred"].items()})
    return int(state["step"]), float(state["best_val_loss"])


def _summary_writer(output_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    except Exception:
        return None


def _log_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
