from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class DenoisingRecord:
    query_index: int
    timestep: int
    noisy_action: np.ndarray
    unguided_noise_pred: np.ndarray
    guided_noise_pred: np.ndarray
    guidance_grad: np.ndarray


@dataclass
class QueryTrace:
    obs: dict[str, np.ndarray]
    obs_rgb: dict[str, np.ndarray]
    obs_low_dim: dict[str, np.ndarray]
    teacher_guided_action_chunk: np.ndarray
    denoising_records: list[DenoisingRecord]


@dataclass
class QueryTraceSink:
    rgb_keys: tuple[str, ...] = ("third_person", "eye_in_hand")
    low_dim_keys: tuple[str, ...] = ("proprio",)
    queries: list[QueryTrace] = field(default_factory=list)
    _raw_obs: dict[str, Any] | None = None

    @property
    def query_index(self) -> int:
        return len(self.queries)

    def begin(self, obs: dict[str, Any]) -> None:
        self._raw_obs = {key: np.asarray(value).copy() for key, value in obs.items()}

    def finish(self, guided_action_chunk: Any, records: list[DenoisingRecord]) -> None:
        if self._raw_obs is None:
            raise RuntimeError("begin must be called before finish")
        self.queries.append(
            QueryTrace(
                obs=_policy_obs(self._raw_obs, self.rgb_keys + self.low_dim_keys),
                obs_rgb=_rgb_obs(self._raw_obs, self.rgb_keys),
                obs_low_dim=_low_dim_obs(self._raw_obs, self.low_dim_keys),
                teacher_guided_action_chunk=_squeeze_batch(guided_action_chunk).astype(np.float32),
                denoising_records=records,
            )
        )
        self._raw_obs = None


def policy_will_query(policy_algo: Any) -> bool:
    queue = getattr(policy_algo, "action_queue", None)
    if queue is None:
        return True
    try:
        return len(queue) == 0
    except TypeError:
        return not bool(queue)


def trace_guided_denoising_at_obs(
    policy_algo: Any,
    obs: dict[str, Any],
    sink: QueryTraceSink,
    guidance_function: Callable[[dict[str, Any], Any], Any],
    *,
    ss: int,
    num_inference_timesteps: int,
) -> None:
    """Trace DynaGuide targets without changing the following unguided policy call."""

    rng_state = _torch_rng_state()
    sink.begin(obs)
    try:
        policy_obs = _prepare_policy_obs(policy_algo, obs)
        _run_guided_denoising(
            policy_algo,
            policy_obs,
            sink,
            guidance_function,
            ss=ss,
            num_inference_timesteps=num_inference_timesteps,
        )
    finally:
        _restore_torch_rng_state(rng_state)


def _run_guided_denoising(
    policy_algo: Any,
    inputs: dict[str, Any],
    sink: QueryTraceSink,
    guidance_function: Callable[[dict[str, Any], Any], Any],
    *,
    ss: int,
    num_inference_timesteps: int,
) -> None:
    import torch
    import tqdm
    import robomimic.utils.tensor_utils as TensorUtils

    if policy_algo.nets.training:
        raise RuntimeError("trace collection expects an eval-mode policy")

    nets = policy_algo.nets
    if policy_algo.ema is not None:
        policy_algo.ema.copy_to(parameters=policy_algo._shadow_nets.parameters())
        nets = policy_algo._shadow_nets

    observation_horizon = int(policy_algo.algo_config.horizon.observation_horizon)
    action_horizon = int(policy_algo.algo_config.horizon.action_horizon)
    prediction_horizon = int(policy_algo.algo_config.horizon.prediction_horizon)
    action_dim = int(policy_algo.ac_dim)

    for key in policy_algo.obs_shapes:
        if inputs[key].ndim - 1 == len(policy_algo.obs_shapes[key]):
            inputs[key] = torch.unsqueeze(inputs[key], dim=1)
        if inputs[key].ndim - 2 != len(policy_algo.obs_shapes[key]):
            raise ValueError(f"unexpected observation rank for {key}")

    with torch.no_grad():
        obs_features = TensorUtils.time_distributed(
            {"obs": inputs, "goal": None},
            nets["policy"]["obs_encoder"],
            inputs_as_kwargs=True,
        )
    obs_cond = obs_features.flatten(start_dim=1)

    naction = torch.randn(
        (1, prediction_horizon, action_dim),
        device=policy_algo.device,
        requires_grad=True,
    )
    policy_algo.noise_scheduler.set_timesteps(num_inference_timesteps)

    records = []
    query_index = sink.query_index
    for timestep in tqdm.tqdm(policy_algo.noise_scheduler.timesteps, leave=False):
        for _ in range(ss):
            noisy_action = naction
            with torch.no_grad():
                unguided_noise_pred = nets["policy"]["noise_pred_net"](
                    sample=naction,
                    timestep=timestep,
                    global_cond=obs_cond,
                )
            guidance_grad = guidance_function(inputs, naction)
            timestep_index = int(timestep.item() if hasattr(timestep, "item") else timestep)
            scheduler_scale = (1 - policy_algo.noise_scheduler.alphas_cumprod[timestep_index]).sqrt()
            scheduler_scale = scheduler_scale.to(device=naction.device)
            guided_noise_pred = unguided_noise_pred - scheduler_scale * guidance_grad
            records.append(
                DenoisingRecord(
                    query_index=query_index,
                    timestep=timestep_index,
                    noisy_action=_squeeze_batch(noisy_action).astype(np.float32),
                    unguided_noise_pred=_squeeze_batch(unguided_noise_pred).astype(np.float32),
                    guided_noise_pred=_squeeze_batch(guided_noise_pred).astype(np.float32),
                    guidance_grad=_squeeze_batch(guidance_grad).astype(np.float32),
                )
            )
            naction = policy_algo.noise_scheduler.step(
                model_output=guided_noise_pred,
                timestep=timestep,
                sample=naction,
            ).prev_sample.detach()
            naction.requires_grad_(True)

    start = observation_horizon - 1
    guided_action_chunk = naction[:, start : start + action_horizon]
    sink.finish(guided_action_chunk, records)


def _prepare_policy_obs(policy_algo: Any, obs: dict[str, Any]) -> dict[str, Any]:
    import robomimic.utils.tensor_utils as TensorUtils

    obs = TensorUtils.to_tensor(obs)
    obs = TensorUtils.to_batch(obs)
    obs = TensorUtils.to_device(obs, policy_algo.device)
    return TensorUtils.to_float(obs)


def _torch_rng_state() -> tuple[Any, Any]:
    import torch

    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return cpu_state, cuda_state


def _restore_torch_rng_state(state: tuple[Any, Any]) -> None:
    import torch

    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _squeeze_batch(value: Any) -> np.ndarray:
    value = _to_numpy(value)
    if value.shape and value.shape[0] == 1:
        return value[0]
    return value


def _policy_obs(obs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    for key in keys:
        if key in obs:
            out[key] = np.asarray(obs[key]).copy()
    return out


def _rgb_obs(obs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    for key in keys:
        if key not in obs:
            continue
        value = np.asarray(obs[key])
        if value.ndim == 4:
            value = value[-1]
        if value.ndim == 3 and value.shape[0] in (1, 3):
            value = np.transpose(value, (1, 2, 0))
        if value.dtype != np.uint8:
            scale = 255 if value.max(initial=0) <= 1.0 else 1
            value = np.clip(value * scale, 0, 255).astype(np.uint8)
        out[key] = value
    return out


def _low_dim_obs(obs: dict[str, Any], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    out = {}
    for key in keys:
        if key not in obs:
            continue
        value = np.asarray(obs[key])
        if value.ndim == 2:
            value = value[-1]
        out[key] = value.astype(np.float32)
    return out
