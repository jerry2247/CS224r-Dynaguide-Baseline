# DynaGuide Self-Distillation

This folder contains the trace-collection code for DynaGuide self-distillation
on CALVIN `switch_on`. The job runs the released base diffusion policy without
guidance, records DynaGuide teacher targets at the same visited states, and
writes a plain HDF5 dataset for the distillation stage.

The implementation is intentionally close to the released DynaGuide workflow:
simple JSON configs, a Modal launch file, and HDF5 outputs.

## Repository Layout

```text
dynaguide_self_distillation/
├── README.md
├── modal_app.py
├── pyproject.toml
├── configs/
│   ├── switch_on_trace.json
│   └── switch_on_trace_smoke.json
└── src/dynaguide_self_distillation/
    ├── __init__.py
    ├── calvin_labels.py
    ├── collect_traces.py
    └── trace_diffusion.py
```

## Inputs

The Modal volume is named:

```text
dynaguide-self-distillation
```

Upload the released artifacts to these paths:

```bash
modal volume create dynaguide-self-distillation
modal volume put dynaguide-self-distillation /path/to/base_policy.pth /inputs/base_policy.pth
modal volume put dynaguide-self-distillation /path/to/dynaguide_model.pth /inputs/dynaguide_model.pth
modal volume put dynaguide-self-distillation /path/to/switch_on_guidance.hdf5 /inputs/switch_on_guidance.hdf5
```

Inside the Modal container, the volume is mounted at `/artifacts`, so the config
entry `/inputs/base_policy.pth` resolves to:

```text
/artifacts/inputs/base_policy.pth
```

## Run Trace Collection

Production trace collection is launched with one command:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::collect_switch_on_traces
```

That entrypoint submits fifteen independent A100-80GB jobs:

```text
3 seeds x 5 shards per seed x 10 rollouts per shard = 150 episodes
```

Each shard writes its own HDF5 file and commits the Modal volume when that
shard finishes. This avoids a single five-hour failure point: if one shard
fails, rerun only that ten-rollout shard rather than the whole experiment.

To rerun one shard manually:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::collect_switch_on_shard --seed 2 --rollout-start 20 --n-rollouts 10
```

Smoke run:

```bash
modal run dynaguide_self_distillation/modal_app.py::collect_switch_on_smoke
```

All trace workers request a single `A100-80GB` GPU. The DynaGuide paper used
single RTX 3090-class GPUs; `A100-80GB` is used here for runtime and memory
headroom while preserving the single-GPU experimental setup.

## Output

The production launcher writes one shard per seed/rollout slice:

```text
/artifacts/traces/switch_on_teacher/seed_1_rollouts_00_09.hdf5
/artifacts/traces/switch_on_teacher/seed_1_rollouts_00_09_summary.json
/artifacts/traces/switch_on_teacher/seed_1_rollouts_10_19.hdf5
...
/artifacts/traces/switch_on_teacher/seed_3_rollouts_40_49.hdf5
/artifacts/traces/switch_on_teacher/seed_3_rollouts_40_49_summary.json
```

The smoke job writes:

```text
/artifacts/traces/switch_on_teacher_smoke/trace.hdf5
/artifacts/traces/switch_on_teacher_smoke/summary.json
```

Each HDF5 file contains rollout data and denoising targets. Each summary JSON
contains the config used for the shard, the number of episodes, the number of
`switch_on` successes, and the behavior histogram.

## Production Config

The production config is:

```text
configs/switch_on_trace.json
```

Key settings:

```json
{
  "task": "switch_on",
  "success_label": "switch_on",
  "seeds": [1, 2, 3],
  "n_rollouts_per_seed": 50,
  "horizon": 400,
  "env_setup": {"switch": 0},
  "sampler": "ddim",
  "num_inference_timesteps": 10,
  "scale": 1.5,
  "alpha": 30,
  "ss": 4
}
```

These match the released DynaGuide `switch_on` settings:

- CALVIN horizon is `400`.
- The switch starts off with `env_setup = {"switch": 0}`.
- The diffusion sampler is DDIM.
- DDIM inference uses `10` steps.
- DynaGuide uses `scale = 1.5`.
- The released code calls the latent-distance temperature `alpha`; for
  `switch_on`, it is `30`.
- The released code calls stochastic sampling `ss`; this corresponds to
  `M = 4` in the paper.

The policy loader requires DDIM for this job and sets the checkpoint's DDIM
inference step count from the config.

## Rollout Semantics

Each CALVIN rollout executes the student action as:

```python
action = policy(ob=obs)
```

No DynaGuide guidance arguments are passed into the action that steps the
environment.

When the policy is about to run a fresh diffusion query, the trace job performs
a separate DynaGuide denoising pass at the same observation. That pass records
the teacher targets and then restores PyTorch RNG state before the unguided
student action is sampled. This keeps the simulator trajectory on-policy for the
unguided student.

For each denoising record, the saved target is:

```text
guided_noise_pred = unguided_noise_pred - sqrt(1 - alpha_bar_k) * guidance_grad
```

This is the DynaGuide guidance update used during diffusion denoising.

Each episode is seeded from its configured seed and rollout index. This makes
the shards order-independent and rerunnable: `seed_2_rollouts_20_29.hdf5` is
the same shard whether it is collected before or after any other shard.

## HDF5 Structure

Episodes are stored under:

```text
data/demo_<episode_index>/
```

Each episode contains:

```text
query/rgb/<camera>
query/low_dim/proprio
teacher/guided_action_chunk
diffusion/query_index
diffusion/timestep
diffusion/noisy_action
diffusion/unguided_noise_pred
diffusion/guided_noise_pred
diffusion/guidance_grad
rollout/actions
rollout/states
rollout/proprios
```

Episode attributes:

```text
seed
rollout_index
success
behavior_label
action_source = "student_unguided"
```

Root attributes include the task name, success label, action source, horizon,
sampler, DynaGuide hyperparameters, seeds, and rollout count.

## Code Map

- `modal_app.py` defines the Modal image, mounts the artifact volume, installs
  upstream runtime packages, and exposes the production and smoke functions.
- `collect_traces.py` loads the policy, CALVIN environment, DynaGuide dynamics
  model, and guidance conditions; runs rollouts; and writes HDF5 plus summary
  outputs.
- `trace_diffusion.py` implements the DynaGuide side computation for each fresh
  diffusion-policy query.
- `calvin_labels.py` classifies the first behavior expressed in each rollout
  from privileged CALVIN state.

## Expected Dataset Size

The production run collects:

```text
3 seeds x 5 shards per seed x 10 rollouts per shard = 150 episodes
```

Each episode has a maximum of `400` environment steps. The policy acts through
action chunks, so fresh diffusion queries are less frequent than environment
steps. Each fresh query records:

```text
10 DDIM timesteps x 4 stochastic samples = 40 denoising records
```

At the horizon limit, each shard contains at most:

```text
10 episodes x 400 steps = 4,000 environment steps
```

The full production dataset contains at most:

```text
150 episodes x 400 steps = 60,000 environment steps
```

The trace is therefore a diffusion-target dataset for training, not just a set
of episode-level labels.
