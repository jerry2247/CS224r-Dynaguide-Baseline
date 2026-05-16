# DynaGuide Self-Distillation

This folder contains the DynaGuide self-distillation experiments for CALVIN. The
main experiment follows the project proposal: run the base diffusion policy
unguided, record DynaGuide denoising targets at the states that policy actually
visits, train the diffusion model to predict those guided denoising targets, and
evaluate the resulting policy with no guidance at inference time.

The implementation is intentionally compact: one Modal app, JSON experiment
configs, trace collection, diffusion-target training, and evaluation.

## Headline Result

All rows below use the same evaluation protocol: CALVIN `switch_on`, reset poses
from the reproduced DynaGuide setup, horizon 400, seeds `[1, 2, 3]`, and 100
rollouts per seed.

| Method | Guidance at evaluation | Success | Rollout time |
| --- | --- | ---: | ---: |
| Base diffusion policy | none | 14.3% ± 2.3% SE | 9.09 s |
| DynaGuide | online guidance | 72.0% ± 2.3% SE | 15.87 s |
| Distilled, off-policy guided traces | none | 73.0% ± 3.1% SE | 9.34 s |
| **Distilled, on-policy student traces** | **none** | **78.0% ± 2.1% SE** | **7.97 s** |

The on-policy distilled policy is the proposal method. It exceeds the reproduced
DynaGuide success rate while removing inference-time guidance.

## Repository Layout

```text
dynaguide_self_distillation/
├── README.md
├── modal_app.py
├── pyproject.toml
├── configs/
│   ├── switch_on_trace.json
│   ├── switch_on_trace_guided.json
│   ├── switch_on_trace_smoke.json
│   ├── switch_on_distill.json
│   ├── switch_on_distill_offpolicy.json
│   ├── switch_on_eval.json
│   ├── switch_on_eval_base.json
│   ├── switch_on_eval_dynaguide.json
│   ├── switch_on_eval_offpolicy.json
│   ├── drawer_open_trace_after_switch_on.json
│   ├── drawer_open_distill_after_switch_on.json
│   ├── drawer_open_eval_after_switch_on.json
│   └── switch_on_eval_retention_after_drawer_open.json
└── src/dynaguide_self_distillation/
    ├── __init__.py
    ├── calvin_labels.py
    ├── collect_traces.py
    ├── eval.py
    ├── trace_diffusion.py
    └── train.py
```

## Modal Inputs

Create the Modal volume and upload the released DynaGuide artifacts:

```bash
modal volume create dynaguide-self-distillation
modal volume put dynaguide-self-distillation /path/to/base_policy.pth /inputs/base_policy.pth
modal volume put dynaguide-self-distillation /path/to/dynaguide_model.pth /inputs/dynaguide_model.pth
modal volume put dynaguide-self-distillation /path/to/switch_on_guidance.hdf5 /inputs/switch_on_guidance.hdf5
```

Inside Modal, the volume is mounted at `/artifacts`, so
`/inputs/base_policy.pth` resolves to `/artifacts/inputs/base_policy.pth`.

The sequential `drawer_open` experiment also requires:

```bash
modal volume put dynaguide-self-distillation /path/to/drawer_open_guidance.hdf5 /inputs/drawer_open_guidance.hdf5
```

That file is not derived from the `switch_on` guidance set; it must be the
task-specific DynaGuide guidance-condition HDF5 for `drawer_open`.

## End-to-End Commands

Collect on-policy self-distillation traces:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::collect_switch_on_traces
```

Train the on-policy distilled policy:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::train_switch_on_distilled
```

Evaluate the main result:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_switch_on_base
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_switch_on_dynaguide
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_switch_on_distilled
```

Run the off-policy guided-rollout baseline:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::collect_switch_on_guided_traces
modal run --detach dynaguide_self_distillation/modal_app.py::train_switch_on_offpolicy_distilled
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_switch_on_offpolicy_distilled
```

Download the trained policies and metrics:

```bash
modal volume get dynaguide-self-distillation /distilled/switch_on ./dynaguide_self_distillation/outputs/distilled_switch_on
modal volume get dynaguide-self-distillation /distilled/switch_on_offpolicy_guided_rollouts ./dynaguide_self_distillation/outputs/distilled_switch_on_offpolicy
modal volume get dynaguide-self-distillation /metrics ./dynaguide_self_distillation/outputs/metrics
```

## Trace Collection

The production trace jobs launch 15 independent A100-80GB shard jobs:

```text
3 seeds x 5 shards per seed x 10 rollouts per shard = 150 episodes
```

Each shard writes independently, so completed shards survive worker failures:

```text
/artifacts/traces/switch_on_teacher/seed_1_rollouts_00_09.hdf5
/artifacts/traces/switch_on_teacher/seed_1_rollouts_00_09_summary.json
...
/artifacts/traces/switch_on_teacher/seed_3_rollouts_40_49.hdf5
/artifacts/traces/switch_on_teacher/seed_3_rollouts_40_49_summary.json
```

The on-policy trace semantics are strict: the simulator is stepped with
`policy(ob=obs)` and DynaGuide is only a side computation. Each query records:

```text
query/obs/<policy_obs_key>       full observation history used for training
query/rgb/<camera>               last-frame visual record for inspection
query/low_dim/proprio            last-frame proprio record for inspection
diffusion/noisy_action
diffusion/timestep
diffusion/unguided_noise_pred
diffusion/guided_noise_pred
diffusion/guidance_grad
teacher/guided_action_chunk
rollout/actions
rollout/states
rollout/proprios
```

The DynaGuide hyperparameters match the reproduced `switch_on` setup:

```json
{
  "horizon": 400,
  "sampler": "ddim",
  "num_inference_timesteps": 10,
  "scale": 1.5,
  "alpha": 30,
  "ss": 4
}
```

The completed on-policy trace set on the Modal volume contains 150 episodes,
1,907 policy queries, and 76,280 denoising supervision records. Its behavior
histogram has 27 `switch_on` successes, consistent with unguided base-policy
data rather than guided-policy leakage.

The off-policy baseline trace set uses the same schema but sets
`action_source = "dynaguide_guided"` and steps the environment with DynaGuide.
It contains 150 episodes, 913 policy queries, and 36,520 denoising supervision
records.

## Distillation Training

Training consumes trace shards directly from the Modal volume. Each supervision
sample is:

```text
(full_obs_history, noisy_action, timestep) -> guided_noise_pred
```

The trainer:

- loads a robomimic diffusion-policy checkpoint;
- starts from the deployed base-policy weights;
- freezes the observation encoder;
- trains only `policy.nets["policy"]["noise_pred_net"]`;
- minimizes MSE to the recorded DynaGuide-guided noise prediction;
- maintains an EMA copy of the trained noise predictor for saved checkpoints;
- writes `training_state.pth` every 2,000 steps so training can resume;
- saves robomimic-compatible `best.pth` and `final.pth`.

Default training config:

```json
{
  "batch_size": 64,
  "max_steps": 20000,
  "learning_rate": 0.00001,
  "weight_decay": 0.0,
  "validation_fraction": 0.1,
  "ema_decay": 0.999,
  "grad_clip": 1.0,
  "save_every": 2000
}
```

Completed training artifacts:

```text
/artifacts/distilled/switch_on/best.pth
/artifacts/distilled/switch_on/final.pth
/artifacts/distilled/switch_on/training_summary.json

/artifacts/distilled/switch_on_offpolicy_guided_rollouts/best.pth
/artifacts/distilled/switch_on_offpolicy_guided_rollouts/final.pth
/artifacts/distilled/switch_on_offpolicy_guided_rollouts/training_summary.json
```

The on-policy run used 68,652 training samples and 7,628 validation samples;
its best validation loss was `0.027430339755179983`. The off-policy run used
32,868 training samples and 3,652 validation samples; its best validation loss
was `0.011793498214783853`.

## Evaluation Artifacts

Completed metrics:

```text
/artifacts/metrics/switch_on_base/metrics.json
/artifacts/metrics/switch_on_dynaguide/metrics.json
/artifacts/metrics/switch_on_distilled/metrics.json
/artifacts/metrics/switch_on_offpolicy_guided_rollouts/metrics.json
```

Per-seed results:

| Method | Seed 1 | Seed 2 | Seed 3 | Total |
| --- | ---: | ---: | ---: | ---: |
| Base diffusion policy | 12/100 | 19/100 | 12/100 | 43/300 |
| DynaGuide | 76/100 | 68/100 | 72/100 | 216/300 |
| Distilled, off-policy guided traces | 79/100 | 69/100 | 71/100 | 219/300 |
| Distilled, on-policy student traces | 81/100 | 74/100 | 79/100 | 234/300 |

Evaluation writes `metrics.partial.json` after each completed seed and
`metrics.json` after all seeds finish.

## Sequential Experiment

The proposal's next experiment is sequential distillation. The code path is
implemented:

```bash
modal run --detach dynaguide_self_distillation/modal_app.py::collect_drawer_open_traces_after_switch_on
modal run --detach dynaguide_self_distillation/modal_app.py::train_drawer_open_after_switch_on
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_drawer_open_after_switch_on
modal run --detach dynaguide_self_distillation/modal_app.py::evaluate_switch_on_retention_after_drawer_open
```

Do not run those commands until `/inputs/drawer_open_guidance.hdf5` exists on
the Modal volume. The expected `drawer_open` DynaGuide hyperparameters are:

```json
{
  "horizon": 400,
  "sampler": "ddim",
  "num_inference_timesteps": 10,
  "scale": 1.0,
  "alpha": 40,
  "ss": 4
}
```
