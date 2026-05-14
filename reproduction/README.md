# DynaGuide reproduction — CALVIN `switch_on`

Exact reproduction of the headline DynaGuide result on `switch_on`. The
vendored DynaGuide repository at the parent level is **never modified** —
we only call it.

## Headline result

| Metric | Paper | Ours | Verdict |
|---|---|---|---|
| DynaGuide `switch_on` success rate (mean ± SE, n = 3 seeds) | **74.0 % ± 2.7 %** | **75.0 % ± 4.0 %** | ✅ within pass band [68 %, 78 %] |
| Base policy `switch_on` (no guidance) | ≈ 8 % | 13 % | ✓ within single-seed Bernoulli noise |
| Boost factor | ≈ 9 × | 5.8 × | ✓ lower because our baseline ran higher |

Per-seed DynaGuide rates: 79 %, 67 %, 79 %. Source: paper Table 5 / Table 4 (Appendix A.4).

Full numerical breakdown → `results/behavior_report.json`.

## Layout (7 files, every one load-bearing)

```
reproduction/
├── README.md                       ← this file
├── modal_app.py                    ← only entrypoint; runs on Modal (GPU)
├── configs/switch_on_paper.json    ← pinned hyperparameters
├── artifacts/REGISTRY.json         ← SHA-256 of the 3 binary inputs (gitignored)
├── scripts/verify.py               ← local pre-flight (enforces REGISTRY hashes)
├── scripts/analyze_behaviors.py    ← post-hoc behavior classification → result
└── results/behavior_report.json    ← the headline number
```

## Reproduce end-to-end

```bash
# --- one-time, on your Mac ---
pip install modal
modal token new

# Download the three release artifacts (links in the upstream README) and
# place them at reproduction/artifacts/ with canonical names:
#   dynaguide_model.pth, base_policy.pth, switch_on_guidance.hdf5

# Pre-flight: enforces SHA-256 in REGISTRY.json against local files
python reproduction/scripts/verify.py

# Upload artifacts to the Modal volume
modal volume create dynaguide-artifacts
modal volume put dynaguide-artifacts reproduction/artifacts/dynaguide_model.pth     /
modal volume put dynaguide-artifacts reproduction/artifacts/base_policy.pth         /
modal volume put dynaguide-artifacts reproduction/artifacts/switch_on_guidance.hdf5 /

# --- sanity checks on Modal (~$0.10) ---
modal run reproduction/modal_app.py::verify_environment
modal run reproduction/modal_app.py::verify_artifacts
modal run reproduction/modal_app.py::sanity_check_dynamics_model

# --- full reproduction: 1 baseline + 3 DynaGuide seeds (~75 min, ~$2.50) ---
modal run --detach reproduction/modal_app.py::reproduce

# --- pull results back and aggregate ---
modal volume get dynaguide-artifacts /results reproduction/results
python reproduction/scripts/analyze_behaviors.py
```

## Pinned hyperparameters

From `configs/switch_on_paper.json` (paper Table 4, Appendix A.4):
`scale=1.5, alpha_sigma=30, stochastic_sampling_M=4, |g+|=20, K=10, T_p/T_a/T_o = 16/14/2`.

## Deviations from the paper

None of substance. Minor, accepted differences:

- **Seeds**: paper uses 6, we use 3 (budget). We report SE accordingly.
- **Hardware**: paper uses RTX 3090, we use A10G on Modal. Same arch family.
- **Action prefix `T_a`**: paper appendix says 14; upstream config defaults to 8 but `diffusion_policy.py:658` hardcodes 14 for the guidance path. We use 14.

## Why `Success_Rate: 0.0` in `summary.json` is OK

`run_dynaguide.py` writes `summary.json` with `Success_Rate` from CALVIN's
generic `env.is_success()` — which does NOT detect specific behaviors like
`switch_on`. The proper score is computed post-hoc from the per-step state
traces in `data.hdf5` via `analyze_behaviors.py::classify_demo` (mirroring
`analyze_calvin_touch.py:generate_detailed_behavior_distribution` in the
upstream repo). All headline numbers come from that classifier.

## Provenance

Every Modal run writes a `run_manifest.json` alongside its rollout output
(git commit, artifact SHA-256s, command line, UTC timestamps). The
combination of `REGISTRY.json` and these per-run manifests is sufficient
to prove which input file produced which output number.
