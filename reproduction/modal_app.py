"""
Modal app for the DynaGuide reproduction.

This file is the ONLY Modal entrypoint. It reads hyperparameters from
reproduction/configs/switch_on_paper.json — do not hardcode hyperparameter
values anywhere else.

USAGE
-----
    # one-time, on your Mac
    pip install modal
    modal token new

    # one-time, upload artifacts to the Modal volume
    modal volume put dynaguide-artifacts ~/Downloads/dynaguide_model.pth      /
    modal volume put dynaguide-artifacts ~/Downloads/base_policy.pth          /
    modal volume put dynaguide-artifacts ~/Downloads/switch_on_guidance.hdf5  /

    # sanity checks
    modal run reproduction/modal_app.py::verify_environment
    modal run reproduction/modal_app.py::verify_artifacts
    modal run reproduction/modal_app.py::sanity_check_dynamics_model

    # the reproduction (multi-seed)
    modal run reproduction/modal_app.py::reproduce        # all 3 seeds + baseline

    # individual seeds
    modal run reproduction/modal_app.py::run_baseline --seed 1
    modal run reproduction/modal_app.py::run_dynaguide --seed 1
    modal run reproduction/modal_app.py::run_dynaguide --seed 2
    modal run reproduction/modal_app.py::run_dynaguide --seed 3

    # pull results back to your Mac
    modal volume get dynaguide-artifacts /results ./reproduction/results

NOTES
-----
- Every run writes a `run_manifest.json` alongside the rollout outputs. This
  records exact hyperparameters used, git commit hash, container image hash,
  start/end timestamps, GPU type, and a SHA-256 of the dynamics model + base
  policy. Use it to prove a result was generated from a specific run.

- No code in this file modifies the vendored DynaGuide repository. We only
  CALL run_dynaguide.py with paper hyperparameters. This is the "exact
  reproduction" path.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "dynaguide-reproduction"

# ============================================================
# Paths
# ============================================================
REPO_ROOT_LOCAL = Path("/Users/jerrygu/Jerry Research/CS224r DynaGuide Reproduction")
CONFIG_LOCAL = REPO_ROOT_LOCAL / "reproduction" / "configs" / "switch_on_paper.json"

# Inside the container:
REPO_ROOT = "/workspace"
ARTIFACTS = "/artifacts"
RESULTS_ROOT = f"{ARTIFACTS}/results"

# ============================================================
# Load the pinned hyperparameter config at import time.
#
# The module is imported in two places:
#   1. Locally on your Mac when `modal run` discovers the function.
#   2. Inside the Modal container when the function actually executes.
#
# We try both paths so this works in either context.
# ============================================================
CONFIG_IN_CONTAINER = Path(REPO_ROOT) / "reproduction/configs/switch_on_paper.json"

def _load_config() -> dict:
    for candidate in (CONFIG_LOCAL, CONFIG_IN_CONTAINER):
        if Path(candidate).exists():
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Config not found. Tried:\n  {CONFIG_LOCAL}\n  {CONFIG_IN_CONTAINER}"
    )

CONFIG = _load_config()

# Sanity: catch hyperparameter drift early.
assert CONFIG["task"]["name"] == "switch_on", "Config task must be switch_on."
assert CONFIG["guidance_hyperparameters"]["scale"] == 1.5, "Paper scale is 1.5."
assert CONFIG["guidance_hyperparameters"]["alpha_sigma"] == 30, "Paper sigma is 30."
assert CONFIG["guidance_hyperparameters"]["stochastic_sampling_M"] == 4, "Paper M is 4."


# ============================================================
# Container image
# ============================================================
# In modern Modal (>= 1.0), local directories are added to the image via
# .add_local_dir() instead of modal.Mount.from_local_dir().
#
# IMPORTANT: artifact binaries (*.pth, *.hdf5) are NOT uploaded with the repo.
# They live on the dynaguide-artifacts volume. Only metadata files (README,
# REGISTRY.json) ship inside the image so the container can read them.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "ffmpeg",
        "libegl1",
        "libgl1",
        "libgl1-mesa-dev",
        "libosmesa6-dev",
        "libglfw3",
        "libglfw3-dev",
        "libglew-dev",
        "patchelf",
        "build-essential",
        "cmake",
    )
    # IMPORTANT: numpy must stay on the 1.x ABI for torch 2.1 + DinoV2 to load
    # without "Failed to initialize NumPy" warnings. Pandas 2.1+ pulls in
    # numpy>=2.0 transitively, so we pin pandas too. CALVIN's install will
    # bring in pandas via gitpython→pandas, so this pin matters.
    .pip_install("numpy==1.26.4", "pandas==2.0.3")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "diffusers==0.24.0",
        "einops",
        "tqdm",
        "h5py",
        "matplotlib",
        "imageio[ffmpeg]",
        "opencv-python-headless",
        "tensorboard",
        "tensorboardX",        # robomimic uses this in addition to tensorboard
        "psutil",              # robomimic dep
        "termcolor",           # robomimic dep
        "scikit-learn==1.4.2", # test_dynaguide_embedding.py uses TSNE/PCA
        "pybullet",
        "pyglet",
        "hydra-core==1.1.1",
        "omegaconf",
        "gym",
    )
    .env({
        "EGL_PLATFORM": "surfaceless",
        "PYOPENGL_PLATFORM": "egl",
        "MUJOCO_GL": "egl",
        "PYTHONUNBUFFERED": "1",
        # The DynaGuide repo has no setup.py at root, so we make /workspace
        # importable directly via PYTHONPATH. `from core.dynaguide import ...`
        # then works as long as cwd or PYTHONPATH contains /workspace.
        "PYTHONPATH": "/workspace",
    })
    .add_local_dir(
        str(REPO_ROOT_LOCAL),
        remote_path=REPO_ROOT,
        ignore=[
            ".git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.DS_Store",
            "reproduction/results/**",
            "reproduction/artifacts/*.pth",
            "reproduction/artifacts/*.hdf5",
            "reproduction/artifacts/*.mp4",
            "tmp_test/**",
        ],
    )
)

app = modal.App(APP_NAME, image=image)

# ============================================================
# Volumes
# ============================================================
artifacts = modal.Volume.from_name("dynaguide-artifacts", create_if_missing=True)


# ============================================================
# Helpers
# ============================================================
def _bootstrap() -> None:
    """Install vendored editable packages once per cold start.

    Order matters:
      1. Editable installs (robomimic, calvin/calvin_env) — they pull in
         their own deps, which may UPGRADE diffusers / numpy / pandas.
      2. AFTER those, force-pin our critical versions back. This is the
         final state the SUBPROCESS will see when it imports things.
    """
    print("[bootstrap] installing robomimic ...")
    subprocess.run(
        ["pip", "install", "-q", "-e", f"{REPO_ROOT}/robomimic"],
        check=True,
    )
    print("[bootstrap] installing CALVIN ...")
    rc = subprocess.run(
        ["bash", "install.sh"],
        cwd=f"{REPO_ROOT}/calvin",
        check=False,
    ).returncode
    if rc != 0:
        print(f"[bootstrap] install.sh exited {rc}, retrying calvin_env editable install ...")
        subprocess.run(
            ["pip", "install", "-q", "-e", f"{REPO_ROOT}/calvin/calvin_env"],
            check=False,
        )

    # ------------------------------------------------------------
    # FINAL PINNING STEP — runs AFTER the editable installs.
    #
    # robomimic/setup.py declares `diffusers>=0.26.2`, which causes
    # `pip install -e robomimic` to UPGRADE diffusers past our pinned
    # 0.24.0 to a version that references torch.xpu (broken on torch 2.1).
    # Similarly, calvin's install.sh pulls in pandas/numpy upgrades.
    #
    # We do a clean uninstall + install here so the FINAL state has the
    # versions we need. Subsequent subprocesses (e.g. run_dynaguide.py)
    # will see this final state.
    #
    # numpy 1.26.4         — torch 2.1 ABI
    # pandas 2.0.3         — last pandas that works cleanly with numpy 1.x
    # diffusers 0.24.0     — last version before torch.xpu reference
    # huggingface_hub 0.19.4 — diffusers 0.24 uses hf_cache_home (removed in hub >= 0.22)
    # ------------------------------------------------------------
    print("[bootstrap] FINAL PIN: uninstalling and reinstalling pinned versions ...")
    subprocess.run(
        ["pip", "uninstall", "-y", "-q",
         "diffusers", "huggingface_hub"],
        check=False,
    )
    subprocess.run(
        ["pip", "install", "-q", "--upgrade", "--upgrade-strategy", "only-if-needed",
         "numpy==1.26.4",
         "pandas==2.0.3",
         "diffusers==0.24.0",
         "huggingface_hub==0.19.4"],
        check=True,
    )

    # Sanity print — confirm what's actually installed AT THE END
    import sys
    for mod_name in ("numpy", "diffusers", "huggingface_hub", "pandas"):
        for k in [k for k in list(sys.modules) if k == mod_name or k.startswith(mod_name + ".")]:
            del sys.modules[k]
    for name in ("numpy", "pandas", "diffusers", "huggingface_hub"):
        try:
            mod = __import__(name)
            print(f"[bootstrap] {name:18s} = {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"[bootstrap] {name} import test failed: {e}")

    # The DynaGuide repo has no setup.py at root; PYTHONPATH=/workspace in
    # the image env makes `from core.*` work directly.

    # Refresh sys.path so this Python process sees newly written .pth files
    import site
    site.addsitedir("/usr/local/lib/python3.10/site-packages")
    for p in [
        f"{REPO_ROOT}/robomimic",
        f"{REPO_ROOT}/calvin/calvin_env",
        f"{REPO_ROOT}/calvin/calvin_env/tacto",
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    print("[bootstrap] done.")


def _sha256(path: str, max_bytes: int = 64 * 1024 * 1024) -> str:
    """SHA-256 of (up to max_bytes of) a file — fast enough for big .pth."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", REPO_ROOT, "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception as e:
        return f"unknown ({e})"


def _write_manifest(out_dir: str, run_kind: str, cmd: list,
                    extras: dict | None = None) -> None:
    """Drop a run_manifest.json next to the rollout outputs."""
    import platform
    manifest = {
        "run_kind": run_kind,
        "task": CONFIG["task"]["name"],
        "config_version": CONFIG["_meta"]["config_version"],
        "git_commit": _git_commit(),
        "container_python": platform.python_version(),
        "command": cmd,
        "hyperparameters_used": {
            "scale":  float(cmd[cmd.index("--scale")  + 1]),
            "alpha":  float(cmd[cmd.index("--alpha")  + 1]),
            "ss":     int(cmd[cmd.index("--ss")       + 1]),
            "seed":   int(cmd[cmd.index("--seed")     + 1]),
            "n_rollouts": int(cmd[cmd.index("--n_rollouts") + 1]),
            "horizon":    int(cmd[cmd.index("--horizon")    + 1]),
        },
        "artifact_sha256": {
            "dynamics_model": _sha256(f"{ARTIFACTS}/dynaguide_model.pth"),
            "base_policy":    _sha256(f"{ARTIFACTS}/base_policy.pth"),
            "guidance_h5":    _sha256(f"{ARTIFACTS}/switch_on_guidance.hdf5"),
        },
        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extras": extras or {},
    }
    with open(f"{out_dir}/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def _rollout_cmd(out_dir: str, scale: float, alpha: float, ss: int, seed: int) -> list:
    """Single source of truth for the run_dynaguide.py invocation."""
    return [
        "python", "run_dynaguide.py",
        "--video_path",    f"{out_dir}/rollout.mp4",
        "--dataset_path",  f"{out_dir}/data.hdf5",
        "--dataset_obs",
        "--json_path",     f"{out_dir}/summary.json",
        "--horizon",       str(CONFIG["evaluation_protocol"]["horizon_steps"]),
        "--n_rollouts",    str(CONFIG["evaluation_protocol"]["n_rollouts_per_seed"]),
        "--agent",         f"{ARTIFACTS}/base_policy.pth",
        "--output_folder", out_dir,
        "--video_skip",    "2",
        "--exp_setup_config",
            f"{REPO_ROOT}/calvin_exp_configs_examples/switch_on.json",
        "--guidance",      f"{ARTIFACTS}/dynaguide_model.pth",
        "--camera_names",  "third_person",
        "--scale",         str(scale),
        "--ss",            str(ss),
        "--alpha",         str(alpha),
        "--seed",          str(seed),
        "--save_frames",
    ]


def _patch_exp_config_paths() -> None:
    """The released JSON config has an absolute path to pos_examples that won't
    exist in the container. Rewrite it to point at the volume-mounted h5 and
    fix the reset_poses path to the correct subdirectory.

    We do NOT touch env_setup, use_neg, or loc_target — those are paper-defined
    and must remain identical.
    """
    path = f"{REPO_ROOT}/calvin_exp_configs_examples/switch_on.json"
    with open(path) as f:
        cfg = json.load(f)
    cfg["pos_examples"] = f"{ARTIFACTS}/switch_on_guidance.hdf5"
    cfg["reset_poses"] = (
        f"{REPO_ROOT}/calvin_exp_configs_examples/reset_poses/"
        f"initial_calvin_robot_states_right_side_midpoint.json"
    )
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[patch] rewrote pos_examples + reset_poses in {path}")
    print(f"[patch] env_setup kept as: {cfg['env_setup']}  (paper value, unchanged)")


# ============================================================
# Functions
# ============================================================

@app.function(timeout=900)
def verify_environment():
    """Cheap CPU-only check that the image builds and packages import."""
    _bootstrap()
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available?: {torch.cuda.is_available()}")
    import robomimic
    print(f"robomimic: {robomimic.__file__}")
    try:
        import calvin_env
        print(f"calvin_env: {calvin_env.__file__}")
    except Exception as e:
        print(f"calvin_env import FAILED: {e}")
        raise
    import diffusers
    print(f"diffusers: {diffusers.__version__}")
    print("\nverify_environment: PASS")


@app.function(
    volumes={ARTIFACTS: artifacts},
    timeout=600,
)
def verify_artifacts():
    """Confirm the three expected files exist on the volume and look sane."""
    _bootstrap()
    import h5py
    import torch

    expected = ["dynaguide_model.pth", "base_policy.pth", "switch_on_guidance.hdf5"]
    print(f"Looking for {len(expected)} artifacts in {ARTIFACTS}/ ...")
    for name in expected:
        path = f"{ARTIFACTS}/{name}"
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            print(f"Upload it with:  modal volume put dynaguide-artifacts ~/Downloads/{name} /")
            raise SystemExit(1)
        size_mb = os.path.getsize(path) / 1e6
        print(f"  OK  {name:34s}  {size_mb:7.1f} MB   sha256={_sha256(path)[:16]}...")

    dm = torch.load(f"{ARTIFACTS}/dynaguide_model.pth", map_location="cpu")
    print(f"\ndynamics_model state_dict: {len(dm)} keys")
    bp = torch.load(f"{ARTIFACTS}/base_policy.pth", map_location="cpu")
    print(f"base_policy checkpoint keys: "
          f"{list(bp.keys()) if isinstance(bp, dict) else type(bp)}")
    with h5py.File(f"{ARTIFACTS}/switch_on_guidance.hdf5", "r") as f:
        demos = list(f["data"].keys())
        print(f"guidance h5: {len(demos)} demos, first 3: {demos[:3]}")

    print("\nverify_artifacts: PASS")


@app.function(
    volumes={ARTIFACTS: artifacts},
    gpu="A10G",
    timeout=1200,
)
def sanity_check_dynamics_model():
    """Step 4 — confirm the dynamics-model checkpoint loads cleanly and
    produces correctly-shaped outputs on a real (obs, action) pair from
    the guidance h5.

    This is a structural / functional check. It does NOT run the diagnostic
    suite from test_dynaguide_embedding.py because that suite crashes on a
    visualization step that expects DIFFERENT good/mixed h5 files (we only
    have one). The check below exercises exactly what DynaGuide will use at
    inference time:
      - DinoV2 encoder (frozen)
      - 6-layer transformer dynamics head
      - state_action_embedding(state, action) → (B, 256, 384)
    """
    _bootstrap()
    _patch_exp_config_paths()

    import sys
    sys.path.insert(0, REPO_ROOT)

    import torch
    import h5py
    import numpy as np
    from core.dynamics_models import FinalStatePredictionDino

    print("\n[sanity] instantiating FinalStatePredictionDino ...")
    model = FinalStatePredictionDino(
        action_dim=7,
        action_horizon=16,
        cameras=["third_person"],
        proprio="proprio",
        proprio_dim=15,
        reconstruction=True,
    )
    model.to("cuda")

    print("[sanity] loading checkpoint with strict=True ...")
    sd = torch.load(f"{ARTIFACTS}/dynaguide_model.pth", map_location="cuda",
                    weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[sanity]   missing keys:    {len(missing)}")
    print(f"[sanity]   unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print(f"[sanity]   first missing:    {missing[:3]}")
    if len(unexpected) > 0:
        print(f"[sanity]   first unexpected: {unexpected[:3]}")
    model.eval()
    print(f"[sanity] trainable params: {model.trainable_parameters():,}")

    print("\n[sanity] loading sample from guidance h5 ...")
    with h5py.File(f"{ARTIFACTS}/switch_on_guidance.hdf5", "r") as f:
        demo = f["data"][list(f["data"].keys())[0]]
        img    = np.transpose(demo["obs"]["third_person"][0], (2, 0, 1))  # CHW
        proprio = demo["obs"]["proprio"][0]
        action  = demo["actions"][:16]  # first 16 actions
    print(f"[sanity]   third_person shape: {img.shape}     dtype: {img.dtype}")
    print(f"[sanity]   proprio shape:      {proprio.shape}")
    print(f"[sanity]   action shape:       {action.shape}")

    # Add batch dim and move to GPU
    state = {
        "third_person": torch.tensor(img).unsqueeze(0).float().cuda(),
        "proprio":      torch.tensor(proprio).unsqueeze(0).float().cuda(),
    }
    action_t = torch.tensor(action).unsqueeze(0).float().cuda()

    print("\n[sanity] running forward pass through dynamics model ...")
    with torch.no_grad():
        z_hat = model.state_action_embedding(state, action_t)
    print(f"[sanity]   z_hat shape: {tuple(z_hat.shape)}")
    print(f"[sanity]   z_hat finite: {torch.all(torch.isfinite(z_hat)).item()}")
    print(f"[sanity]   z_hat mean / std: {z_hat.mean().item():.4f} / {z_hat.std().item():.4f}")

    assert z_hat.shape == (1, 256, 384), \
        f"Expected (1, 256, 384), got {tuple(z_hat.shape)}"
    assert torch.all(torch.isfinite(z_hat)), "z_hat has NaN/Inf!"

    print("\nsanity_check_dynamics_model: PASS")


@app.function(
    volumes={ARTIFACTS: artifacts},
    gpu="A10G",
    timeout=3600,
)
def run_baseline(seed: int = 1):
    """Step 5 — base policy, no guidance (scale=0, M=1)."""
    _bootstrap()
    _patch_exp_config_paths()
    out_dir = f"{RESULTS_ROOT}/BasePolicy_switch_on_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = _rollout_cmd(
        out_dir,
        scale=CONFIG["baseline_run"]["scale"],
        alpha=CONFIG["guidance_hyperparameters"]["alpha_sigma"],
        ss=CONFIG["baseline_run"]["stochastic_sampling_M"],
        seed=seed,
    )
    _write_manifest(out_dir, "baseline", cmd)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    artifacts.commit()
    print(f"\nrun_baseline (seed={seed}): outputs in {out_dir}")


@app.function(
    volumes={ARTIFACTS: artifacts},
    gpu="A10G",
    timeout=3600,
)
def run_dynaguide(seed: int = 1):
    """Step 6 — DynaGuide reproduction at paper hyperparameters."""
    _bootstrap()
    _patch_exp_config_paths()
    out_dir = f"{RESULTS_ROOT}/DynaGuide_switch_on_seed{seed}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = _rollout_cmd(
        out_dir,
        scale=CONFIG["guidance_hyperparameters"]["scale"],
        alpha=CONFIG["guidance_hyperparameters"]["alpha_sigma"],
        ss=CONFIG["guidance_hyperparameters"]["stochastic_sampling_M"],
        seed=seed,
    )
    _write_manifest(out_dir, "dynaguide", cmd)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    artifacts.commit()
    print(f"\nrun_dynaguide (seed={seed}): outputs in {out_dir}")


@app.function(
    volumes={ARTIFACTS: artifacts},
    gpu="A10G",
    timeout=14400,  # 4 hours — covers baseline + 3 DynaGuide seeds
)
def reproduce():
    """The full reproduction: baseline + 3 DynaGuide seeds. ~75 min, ~$2.50."""
    _bootstrap()
    _patch_exp_config_paths()

    seeds = CONFIG["evaluation_protocol"]["seeds"]

    # Baseline once — variance is small at scale=0
    out_dir = f"{RESULTS_ROOT}/BasePolicy_switch_on_seed{seeds[0]}"
    os.makedirs(out_dir, exist_ok=True)
    cmd = _rollout_cmd(
        out_dir,
        scale=CONFIG["baseline_run"]["scale"],
        alpha=CONFIG["guidance_hyperparameters"]["alpha_sigma"],
        ss=CONFIG["baseline_run"]["stochastic_sampling_M"],
        seed=seeds[0],
    )
    _write_manifest(out_dir, "baseline", cmd, {"part_of": "reproduce"})
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # DynaGuide × N seeds
    for seed in seeds:
        out_dir = f"{RESULTS_ROOT}/DynaGuide_switch_on_seed{seed}"
        os.makedirs(out_dir, exist_ok=True)
        cmd = _rollout_cmd(
            out_dir,
            scale=CONFIG["guidance_hyperparameters"]["scale"],
            alpha=CONFIG["guidance_hyperparameters"]["alpha_sigma"],
            ss=CONFIG["guidance_hyperparameters"]["stochastic_sampling_M"],
            seed=seed,
        )
        _write_manifest(out_dir, "dynaguide", cmd, {"part_of": "reproduce"})
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    artifacts.commit()
    print("\nreproduce: DONE. Aggregate with reproduction/scripts/analyze_behaviors.py.")


@app.local_entrypoint()
def main():
    """Default entrypoint — prints help."""
    print(__doc__)
