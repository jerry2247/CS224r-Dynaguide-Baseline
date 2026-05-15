from __future__ import annotations

import json
import site
import subprocess
import sys
from pathlib import Path

import modal


APP_NAME = "dynaguide-self-distillation"
VOLUME_NAME = "dynaguide-self-distillation"

LOCAL_REPO = Path(__file__).resolve().parents[1]
LOCAL_PROJECT = LOCAL_REPO / "dynaguide_self_distillation"

REMOTE_REPO = "/workspace"
REMOTE_PROJECT = f"{REMOTE_REPO}/dynaguide_self_distillation"
ARTIFACTS = "/artifacts"
TRACE_GPU = "A100-80GB"
TRACE_SEEDS = (1, 2, 3)
ROLLOUTS_PER_SEED = 50
ROLLOUTS_PER_SHARD = 10
SHARD_TIMEOUT = 7200
TRACE_SHARDS = tuple(
    (seed, start, min(ROLLOUTS_PER_SHARD, ROLLOUTS_PER_SEED - start))
    for seed in TRACE_SEEDS
    for start in range(0, ROLLOUTS_PER_SEED, ROLLOUTS_PER_SHARD)
)

TRACE_CONFIG = f"{REMOTE_PROJECT}/configs/switch_on_trace.json"
SMOKE_CONFIG = f"{REMOTE_PROJECT}/configs/switch_on_trace_smoke.json"

PYTHONPATH = (
    f"{REMOTE_PROJECT}/src",
    REMOTE_REPO,
    f"{REMOTE_REPO}/robomimic",
    f"{REMOTE_REPO}/calvin/calvin_env",
    f"{REMOTE_REPO}/calvin/calvin_env/tacto",
)


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "bash",
        "build-essential",
        "cmake",
        "ffmpeg",
        "git",
        "libegl1",
        "libgl1",
        "libgl1-mesa-dev",
        "libglew-dev",
        "libglfw3",
        "libglfw3-dev",
        "libosmesa6-dev",
        "patchelf",
        "wget",
    )
    .uv_pip_install("numpy==1.26.4", "pandas==2.0.3")
    .uv_pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "diffusers==0.24.0",
        "huggingface_hub==0.19.4",
        "einops",
        "tqdm",
        "h5py",
        "imageio[ffmpeg]",
        "opencv-python-headless",
        "tensorboard",
        "tensorboardX",
        "psutil",
        "termcolor",
        "scikit-learn==1.4.2",
        "pybullet",
        "pyglet",
        "hydra-core==1.1.1",
        "omegaconf",
        "gym",
    )
    .env(
        {
            "EGL_PLATFORM": "surfaceless",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_GL": "egl",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": ":".join(PYTHONPATH),
        }
    )
    .add_local_dir(
        str(LOCAL_PROJECT),
        remote_path=REMOTE_PROJECT,
        ignore=["**/__pycache__/**", "**/*.pyc", "outputs/**", "runs/**"],
    )
    .add_local_dir(
        str(LOCAL_REPO / "core"),
        remote_path=f"{REMOTE_REPO}/core",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        str(LOCAL_REPO / "robomimic"),
        remote_path=f"{REMOTE_REPO}/robomimic",
        ignore=["**/__pycache__/**", "**/*.pyc", "docs/**", "tests/**"],
    )
    .add_local_dir(
        str(LOCAL_REPO / "calvin"),
        remote_path=f"{REMOTE_REPO}/calvin",
        ignore=["**/__pycache__/**", "**/*.pyc", ".git/**"],
    )
    .add_local_dir(
        str(LOCAL_REPO / "calvin_exp_configs_examples" / "reset_poses"),
        remote_path=f"{REMOTE_REPO}/calvin_exp_configs_examples/reset_poses",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run(cmd: list[str], *, cwd: str | None = None, check: bool = True) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=check)


def _bootstrap() -> None:
    _run(["pip", "install", "-q", "-e", f"{REMOTE_REPO}/robomimic"])
    rc = subprocess.run(["bash", "install.sh"], cwd=f"{REMOTE_REPO}/calvin", check=False).returncode
    if rc != 0:
        _run(["pip", "install", "-q", "-e", f"{REMOTE_REPO}/calvin/calvin_env"], check=False)

    subprocess.run(["pip", "uninstall", "-y", "-q", "diffusers", "huggingface_hub"], check=False)
    _run(
        [
            "pip",
            "install",
            "-q",
            "--upgrade",
            "--upgrade-strategy",
            "only-if-needed",
            "numpy==1.26.4",
            "pandas==2.0.3",
            "diffusers==0.24.0",
            "huggingface_hub==0.19.4",
        ]
    )
    _run(["pip", "install", "-q", "-e", REMOTE_PROJECT])

    site.addsitedir("/usr/local/lib/python3.10/site-packages")
    for path in reversed(PYTHONPATH):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _collect(
    config_path: str,
    *,
    seed: int | None = None,
    rollout_start: int = 0,
    n_rollouts: int | None = None,
    output_name: str | None = None,
) -> dict:
    _bootstrap()
    from dynaguide_self_distillation.collect_traces import collect_traces

    result = collect_traces(
        config_path,
        repo_root=REMOTE_REPO,
        artifact_root=ARTIFACTS,
        seed=seed,
        rollout_start=rollout_start,
        n_rollouts=n_rollouts,
        output_name=output_name,
    )
    volume.commit()
    print(json.dumps(result, indent=2))
    return result


@app.function(volumes={ARTIFACTS: volume}, gpu=TRACE_GPU, timeout=SHARD_TIMEOUT)
def collect_switch_on_shard_worker(seed: int, rollout_start: int, n_rollouts: int) -> dict:
    rollout_stop = rollout_start + n_rollouts
    output_name = f"seed_{seed}_rollouts_{rollout_start:02d}_{rollout_stop - 1:02d}"
    return _collect(
        TRACE_CONFIG,
        seed=seed,
        rollout_start=rollout_start,
        n_rollouts=n_rollouts,
        output_name=output_name,
    )


@app.local_entrypoint()
def collect_switch_on_traces() -> None:
    handles = [collect_switch_on_shard_worker.spawn(seed, start, count) for seed, start, count in TRACE_SHARDS]
    print("Submitted switch_on trace shards:")
    for (seed, start, count), handle in zip(TRACE_SHARDS, handles, strict=True):
        stop = start + count
        print(f"  seed={seed}, rollouts=[{start}, {stop}): {handle.object_id}")
    print("Each shard writes /artifacts/traces/switch_on_teacher/seed_<seed>_rollouts_<start>_<end>.hdf5")


@app.local_entrypoint()
def collect_switch_on_shard(seed: int, rollout_start: int, n_rollouts: int = ROLLOUTS_PER_SHARD) -> None:
    result = collect_switch_on_shard_worker.remote(seed, rollout_start, n_rollouts)
    print(json.dumps(result, indent=2))


@app.function(volumes={ARTIFACTS: volume}, gpu=TRACE_GPU, timeout=7200)
def collect_switch_on_smoke() -> dict:
    return _collect(SMOKE_CONFIG)
