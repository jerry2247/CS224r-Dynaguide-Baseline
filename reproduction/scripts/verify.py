"""
Pre-flight checks for the reproduction. Runs on your local Mac.

Confirms:
    - reproduction/configs/switch_on_paper.json parses and has expected fields
    - reproduction/modal_app.py imports without errors (and `modal` is installed)
    - DynaGuide source files we expect to call are present
    - reproduction/artifacts/ files match the SHA-256 hashes in REGISTRY.json
    - .gitignore covers reproduction/results/

Does NOT touch GPUs or run rollouts. Cheap. Run this before `modal run`.

Usage:
    python reproduction/scripts/verify.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    print(f"Repo root: {REPO_ROOT}\n")
    all_ok = True

    # 1. config
    print("Config")
    cfg_path = REPO_ROOT / "reproduction/configs/switch_on_paper.json"
    if check("config file present", cfg_path.exists(), str(cfg_path)):
        with open(cfg_path) as f:
            cfg = json.load(f)
        all_ok &= check(
            "task == switch_on",
            cfg["task"]["name"] == "switch_on",
            cfg["task"]["name"],
        )
        all_ok &= check(
            "scale == 1.5",
            cfg["guidance_hyperparameters"]["scale"] == 1.5,
            str(cfg["guidance_hyperparameters"]["scale"]),
        )
        all_ok &= check(
            "alpha (sigma) == 30",
            cfg["guidance_hyperparameters"]["alpha_sigma"] == 30,
            str(cfg["guidance_hyperparameters"]["alpha_sigma"]),
        )
        all_ok &= check(
            "stochastic_sampling_M == 4",
            cfg["guidance_hyperparameters"]["stochastic_sampling_M"] == 4,
            str(cfg["guidance_hyperparameters"]["stochastic_sampling_M"]),
        )
        all_ok &= check(
            "evaluation has >=3 seeds",
            len(cfg["evaluation_protocol"]["seeds"]) >= 3,
            f"{cfg['evaluation_protocol']['seeds']}",
        )
        all_ok &= check(
            "paper_targets.dynaguide_band defined",
            "switch_on_dynaguide_band_lower" in cfg["paper_targets"],
        )
    else:
        all_ok = False
    print()

    # 2. vendored files we will CALL
    print("Vendored DynaGuide files (untouched)")
    must_exist = [
        "run_dynaguide.py",
        "test_dynaguide_embedding.py",
        "core/dynamics_models.py",
        "core/dynaguide.py",
        "core/calvin_utils.py",
        "calvin_exp_configs_examples/switch_on.json",
        "robomimic/robomimic/algo/diffusion_policy.py",
        "calvin_exp_configs_examples/reset_poses/initial_calvin_robot_states_right_side_midpoint.json",
    ]
    for rel in must_exist:
        p = REPO_ROOT / rel
        all_ok &= check(rel, p.exists())
    print()

    # 3. modal app
    print("Modal app")
    modal_installed = importlib.util.find_spec("modal") is not None
    all_ok &= check(
        "modal package importable",
        modal_installed,
        "" if modal_installed else "pip install modal",
    )
    modal_app_path = REPO_ROOT / "reproduction/modal_app.py"
    all_ok &= check("modal_app.py present", modal_app_path.exists())
    if modal_installed and modal_app_path.exists():
        # Compile-check (no execution).
        import py_compile
        try:
            py_compile.compile(str(modal_app_path), doraise=True)
            all_ok &= check("modal_app.py compiles", True)
        except py_compile.PyCompileError as e:
            all_ok &= check("modal_app.py compiles", False, str(e)[:80])
    print()

    # 4. artifacts registry
    print("Artifacts (reproduction/artifacts/)")
    reg_path = REPO_ROOT / "reproduction/artifacts/REGISTRY.json"
    if check("REGISTRY.json present", reg_path.exists()):
        with open(reg_path) as f:
            reg = json.load(f)
        for canonical, meta in reg["expected_artifacts"].items():
            artifact_path = REPO_ROOT / "reproduction/artifacts" / canonical
            if not artifact_path.exists():
                expected_status = meta.get("status", "")
                if expected_status == "MISSING":
                    print(f"  [WARN] {canonical:34s}  not on disk yet "
                          f"(registry: {expected_status})")
                    all_ok = False
                else:
                    all_ok &= check(canonical + " on disk", False)
                continue
            # File is present — check hash
            expected_hash = meta.get("sha256")
            if expected_hash is None:
                print(f"  [WARN] {canonical:34s}  on disk but registry has no sha256")
                all_ok = False
                continue
            actual = sha256(artifact_path)
            match = actual == expected_hash
            all_ok &= check(
                f"{canonical} sha256 matches registry",
                match,
                f"{actual[:16]}..." if match else f"got {actual[:16]}...",
            )
    else:
        all_ok = False
    print()

    # 5. .gitignore covers results dir
    print("Git hygiene")
    gi = REPO_ROOT / ".gitignore"
    if gi.exists():
        text = gi.read_text()
        all_ok &= check(
            "reproduction/results/ is gitignored",
            "reproduction/results" in text or "results/" in text,
        )
    else:
        all_ok &= check(".gitignore present", False)
    print()

    # 5. instructions
    print("Outcome")
    if all_ok:
        print("  ALL CHECKS PASSED.")
        print("  Next:  modal run reproduction/modal_app.py::verify_environment")
        return 0
    else:
        print("  Some checks failed — fix the FAILs above before running on Modal.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
