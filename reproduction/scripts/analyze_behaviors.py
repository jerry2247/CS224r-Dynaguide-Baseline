"""
Post-hoc behavior analysis for the switch_on reproduction.

The Success_Rate in run_dynaguide.py's summary.json reports CALVIN's generic
env.is_success() which does NOT detect specific behaviors. The proper way to
score switch_on success is to inspect the per-step `obs/states` array in
data.hdf5 and check whether the switch crossed the ON threshold.

This script adapts the detection logic from analyze_calvin_touch.py to compute:
  - For each rollout: which behavior (if any) occurred
  - For each run dir: the breakdown of behaviors across 100 rollouts
  - Aggregate switch_on success rate per condition + mean/SE across seeds

Usage:
    python reproduction/scripts/analyze_behaviors.py
"""
import h5py
import json
import math
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "reproduction" / "results"
CONFIG_PATH = REPO_ROOT / "reproduction" / "configs" / "switch_on_paper.json"

# State indices per analyze_calvin_touch.py:segment_states
IDX_SLIDING_DOOR = 0
IDX_DRAWER       = 1
IDX_BUTTON       = 2
IDX_SWITCH       = 3
IDX_GREEN_LIGHT  = 5
IDX_RED_BLOCK    = slice(6, 9)
IDX_BLUE_BLOCK   = slice(12, 15)
IDX_PINK_BLOCK   = slice(18, 21)


def classify_demo(states: np.ndarray, proprios: np.ndarray) -> str:
    """Mirror analyze_calvin_touch.py:generate_detailed_behavior_distribution.
    Returns the first detected behavior, or 'no_behavior' if none triggered.

    States are the privileged 24-d simulator state per step.
    Proprios are 15-d (xyz at [0:3]).
    """
    first = states[0]
    for step in range(states.shape[0]):
        cur = states[step]
        robot_xyz = proprios[step, 0:3]
        delta = cur - first

        # Block touches (require proximity AND displacement)
        if (np.linalg.norm(robot_xyz - cur[IDX_RED_BLOCK]) < 0.1
                and np.linalg.norm(delta[IDX_RED_BLOCK]) > 0.001):
            return "red_displace"
        if (np.linalg.norm(robot_xyz - cur[IDX_PINK_BLOCK]) < 0.1
                and np.linalg.norm(delta[IDX_PINK_BLOCK]) > 0.001):
            return "pink_displace"
        if (np.linalg.norm(robot_xyz - cur[IDX_BLUE_BLOCK]) < 0.1
                and np.linalg.norm(delta[IDX_BLUE_BLOCK]) > 0.001):
            return "blue_displace"

        # Articulated parts: pick the first one over its threshold
        if abs(delta[IDX_SLIDING_DOOR]) > 0.05:
            return "door_left" if cur[IDX_SLIDING_DOOR] > first[IDX_SLIDING_DOOR] else "door_right"
        if abs(delta[IDX_DRAWER]) > 0.05:
            return "drawer_open" if cur[IDX_DRAWER] > first[IDX_DRAWER] else "drawer_close"
        if abs(delta[IDX_SWITCH]) > 0.02:
            return "switch_on"  if cur[IDX_SWITCH]  > first[IDX_SWITCH]  else "switch_off"
        if abs(delta[IDX_GREEN_LIGHT]) > 0.01:
            return "button_on"  if cur[IDX_GREEN_LIGHT] > first[IDX_GREEN_LIGHT] else "button_off"

    return "no_behavior"


def analyze_run(h5_path: Path) -> dict:
    """Read a rollout dataset, classify each demo, return a behavior histogram."""
    counts = {}
    n_demos = 0
    avg_horizon = 0
    with h5py.File(h5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for demo in demos:
            states  = f["data"][demo]["obs"]["states"][:]
            proprio = f["data"][demo]["obs"]["proprio"][:]
            if states.ndim == 3:    # frame-stack squashed
                states = states[:, -1]
            if proprio.ndim == 3:
                proprio = proprio[:, -1]
            n_demos += 1
            avg_horizon += states.shape[0]
            label = classify_demo(states, proprio)
            counts[label] = counts.get(label, 0) + 1
    return {
        "n_rollouts": n_demos,
        "avg_horizon": avg_horizon / max(n_demos, 1),
        "counts": counts,
    }


def main():
    cfg = json.loads(CONFIG_PATH.read_text())
    paper = cfg["paper_targets"]

    print(f"{'Run':40s}  {'switch_on':>10s}  {'horizon':>8s}")
    print("-" * 64)

    by_kind: dict[str, list[float]] = {"baseline": [], "dynaguide": []}
    all_results = {}

    for run_dir in sorted(RESULTS.glob("*_switch_on_seed*")):
        h5 = run_dir / "data.hdf5"
        if not h5.exists():
            continue
        result = analyze_run(h5)
        all_results[run_dir.name] = result

        success_count = result["counts"].get("switch_on", 0)
        n = result["n_rollouts"]
        rate = success_count / max(n, 1)

        kind = "baseline" if run_dir.name.startswith("BasePolicy") else "dynaguide"
        by_kind[kind].append(rate)

        print(f"{run_dir.name:40s}  {success_count:3d}/{n:<3d} = "
              f"{rate*100:5.1f}%   {result['avg_horizon']:6.1f}")

    print("-" * 64)

    # Aggregate
    def aggregate(rates):
        """Mean and standard error across seeds.

        Returns (mean, se). `se` is None when n_seeds < 2 (no variance estimate
        possible). Using None rather than NaN keeps the emitted JSON valid.
        """
        if not rates:
            return None, None
        mean = sum(rates) / len(rates)
        if len(rates) >= 2:
            var = sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)
            se = math.sqrt(var) / math.sqrt(len(rates))
        else:
            se = None
        return mean, se

    base_mean, base_se = aggregate(by_kind["baseline"])
    dyna_mean, dyna_se = aggregate(by_kind["dynaguide"])

    print()
    print("=" * 64)
    print("AGGREGATE")
    print("=" * 64)
    if base_mean is not None:
        se_str = f" ± {base_se*100:.1f}%" if base_se is not None else ""
        print(f"  Baseline   switch_on:  {base_mean*100:5.1f}%{se_str} (n_seeds={len(by_kind['baseline'])})")
    if dyna_mean is not None:
        se_str = f" ± {dyna_se*100:.1f}%" if dyna_se is not None else ""
        print(f"  DynaGuide  switch_on:  {dyna_mean*100:5.1f}%{se_str}  (n_seeds={len(by_kind['dynaguide'])})")
    if base_mean and dyna_mean:
        print(f"  Boost factor:           {dyna_mean/max(base_mean, 1e-9):.1f}x")

    print()
    print(f"  Paper target (DynaGuide): {paper['switch_on_dynaguide_mean']*100:.1f}% "
          f"± {paper['switch_on_dynaguide_se']*100:.1f}%")
    print(f"  Paper target (baseline):  ~{paper['switch_on_base_policy_approx']*100:.0f}%")
    print(f"  Paper pass band:          "
          f"[{paper['switch_on_dynaguide_band_lower']*100:.0f}%, "
          f"{paper['switch_on_dynaguide_band_upper']*100:.0f}%]")

    # Print full behavior breakdown for one DynaGuide run (informational)
    print()
    print("=" * 64)
    print("BEHAVIOR BREAKDOWN per run")
    print("=" * 64)
    for name, r in all_results.items():
        print(f"\n  {name}:")
        for b, c in sorted(r["counts"].items(), key=lambda x: -x[1]):
            print(f"    {b:18s}  {c:3d}  ({c/r['n_rollouts']*100:5.1f}%)")

    # Save machine-readable
    out = {
        "per_run": all_results,
        "aggregate": {
            "baseline":  {"mean": base_mean, "se": base_se,
                          "rates": by_kind["baseline"]},
            "dynaguide": {"mean": dyna_mean, "se": dyna_se,
                          "rates": by_kind["dynaguide"]},
            "boost_factor": (dyna_mean / max(base_mean, 1e-9)) if base_mean and dyna_mean else None,
        },
        "paper_targets": paper,
    }
    out_path = RESULTS / "behavior_report.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nFull report → {out_path}")


if __name__ == "__main__":
    main()
