#!/usr/bin/env python3
import argparse
import copy
import importlib
import sys
from pathlib import Path

import yaml


DEFAULT_MODES = [
    "track_goal",
    "track_reference_nmpc_standard",
    "track_reference_nmpc_refwarm",
    "track_reference_policy",
    "track_linear_hover",
]


def _resolve_existing(raw_value: str, prob_dir: Path, agile_root: Path) -> str:
    p = Path(raw_value)
    if p.is_absolute() and p.exists():
        return str(p)
    candidates = [
        (prob_dir / p).resolve(),
        (agile_root / p).resolve(),
        (agile_root / "build" / p).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return str(candidates[0])


def _load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dump_yaml(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="NMPC benchmark via Python bindings")
    parser.add_argument("--prob", required=True, help="Base problem yaml")
    parser.add_argument("--cfg", required=True, help="Base optimization yaml")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument(
        "--module-dir",
        default="deps/agile-payload-transport/build",
        help="Directory containing nmpc_controller_py module",
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma separated nmpc modes",
    )
    parser.add_argument(
        "--policy-onnx",
        default="",
        help="Required when mode includes track_reference_policy",
    )
    args = parser.parse_args()

    prob_path = Path(args.prob).resolve()
    cfg_path = Path(args.cfg).resolve()
    out_dir = Path(args.out_dir).resolve()
    if "runs" not in out_dir.parts:
        raise ValueError(f"--out-dir must be inside a 'runs' directory, got: {out_dir}")
    module_dir = Path(args.module_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(module_dir))
    nmpc_mod = importlib.import_module("nmpc_controller_py")

    base_prob = _load_yaml(prob_path)
    base_cfg = _load_yaml(cfg_path)
    prob_dir = prob_path.parent
    agile_root = prob_path.parents[2]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        print(f"[nmpc-bindings] mode={mode}")
        mode_prob = copy.deepcopy(base_prob)
        mode_cfg = copy.deepcopy(base_cfg)

        result_yaml = out_dir / f"{mode}_result.yaml"
        timing_json = out_dir / f"{mode}_result_timing.json"
        mode_prob["results_path"] = str(result_yaml)
        mode_prob["visualize"] = False
        mode_cfg["nmpc_mode"] = mode
        for key in ("env_file", "init_file", "ref_file"):
            if mode_prob.get(key):
                resolved = Path(_resolve_existing(mode_prob[key], prob_dir, agile_root))
                mode_prob[key] = str(resolved) if resolved.exists() else ""
        if mode_prob.get("models_dir"):
            resolved_models = Path(_resolve_existing(mode_prob["models_dir"], prob_dir, agile_root))
            mode_prob["models_dir"] = (
                str(resolved_models) if resolved_models.exists() else str((agile_root / "models").resolve())
            )
        else:
            mode_prob["models_dir"] = str((agile_root / "models").resolve())

        if mode in {"track_reference_nmpc_standard", "track_reference_nmpc_refwarm"} and not mode_prob.get("ref_file"):
            print(f"[nmpc-bindings] skip {mode} (missing ref_file)")
            continue

        if mode == "track_reference_policy":
            if not args.policy_onnx:
                print("[nmpc-bindings] skip track_reference_policy (missing --policy-onnx)")
                continue
            policy_onnx = Path(args.policy_onnx).resolve()
            if not policy_onnx.exists():
                raise FileNotFoundError(f"--policy-onnx not found: {policy_onnx}")
            mode_cfg["use_policy_onnx"] = True
            mode_cfg["policy_onnx_path"] = str(policy_onnx)

        mode_prob_file = out_dir / f"{mode}_prob_cfg.yaml"
        mode_cfg_file = out_dir / f"{mode}_opt_cfg.yaml"
        _dump_yaml(mode_prob_file, mode_prob)
        _dump_yaml(mode_cfg_file, mode_cfg)

        controller = nmpc_mod.Controller(str(mode_prob_file), str(mode_cfg_file))
        controller.run(
            mode=mode,
            out_yaml=str(result_yaml),
            out_timing_json=str(timing_json),
            visualize=False,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
