import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def _to_np(value: Any):
    """Convert YAML-loaded value into a NumPy array if it looks numeric/list-like; else keep as-is."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        try:
            arr = np.asarray(value)
            # Only convert if it's numeric-ish or at least array-like
            if arr.dtype != object:
                return arr
        except Exception:
            pass
    return value


def yaml_dict_to_npz(yaml_path: str, npz_path: str) -> None:
    yaml_path = str(yaml_path)
    npz_path = str(npz_path)

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a dict, got {type(data)}")

    out: Dict[str, Any] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError(f"NPZ keys must be strings. Got key {k} of type {type(k)}")
        out[k] = _to_np(v)

    # np.savez can store arrays + picklable python objects (object arrays)
    np.savez(npz_path, **out)


def main():
    p = argparse.ArgumentParser(description="Convert a YAML dict into an NPZ file.")
    p.add_argument("--yaml", required=True, help="Input YAML file path")
    p.add_argument("--out", required=True, help="Output NPZ file path")
    args = p.parse_args()

    yaml_dict_to_npz(args.yaml, args.out)
    print(f"Wrote NPZ: {args.out}")


if __name__ == "__main__":
    main()
