#!/usr/bin/env python3
import argparse
import importlib
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prob_file")
    ap.add_argument("cfg_file")
    args = ap.parse_args()

    sys.path.insert(0, "deps/agile-payload-transport/build")
    nmpc_mod = importlib.import_module("nmpc_controller_py")
    controller = nmpc_mod.Controller(args.prob_file, args.cfg_file)
    controller.run()
    controller.maybe_visualize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
