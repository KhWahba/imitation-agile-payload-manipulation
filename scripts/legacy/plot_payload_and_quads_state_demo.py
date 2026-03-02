import yaml
import numpy as np
import argparse
from utils import plot_payload_and_quads_state, plot_all_state

def main():
    parser = argparse.ArgumentParser(description="Plot payload and quads state from YAML file.")
    parser.add_argument("--yaml", type=str, required=True, help="Path to input YAML file")
    args = parser.parse_args()

    # Load YAML
    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    # Get start state and number of bodies
    joint_robot = cfg["joint_robot"][0]
    start_state = np.array(joint_robot["start"], dtype=np.float64)
    n_bodies = joint_robot["quadsNum"] + 1  # payload + quads

    # Original plot for comparison
    plot_payload_and_quads_state(
        state=start_state,
        n_bodies=n_bodies,
        title="Payload + Quads State",
        save_path="payload_quads_state.pdf",
        show=False,
    )

    # Enhanced plots (3D and 2D with velocities)
    plot_all_state(
        state=start_state,
        n_bodies=n_bodies,
        title="Payload + Quads State (Enhanced)",
        save_path_prefix="payload_quads_state",
        show=True,
    )

if __name__ == "__main__":
    main()
