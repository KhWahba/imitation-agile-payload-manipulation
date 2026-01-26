import yaml
import numpy as np
import mujoco



def get_obs_from_qpos_qvel(data, n_bodies: int, quat_out: str = "xyzw") -> np.ndarray:
    """
    Returns state = [pose_all (7*n), vel_all (6*n)] where
      pose_i = [pos3, quat4] for i=0..n-1  (quat order selectable)
      vel_i  = [lin3, ang3] for i=0..n-1

    Assumes qpos layout is free joints: [pos3, quat(wxyz)] repeated.
    Assumes qvel layout is free joints: [lin3, ang3] repeated.
    """
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)

    poses_wxyz = qpos[:7 * n_bodies].copy()   # (7n,)
    vels       = qvel[:6 * n_bodies].copy()   # (6n,)

    if quat_out == "wxyz":
        poses = poses_wxyz
    elif quat_out == "xyzw":
        poses = poses_wxyz.copy()
        for i in range(n_bodies):
            p0 = 7 * i
            qw, qx, qy, qz = poses[p0 + 3 : p0 + 7]
            poses[p0 + 3 : p0 + 7] = [qx, qy, qz, qw]
    else:
        raise ValueError("quat_out must be 'wxyz' or 'xyzw'")

    state = np.concatenate([poses, vels], axis=0).astype(np.float32)
    return state


def set_mujoco_state_from_joint_robot_start(model, data, input_yaml):
    with open(input_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    start = np.asarray(cfg["joint_robot"][0]["start"], dtype=np.float64)

    assert start.size % 13 == 0, f"start length {start.size} not divisible by 13"
    n = start.size // 13

    poses = start[:7 * n]    # shape (7n,)
    vels  = start[7 * n:]    # shape (6n,)

  
    qpos = np.zeros(7 * n, dtype=np.float64)

    for i in range(n):
        p0 = 7 * i
        # position
        qpos[p0:p0+3] = poses[p0:p0+3]

        # quaternion in YAML order (xyzw)
        qx, qy, qz, qw = poses[p0+3:p0+7]
        # write MuJoCo order (wxyz)
        qpos[p0+3:p0+7] = [qw, qx, qy, qz]

    # write to mujoco
    data.qpos[:7 * n] = qpos
    data.qvel[:6 * n] = vels
    mujoco.mj_forward(model, data)

    return data.qpos.copy(), data.qvel.copy()



def sample_bounded_quat(max_tilt_deg: float, rng: np.random.Generator):
    max_tilt = np.deg2rad(max_tilt_deg)

    # sample roll/pitch bounded, yaw free
    roll  = rng.uniform(-max_tilt, max_tilt)
    pitch = rng.uniform(-max_tilt, max_tilt)
    yaw   = rng.uniform(-np.pi, np.pi)

    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)

    # ZYX convention
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    return np.array([qw, qx, qy, qz], dtype=np.float64)



def sample_unit_vector_with_elevation_limits(
    rng: np.random.Generator,
    elev_min_deg: float = 30.0,
    elev_max_deg: float = 89.0,
) -> np.ndarray:
    """
    Sample a unit vector with elevation φ relative to the xy-plane:
      φ = 0°  -> in the plane
      φ = 90° -> straight up (+z)

    elev_min_deg prevents quads from being too sideways.
    elev_max_deg avoids being exactly vertical if you want.
    """
    elev_min = np.deg2rad(elev_min_deg)
    elev_max = np.deg2rad(elev_max_deg)

    theta = rng.uniform(0.0, 2.0 * np.pi)        # yaw
    phi = rng.uniform(elev_min, elev_max)       # elevation above xy-plane

    # spherical coords: z = sin(phi), r_xy = cos(phi)
    r_xy = np.cos(phi)
    z = np.sin(phi)

    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    return np.array([x, y, z], dtype=np.float64)


def sample_velocities(
    n_quads: int,
    lin_vel_max=0.0001,
    ang_vel_max=0.0001,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    payload_lin = rng.uniform(-lin_vel_max, lin_vel_max, size=3)
    payload_ang = np.zeros(3)  # point-mass payload

    quad_lin = rng.uniform(-lin_vel_max, lin_vel_max, size=(n_quads, 3))
    quad_ang = rng.uniform(-ang_vel_max, ang_vel_max, size=(n_quads, 3))

    return payload_lin, payload_ang, quad_lin, quad_ang



def sample_payload_and_quads(
    payload_bounds,   # (low, high) arrays shape (3,)
    n_quads: int,
    cable_min: float,
    cable_max: float,
    rng: np.random.Generator,
    quad_radius: float = 0.1,
    min_pair_clearance: float = 0.02,   # extra margin
    max_tries: int = 2000,
):
    low, high = payload_bounds
    payload_pos = rng.uniform(low, high)

    quad_pos = np.zeros((n_quads, 3), dtype=np.float64)

    min_dist_between_quads = 2.0 * quad_radius + min_pair_clearance

    for i in range(n_quads):
        ok = False
        for _ in range(max_tries):
            d = rng.uniform(cable_min, cable_max)

            # IMPORTANT: upper hemisphere so quad tends to be above payload
            q = sample_unit_vector_with_elevation_limits(rng, elev_min_deg=65.0, elev_max_deg=68.0)

            p = payload_pos + d * q

            # collision check with already placed quads (sphere radius 0.1)
            if i > 0:
                if np.any(np.linalg.norm(quad_pos[:i] - p[None, :], axis=1) < min_dist_between_quads):
                    continue

            quad_pos[i] = p
            ok = True
            break

        if not ok:
            raise RuntimeError(f"Failed to sample non-colliding quad position for quad {i} after {max_tries} tries.")

    return payload_pos.astype(np.float64), quad_pos



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_payload_and_quads_state(
    state,
    n_bodies,
    radius=0.1,
    title="Payload + Quads",
    save_path="pos.pdf",
    show=False,
):
    """
    state: (13 * n_bodies,)
           [pos3, quat4(xyzw), lin3, ang3] repeated
    """

    state = np.asarray(state).reshape(-1)
    assert state.size == 13 * n_bodies

    # poses-first layout
    poses = state[: 7 * n_bodies]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # sphere mesh
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    sx = radius * np.outer(np.cos(u), np.sin(v))
    sy = radius * np.outer(np.sin(u), np.sin(v))
    sz = radius * np.outer(np.ones_like(u), np.cos(v))

    positions = []

    for i in range(n_bodies):
        pos = poses[7 * i : 7 * i + 3]
        positions.append(pos)

        color = "tab:red" if i == 0 else "tab:blue"
        label = "payload" if i == 0 else ("quad" if i == 1 else None)

        ax.plot_surface(
            sx + pos[0],
            sy + pos[1],
            sz + pos[2],
            color=color,
            alpha=0.4,
            linewidth=0,
        )
        ax.scatter(*pos, color=color, s=60, label=label)

    positions = np.array(positions)

    # formatting
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()

    mins = positions.min(axis=0) - radius
    maxs = positions.max(axis=0) + radius
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)

    # save
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[plot] saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
