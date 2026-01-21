# scripts/videos_from_log.py
#
# Modular MuJoCo → MP4 renderer that supports TWO modes:
#   1) Replay states: provide x_traj = [qpos|qvel] for each step
#   2) Rollout actions: provide (qpos0, qvel0, u_traj) and it will mj_step()
#
# You can import and call:
#   - render_from_states(...)
#   - render_from_actions(...)
#
# Both accept numpy arrays directly OR a path to a saved .npz file.

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import mujoco as mj
from mujoco.glfw import glfw


ArrayLike = Union[np.ndarray, "np.typing.NDArray"]


@dataclass
class VideoConfig:
    out_dir: str
    width: int = 1280
    height: int = 720
    fps: int = 60
    bitrate: str = "6M"
    loops: int = 1
    scale_factor: float = 1.0
    views: Union[str, List[str]] = "auto"  # "auto" or e.g. ["side","top","front","diag"]
    env_min: Optional[np.ndarray] = None   # (3,)
    env_max: Optional[np.ndarray] = None   # (3,)


# --------------------------- internal helpers ---------------------------

def _ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg not found on PATH. Please install ffmpeg.")


def _open_ffmpeg(out_path: str, width: int, height: int, fps: int, bitrate: str):
    _ensure_ffmpeg()
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}",
        "-framerate", str(fps),
        "-i", "-",
        "-vf", "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _init_hidden_glfw_context():
    if not glfw.init():
        raise RuntimeError("GLFW init failed (needed for OpenGL context).")

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    win = glfw.create_window(64, 64, "", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Failed to create hidden GLFW window/context.")
    glfw.make_context_current(win)
    return win


def _close_hidden_glfw_context(win):
    try:
        glfw.destroy_window(win)
    except Exception:
        pass
    try:
        glfw.terminate()
    except Exception:
        pass


def _get_views(views_cfg: Union[str, List[str]]) -> List[str]:
    if isinstance(views_cfg, str) and views_cfg.lower() == "auto":
        return ["side", "top", "front", "diag"]
    if isinstance(views_cfg, list):
        return [str(v).lower() for v in views_cfg]
    return ["side", "top", "front", "diag"]


def apply_camera_preset(cam: mj.MjvCamera,
                        env_center: np.ndarray,
                        env_size: np.ndarray,
                        view: str,
                        scale_factor: float = 1.0):
    """Same preset logic you used, extracted for reuse."""
    cam.lookat[:] = [float(env_center[0]), float(env_center[1]), float(env_center[2])]
    xy_diag = float(np.sqrt(env_size[0] ** 2 + env_size[1] ** 2))
    base_dist = max(1.0, xy_diag * 0.65) * float(scale_factor)

    v = (view or "").lower()
    if v == "top":
        cam.azimuth = 0.0
        cam.elevation = -90.0
        cam.distance = max(env_size[0], env_size[1]) * 0.8 * float(scale_factor)
    elif v == "front":
        cam.azimuth = 180.0
        cam.elevation = -15.0
        cam.distance = base_dist
    elif v == "side":
        cam.azimuth = 90.0
        cam.elevation = -15.0
        cam.distance = base_dist
    elif v == "diag":
        cam.azimuth = 202.5
        cam.elevation = -15.0
        cam.distance = base_dist
    else:
        cam.azimuth = 45.0
        cam.elevation = -35.0
        cam.distance = base_dist


def _prepare_renderer(model: mj.MjModel, width: int, height: int):
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)

    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, context)
    if context.offWidth != width or context.offHeight != height:
        mj.mjr_resizeOffscreen(width, height, context)

    viewport = mj.MjrRect(0, 0, width, height)
    rgb_buf = np.empty((height, width, 3), dtype=np.uint8)

    return cam, opt, scene, context, viewport, rgb_buf


def _free_renderer(scene, context):
    try:
        mj.mjr_freeContext(context)
    except Exception:
        pass
    try:
        mj.mjv_freeScene(scene)
    except Exception:
        pass


def _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf) -> np.ndarray:
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    mj.mjr_readPixels(rgb_buf, None, viewport, context)
    return rgb_buf


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _require_env_bounds(cfg: VideoConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cfg.env_min is None or cfg.env_max is None:
        raise ValueError("VideoConfig.env_min and env_max must be provided (3D vectors).")
    env_min = np.asarray(cfg.env_min, dtype=float).reshape(3,)
    env_max = np.asarray(cfg.env_max, dtype=float).reshape(3,)
    env_center = 0.5 * (env_min + env_max)
    env_size = env_max - env_min
    return env_min, env_max, env_center, env_size


# --------------------------- public API ---------------------------

def render_from_states(
    xml_path: str,
    x_traj_or_npz: Union[np.ndarray, str],
    cfg: VideoConfig,
    *,
    key: str = "x_traj",
    set_state_fn: Optional[Callable[[mj.MjModel, mj.MjData, np.ndarray], None]] = None,
) -> List[str]:
    """
    Render MP4(s) from a saved trajectory of states.

    Inputs:
      - xml_path: path to MuJoCo XML
      - x_traj_or_npz:
          * numpy array of shape (N, nx) OR
          * path to .npz containing array under `key` (default: "x_traj")
      - cfg: VideoConfig (must include out_dir, env_min, env_max)
      - set_state_fn:
          If provided, called as set_state_fn(model, data, x) to set MuJoCo state.
          If None, assumes x is concatenated [qpos|qvel] with shape (nq+nv,).

    Returns:
      - list of output mp4 paths written (one per view)
    """
    if isinstance(x_traj_or_npz, str):
        d = _load_npz(x_traj_or_npz)
        if key not in d:
            raise KeyError(f"NPZ missing key '{key}'. Available keys: {list(d.keys())}")
        x_traj = d[key]
    else:
        x_traj = np.asarray(x_traj_or_npz)

    if x_traj.ndim != 2:
        raise ValueError(f"x_traj must be 2D (N, nx). Got shape {x_traj.shape}.")

    _require_env_bounds(cfg)
    views = _get_views(cfg.views)
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # Default state setter: x = [qpos|qvel]
    if set_state_fn is None:
        expected = model.nq + model.nv
        if x_traj.shape[1] != expected:
            raise ValueError(f"Default set_state expects nx=nq+nv={expected}, got {x_traj.shape[1]}. "
                             f"Provide set_state_fn to map your x into qpos/qvel.")
        def set_state_fn(model_, data_, x_):
            data_.qpos[:] = x_[:model_.nq]
            data_.qvel[:] = x_[model_.nq:model_.nq + model_.nv]

    # Offscreen init
    win = _init_hidden_glfw_context()
    cam, opt, scene, context, viewport, rgb_buf = _prepare_renderer(model, cfg.width, cfg.height)
    _, _, env_center, env_size = _require_env_bounds(cfg)

    written: List[str] = []
    try:
        for view in views:
            apply_camera_preset(cam, env_center, env_size, view, scale_factor=cfg.scale_factor)
            out_path = os.path.join(cfg.out_dir, f"{view}.mp4")
            proc = _open_ffmpeg(out_path, cfg.width, cfg.height, cfg.fps, cfg.bitrate)

            try:
                for _ in range(max(1, cfg.loops)):
                    for x in x_traj:
                        set_state_fn(model, data, x)
                        mj.mj_forward(model, data)
                        frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf)
                        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            finally:
                proc.stdin.close()
                proc.wait()

            written.append(out_path)
    finally:
        _free_renderer(scene, context)
        _close_hidden_glfw_context(win)

    return written


def render_from_actions(
    xml_path: str,
    init_and_actions_or_npz: Union[Tuple[np.ndarray, np.ndarray, np.ndarray], str],
    cfg: VideoConfig,
    *,
    keys: Tuple[str, str, str] = ("qpos0", "qvel0", "u_traj"),
    set_init_fn: Optional[Callable[[mj.MjModel, mj.MjData, np.ndarray, np.ndarray], None]] = None,
    apply_ctrl_fn: Optional[Callable[[mj.MjModel, mj.MjData, np.ndarray], None]] = None,
    include_initial_frame: bool = True,
) -> List[str]:
    """
    Roll out controls with mj_step() and render MP4(s).

    Inputs:
      - init_and_actions_or_npz:
          * tuple (qpos0, qvel0, u_traj) OR
          * path to .npz containing those arrays under `keys`
      - set_init_fn:
          If None, uses default: data.qpos=qpos0, data.qvel=qvel0.
      - apply_ctrl_fn:
          If None, uses default: data.ctrl[:] = u.
    """
    if isinstance(init_and_actions_or_npz, str):
        d = _load_npz(init_and_actions_or_npz)
        kq, kv, ku = keys
        for k in keys:
            if k not in d:
                raise KeyError(f"NPZ missing key '{k}'. Available keys: {list(d.keys())}")
        qpos0 = np.asarray(d[kq])
        qvel0 = np.asarray(d[kv])
        u_traj = np.asarray(d[ku])
    else:
        qpos0, qvel0, u_traj = init_and_actions_or_npz
        qpos0 = np.asarray(qpos0)
        qvel0 = np.asarray(qvel0)
        u_traj = np.asarray(u_traj)

    _require_env_bounds(cfg)
    views = _get_views(cfg.views)
    os.makedirs(cfg.out_dir, exist_ok=True)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    if qpos0.shape != (model.nq,):
        raise ValueError(f"qpos0 must have shape ({model.nq},), got {qpos0.shape}")
    if qvel0.shape != (model.nv,):
        raise ValueError(f"qvel0 must have shape ({model.nv},), got {qvel0.shape}")
    if u_traj.ndim != 2 or u_traj.shape[1] != model.nu:
        raise ValueError(f"u_traj must have shape (T, {model.nu}), got {u_traj.shape}")

    if set_init_fn is None:
        def set_init_fn(model_, data_, qpos0_, qvel0_):
            mj.mj_resetData(model_, data_)
            data_.qpos[:] = qpos0_
            data_.qvel[:] = qvel0_

    if apply_ctrl_fn is None:
        def apply_ctrl_fn(model_, data_, u_):
            data_.ctrl[:] = u_

    win = _init_hidden_glfw_context()
    cam, opt, scene, context, viewport, rgb_buf = _prepare_renderer(model, cfg.width, cfg.height)
    _, _, env_center, env_size = _require_env_bounds(cfg)

    written: List[str] = []
    try:
        for view in views:
            apply_camera_preset(cam, env_center, env_size, view, scale_factor=cfg.scale_factor)
            out_path = os.path.join(cfg.out_dir, f"{view}.mp4")
            proc = _open_ffmpeg(out_path, cfg.width, cfg.height, cfg.fps, cfg.bitrate)

            try:
                for _ in range(max(1, cfg.loops)):
                    set_init_fn(model, data, qpos0, qvel0)
                    mj.mj_forward(model, data)

                    if include_initial_frame:
                        frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf)
                        proc.stdin.write(np.ascontiguousarray(frame).tobytes())

                    for u in u_traj:
                        apply_ctrl_fn(model, data, u)
                        mj.mj_step(model, data)
                        frame = _render_frame(model, data, opt, cam, scene, context, viewport, rgb_buf)
                        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            finally:
                proc.stdin.close()
                proc.wait()

            written.append(out_path)
    finally:
        _free_renderer(scene, context)
        _close_hidden_glfw_context(win)

    return written
