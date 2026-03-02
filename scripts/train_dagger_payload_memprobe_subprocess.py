#!/usr/bin/env python3
"""
Run train_dagger_payload.py with expert-call memory instrumentation, but using
the subprocess-backed pc-dbCBS expert wrapper.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Tuple


LOG_PATH = Path("scripts/expert_memprobe.jsonl")
MEMPROBE_REPLAN_K = os.environ.get("MEMPROBE_REPLAN_K")
MEMPROBE_USE_DBG_EXPERT = os.environ.get("MEMPROBE_USE_DBG_EXPERT") == "1"


def _read_proc_mem_kb() -> Tuple[int, int]:
    rss_kb = -1
    hwm_kb = -1
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = int(line.split()[1])
                if rss_kb >= 0 and hwm_kb >= 0:
                    break
    except Exception:
        pass
    return rss_kb, hwm_kb


class _JsonlLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fp = open(self.path, "a", encoding="utf-8")

    def log(self, payload: Dict) -> None:
        with self._lock:
            self._fp.write(json.dumps(payload, sort_keys=True) + "\n")
            self._fp.flush()


def main() -> None:
    import train_dagger_payload as base

    if MEMPROBE_USE_DBG_EXPERT:
        import expert_pcdbcbs_subprocess_dbg as expert_mod
        expert_base_cls = expert_mod.PcDbCBSExpert
    else:
        import expert_pcdbcbs_subprocess as expert_mod
        expert_base_cls = expert_mod.PcDbCBSExpert

    logger = _JsonlLogger(LOG_PATH)

    class InstrumentedPcDbCBSExpert(expert_base_cls):  # type: ignore[misc]
        _next_id = 0
        _next_call = 0
        _class_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if MEMPROBE_REPLAN_K is not None and int(getattr(self, "replan_every_k", 0)) > 0:
                self.replan_every_k = int(MEMPROBE_REPLAN_K)
            with self._class_lock:
                self._expert_instance_id = InstrumentedPcDbCBSExpert._next_id
                InstrumentedPcDbCBSExpert._next_id += 1
            self._expert_call_count = 0
            self._expert_label = (
                f"seed_k{self.replan_every_k}"
                if int(getattr(self, "replan_every_k", 0)) == 0
                else f"dagger_k{self.replan_every_k}"
            )

        def act(self, obs):
            rss_before, hwm_before = _read_proc_mem_kb()
            t0 = time.perf_counter()
            err = None
            try:
                out = super().act(obs)
                return out
            except Exception as e:
                err = repr(e)
                raise
            finally:
                dt_ms = (time.perf_counter() - t0) * 1000.0
                rss_after, hwm_after = _read_proc_mem_kb()
                with self._class_lock:
                    global_call_id = InstrumentedPcDbCBSExpert._next_call
                    InstrumentedPcDbCBSExpert._next_call += 1
                self._expert_call_count += 1

                record = {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "expert_instance_id": self._expert_instance_id,
                    "expert_label": self._expert_label,
                    "expert_call_idx": self._expert_call_count,
                    "global_call_id": global_call_id,
                    "replan_every_k": int(getattr(self, "replan_every_k", -1)),
                    "episode_id": int(getattr(self, "_episode_id", -1)),
                    "global_step": int(getattr(self, "_global_step", -1)),
                    "plan_t": int(getattr(self, "_t", -1)),
                    "plan_len": int(self._U.shape[0]) if getattr(self, "_U", None) is not None else -1,
                    "just_replanned": bool(getattr(self, "just_replanned", False)),
                    "replan_reason": getattr(self, "last_replan_reason", None),
                    "rss_before_kb": rss_before,
                    "rss_after_kb": rss_after,
                    "rss_delta_kb": (rss_after - rss_before)
                    if rss_before >= 0 and rss_after >= 0
                    else None,
                    "hwm_before_kb": hwm_before,
                    "hwm_after_kb": hwm_after,
                    "dur_ms": round(dt_ms, 3),
                    "error": err,
                    "expert_impl": "subprocess_dbg" if MEMPROBE_USE_DBG_EXPERT else "subprocess",
                }

                logger.log(record)

                delta_mb = (record["rss_delta_kb"] or 0) / 1024.0
                print(
                    "[MEMPROBE-SUBPROC] "
                    f"{self._expert_label} call={self._expert_call_count} "
                    f"ep={record['episode_id']} gs={record['global_step']} "
                    f"replan={int(record['just_replanned'])} "
                    f"reason={record['replan_reason']} "
                    f"dt={record['dur_ms']:.1f}ms "
                    f"rss={rss_before}->{rss_after} kB "
                    f"(delta={delta_mb:+.1f} MB) "
                    f"hwm={hwm_after}",
                    flush=True,
                )

    base.PcDbCBSExpert = InstrumentedPcDbCBSExpert

    if MEMPROBE_USE_DBG_EXPERT:
        print("[MEMPROBE-SUBPROC] using debug subprocess expert", flush=True)
    else:
        print("[MEMPROBE-SUBPROC] using subprocess expert", flush=True)
    if MEMPROBE_REPLAN_K is not None:
        print(f"[MEMPROBE-SUBPROC] overriding dagger replan_every_k -> {int(MEMPROBE_REPLAN_K)}", flush=True)
    print(f"[MEMPROBE-SUBPROC] logging expert-call memory to {LOG_PATH}", flush=True)
    base.main()


if __name__ == "__main__":
    main()
