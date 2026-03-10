#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import sqlite3
import sys
import time
import signal
import threading
import queue
import json
import os
import smtplib
import subprocess
import shutil
import struct
import zlib
import io
from email.message import EmailMessage

STOP_REQUESTED = False

def _signal_handler(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Heavy deps (pandas/pyarrow) are imported lazily to speed GUI startup.
_pd = None
_pa = None
_pq = None

def _lazy_pandas():
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd

def _lazy_arrow():
    global _pa, _pq
    if _pa is None or _pq is None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        _pa, _pq = pa, pq
    return _pa, _pq

# GUI + plotting (optional)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib is expensive to import; defer it until a plot window is actually created.
_MPL = None
def _lazy_mpl():
    global _MPL
    if _MPL is not None:
        return _MPL
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # Keep appearance consistent with your previous styling
    try:
        plt.style.use("dark_background")
    except Exception:
        pass
    _MPL = (matplotlib, plt, FigureCanvasTkAgg)
    return _MPL


# ── Theme palette ─────────────────────────────────────────────────────────
# Deep-space dark theme with color-coded functional zones.
#   Channel A : Cyan        Channel B : Gold/Amber
#   Trigger   : Magenta     Integration : Teal
#   Surfaces  : Navy/slate  Accents     : bright on dark
T_BG        = "#0b0e14"   # deepest background
T_SURFACE   = "#111620"   # panel / card surface
T_SURFACE2  = "#151b28"   # slightly lighter surface
T_BORDER    = "#1e2636"   # subtle border
T_BORDER_HI = "#2a3550"   # highlighted border
T_TEXT      = "#c8d4e6"   # primary text
T_TEXT_DIM  = "#6e7f99"   # secondary / dim text
T_TEXT_BRIGHT = "#eef2f8"  # bright labels

# Functional accent colors
T_CYAN      = "#00d4ff"   # Channel A / primary accent
T_GOLD      = "#ffb700"   # Channel B / trigger-related
T_MAGENTA   = "#e040fb"   # Trigger / alerts
T_GREEN     = "#00e676"   # Status OK / runtime
T_TEAL      = "#1de9b6"   # Integration history
T_RED       = "#ff5252"   # Errors / stop
T_ORANGE    = "#ff9100"   # Warnings

# Selection / interactive
T_SEL       = "#0d2240"   # selection background
T_SEL_HI    = "#143360"   # active selection
T_BTN       = "#1a2740"   # button background
T_BTN_HI    = "#243654"   # button hover
T_BTN_ACT   = "#00d4ff"   # active / primary button

# Legacy aliases for plot compatibility
NEON_PINK = T_CYAN         # Channel A plots
NEON_GREEN = T_GOLD        # Channel B plots

SAMPLE_RATE_OPTIONS_MSPS = [20.0, 50.0, 100.0, 125.0, 160.0, 180.0, 200.0, 250.0, 500.0, 1000.0]
INPUT_RANGE_OPTIONS = [
    "PM_20_MV", "PM_40_MV", "PM_50_MV", "PM_80_MV", "PM_100_MV", "PM_200_MV",
    "PM_400_MV", "PM_500_MV", "PM_800_MV", "PM_1_V", "PM_2_V", "PM_4_V",
]
COUPLING_OPTIONS = ["DC", "AC"]
IMPEDANCE_OPTIONS = ["50_OHM", "1_MOHM"]
CHANNEL_MASK_OPTIONS = ["CHANNEL_A", "CHANNEL_B", "CHANNEL_A|CHANNEL_B"]
TRIGGER_SOURCE_LABEL_TO_CONST = {
    "External": "TRIG_EXTERNAL",
    "Channel A": "TRIG_CHAN_A",
    "Channel B": "TRIG_CHAN_B",
}
TRIGGER_SOURCE_CONST_TO_LABEL = {v: k for k, v in TRIGGER_SOURCE_LABEL_TO_CONST.items()}
TRIGGER_SOURCEK_LABEL_TO_CONST = {
    "Disabled": "TRIG_DISABLE",
    "External": "TRIG_EXTERNAL",
    "Channel A": "TRIG_CHAN_A",
    "Channel B": "TRIG_CHAN_B",
}
TRIGGER_SOURCEK_CONST_TO_LABEL = {v: k for k, v in TRIGGER_SOURCEK_LABEL_TO_CONST.items()}
TRIGGER_SLOPE_LABEL_TO_CONST = {
    "Positive": "TRIGGER_SLOPE_POSITIVE",
    "Negative": "TRIGGER_SLOPE_NEGATIVE",
}
TRIGGER_SLOPE_CONST_TO_LABEL = {v: k for k, v in TRIGGER_SLOPE_LABEL_TO_CONST.items()}
TRIGGER_MODE_OPTIONS = ["TRIG_ENGINE_OP_J", "TRIG_ENGINE_OP_J_OR_K", "TRIG_ENGINE_OP_J_AND_K"]
EXT_TRIGGER_RANGE_OPTIONS = ["ETR_5V", "ETR_1V"]
TRIGGER_LEVEL_PRESETS_PCT = [-50.0, -25.0, 0.0, 25.0, 50.0]
RUNTIME_PROFILE_PRESETS = {
    "Mu2e Spill": {
        "rearm_if_no_trigger_s": 3,
        "rearm_cooldown_s": 2,
        "max_rearms_per_hour": 3600,
        "flush_every_records": 20000,
        "flush_every_seconds": 2.0,
        "sqlite_commit_every_snips": 200,
        "stream_window_points": 100000,
        "stream_window_seconds": 2.0,
        "max_waveforms_per_tick": 20,
    },
    "Balanced": {
        "rearm_if_no_trigger_s": 300,
        "rearm_cooldown_s": 30,
        "max_rearms_per_hour": 12,
        "flush_every_records": 20000,
        "flush_every_seconds": 2.0,
        "sqlite_commit_every_snips": 200,
        "stream_window_points": 100000,
        "stream_window_seconds": 2.0,
        "max_waveforms_per_tick": 20,
    },
    "Low Latency": {
        "rearm_if_no_trigger_s": 60,
        "rearm_cooldown_s": 10,
        "max_rearms_per_hour": 60,
        "flush_every_records": 4000,
        "flush_every_seconds": 0.5,
        "sqlite_commit_every_snips": 50,
        "stream_window_points": 12000,
        "stream_window_seconds": 1.0,
        "max_waveforms_per_tick": 80,
    },
    "Throughput": {
        "rearm_if_no_trigger_s": 600,
        "rearm_cooldown_s": 60,
        "max_rearms_per_hour": 6,
        "flush_every_records": 80000,
        "flush_every_seconds": 5.0,
        "sqlite_commit_every_snips": 1000,
        "stream_window_points": 50000,
        "stream_window_seconds": 5.0,
        "max_waveforms_per_tick": 10,
    },
    "Low CPU": {
        "rearm_if_no_trigger_s": 300,
        "rearm_cooldown_s": 30,
        "max_rearms_per_hour": 12,
        "flush_every_records": 80000,
        "flush_every_seconds": 8.0,
        "sqlite_commit_every_snips": 1024,
        "stream_window_points": 8000,
        "stream_window_seconds": 4.0,
        "max_waveforms_per_tick": 2,
    },
}
RUNTIME_PROFILE_OPTIONS = list(RUNTIME_PROFILE_PRESETS.keys()) + ["Custom"]
LIVE_UI_FPS_MAX = 15.0
LIVE_MAX_CATCHUP_WAVEFORMS_PER_TICK = 12
WAVEFORM_EVERY_N_MAX = 3000
WAVE_ARCHIVE_CODEC_OPTIONS = {"none", "f32_zlib", "delta_i16_zlib"}

# ── Dummy / Admin mode ────────────────────────────────────────────────────
ADMIN_PASSWORD = "FermiAdmin1234"


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    x = _to_int(v, default)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _clamp_float(v: Any, lo: float, hi: float, default: float) -> float:
    x = _to_float(v, default)
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _trigger_pct_to_level_code(pct: Any, default_pct: float = 0.0) -> int:
    """Map signed trigger percent [-100, 100] to ATS code [0, 255] with 0% centered at code 128."""
    p = _clamp_float(pct, -100.0, 100.0, default_pct)
    if p >= 0.0:
        code = 128 + int(round(p * 127.0 / 100.0))
    else:
        code = 128 + int(round(p * 128.0 / 100.0))
    return _clamp_int(code, 0, 255, 128)


def _level_code_to_trigger_pct(code: Any, default_code: int = 128) -> float:
    """Map ATS trigger code [0, 255] to signed percent [-100, 100] with code 128 at 0%."""
    c = _clamp_int(code, 0, 255, default_code)
    if c == 128:
        return 0.0
    if c > 128:
        return (float(c - 128) / 127.0) * 100.0
    return (float(c - 128) / 128.0) * 100.0

ATS_AVAILABLE = False
ats = None
try:
    sys.path.append('/usr/local/AlazarTech/samples/Samples_Python/Library/')
    import atsapi as ats  # type: ignore
    ATS_AVAILABLE = True
except Exception:
    ATS_AVAILABLE = False
    ats = None

DEFAULT_YAML = r"""# =========================
# CAPPY v1.3 Configuration
# =========================
board:
  system_id: 2
  board_id: 1


clock:
  source: INTERNAL_CLOCK
  sample_rate_msps: 250.0
  edge: CLOCK_EDGE_RISING

channels:
  A:
    coupling: DC
    range: PM_1_V          # maps to INPUT_RANGE_PM_1_V (old-script style)
    impedance: 50_OHM
  B:
    coupling: DC
    range: PM_1_V
    impedance: 50_OHM

trigger:
  operation: TRIG_ENGINE_OP_J
  engine1: TRIG_ENGINE_J
  engine2: TRIG_ENGINE_K

  sourceJ: TRIG_EXTERNAL
  slopeJ: TRIGGER_SLOPE_POSITIVE
  levelJ: 128

  sourceK: TRIG_DISABLE
  slopeK: TRIGGER_SLOPE_POSITIVE
  levelK: 128

  ext_coupling: DC_COUPLING
  ext_range: ETR_5V
  delay_samples: 0
  trigger_delay_us: 0.0           # convenience: trigger delay in microseconds (overrides delay_samples if > 0)
  timeout_ms: 0                   # 0 = wait forever (unless runtime.noise_test=true)
  timeout_pause_s: 0.0            # if >0 and no triggers arrive for this long, pause then rearm

  external_startcapture: false

acquisition:
  channels_mask: CHANNEL_A|CHANNEL_B
  pre_trigger_samples: 0
  samples_per_record: 256      # optional convenience key; post = samples_per_record - pre
  post_trigger_samples: 256       # full record = pre + post
  records_per_buffer: 128
  buffers_allocated: 16
  buffers_per_acquisition: 0      # 0 = run forever (until stop)
  wait_timeout_ms: 1000           # DMA wait timeout; timeouts are handled (no crash)

integration:
  baseline_window_samples: [0, 64]
  integral_window_samples:  [64, 128]

waveforms:
  enable: true
  full_record: true
  mode: every_n
  every_n: 16
  threshold_integral_Vs: 0.0
  threshold_peak_V: 0.0
  max_waveforms_per_sec: 120
  store_volts: true
  archive_codec: delta_i16_zlib    # none | f32_zlib | delta_i16_zlib (best compression)
  archive_quant_bits: 12          # used by delta_i16_zlib (8..14)
  archive_zlib_level: 3           # zlib compression level (1..9)

storage:
  data_dir: dataFile
  session_tag: ""
  rollover_minutes: 60
  session_rotate_hours: 24
  flush_every_records: 20000
  flush_every_seconds: 2
  sqlite_commit_every_snips: 200

notify:
  enabled: false
  to: "user@example.com"
  from: "cappy@localhost"
  method: "sendmail"
  sendmail_path: "/usr/sbin/sendmail"
  subject_prefix: "[CAPPY]"
  heartbeat_seconds: 0.25
  interval_minutes: 120

runtime:
  noise_test: false
  autotrigger_timeout_ms: 10
  rearm_if_no_trigger_s: 3         # ≈2× the MI beam-off gap (1.02s) for Mu2e spill timing
  rearm_cooldown_s: 2              # allow rapid rearm across spill boundaries
  max_rearms_per_hour: 3600        # ≈1/s is fine for Mu2e cycle (1.4s period)

live:
  ring_slots: 4096
  ring_points: 4096
  waveform_every_n_buffers: 1
  stream_window_points: 100000
  stream_window_seconds: 2.0
  max_waveforms_per_tick: 6
  ui_fps: 4
  show_channel_b: false
  preview_mode: archive_match
"""

SESSION_INDEX_COLUMNS = [
    "session_id",
    "date",
    "first_timestamp_ns",
    "last_timestamp_ns",
    "reduced_rows",
    "waveform_snips",
    "channels_mask",
]

REDUCED_SCHEMA = None
SESSION_INDEX_SCHEMA = None
try:
    _pa0 = _lazy_arrow()[0]
    REDUCED_SCHEMA = _pa0.schema([
        ("session_id", _pa0.string()),
        ("buffer_index", _pa0.int32()),
        ("record_in_buffer", _pa0.int32()),
        ("record_global", _pa0.int64()),
        ("timestamp_ns", _pa0.int64()),
        ("sample_rate_hz", _pa0.float64()),
        ("samples_per_record", _pa0.int32()),
        ("records_per_buffer", _pa0.int32()),
        ("channels_mask", _pa0.string()),
        ("area_A_Vs", _pa0.float64()),
        ("peak_A_V", _pa0.float64()),
        ("baseline_A_V", _pa0.float64()),
        ("area_B_Vs", _pa0.float64()),
        ("peak_B_V", _pa0.float64()),
        ("baseline_B_V", _pa0.float64()),
    ])

    SESSION_INDEX_SCHEMA = _pa0.schema([
        ("session_id", _pa0.string()),
        ("date", _pa0.string()),
        ("first_timestamp_ns", _pa0.int64()),
        ("last_timestamp_ns", _pa0.int64()),
        ("reduced_rows", _pa0.int64()),
        ("waveform_snips", _pa0.int64()),
        ("channels_mask", _pa0.string()),
    ])
except Exception:
    REDUCED_SCHEMA = None
    SESSION_INDEX_SCHEMA = None

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clear_pycache(root: Path) -> int:
    """Delete __pycache__ folders and .pyc/.pyo files under root. Returns count removed (best-effort)."""
    root = Path(root)
    removed = 0
    try:
        # Remove __pycache__ directories
        for d in root.rglob("__pycache__"):
            try:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
            except Exception:
                pass

        # Remove stray bytecode files
        for ext in ("*.pyc", "*.pyo"):
            for f in root.rglob(ext):
                try:
                    f.unlink(missing_ok=True)  # py3.8+
                    removed += 1
                except Exception:
                    try:
                        if f.exists():
                            f.unlink()
                            removed += 1
                    except Exception:
                        pass
    except Exception:
        return removed
    return removed


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic text write with backup. Creates .bak of previous version for YAML configs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep a .bak for YAML configs (safety net)
    if path.exists() and path.suffix in ('.yaml', '.yml'):
        try:
            bak = path.with_suffix(path.suffix + '.bak')
            shutil.copy2(str(path), str(bak))
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, text.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        tmp.replace(path)
    except Exception:
        # Fallback to simple write
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def _auto_time_axis(n: int, sample_rate_hz: float) -> Tuple[np.ndarray, str]:
    """Return (t, unit) for n samples at sample_rate_hz, using a human scale (ns/us/ms/s)."""
    sr = float(sample_rate_hz) if sample_rate_hz else 1.0
    dt = 1.0 / sr
    t = np.arange(int(n), dtype=np.float64) * dt
    tmax = float(t[-1]) if t.size else 0.0
    if tmax < 1e-6:
        return t * 1e9, "ns"
    if tmax < 1e-3:
        return t * 1e6, "µs"
    if tmax < 1.0:
        return t * 1e3, "ms"
    return t, "s"


def _active_tail_index(y: np.ndarray) -> int:
    """
    Estimate the last meaningfully active sample index in a waveform.
    Used only for display trimming of long flat tails.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(arr.size)
    if n <= 8:
        return max(0, n - 1)

    tail_start = int(max(0, n * 0.8))
    tail = arr[tail_start:] if tail_start < n else arr
    baseline = float(np.median(tail))
    dev = np.abs(arr - baseline)
    mad = float(np.median(np.abs(tail - baseline)))
    noise = 1.4826 * mad
    amp = float(np.percentile(dev, 99)) if n > 0 else 0.0
    thr = max(5.0 * noise, 0.02 * amp, 1e-6)
    idx = np.flatnonzero(dev > thr)
    if idx.size == 0:
        return n - 1
    pad = max(4, n // 100)
    return min(n - 1, int(idx[-1]) + pad)

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def _ns_to_iso(ts_ns: int) -> str:
    try:
        return datetime.fromtimestamp(int(ts_ns) / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return ""


def _format_size_gib(n_bytes: int) -> str:
    b = max(0, int(n_bytes or 0))
    gib = b / float(1024 ** 3)
    gb = b / 1e9
    if gib >= 1.0:
        return f"{gib:.3f} GiB ({gb:.3f} GB)"
    mib = b / float(1024 ** 2)
    return f"{mib:.1f} MiB ({gb:.3f} GB)"


def _dir_size_bytes(root: Path) -> int:
    total = 0
    p = Path(root)
    if not p.exists():
        return 0
    for dpath, _dirs, files in os.walk(p):
        for name in files:
            try:
                total += (Path(dpath) / name).stat().st_size
            except Exception:
                pass
    return int(total)


def _preferred_data_dir(local_name: str) -> str:
    try:
        data_root = Path("/Data")
        if data_root.exists() and data_root.is_dir() and os.access(str(data_root), os.W_OK):
            return str(data_root / local_name)
    except Exception:
        pass
    return local_name

def ats_const(prefix_or_name: str, maybe_name: str | None = None) -> int:
    """Resolve atsapi constants with the same prefix-mapping style as your old script."""
    if not ATS_AVAILABLE or ats is None:
        raise RuntimeError("atsapi not available.")

    if maybe_name is None:
        prefix = ""
        name = prefix_or_name
    else:
        prefix = prefix_or_name
        name = maybe_name

    if prefix and name.startswith(prefix):
        target = name
    else:
        target = f"{prefix}{name}"

    if hasattr(ats, target):
        return int(getattr(ats, target))

    if prefix == "INPUT_RANGE_" and hasattr(ats, name):
        print(f"[WARN] Using '{name}' instead of '{target}'")
        return int(getattr(ats, name))

    opts = [a for a in dir(ats) if (a.startswith(prefix) if prefix else True)]
    raise AttributeError(f"atsapi constant not found: {target} (prefix={prefix}). Example options: {opts[:40]} ...")

def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write to survive crashes/power loss."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)

def _send_status_email(cfg: dict, subject: str, body: str) -> None:
    """Send status email. Supports sendmail (local MTA) or SMTP."""
    notify = (cfg.get("notify", {}) or {})
    if not bool(notify.get("enabled", False)):
        return

    to_addr = str(notify.get("to", "")).strip()
    if not to_addr:
        return
    from_addr = str(notify.get("from", "cappy@localhost")).strip() or "cappy@localhost"
    method = str(notify.get("method", "sendmail")).strip().lower()

    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = from_addr
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if method == "sendmail":
            sendmail_path = str(notify.get("sendmail_path", "/usr/sbin/sendmail"))
            p = subprocess.Popen([sendmail_path, "-t", "-i"], stdin=subprocess.PIPE)
            p.communicate(msg.as_bytes())
        elif method == "smtp":
            host = str(notify.get("smtp_host", "")).strip()
            port = int(notify.get("smtp_port", 587))
            user = str(notify.get("smtp_user", "")).strip()
            pwd_env = str(notify.get("smtp_password_env", "CAPPY_SMTP_PASSWORD")).strip() or "CAPPY_SMTP_PASSWORD"
            password = os.environ.get(pwd_env, "")
            starttls = bool(notify.get("smtp_starttls", True))

            if not host:
                print("[WARN] SMTP host not configured, skipping email")
                return

            with smtplib.SMTP(host, port, timeout=20) as s:
                s.ehlo()
                if starttls:
                    s.starttls()
                    s.ehlo()
                if user:
                    if not password:
                        raise RuntimeError(f"SMTP password env var not set: {pwd_env}")
                    s.login(user, password)
                s.send_message(msg)
    except Exception as e:
        print(f"[WARN] Failed to send status email: {e}")


class LiveRingWriter:
    """
    Rolling on-disk ring buffer for live waveform display.

    Why: You want EVERY buffer's waveform captured, but you only want to *render* at a stable UI cadence.
    The capture process writes one downsampled waveform per buffer into a fixed-size ring file.
    The GUI reads forward at its own pace (e.g., 10–30 fps) for smooth, consistent visualization.

    File format (little-endian, fixed record size):
      header: 32 bytes
        - magic b'CPRING1\0' (8)
        - version u32 (4)
        - nslots u32 (4)
        - npts u32 (4)   (fixed points per waveform)
        - reserved u32 (4)
        - write_seq u64 (8)  (monotonic sequence number)
      record slot i:
        - seq u64
        - t_unix f64
        - buf_idx u64
        - chmask u32  (bit0=A, bit1=B)
        - reserved u32
        - wfA float32[npts]
        - wfB float32[npts]  (NaN if disabled)
    """
    MAGIC = b"CPRING1\0"
    VERSION = 1

    def __init__(self, path: Path, nslots: int = 4096, npts: int = 1024):
        self.path = path
        self.nslots = int(nslots)
        self.npts = int(npts)
        self._lock = threading.Lock()
        self._seq = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        self._rec_bytes = (8 + 8 + 8 + 4 + 4) + (self.npts * 4) + (self.npts * 4)
        self._hdr_bytes = 32
        total = self._hdr_bytes + self.nslots * self._rec_bytes

        # create/size file
        if not path.exists() or path.stat().st_size != total:
            with open(path, "wb") as f:
                f.truncate(total)
            self._write_header()

        # open for random access writes
        self._fh = open(path, "r+b", buffering=0)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass

    def _write_header(self):
        # write initial header with seq=0
        import struct
        hdr = struct.pack("<8sIIIIQ", self.MAGIC, self.VERSION, self.nslots, self.npts, 0, 0)
        with open(self.path, "r+b", buffering=0) as f:
            f.seek(0)
            f.write(hdr)

    def write(self, wfA: np.ndarray, wfB: Optional[np.ndarray], buf_idx: int, chmask: int):
        import struct
        # Downsample / coerce to float32 length npts
        def to_npts(w: np.ndarray) -> np.ndarray:
            w = np.asarray(w, dtype=np.float32)
            if w.size == self.npts:
                return w
            if w.size < 2:
                return np.zeros((self.npts,), dtype=np.float32)

            if w.size < self.npts:
                # Upsample by linear interpolation to avoid flat padded tails
                x_old = np.linspace(0.0, 1.0, num=w.size, dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, num=self.npts, dtype=np.float32)
                return np.interp(x_new, x_old, w).astype(np.float32, copy=False)

            # Downsample by decimation then trim
            step = max(1, int(w.size // self.npts))
            w2 = w[::step]
            if w2.size > self.npts:
                w2 = w2[: self.npts]
            elif w2.size < self.npts:
                # If decimation undershot (rare), interpolate to exact length
                x_old = np.linspace(0.0, 1.0, num=w2.size, dtype=np.float32)
                x_new = np.linspace(0.0, 1.0, num=self.npts, dtype=np.float32)
                w2 = np.interp(x_new, x_old, w2).astype(np.float32, copy=False)
            return w2

        a = to_npts(wfA)
        if wfB is None:
            b = np.full((self.npts,), np.nan, dtype=np.float32)
        else:
            b = to_npts(wfB)

        with self._lock:
            self._seq += 1
            seq = self._seq
            slot = (seq - 1) % self.nslots
            off = self._hdr_bytes + slot * self._rec_bytes

            t_unix = time.time()
            rec_hdr = struct.pack("<QdQII", seq, t_unix, int(buf_idx), int(chmask), 0)
            # Two-phase slot commit:
            #   1) mark slot invalid (seq=0)
            #   2) write payload bytes
            #   3) write final header with real seq
            # Reader only accepts records whose slot seq matches expected seq, so this
            # prevents torn reads while capture and GUI access the same file concurrently.
            invalid_hdr = struct.pack("<QdQII", 0, t_unix, int(buf_idx), int(chmask), 0)

            try:
                self._fh.seek(off)
                self._fh.write(invalid_hdr)
                self._fh.write(a.tobytes(order="C"))
                self._fh.write(b.tobytes(order="C"))
                self._fh.flush()
                self._fh.seek(off)
                self._fh.write(rec_hdr)
                # also update header write_seq for reader to know latest
                self._fh.seek(24)  # write_seq offset in header
                self._fh.write(struct.pack("<Q", seq))
                self._fh.flush()
            except IOError:
                # Retry once on transient IO error
                try:
                    self._fh.close()
                except Exception:
                    pass
                try:
                    self._fh = open(self.path, "r+b", buffering=0)
                    self._fh.seek(off)
                    self._fh.write(rec_hdr)
                    self._fh.write(a.tobytes(order="C"))
                    self._fh.write(b.tobytes(order="C"))
                    self._fh.flush()
                    self._fh.seek(24)
                    self._fh.write(struct.pack("<Q", seq))
                    self._fh.flush()
                except Exception:
                    pass
            except Exception:
                pass


class StatusNotifier:
    """Heartbeat file + periodic status email, non-blocking."""
    def __init__(self, cfg: dict, data_dir: Path):
        self.cfg = cfg
        self.data_dir = data_dir
        self.notify = (cfg.get("notify", {}) or {})
        self.hb_seconds = max(0.05, float(self.notify.get("heartbeat_seconds", 0.25)))
        self.email_seconds = float(self.notify.get("interval_minutes", 120)) * 60.0
        self._last_hb = 0.0
        self._last_email = 0.0
        self._lock = threading.Lock()
        self._latest: dict = {}
        self._seq = 0

    def update(self, **kw) -> None:
        with self._lock:
            self._latest.update(kw)

    def _snapshot(self):
        self._seq += 1
        self._latest["status_seq"] = self._seq
        self._latest["status_unix"] = time.time()
        return dict(self._latest)

    def emit_now(self) -> None:
        with self._lock:
            snap = self._snapshot()
        try:
            _write_json_atomic(self.data_dir / "status" / "cappy_status.json", snap)
        except Exception:
            pass

    def maybe_emit(self) -> None:
        now = time.time()
        if now - self._last_hb >= self.hb_seconds:
            self._last_hb = now
            with self._lock:
                snap = self._snapshot()
            try:
                _write_json_atomic(self.data_dir / "status" / "cappy_status.json", snap)
            except Exception:
                pass

        if bool(self.notify.get("enabled", False)) and (now - self._last_email >= self.email_seconds):
            self._last_email = now
            try:
                to_addr = str(self.notify.get("to", "")).strip()
                if to_addr:
                    with self._lock:
                        snap = dict(self._latest)
                    subject_prefix = str(self.notify.get("subject_prefix", "[CAPPY]")).strip() or "[CAPPY]"
                    state = str(snap.get("state", "status")).strip() or "status"
                    subject = f"{subject_prefix} {state}"
                    body = json.dumps(snap, indent=2, sort_keys=True, default=str)
                    _send_status_email(cfg=self.cfg, subject=subject, body=body)
            except Exception:
                pass

class ParquetRollingWriter:
    """
    Parquet writer that rolls files by time and stores them in an hourly hierarchy.

    Directory layout (under captures/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>/<HH:00>/):
      - reduced/<prefix>_<YYYYMMDD_HHMM>.parquet
    """
    def __init__(self, day_dir: Path, prefix: str, schema: _lazy_arrow()[0].Schema, rollover_minutes: int):
        self.day_dir = day_dir
        self.prefix = prefix
        self.schema = schema
        self.rollover_minutes = max(1, int(rollover_minutes))
        _ensure_dir(day_dir)
        self._writer: Optional[_lazy_arrow()[1].ParquetWriter] = None
        self._open_key: Optional[str] = None  # YYYYMMDD_HHMM
        self._open_hour: Optional[str] = None  # HH:00

    def _hour_dir(self, ts_ns: int) -> Tuple[str, Path]:
        dt = datetime.fromtimestamp(ts_ns / 1e9)
        hour = dt.strftime("%H:00")
        p = self.day_dir / hour / "reduced"
        _ensure_dir(p)
        return hour, p

    def _minute_key(self, ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9).strftime("%Y%m%d_%H%M")

    def _open_new(self, ts_ns: int, key: str) -> None:
        hour, base = self._hour_dir(ts_ns)
        path = base / f"{self.prefix}_{key}.parquet"
        self._writer = _lazy_arrow()[1].ParquetWriter(path, self.schema, compression="snappy", use_dictionary=True)
        self._open_key = key
        self._open_hour = hour

    def _maybe_roll(self, ts_ns: int) -> None:
        key = self._minute_key(ts_ns)
        hour, _ = self._hour_dir(ts_ns)

        if self._open_key is None or self._writer is None or self._open_hour is None:
            self._open_new(ts_ns, key)
            return

        # roll if hour changed
        if hour != self._open_hour:
            self.close()
            self._open_new(ts_ns, key)
            return

        # roll if minutes exceeded rollover window
        t0 = datetime.strptime(self._open_key, "%Y%m%d_%H%M")
        t1 = datetime.strptime(key, "%Y%m%d_%H%M")
        if (t1 - t0).total_seconds() >= 60 * self.rollover_minutes:
            self.close()
            self._open_new(ts_ns, key)

    def write_rows(self, rows: List[Dict[str, Any]], ts_ns: int) -> int:
        if not rows:
            return 0
        self._maybe_roll(ts_ns)
        assert self._writer is not None

        df = _lazy_pandas().DataFrame(rows)
        for name in self.schema.names:
            if name not in df.columns:
                df[name] = np.nan
        df = df[self.schema.names]
        tbl = _lazy_arrow()[0].Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self._writer.write_table(tbl)
        return int(tbl.num_rows)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None
                self._open_key = None
                self._open_hour = None


def _normalize_waveform_codec(codec: Any) -> str:
    c = str(codec or "none").strip().lower()
    if c in WAVE_ARCHIVE_CODEC_OPTIONS:
        return c
    return "none"


_WAVE_CODEC_MAGIC = b"CWZ1"
_WAVE_CODEC_HEADER_FMT = "<4sBBIf"
_WAVE_CODEC_HEADER_SIZE = struct.calcsize(_WAVE_CODEC_HEADER_FMT)


def _encode_wave_payload(wave: np.ndarray, codec: str, zlib_level: int = 3,
                         quant_bits: int = 12) -> bytes:
    """Encode a float32 waveform with the chosen codec.

    Codecs:
      none          – raw float32 bytes (no compression)
      f32_zlib      – zlib-compressed float32 (lossless, ~2:1 typical)
      delta_i16_zlib – quantize to int16, delta-code, zlib  (~4-8:1 typical)
    """
    wf = np.asarray(wave, dtype=np.float32)
    raw = wf.tobytes(order="C")
    c = _normalize_waveform_codec(codec)

    if c == "f32_zlib":
        try:
            return zlib.compress(raw, _clamp_int(zlib_level, 1, 9, 3))
        except Exception:
            return raw

    if c == "delta_i16_zlib":
        try:
            qbits = int(max(8, min(14, int(quant_bits))))
            qmax = (1 << (qbits - 1)) - 1
            max_abs = float(np.max(np.abs(wf))) if wf.size else 0.0
            scale = (max_abs / qmax) if (max_abs > 0.0 and qmax > 0) else 1.0
            q = np.clip(np.rint(wf / scale), -qmax, qmax).astype(np.int16, copy=False)
            deltas = np.empty_like(q, dtype=np.int16)
            if q.size:
                deltas[0] = q[0]
                if q.size > 1:
                    deltas[1:] = (q[1:].astype(np.int32) - q[:-1].astype(np.int32)).astype(np.int16)
            comp = zlib.compress(deltas.tobytes(order="C"),
                                 _clamp_int(zlib_level, 1, 9, 3))
            header = struct.pack(_WAVE_CODEC_HEADER_FMT,
                                 _WAVE_CODEC_MAGIC, 1, qbits, int(q.size), float(scale))
            return header + comp
        except Exception:
            return raw

    # codec == "none"
    return raw


def _decode_wave_payload(payload: bytes, n_samples: int, codec: Any = None) -> np.ndarray:
    """Decode waveform payload.  Detects CWZ1 header for delta_i16_zlib regardless of
    the *codec* hint so archives from either v1.0 or v1.3 are always readable."""
    # Try CWZ1 (delta_i16_zlib) first – works even when codec hint is wrong/missing
    if len(payload) >= _WAVE_CODEC_HEADER_SIZE and payload[:4] == _WAVE_CODEC_MAGIC:
        try:
            _, _ver, _qb, nq, sc = struct.unpack(
                _WAVE_CODEC_HEADER_FMT, payload[:_WAVE_CODEC_HEADER_SIZE])
            d = np.frombuffer(zlib.decompress(payload[_WAVE_CODEC_HEADER_SIZE:]),
                              dtype=np.int16, count=int(nq))
            out = np.cumsum(d.astype(np.int32), dtype=np.int32).astype(np.float32)
            out *= np.float32(sc)
            return out[:n_samples] if n_samples > 0 else out
        except Exception:
            pass

    codec_norm = _normalize_waveform_codec(codec)
    data = payload
    if codec_norm == "f32_zlib":
        try:
            data = zlib.decompress(payload)
        except Exception:
            data = payload
    elif codec_norm == "none":
        # Also try zlib decompress as a fallback for mis-labeled f32_zlib
        if len(payload) >= 2 and payload[0:2] in (b'\x78\x01', b'\x78\x5e', b'\x78\x9c', b'\x78\xda'):
            try:
                data = zlib.decompress(payload)
            except Exception:
                data = payload
    arr = np.frombuffer(data, dtype=np.float32)
    if n_samples > 0:
        return arr[:n_samples]
    return arr


class WaveBinSqliteStore:
    """
    Store waveform snippets as:
      - binary channel-separated waveform payloads (raw float32 or zlib-compressed float32)
        appended to time-rolled .bin files inside hourly folders
      - SQLite index (WAL) at day_dir/index/snips_<session>.sqlite pointing to file+offset per channel

    Directory layout (under captures/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>/):
      - index/snips_<session>.sqlite
      - <HH:00>/waveforms/A_snips_<session>_<YYYYMMDD_HHMM>.bin
      - <HH:00>/waveforms/B_snips_<session>_<YYYYMMDD_HHMM>.bin
    """
    def __init__(
        self,
        day_dir: Path,
        session_id: str,
        rollover_minutes: int,
        commit_every: int,
        waveform_codec: str = "none",
        waveform_zlib_level: int = 3,
        waveform_quant_bits: int = 12,
    ):
        self.day_dir = day_dir
        self.session_id = session_id
        self.rollover_minutes = max(1, int(rollover_minutes))
        self.commit_every = max(1, int(commit_every))
        self.waveform_codec = _normalize_waveform_codec(waveform_codec)
        self.waveform_zlib_level = _clamp_int(waveform_zlib_level, 1, 9, 3)
        self.waveform_quant_bits = int(max(8, min(14, int(waveform_quant_bits))))
        _ensure_dir(day_dir)

        idx_dir = day_dir / "index"
        _ensure_dir(idx_dir)
        self.db_path = idx_dir / f"snips_{session_id}.sqlite"

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS snips (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT,
              timestamp_ns INTEGER,
              buffer_index INTEGER,
              record_in_buffer INTEGER,
              record_global INTEGER,
              channels_mask TEXT,
              sample_rate_hz REAL,
              n_samples INTEGER,

              -- legacy combined payload (kept for backward compatibility)
              n_channels INTEGER,
              file TEXT,
              offset_bytes INTEGER,
              nbytes INTEGER,

              -- channel-separated payload (preferred)
              file_A TEXT,
              offset_A INTEGER,
              nbytes_A INTEGER,
              codec_A TEXT,
              file_B TEXT,
              offset_B INTEGER,
              nbytes_B INTEGER,
              codec_B TEXT,

              area_A_Vs REAL,
              peak_A_V REAL,
              baseline_A_V REAL,
              area_B_Vs REAL,
              peak_B_V REAL,
              baseline_B_V REAL
            );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_snips_session_time ON snips(session_id, timestamp_ns);")
        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_snips_record_global ON snips(session_id, record_global);")
        self.conn.commit()

        # Migration: add new columns if the DB already existed without them
        try:
            cols = {r[1] for r in self.conn.execute("PRAGMA table_info(snips);").fetchall()}
            need = {
                "file_A": "ALTER TABLE snips ADD COLUMN file_A TEXT",
                "offset_A": "ALTER TABLE snips ADD COLUMN offset_A INTEGER",
                "nbytes_A": "ALTER TABLE snips ADD COLUMN nbytes_A INTEGER",
                "codec_A": "ALTER TABLE snips ADD COLUMN codec_A TEXT",
                "file_B": "ALTER TABLE snips ADD COLUMN file_B TEXT",
                "offset_B": "ALTER TABLE snips ADD COLUMN offset_B INTEGER",
                "nbytes_B": "ALTER TABLE snips ADD COLUMN nbytes_B INTEGER",
                "codec_B": "ALTER TABLE snips ADD COLUMN codec_B TEXT",
                "baseline_A_V": "ALTER TABLE snips ADD COLUMN baseline_A_V REAL",
                "baseline_B_V": "ALTER TABLE snips ADD COLUMN baseline_B_V REAL",
            }
            for c, ddl in need.items():
                if c not in cols:
                    self.conn.execute(ddl)
            self.conn.commit()
        except Exception:
            pass

        self._fhA = None
        self._fhB = None
        self._bin_key = None
        self._bin_hour = None
        self._rows_since_commit = 0

    def _hour_dir(self, ts_ns: int) -> Tuple[str, Path]:
        dt = datetime.fromtimestamp(ts_ns / 1e9)
        hour = dt.strftime("%H:00")
        p = self.day_dir / hour / "waveforms"
        _ensure_dir(p)
        return hour, p

    def _minute_key(self, ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9).strftime("%Y%m%d_%H%M")

    def _open_bins(self, ts_ns: int, key: str):
        hour, base = self._hour_dir(ts_ns)
        pathA = base / f"A_snips_{self.session_id}_{key}.bin"
        pathB = base / f"B_snips_{self.session_id}_{key}.bin"
        # Use 64 KiB write buffer to reduce syscalls and IO errors
        self._fhA = open(pathA, "ab", buffering=65536)
        self._fhB = open(pathB, "ab", buffering=65536)
        self._bin_key = key
        self._bin_hour = hour

    def _maybe_roll(self, ts_ns: int):
        key = self._minute_key(ts_ns)
        hour, _ = self._hour_dir(ts_ns)

        if self._bin_key is None or self._fhA is None or self._fhB is None or self._bin_hour is None:
            self._open_bins(ts_ns, key)
            return

        # roll on hour boundary
        if hour != self._bin_hour:
            self.close_bin()
            self._open_bins(ts_ns, key)
            return

        # roll on rollover window
        try:
            t0 = datetime.strptime(self._bin_key, "%Y%m%d_%H%M")
            t1 = datetime.strptime(key, "%Y%m%d_%H%M")
            if (t1 - t0).total_seconds() >= 60 * self.rollover_minutes:
                self.close_bin()
                self._open_bins(ts_ns, key)
        except Exception:
            # if parsing fails, reopen
            self.close_bin()
            self._open_bins(ts_ns, key)

    def append(self, *, ts_ns: int, buffer_index: int, record_in_buffer: int, record_global: int,
               channels_mask: str, sample_rate_hz: float, wfA_V: np.ndarray, wfB_V: Optional[np.ndarray],
               area_A_Vs: float, peak_A_V: float, area_B_Vs: float, peak_B_V: float,
               baseline_A_V: float = 0.0, baseline_B_V: float = 0.0):
        self._maybe_roll(ts_ns)
        assert self._fhA is not None and self._fhB is not None

        wfA = wfA_V.astype(np.float32, copy=False)
        payloadA = _encode_wave_payload(
            wfA,
            self.waveform_codec,
            zlib_level=self.waveform_zlib_level,
            quant_bits=self.waveform_quant_bits,
        )
        codecA = self.waveform_codec

        offA = self._fhA.tell()
        self._fhA.write(payloadA)

        if wfB_V is not None:
            wfB = wfB_V.astype(np.float32, copy=False)
            payloadB = _encode_wave_payload(
                wfB,
                self.waveform_codec,
                zlib_level=self.waveform_zlib_level,
                quant_bits=self.waveform_quant_bits,
            )
            codecB = self.waveform_codec
            offB = self._fhB.tell()
            self._fhB.write(payloadB)
            n_channels = 2
        else:
            payloadB = b""
            codecB = None
            offB = 0
            n_channels = 1

        # Store file paths relative to day_dir so the archive can relocate the root
        try:
            fileA = str(Path(self._fhA.name).relative_to(self.day_dir))
        except Exception:
            fileA = str(Path(self._fhA.name).name)
        try:
            fileB = str(Path(self._fhB.name).relative_to(self.day_dir))
        except Exception:
            fileB = str(Path(self._fhB.name).name)

        # Legacy combined fields: keep pointing at A only (so old viewers still show A)
        file_legacy = fileA
        off_legacy = offA
        nbytes_legacy = len(payloadA)

        self.conn.execute(
            "INSERT OR IGNORE INTO snips(session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
            "file,offset_bytes,nbytes,file_A,offset_A,nbytes_A,codec_A,file_B,offset_B,nbytes_B,codec_B,area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.session_id, int(ts_ns), int(buffer_index), int(record_in_buffer), int(record_global),
             str(channels_mask), float(sample_rate_hz), int(wfA.shape[0]), int(n_channels),
             file_legacy, int(off_legacy), int(nbytes_legacy),
             fileA, int(offA), int(len(payloadA)), codecA,
             (fileB if wfB_V is not None else None),
             (int(offB) if wfB_V is not None else None),
             (int(len(payloadB)) if wfB_V is not None else None),
             codecB,
             float(area_A_Vs), float(peak_A_V), float(baseline_A_V), float(area_B_Vs), float(peak_B_V), float(baseline_B_V))
        )

        self._rows_since_commit += 1
        if self._rows_since_commit >= self.commit_every:
            self.conn.commit()
            self._rows_since_commit = 0

    def close_bin(self):
        for fh_name in ("_fhA", "_fhB"):
            fh = getattr(self, fh_name, None)
            if fh is not None:
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                except Exception:
                    pass
                try:
                    fh.close()
                except Exception:
                    pass
                setattr(self, fh_name, None)
        self._bin_key = None
        self._bin_hour = None

    def close(self):
        try:
            if self._rows_since_commit:
                self.conn.commit()
                self._rows_since_commit = 0
        except Exception:
            pass
        try:
            self.close_bin()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def load_waveforms(self, row: _lazy_pandas().Series, day_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load channel-separated waveforms if available; otherwise fall back to legacy combined payload.
        """
        n_samples = int(row.get("n_samples", 0))

        # Prefer channel-separated fields when present and non-null
        fileA = row.get("file_A", None)
        offA = row.get("offset_A", None)
        nbytesA = row.get("nbytes_A", None)
        codecA = row.get("codec_A", "none")

        if isinstance(fileA, str) and fileA and _lazy_pandas().notna(offA) and _lazy_pandas().notna(nbytesA):
            binA = day_dir / str(fileA)
            with open(binA, "rb") as fh:
                fh.seek(int(offA))
                payloadA = fh.read(int(nbytesA))
            a = _decode_wave_payload(payloadA, n_samples, codecA)

            fileB = row.get("file_B", None)
            offB = row.get("offset_B", None)
            nbytesB = row.get("nbytes_B", None)
            codecB = row.get("codec_B", "none")
            if isinstance(fileB, str) and fileB and _lazy_pandas().notna(offB) and _lazy_pandas().notna(nbytesB):
                binB = day_dir / str(fileB)
                with open(binB, "rb") as fh:
                    fh.seek(int(offB))
                    payloadB = fh.read(int(nbytesB))
                b = _decode_wave_payload(payloadB, n_samples, codecB)
                return a, b
            return a, None

        # Legacy fallback
        bin_path = day_dir / str(row["file"])
        offset = int(row["offset_bytes"])
        nbytes = int(row["nbytes"])
        n_channels = int(row.get("n_channels", 1))
        with open(bin_path, "rb") as fh:
            fh.seek(offset)
            payload = fh.read(nbytes)
        legacy_codec = row.get("codec_A", "none")
        arr = _decode_wave_payload(payload, n_samples * max(1, n_channels), legacy_codec)
        if n_channels == 2:
            return arr[:n_samples], arr[n_samples:2*n_samples]
        return arr[:n_samples], None


def load_waveforms_from_row(row: _lazy_pandas().Series, day_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read waveform payloads referenced by a snip row WITHOUT opening/creating any SQLite DB."""
    n_samples = int(row.get("n_samples", 0))

    # Prefer channel-separated fields when present and non-null
    fileA = row.get("file_A", None)
    offA = row.get("offset_A", None)
    nbytesA = row.get("nbytes_A", None)
    codecA = row.get("codec_A", "none")

    if isinstance(fileA, str) and fileA and _lazy_pandas().notna(offA) and _lazy_pandas().notna(nbytesA):
        binA = day_dir / str(fileA)
        with open(binA, "rb") as fh:
            fh.seek(int(offA))
            payloadA = fh.read(int(nbytesA))
        a = _decode_wave_payload(payloadA, n_samples, codecA)

        fileB = row.get("file_B", None)
        offB = row.get("offset_B", None)
        nbytesB = row.get("nbytes_B", None)
        codecB = row.get("codec_B", "none")
        if isinstance(fileB, str) and fileB and _lazy_pandas().notna(offB) and _lazy_pandas().notna(nbytesB):
            binB = day_dir / str(fileB)
            with open(binB, "rb") as fh:
                fh.seek(int(offB))
                payloadB = fh.read(int(nbytesB))
            b = _decode_wave_payload(payloadB, n_samples, codecB)
            return a, b
        return a, None

    # Legacy fallback
    bin_path = day_dir / str(row["file"])
    offset = int(row["offset_bytes"])
    nbytes = int(row["nbytes"])
    n_channels = int(row.get("n_channels", 1))
    with open(bin_path, "rb") as fh:
        fh.seek(offset)
        payload = fh.read(nbytes)
    legacy_codec = row.get("codec_A", "none")
    arr = _decode_wave_payload(payload, n_samples * max(1, n_channels), legacy_codec)
    if n_channels == 2:
        return arr[:n_samples], arr[n_samples:2*n_samples]
    return arr[:n_samples], None



class CappyArchive:

    def __init__(self, data_dir: Path, rollover_minutes: int, flush_every_records: int,
                 session_rotate_hours: float, sqlite_commit_every_snips: int, flush_every_seconds: float = 10.0,
                 waveform_codec: str = "none", waveform_zlib_level: int = 3,
                 waveform_quant_bits: int = 12):
        self.data_dir = data_dir
        _ensure_dir(self.data_dir)
        self.captures = data_dir / "captures"
        _ensure_dir(self.captures)
        self.rollover_minutes = int(rollover_minutes)
        self.flush_every_records = int(flush_every_records)
        self.flush_every_seconds = float(flush_every_seconds or 0.0)
        self._last_flush_unix = 0.0
        self.session_rotate_hours = float(session_rotate_hours or 0.0)
        self.sqlite_commit_every_snips = int(sqlite_commit_every_snips)
        self.waveform_codec = _normalize_waveform_codec(waveform_codec)
        self.waveform_zlib_level = _clamp_int(waveform_zlib_level, 1, 9, 3)
        self.waveform_quant_bits = _clamp_int(waveform_quant_bits, 8, 14, 12)
        self.session_id = ""
        self.day_dir: Optional[Path] = None
        self.reduced_writer: Optional[ParquetRollingWriter] = None
        self.wave_store: Optional[WaveBinSqliteStore] = None
        self._reduced_buf: List[Dict[str, Any]] = []
        self._first_ts = 0
        self._last_ts = 0
        self._n_reduced = 0
        self._n_snips = 0
        self.session_start_ns = 0

    def start(self, tag: str, channels_mask: str) -> str:
        sid = datetime.now().strftime("%Y%m%d_%H%M%S")
        if tag:
            sid = f"{sid}_{tag}"
        self.session_id = sid
        self.session_start_ns = time.time_ns()

        # Hierarchical capture layout:
        # captures/YYYY/YYYY-MM/YYYY-MM-DD/HH:00/{reduced,waveforms}/...
        now = datetime.now()
        year = now.strftime("%Y")
        ym = now.strftime("%Y-%m")
        ymd = now.strftime("%Y-%m-%d")
        self.day_dir = self.captures / year / ym / ymd
        _ensure_dir(self.day_dir)

        idx_dir = self.day_dir / "index"
        _ensure_dir(idx_dir)

        self.reduced_writer = ParquetRollingWriter(self.day_dir, f"reduced_{sid}", REDUCED_SCHEMA, self.rollover_minutes)
        self.wave_store = WaveBinSqliteStore(
            self.day_dir,
            sid,
            self.rollover_minutes,
            self.sqlite_commit_every_snips,
            waveform_codec=self.waveform_codec,
            waveform_zlib_level=self.waveform_zlib_level,
            waveform_quant_bits=self.waveform_quant_bits,
        )

        _atomic_write_text(idx_dir / f"session_{sid}.txt", f"CAPPY v1.3 session {sid}\nchannels={channels_mask}\n")
        print(f"[CAPPY] Started session {sid} in {self.day_dir}")
        return sid

    def should_rotate(self) -> bool:
        if self.session_rotate_hours <= 0:
            return False
        return (time.time_ns() - self.session_start_ns) >= int(self.session_rotate_hours * 3600 * 1e9)

    def _touch(self, ts: int) -> None:
        if self._first_ts == 0:
            self._first_ts = ts
        self._last_ts = max(self._last_ts, ts)

    def append_reduced(self, rows: List[Dict[str, Any]], ts: int) -> None:
        if not rows:
            return
        self._reduced_buf.extend(rows)
        self._touch(ts)
        now = time.time()
        if self.flush_every_seconds > 0 and (now - self._last_flush_unix) >= self.flush_every_seconds:
            self._last_flush_unix = now
            self.flush_reduced(ts)
            return
        if len(self._reduced_buf) >= self.flush_every_records:
            self.flush_reduced(ts)

    def flush_reduced(self, ts: int) -> None:
        if not self._reduced_buf:
            return
        assert self.reduced_writer is not None
        self._n_reduced += self.reduced_writer.write_rows(self._reduced_buf, ts)
        self._reduced_buf.clear()

    def append_snip(self, **kw) -> None:
        assert self.wave_store is not None
        self.wave_store.append(**kw)
        self._n_snips += 1

    def finalize(self, channels_mask: str) -> None:
        if not self.day_dir:
            return
        ts = self._last_ts or time.time_ns()
        self.flush_reduced(ts)
        if self.reduced_writer:
            self.reduced_writer.close()
        if self.wave_store:
            self.wave_store.close()
        idx = self.day_dir / "session_index.parquet"
        row = dict(
            session_id=self.session_id,
            date=self.day_dir.name,
            first_timestamp_ns=int(self._first_ts or ts),
            last_timestamp_ns=int(self._last_ts or ts),
            reduced_rows=int(self._n_reduced),
            waveform_snips=int(self._n_snips),
            channels_mask=str(channels_mask),
        )
        if idx.exists():
            df = _lazy_pandas().read_parquet(idx)
            df = _lazy_pandas().concat([df, _lazy_pandas().DataFrame([row])], ignore_index=True)
        else:
            df = _lazy_pandas().DataFrame([row])
        df = df.drop_duplicates(subset=["session_id"], keep="last")
        _lazy_arrow()[1].write_table(_lazy_arrow()[0].Table.from_pandas(df, schema=SESSION_INDEX_SCHEMA, preserve_index=False), idx, compression="snappy")
        print(f"[CAPPY] Finalized session {self.session_id} reduced={self._n_reduced} snips={self._n_snips}")

def reduce_u16(raw: np.ndarray, sr_hz: float, b0: int, b1: int, g0: int, g1: int, vpp: float):
    """
    Reduce raw uint16 waveforms into baseline (V), peak (V) and gated integral (V·s).

    This version converts ADC codes -> volts using the configured input range (Vpp),
    then does baseline subtraction in volts, then integrates (sum * dt).
    """
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16, copy=False)

    n = int(raw.shape[1])
    b0 = max(0, min(int(b0), n))
    b1 = max(0, min(int(b1), n))
    g0 = max(0, min(int(g0), n))
    g1 = max(0, min(int(g1), n))

    if b1 <= b0:
        b0, b1 = 0, min(1, n)
    if g1 <= g0:
        g0, g1 = 0, min(1, n)

    V = _codes_to_volts_u16(raw, vpp=vpp)  # float32 volts
    baseline_V = V[:, b0:b1].mean(axis=1, dtype=np.float32).astype(np.float64)

    gate = V[:, g0:g1].astype(np.float64, copy=False)
    gate_bs = gate - baseline_V[:, None]

    dt = 1.0 / float(sr_hz)
    area = gate_bs.sum(axis=1) * dt
    peak = gate_bs.max(axis=1)
    return area.astype(np.float64), peak.astype(np.float64), baseline_V.astype(np.float64)


@dataclass
class WfPolicy:
    mode: str
    every_n: int
    thr_area: float
    thr_peak: float
    max_per_sec: int

    def __post_init__(self):
        self.mode = (self.mode or "every_n").lower()
        self.every_n = max(1, int(self.every_n))
        self.max_per_sec = max(1, int(self.max_per_sec))
        self._sec = int(time.time())
        self._count = 0

    def _ok(self) -> bool:
        s = int(time.time())
        if s != self._sec:
            self._sec = s
            self._count = 0
        if self._count >= self.max_per_sec:
            return False
        self._count += 1
        return True

    def want(self, rec_global: int, area: float, peak: float) -> bool:
        every = (self.mode in ("every_n", "both")) and (rec_global % self.every_n == 0)
        event = (self.mode in ("event", "both")) and (
            (self.thr_area > 0 and abs(area) >= self.thr_area) or (self.thr_peak > 0 and abs(peak) >= self.thr_peak)
        )
        return (every or event) and self._ok()

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        _atomic_write_text(path, DEFAULT_YAML)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("YAML must be a dict.")
    return cfg


def validate_and_normalize_capture_cfg(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Normalize board/capture-critical settings into sane bounds.
    Returns (cfg, warnings, errors).
    """
    out = dict(cfg)
    warnings: List[str] = []
    errors: List[str] = []

    clock = out.setdefault("clock", {}) or {}
    acq = out.setdefault("acquisition", {}) or {}
    timing_obj = out.get("timing", {})
    timing = timing_obj if isinstance(timing_obj, dict) else {}
    trig = out.setdefault("trigger", {}) or {}
    waves = out.setdefault("waveforms", {}) or {}
    runtime = out.setdefault("runtime", {}) or {}
    storage = out.setdefault("storage", {}) or {}
    live = out.setdefault("live", {}) or {}

    out["clock"] = clock
    out["acquisition"] = acq
    out["trigger"] = trig
    out["waveforms"] = waves
    out["runtime"] = runtime
    out["storage"] = storage
    out["live"] = live

    source = str(clock.get("source", "INTERNAL_CLOCK")).strip() or "INTERNAL_CLOCK"
    sr_msps = _clamp_float(clock.get("sample_rate_msps", 250.0), 1.0, 1000.0, 250.0)
    if source == "INTERNAL_CLOCK":
        nearest = min(SAMPLE_RATE_OPTIONS_MSPS, key=lambda x: abs(x - sr_msps))
        if abs(nearest - sr_msps) > 1e-9:
            warnings.append(
                f"clock.sample_rate_msps={sr_msps:g} is not a standard internal rate; using {nearest:g} MS/s."
            )
        sr_msps = float(nearest)
    clock["source"] = source
    clock["sample_rate_msps"] = sr_msps

    max_spr = 8_000_000
    pre = _clamp_int(acq.get("pre_trigger_samples", 0), 0, max_spr - 16, 0)
    post = _clamp_int(acq.get("post_trigger_samples", 256), 16, max_spr, 256)
    spr = pre + post

    # Support explicit acquisition.samples_per_record and keep post-trigger derived from it.
    spr_cfg = _clamp_int(acq.get("samples_per_record", spr), 16, max_spr, spr)
    if spr_cfg != spr:
        spr = spr_cfg
        if spr < (pre + 16):
            warnings.append(
                f"acquisition.samples_per_record={spr_cfg} is too small for pre_trigger_samples={pre}; using {pre + 16}."
            )
            spr = pre + 16
        post = max(16, spr - pre)

    if spr > max_spr:
        warnings.append(f"samples_per_record={spr} exceeds limit {max_spr}; clamping.")
        spr = max_spr
        post = max(16, spr - pre)

    if spr <= 0:
        errors.append("samples_per_record must be > 0.")
    acq["pre_trigger_samples"] = pre
    acq["post_trigger_samples"] = post
    acq["samples_per_record"] = int(spr)

    rpb = _clamp_int(acq.get("records_per_buffer", 128), 1, 100_000, 128)
    bufN = _clamp_int(acq.get("buffers_allocated", 16), 2, 4096, 16)
    if bufN < 4:
        warnings.append("buffers_allocated < 4 can increase overflow risk; 8-32 is recommended.")
    wait_timeout_ms = _clamp_int(acq.get("wait_timeout_ms", 1000), 10, 120_000, 1000)
    bpa = _to_int(acq.get("buffers_per_acquisition", 0), 0)
    if bpa < 0:
        bpa = 0
    acq["records_per_buffer"] = rpb
    acq["buffers_allocated"] = bufN
    acq["wait_timeout_ms"] = wait_timeout_ms
    acq["buffers_per_acquisition"] = bpa

    ch_expr = str(acq.get("channels_mask", "CHANNEL_A")).strip() or "CHANNEL_A"
    ch_mask = channels_from_mask_expr(ch_expr)
    if (ch_mask & 0x3) == 0:
        errors.append("acquisition.channels_mask must include CHANNEL_A and/or CHANNEL_B.")
        ch_expr = "CHANNEL_A"
        ch_mask = channels_from_mask_expr(ch_expr)
    acq["channels_mask"] = channels_mask_to_str(ch_mask)

    # Legacy key; spacing is fixed to samples_per_record now.
    timing.pop("bunch_spacing_samples", None)
    if "timing" in out:
        if timing:
            out["timing"] = timing
        else:
            out.pop("timing", None)

    levelJ = _clamp_int(trig.get("levelJ", 128), 0, 255, 128)
    levelK = _clamp_int(trig.get("levelK", 128), 0, 255, 128)
    trig["levelJ"] = levelJ
    trig["levelK"] = levelK
    trig["timeout_ms"] = _clamp_int(trig.get("timeout_ms", 0), 0, 2_147_483_647, 0)
    trig["timeout_pause_s"] = _clamp_float(trig.get("timeout_pause_s", 0.0), 0.0, 3600.0, 0.0)
    trig["trigger_delay_us"] = _clamp_float(trig.get("trigger_delay_us", 0.0), 0.0, 100_000.0, 0.0)
    trig["allow_autotrigger_with_external"] = _to_bool(
        trig.get("allow_autotrigger_with_external", False), False
    )
    if _to_int(trig.get("timeout_ms", 0), 0) > 0:
        warnings.append(
            "trigger.timeout_ms > 0 enables auto-trigger mode; this can create non-hardware trigger records."
        )

    waves["enable"] = _to_bool(waves.get("enable", True), True)
    waves["mode"] = str(waves.get("mode", "every_n") or "every_n").strip().lower()
    if waves["mode"] not in {"every_n", "event", "both"}:
        waves["mode"] = "every_n"
    waves["every_n"] = _clamp_int(waves.get("every_n", 16), 1, 10_000_000, 16)
    waves["max_waveforms_per_sec"] = _clamp_int(
        waves.get("max_waveforms_per_sec", 120), 1, 50_000, 120
    )
    codec_raw = str(waves.get("archive_codec", "none") or "none").strip().lower()
    codec = _normalize_waveform_codec(codec_raw)
    if codec != codec_raw:
        warnings.append(f"waveforms.archive_codec={codec_raw!r} is invalid; using 'none'.")
    waves["archive_codec"] = codec
    waves["archive_quant_bits"] = _clamp_int(waves.get("archive_quant_bits", 12), 8, 14, 12)
    waves["archive_zlib_level"] = _clamp_int(waves.get("archive_zlib_level", 3), 1, 9, 3)

    runtime["rearm_if_no_trigger_s"] = _clamp_int(runtime.get("rearm_if_no_trigger_s", 300), 0, 86_400, 300)
    runtime["rearm_cooldown_s"] = _clamp_int(runtime.get("rearm_cooldown_s", 30), 0, 86_400, 30)
    runtime["max_rearms_per_hour"] = _clamp_int(runtime.get("max_rearms_per_hour", 12), 1, 100_000, 12)
    profile = str(runtime.get("profile", "Custom") or "Custom").strip()
    profile_lut = {name.lower(): name for name in RUNTIME_PROFILE_OPTIONS}
    runtime["profile"] = profile_lut.get(profile.lower(), "Custom")

    storage["flush_every_records"] = _clamp_int(storage.get("flush_every_records", 20000), 1, 50_000_000, 20000)
    storage["flush_every_seconds"] = _clamp_float(storage.get("flush_every_seconds", 2.0), 0.0, 86_400.0, 2.0)
    storage["sqlite_commit_every_snips"] = _clamp_int(
        storage.get("sqlite_commit_every_snips", 200), 1, 10_000_000, 200
    )

    live["ring_slots"] = _clamp_int(live.get("ring_slots", 4096), 16, 1_000_000, 4096)
    live["ring_points"] = _clamp_int(live.get("ring_points", 4096), 32, 65536, 4096)
    live["waveform_every_n_buffers"] = _clamp_int(
        live.get("waveform_every_n_buffers", 1), 1, WAVEFORM_EVERY_N_MAX, 1
    )
    live["stream_window_points"] = _clamp_int(live.get("stream_window_points", 100000), 256, 5_000_000, 100000)
    live["stream_window_seconds"] = _clamp_float(live.get("stream_window_seconds", 2.0), 0.25, 120.0, 2.0)
    live["max_waveforms_per_tick"] = _clamp_int(live.get("max_waveforms_per_tick", 12), 1, 2000, 12)
    live["ui_fps"] = _clamp_float(live.get("ui_fps", 6.0), 1.0, LIVE_UI_FPS_MAX, 6.0)
    live["show_channel_b"] = _to_bool(live.get("show_channel_b", False), False)
    mode = str(live.get("preview_mode", "archive_match")).strip().lower()
    if mode not in {"archive_match", "record0"}:
        mode = "archive_match"
    live["preview_mode"] = mode

    bytes_per_sample = 2
    ch_count = infer_channel_count_from_mask(ch_mask)
    bytes_per_buffer = bytes_per_sample * spr * rpb * max(1, ch_count)
    total_dma_bytes = bytes_per_buffer * bufN
    if bytes_per_buffer > 128 * 1024 * 1024:
        warnings.append(
            f"Estimated bytes_per_buffer={bytes_per_buffer/1024/1024:.1f} MiB is very large; reduce records_per_buffer or record length."
        )
    if total_dma_bytes > 1024 * 1024 * 1024:
        warnings.append(
            f"Estimated total DMA footprint={total_dma_bytes/1024/1024:.1f} MiB; this may be unstable on some hosts."
        )

    return out, warnings, errors


def _range_name_to_vpp(range_name: str, default_vpp: float = 4.0) -> float:
    """
    Convert Alazar-style range strings like 'PM_1_V' or 'PM_200_MV' to full-scale Vpp.

    NOTE: integration and plotted volts must use the actual ADC full-scale range, otherwise
    integrals/peaks will be numerically wrong.
    """
    rn = (range_name or "").strip().upper()
    rn = rn.replace("INPUT_RANGE_", "")
    if rn.startswith("PM_") and rn.endswith("_MV"):
        try:
            mv = float(rn[3:-3])
            return 2.0 * (mv / 1000.0)
        except Exception:
            return float(default_vpp)
    if rn.startswith("PM_") and rn.endswith("_V"):
        try:
            v = float(rn[3:-2])
            return 2.0 * v
        except Exception:
            return float(default_vpp)

    COMMON = {
        "PM_20_MV": 0.04,
        "PM_40_MV": 0.08,
        "PM_50_MV": 0.10,
        "PM_80_MV": 0.16,
        "PM_100_MV": 0.20,
        "PM_200_MV": 0.40,
        "PM_400_MV": 0.80,
        "PM_500_MV": 1.00,
        "PM_800_MV": 1.60,
        "PM_1_V": 2.00,
        "PM_2_V": 4.00,
        "PM_4_V": 8.00,
        "PM_5_V": 10.00,
        "PM_8_V": 16.00,
        "PM_10_V": 20.00,
        "PM_20_V": 40.00,
        "PM_40_V": 80.00,
    }
    return float(COMMON.get(rn, default_vpp))

def _codes_to_volts_u16(u16: np.ndarray, vpp: float) -> np.ndarray:
    """
    Map uint16 ADC codes to volts for a bipolar input range (mid-scale = 0 V).
    Uses 65536 codes for correct LSB size.
    """
    return (u16.astype(np.float32) - 32768.0) * (float(vpp) / 65536.0)


def channels_mask_to_str(mask: int) -> str:
    parts = []
    if mask & ats.CHANNEL_A:
        parts.append("CHANNEL_A")
    if mask & ats.CHANNEL_B:
        parts.append("CHANNEL_B")
    return "|".join(parts) if parts else "0"


def channels_from_mask_expr(expr: str) -> int:
    """
    Parse channel selection expression into a bitmask.
      - 'A' -> 1, 'B' -> 2, 'AB'/'A|B'/'A,B' -> 3
      - integer strings like '1', '2', '3', '0x3' are accepted as-is
    Defaults to 1 if empty/invalid.
    """
    if expr is None:
        return 1
    e = str(expr).strip().upper()
    if not e:
        return 1
    # numeric
    try:
        if e.startswith("0X"):
            return int(e, 16)
        if e.isdigit():
            return int(e, 10)
    except Exception:
        pass
    # symbolic
    e = e.replace(" ", "").replace("+", "|").replace(",", "|")
    if e in ("A",):
        return 1
    if e in ("B",):
        return 2
    if e in ("AB", "A|B", "B|A"):
        return 3
    # any includes
    mask = 0
    if "A" in e:
        mask |= 1
    if "B" in e:
        mask |= 2
    return mask if mask else 1

def infer_channel_count_from_mask(mask: int) -> int:
    mask = int(mask)
    return 2 if (mask & 3) == 3 else 1

def get_available_boards() -> List[Tuple[int, int, str]]:
    """
    Discover available AlazarTech boards.
    
    Returns:
        List of (system_id, board_id, description) tuples
    """
    if not ATS_AVAILABLE or ats is None:
        return []
    
    boards = []
    try:
        # Try up to 8 systems and 8 boards per system
        for sys_id in range(1, 9):
            for board_id in range(1, 9):
                try:
                    test_board = ats.Board(systemId=sys_id, boardId=board_id)
                    if hasattr(test_board, 'handle') and test_board.handle is not None:
                        try:
                            model_id = test_board.getModelID()
                            boards.append((sys_id, board_id, f"Model ID: {model_id}"))
                        except Exception:
                            boards.append((sys_id, board_id, "Unknown model"))
                except Exception:
                    pass
    except Exception:
        pass
    
    return boards

def configure_board(board: Any, cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    c = cfg.get("clock", {}) or {}
    sr_hz = float(c.get("sample_rate_msps", 250.0)) * 1e6
    source = ats_const(str(c.get("source", "INTERNAL_CLOCK")))
    edge = ats_const(str(c.get("edge", "CLOCK_EDGE_RISING")))

    RATE_MAP = {
        20e6: "SAMPLE_RATE_20MSPS",
        50e6: "SAMPLE_RATE_50MSPS",
        100e6: "SAMPLE_RATE_100MSPS",
        125e6: "SAMPLE_RATE_125MSPS",
        160e6: "SAMPLE_RATE_160MSPS",
        180e6: "SAMPLE_RATE_180MSPS",
        200e6: "SAMPLE_RATE_200MSPS",
        250e6: "SAMPLE_RATE_250MSPS",
        500e6: "SAMPLE_RATE_500MSPS",
        1000e6: "SAMPLE_RATE_1000MSPS",
    }

    if int(source) == int(getattr(ats, "INTERNAL_CLOCK", source)) and sr_hz in RATE_MAP and hasattr(ats, RATE_MAP[sr_hz]):
        board.setCaptureClock(source, getattr(ats, RATE_MAP[sr_hz]), edge, 0)
    else:
        board.setCaptureClock(source, ats.SAMPLE_RATE_USER_DEF, edge, int(sr_hz))

    # Default Vpp fallback (only used if range parsing fails)
    vpp_default = 4.0
    vpp_A = vpp_default
    vpp_B = vpp_default

    def _input_range_const(rname: str) -> Optional[int]:
        try:
            return int(ats_const('INPUT_RANGE_', rname))
        except Exception:
            return None

    def _range_candidates(primary: str) -> List[str]:
        # Start with the requested range. If the hardware rejects it, prefer
        # same-or-larger ranges first to avoid clipping, then try smaller ones.
        rr_primary = str(primary or "").strip().upper().replace("INPUT_RANGE_", "")
        catalog = [
            "PM_20_MV", "PM_40_MV", "PM_50_MV", "PM_80_MV", "PM_100_MV",
            "PM_200_MV", "PM_400_MV", "PM_500_MV", "PM_800_MV",
            "PM_1_V", "PM_2_V", "PM_4_V",
        ]
        supported = [r for r in catalog if _input_range_const(r) is not None]
        if _input_range_const(rr_primary) is not None and rr_primary not in supported:
            supported.append(rr_primary)
        if not supported:
            return [rr_primary] if rr_primary else []

        v0 = max(1e-12, float(_range_name_to_vpp(rr_primary or "PM_1_V", default_vpp=2.0)))

        def _sort_key(rname: str) -> Tuple[int, float]:
            vr = max(1e-12, float(_range_name_to_vpp(rname, default_vpp=2.0)))
            below = 1 if vr < v0 else 0
            ratio_dist = abs((vr / v0) - 1.0)
            return (below, ratio_dist)

        ordered = sorted(supported, key=_sort_key)
        out = [rr_primary] if rr_primary else []
        for rname in ordered:
            if rname != rr_primary:
                out.append(rname)
        return out

    ch_cfg = cfg.get("channels", {}) or {}
    for nm, mask in [("A", ats.CHANNEL_A), ("B", ats.CHANNEL_B)]:
        if nm in ch_cfg:
            cc = ch_cfg[nm] or {}
            coupling_name = str(cc.get('coupling', 'DC'))
            if not coupling_name.endswith('_COUPLING'):
                coupling_name = coupling_name + '_COUPLING'
            coupling = ats_const(coupling_name)
            rng_name = str(cc.get('range', 'PM_1_V'))
            imp = ats_const("IMPEDANCE_", str(cc.get("impedance", "50_OHM")))
            chosen_range = rng_name
            try_ranges = _range_candidates(rng_name)
            configured = False
            last_err: Optional[Exception] = None
            for rname in try_ranges:
                rng = _input_range_const(rname)
                if rng is None:
                    continue
                for attempt in range(2):
                    try:
                        board.inputControlEx(mask, coupling, rng, imp)
                        chosen_range = rname
                        configured = True
                        break
                    except Exception as ex:
                        last_err = ex
                        if "ApiFailed" in str(ex) and attempt == 0:
                            time.sleep(0.05)
                            continue
                        break
                if configured:
                    break
            if not configured:
                print(
                    f"[CAPPY] ERROR: inputControlEx failed for channel {nm} "
                    f"(coupling={coupling_name}, requested_range={rng_name}, impedance={cc.get('impedance', '50_OHM')})."
                )
                raise last_err if last_err is not None else RuntimeError(f"Failed to configure channel {nm}.")
            if chosen_range != rng_name:
                print(f"[CAPPY] Warning: Channel {nm} range {rng_name} failed; using fallback {chosen_range}.")
            if nm == 'A':
                vpp_A = _range_name_to_vpp(chosen_range, default_vpp=vpp_default)
            elif nm == 'B':
                vpp_B = _range_name_to_vpp(chosen_range, default_vpp=vpp_default)
            try:
                board.setBWLimit(mask, 0)
            except Exception as ex:
                print(f"[WARN] setBWLimit failed for channel {nm}: {ex}")

    t = cfg.get("trigger", {}) or {}
    operation = ats_const(str(t.get("operation", "TRIG_ENGINE_OP_J")))
    engine1 = ats_const(str(t.get("engine1", "TRIG_ENGINE_J")))
    engine2 = ats_const(str(t.get("engine2", "TRIG_ENGINE_K")))

    source_j_name = str(t.get("sourceJ", "TRIG_EXTERNAL")).strip()
    slope_j_name = str(t.get("slopeJ", "TRIGGER_SLOPE_POSITIVE")).strip()
    level_j = int(t.get("levelJ", 128))

    # Guardrail: for external edge triggers, thresholds on the "already crossed"
    # side can prevent any further edge crossings and look like "stuck waiting".
    if source_j_name == "TRIG_EXTERNAL":
        if slope_j_name == "TRIGGER_SLOPE_POSITIVE" and level_j < 128:
            print(
                f"[CAPPY] Adjusting trigger.levelJ from {level_j} to 128 for TRIG_EXTERNAL + positive slope "
                "(prevents no-crossing trigger lockout)."
            )
            level_j = 128
        elif slope_j_name == "TRIGGER_SLOPE_NEGATIVE" and level_j > 128:
            print(
                f"[CAPPY] Adjusting trigger.levelJ from {level_j} to 128 for TRIG_EXTERNAL + negative slope "
                "(prevents no-crossing trigger lockout)."
            )
            level_j = 128

    board.setTriggerOperation(
        operation,
        engine1,
        ats_const(source_j_name),
        ats_const(slope_j_name),
        level_j,
        engine2,
        ats_const(str(t.get("sourceK", "TRIG_DISABLE"))),
        ats_const(str(t.get("slopeK", "TRIGGER_SLOPE_POSITIVE"))),
        int(t.get("levelK", 128)),
    )
    ext_coupling_name = str(t.get("ext_coupling", "DC_COUPLING"))
    if not ext_coupling_name.endswith("_COUPLING"):
        ext_coupling_name = ext_coupling_name + "_COUPLING"
    ext_range_name = str(t.get("ext_range", "ETR_2V5"))
    ext_candidates = [ext_range_name]
    for fallback in ["ETR_2V5", "ETR_1V", "ETR_5V", "ETR_TTL"]:
        if fallback != ext_range_name:
            ext_candidates.append(fallback)
    ext_configured = False
    ext_last_err: Optional[Exception] = None
    chosen_ext_range = ext_range_name
    for candidate in ext_candidates:
        try:
            ext_range_const = ats_const(candidate)
        except AttributeError as ex:
            ext_last_err = ex
            continue
        for attempt in range(2):
            try:
                board.setExternalTrigger(
                    ats_const(ext_coupling_name),
                    ext_range_const
                )
                chosen_ext_range = candidate
                ext_configured = True
                break
            except Exception as ex:
                ext_last_err = ex
                if "ApiFailed" in str(ex) and attempt == 0:
                    time.sleep(0.05)
                    continue
                break
        if ext_configured:
            break
    if not ext_configured:
        raise ext_last_err if ext_last_err is not None else RuntimeError(
            f"Failed to configure external trigger range {ext_range_name}."
        )
    if chosen_ext_range != ext_range_name:
        print(f"[CAPPY] Warning: External trigger range {ext_range_name} failed; using fallback {chosen_ext_range}.")

    # Trigger delay: prefer trigger_delay_us if set, otherwise use delay_samples
    delay_us = float(t.get("trigger_delay_us", 0.0) or 0.0)
    delay_samples_cfg = int(t.get("delay_samples", 0))
    if delay_us > 0:
        delay_samples_computed = int(round(delay_us * 1e-6 * sr_hz))
        print(f"[CAPPY] Trigger delay: {delay_us:.3f} µs = {delay_samples_computed} samples at {sr_hz/1e6:.1f} MS/s")
        board.setTriggerDelay(delay_samples_computed)
    else:
        board.setTriggerDelay(delay_samples_cfg)

    timeout_ms = int(t.get("timeout_ms", 0))
    allow_autotrigger_with_external = _to_bool(t.get("allow_autotrigger_with_external", False), False)
    if source_j_name == "TRIG_EXTERNAL" and timeout_ms > 0 and not allow_autotrigger_with_external:
        print(
            f"[CAPPY] trigger.timeout_ms={timeout_ms} would enable auto-trigger (appears continuous). "
            "For external-trigger mode, forcing timeout_ms=0. "
            "Set trigger.allow_autotrigger_with_external=true to keep auto-trigger."
        )
        timeout_ms = 0
    rt = cfg.get("runtime", {}) or {}
    if bool(rt.get("noise_test", False)) and timeout_ms == 0:
        timeout_ms = int(rt.get("autotrigger_timeout_ms", 10))
        print(f"[CAPPY] noise_test enabled -> using trigger.timeout_ms={timeout_ms} for auto-trigger noise captures")
    board.setTriggerTimeOut(timeout_ms)

    try:
        board.configureAuxIO(ats.AUX_OUT_TRIGGER, 0)
    except Exception as ex:
        print(f"[WARN] configureAuxIO failed: {ex}")

    return sr_hz, float(vpp_A), float(vpp_B)

def _should_stop() -> bool:
    if STOP_REQUESTED:
        return True
    try:
        if sys.stdin is not None and hasattr(sys.stdin, 'isatty') and sys.stdin.isatty():
            return bool(ats.enter_pressed())
    except Exception:
        pass
    return False


def _ats_msg(ex: Exception) -> str:
    try:
        return str(ex)
    except Exception:
        return ""


def _is_ats_dma_done(ex: Exception) -> bool:
    msg = _ats_msg(ex)
    # ATS return code 519 == ApiDmaDone (normal DMA completion condition).
    return ("ApiDmaDone" in msg) or ("return code 519" in msg) or (" 519 " in f" {msg} ")


def _is_recoverable_ats_error(ex: Exception) -> bool:
    msg = _ats_msg(ex)
    recoverable_tokens = (
        "ApiBufferOverflow",
        "ApiBufferNotReady",
        "ApiWaitTimeout",
        "ApiWaitCanceled",
        "ApiWaitCancelled",
        "ApiDmaInProgress",
        "ApiFailed",
        "ApiDmaDone",
    )
    return any(tok in msg for tok in recoverable_tokens)

def run_capture(cfg_path: Path) -> int:
    if not ATS_AVAILABLE or ats is None:
        print("[CAPPY] atsapi not available on this machine.")
        return 2
    if REDUCED_SCHEMA is None or SESSION_INDEX_SCHEMA is None:
        print("[CAPPY] pyarrow is required for capture/archive writing. Please install pyarrow.")
        return 2

    # Reduce CPU priority so the DAQ kernel threads get precedence
    try:
        os.nice(5)
        print("[CAPPY] Process niceness set to +5 (lower CPU priority for I/O thread)")
    except Exception:
        pass

    cfg = load_config(cfg_path)
    cfg, cfg_warnings, cfg_errors = validate_and_normalize_capture_cfg(cfg)
    if cfg_errors:
        for e in cfg_errors:
            print(f"[CAPPY] Config error: {e}")
        return 2
    for w in cfg_warnings:
        print(f"[CAPPY] Config warning: {w}")

    acq = cfg.get("acquisition", {})
    integ = cfg.get("integration", {}) or {}
    waves = cfg.get("waveforms", {}) or {}
    storage = cfg.get("storage", {}) or {}
    trig = cfg.get("trigger", {}) or {}

    pre = int(acq.get("pre_trigger_samples", 0))
    post = int(acq.get("post_trigger_samples", 256))
    spr = pre + post
    rpb = int(acq.get("records_per_buffer", 128))
    bufN = int(acq.get("buffers_allocated", 16))
    bpa = int(acq.get("buffers_per_acquisition", 0))
    wait_timeout_ms = int(acq.get("wait_timeout_ms", 1000))
    ch_expr = str(acq.get("channels_mask", "CHANNEL_A"))

    b0, b1 = map(int, integ.get("baseline_window_samples", [0, min(64, spr)]))
    g0, g1 = map(int, integ.get("integral_window_samples", [min(64, spr), min(128, spr)]))

    wf_enable = bool(waves.get("enable", True))
    wf = WfPolicy(
        mode=str(waves.get("mode", "every_n")),
        every_n=int(waves.get("every_n", 20000)),
        thr_area=float(waves.get("threshold_integral_Vs", 0.0)),
        thr_peak=float(waves.get("threshold_peak_V", 0.0)),
        max_per_sec=int(waves.get("max_waveforms_per_sec", 50)),
    )
    store_volts = bool(waves.get("store_volts", True))
    waveform_codec = _normalize_waveform_codec(waves.get("archive_codec", "none"))
    waveform_zlib_level = _clamp_int(waves.get("archive_zlib_level", 3), 1, 9, 3)
    waveform_quant_bits = _clamp_int(waves.get("archive_quant_bits", 12), 8, 14, 12)

    binfo = cfg.get("board", {}) if isinstance(cfg, dict) else {}
    systemId = int(binfo.get("system_id", 2))
    boardId = int(binfo.get("board_id", 1))
    board = ats.Board(systemId=systemId, boardId=boardId)
    
    # Validate board handle
    if not hasattr(board, 'handle') or board.handle is None:
        available = get_available_boards()
        error_msg = (
            f"Failed to initialize board with system_id={systemId}, board_id={boardId}. "
            f"The board handle is None. Check that:\n"
            f"  1. The board is physically connected\n"
            f"  2. The system_id and board_id in config are correct\n"
            f"  3. The atsapi/driver installation is correct\n"
        )
        if available:
            error_msg += f"\nAvailable boards:\n"
            for sys_id, brd_id, desc in available:
                error_msg += f"  - system_id={sys_id}, board_id={brd_id} ({desc})\n"
        else:
            error_msg += f"\nNo boards found. Check physical connection and driver installation.\n"
        raise RuntimeError(error_msg)
    
    sr_hz, vppA, vppB = configure_board(board, cfg)

    ch_mask = channels_from_mask_expr(ch_expr)
    ch_count = infer_channel_count_from_mask(ch_mask)

    _, bps = board.getChannelInfo()
    bytesPerSample = (bps.value + 7) // 8
    sample_type = ctypes.c_uint8 if bytesPerSample == 1 else ctypes.c_uint16
    bytesPerBuffer = bytesPerSample * spr * rpb * ch_count

    buffers = [ats.DMABuffer(board.handle, sample_type, bytesPerBuffer) for _ in range(bufN)]
    board.setRecordSize(pre, post)

    if bpa <= 0:
        recordsPerAcq = 0
        buf_target = 2**31 - 1
    else:
        recordsPerAcq = rpb * bpa
        buf_target = bpa

    archive = CappyArchive(
        data_dir=Path(str(storage.get("data_dir", "dataFile"))),
        rollover_minutes=int(storage.get("rollover_minutes", 60)),
        flush_every_records=int(storage.get("flush_every_records", 200000)),
        session_rotate_hours=float(storage.get("session_rotate_hours", 0) or 0),
        sqlite_commit_every_snips=int(storage.get("sqlite_commit_every_snips", 2000)),
        flush_every_seconds=float(storage.get("flush_every_seconds", 10.0) or 0.0),
        waveform_codec=waveform_codec,
        waveform_zlib_level=waveform_zlib_level,
        waveform_quant_bits=waveform_quant_bits,
    )
    sid = archive.start(tag=str(storage.get("session_tag", "")).strip(), channels_mask=ch_expr)
    notifier = StatusNotifier(cfg, Path(str(storage.get('data_dir', 'dataFile'))))
    # Live waveform ring (writes one downsampled waveform per buffer)
    live_cfg = (cfg.get('live', {}) or {})
    ring_nslots = int(live_cfg.get('ring_slots', 4096))
    ring_npts = int(live_cfg.get('ring_points', 512))
    live_waveform_every_n = _clamp_int(
        live_cfg.get("waveform_every_n_buffers", 1), 1, WAVEFORM_EVERY_N_MAX, 1
    )
    show_channel_b_live = _to_bool(live_cfg.get('show_channel_b', False), False)
    ring_path = Path(str(storage.get('data_dir', 'dataFile'))) / 'status' / 'live_waveforms.ring'
    ring = LiveRingWriter(ring_path, nslots=ring_nslots, npts=ring_npts)

    notifier.update(
        session_id=sid,
        state='running',
        started=time.strftime('%Y-%m-%d %H:%M:%S'),
        started_unix=time.time(),
        data_dir=str(storage.get('data_dir','dataFile')),
        channels_mask=ch_expr,
        sample_rate_hz=sr_hz,
        samples_per_record=spr,
        records_per_buffer=rpb,
        vpp_A=vppA,
        vpp_B=vppB,
        trigger_source=str(trig.get("sourceJ", "TRIG_EXTERNAL")),
        trigger_timeout_ms=int(trig.get("timeout_ms", 0)),
        trigger_level_code=int(trig.get("levelJ", 128)),
        trigger_slope=str(trig.get("slopeJ", "TRIGGER_SLOPE_POSITIVE")),
        live_ring_path=str(ring_path),
        live=dict(
            ring_slots=int(live_cfg.get('ring_slots', ring_nslots)),
            ring_points=int(live_cfg.get('ring_points', ring_npts)),
            waveform_every_n_buffers=int(live_waveform_every_n),
            stream_window_points=int(live_cfg.get('stream_window_points', 100000)),
            stream_window_seconds=float(live_cfg.get('stream_window_seconds', 2.0)),
            max_waveforms_per_tick=int(live_cfg.get('max_waveforms_per_tick', 12)),
            ui_fps=float(live_cfg.get('ui_fps', 6.0)),
            show_channel_b=bool(show_channel_b_live),
            preview_mode=str(live_cfg.get('preview_mode', 'archive_match')),
        ),
    )
    notifier.maybe_emit()

    adma_flags = ats.ADMA_TRADITIONAL_MODE
    if bool(trig.get("external_startcapture", False)):
        adma_flags |= ats.ADMA_EXTERNAL_STARTCAPTURE

    board.beforeAsyncRead(ch_mask, -pre, spr, rpb, recordsPerAcq, adma_flags)

    for b in buffers:
        board.postAsyncBuffer(b.addr, b.size_bytes)

    buf_done = 0
    next_wait_idx = 0
    global_rec = 0
    t0 = time.time()
    last = t0
    dashboard_every_buffers = int((cfg.get('notify', {}) or {}).get('dashboard_every_buffers', 1000))
    gui_every_buffers = max(1, int((cfg.get('notify', {}) or {}).get('gui_every_buffers', 1)))
    last_emit_buf = 0
    last_gui_emit_buf = 0
    timeout_count = 0
    last_buffer_ns = time.time_ns()
    live_cfg = cfg.get('live', {}) or {}
    rt = cfg.get('runtime', {}) or {}
    rearm_if_no_trigger_s = int(rt.get('rearm_if_no_trigger_s', 300))
    rearm_cooldown_s = int(rt.get('rearm_cooldown_s', 30))
    max_rearms_per_hour = int(rt.get('max_rearms_per_hour', 12))
    max_recoverable_errors = int(rt.get('max_recoverable_errors', 200))
    recover_backoff_s = float(rt.get('recover_backoff_s', 0.05) or 0.05)
    timeout_pause_s = float(trig.get("timeout_pause_s", 0.0) or 0.0)
    preview_mode = str(live_cfg.get("preview_mode", "archive_match")).strip().lower()
    if preview_mode not in {"archive_match", "record0"}:
        preview_mode = "archive_match"
    rearm_times: List[float] = []
    recoverable_error_count = 0

    try:
        def _do_rearm(force: bool = False, reason: str = ""):
            nonlocal timeout_count, last_buffer_ns, next_wait_idx
            now = time.time()
            # keep only last hour
            while rearm_times and (now - rearm_times[0]) > 3600:
                rearm_times.pop(0)
            if (not force) and rearm_times and (now - rearm_times[-1]) < rearm_cooldown_s:
                return
            if (not force) and len(rearm_times) >= max_rearms_per_hour:
                return
            rearm_times.append(now)

            reason_str = f" ({reason})" if reason else f" (no completed buffers for {rearm_if_no_trigger_s}s)"
            print(f"[CAPPY] Rearming acquisition{reason_str}. rearms_last_hour={len(rearm_times)}")
            notifier.update(state='rearming', time=time.strftime('%Y-%m-%d %H:%M:%S'), timeouts=timeout_count)
            notifier.maybe_emit()

            try:
                board.abortAsyncRead()
            except Exception:
                pass

            # Brief pause to let the board fully quiesce before re-posting DMA buffers.
            # This prevents ApiBufferOverflow cascades at spill boundaries.
            time.sleep(0.005)

            board.beforeAsyncRead(ch_mask, -pre, spr, rpb, recordsPerAcq, adma_flags)
            for b in buffers:
                board.postAsyncBuffer(b.addr, b.size_bytes)
            board.startCapture()

            timeout_count = 0
            last_buffer_ns = time.time_ns()
            next_wait_idx = 0
            notifier.update(state='running')
            notifier.maybe_emit()

        def _recoverable_rearm(ex: Exception, where: str) -> bool:
            nonlocal recoverable_error_count
            if not _is_recoverable_ats_error(ex):
                return False
            recoverable_error_count += 1
            print(f"[CAPPY] Warning: recoverable ATS error in {where}: {ex}")
            notifier.update(
                state='recovering',
                time=time.strftime('%Y-%m-%d %H:%M:%S'),
                recoverable_errors=recoverable_error_count,
                last_error=str(ex),
            )
            notifier.maybe_emit()
            if recoverable_error_count > max_recoverable_errors:
                print(f"[CAPPY] Too many recoverable errors ({recoverable_error_count}); stopping capture.")
                return False
            try:
                time.sleep(max(0.0, recover_backoff_s))
            except Exception:
                pass
            _do_rearm(force=True, reason=f"recoverable {where}: {ex}")
            return True

        def _pause_for_timeout(no_trigger_for_s: float):
            nonlocal timeout_count, last_buffer_ns
            pause_for = float(timeout_pause_s)
            if pause_for <= 0:
                return
            print(
                f"[CAPPY] No trigger for {no_trigger_for_s:.1f}s -> pausing acquisition for {pause_for:.1f}s before rearm."
            )
            notifier.update(
                state='paused_timeout',
                time=time.strftime('%Y-%m-%d %H:%M:%S'),
                no_trigger_for_s=float(no_trigger_for_s),
                pause_s=float(pause_for),
                timeouts=timeout_count,
            )
            notifier.emit_now()
            try:
                board.abortAsyncRead()
            except Exception:
                pass
            t_end = time.time() + pause_for
            while (time.time() < t_end) and (not _should_stop()):
                time.sleep(0.2)
            if _should_stop():
                return
            _do_rearm(force=True, reason=f"timeout pause ({pause_for:.1f}s)")

        board.startCapture()
        print(f"[CAPPY] Running session {sid}. " + ("Press <enter> to stop." if (sys.stdin is not None and hasattr(sys.stdin,"isatty") and sys.stdin.isatty()) else "Use Stop (GUI) or Ctrl+C to stop."))
        while buf_done < buf_target and not _should_stop():
            buf = buffers[next_wait_idx % len(buffers)]
            try:
                board.waitAsyncBufferComplete(buf.addr, wait_timeout_ms)
            except Exception as ex:
                if _is_ats_dma_done(ex):
                    # Normal termination in finite DMA acquisitions on some ATS SDK builds.
                    break
                if "ApiBufferNotReady" in str(ex):
                    print("[CAPPY] Warning: ApiBufferNotReady while waiting for DMA buffer; resetting DMA queue.")
                    _do_rearm(force=True, reason="ApiBufferNotReady")
                    continue
                if "ApiDmaInProgress" in str(ex):
                    # Non-fatal poll result: DMA transfer is still active for this buffer.
                    timeout_count += 1
                    ago_s = (time.time_ns() - last_buffer_ns) / 1e9
                    notifier.update(state="waiting_for_trigger", time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                   timeouts=timeout_count, last_buffer_ago_s=ago_s,
                                   buffers=buf_done, records=global_rec,
                                   reduced_rows=getattr(archive, "_n_reduced", 0),
                                   snips=getattr(archive, "_n_snips", 0))
                    notifier.maybe_emit()
                    continue
                if "ApiWaitTimeout" in str(ex):
                    timeout_count += 1
                    ago_s = (time.time_ns() - last_buffer_ns) / 1e9
                    notifier.update(state="waiting_for_trigger", time=time.strftime("%Y-%m-%d %H:%M:%S"),
                                   timeouts=timeout_count, last_buffer_ago_s=ago_s,
                                   buffers=buf_done, records=global_rec,
                                   reduced_rows=getattr(archive, "_n_reduced", 0),
                                   snips=getattr(archive, "_n_snips", 0))
                    notifier.maybe_emit()
                    if timeout_count % 100 == 0:
                        print(f"[CAPPY] waiting for triggers... (timeouts={timeout_count})")
                    if timeout_pause_s > 0 and ago_s >= float(timeout_pause_s):
                        _pause_for_timeout(ago_s)
                        continue
                    if rearm_if_no_trigger_s > 0 and ago_s >= float(rearm_if_no_trigger_s):
                        _do_rearm(reason=f"no triggers for {ago_s:.1f}s")
                    continue
                if "ApiWaitCanceled" in str(ex) or "ApiWaitCancelled" in str(ex):
                    # Normal stop path (SIGINT / abortAsyncRead)
                    break
                if _recoverable_rearm(ex, "waitAsyncBufferComplete"):
                    continue
                raise

            timeout_count = 0
            last_buffer_ns = time.time_ns()
            ts_ns = last_buffer_ns
            # Per-record time offset uses the configured record length in samples.
            record_dt_ns = int(round(float(spr) / float(sr_hz) * 1e9))

            # Copy completed DMA data, then immediately recycle the board buffer.
            # Keeping the re-post ahead of reductions/file I/O avoids API buffer overflow.
            raw = buf.buffer.copy()
            if raw.dtype != np.uint16:
                raw = raw.astype(np.uint16, copy=False)
            if _should_stop():
                break
            try:
                board.postAsyncBuffer(buf.addr, buf.size_bytes)
            except Exception as ex:
                msg = str(ex)
                if _is_ats_dma_done(ex):
                    break
                if "ApiBufferOverflow" in msg:
                    print("[CAPPY] Warning: ApiBufferOverflow while recycling DMA buffer; soft rearm.")
                    # Brief quiesce before rearm to prevent cascade at spill boundaries
                    time.sleep(0.01)
                    _do_rearm(force=True, reason="ApiBufferOverflow on postAsyncBuffer")
                    continue
                if "ApiWaitCanceled" in msg or "ApiWaitCancelled" in msg:
                    break
                if _recoverable_rearm(ex, "postAsyncBuffer"):
                    continue
                raise

            if archive.should_rotate():
                archive.finalize(ch_expr)
                sid = archive.start(tag=str(storage.get("session_tag", "")).strip(), channels_mask=ch_expr)

            red_rows: List[Dict[str, Any]] = []
            live_wfA_candidate: Optional[np.ndarray] = None
            live_wfB_candidate: Optional[np.ndarray] = None

            if ch_count == 2:
                A = raw[0::2].reshape(rpb, spr)
                B = raw[1::2].reshape(rpb, spr)
                areaA, peakA, baseA = reduce_u16(A, sr_hz, b0, b1, g0, g1, vppA)
                areaB, peakB, baseB = reduce_u16(B, sr_hz, b0, b1, g0, g1, vppB)
                for r in range(rpb):
                    rec_g = global_rec + r
                    rec_ts_ns = int(ts_ns + r * record_dt_ns)
                    red_rows.append(dict(
                        session_id=sid, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                        timestamp_ns=rec_ts_ns, sample_rate_hz=float(sr_hz),
                        samples_per_record=spr, records_per_buffer=rpb,
                        channels_mask=ch_expr,
                        area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]), baseline_A_V=float(baseA[r]),
                        area_B_Vs=float(areaB[r]), peak_B_V=float(peakB[r]), baseline_B_V=float(baseB[r]),
                    ))
                    if wf_enable and wf.want(rec_g, float(areaA[r]), float(peakA[r])):
                        wfA_V = _codes_to_volts_u16(A[r], vpp=vppA) if store_volts else A[r].astype(np.float32)
                        wfB_V = _codes_to_volts_u16(B[r], vpp=vppB) if store_volts else B[r].astype(np.float32)
                        if preview_mode == "archive_match" and live_wfA_candidate is None:
                            live_wfA_candidate = wfA_V
                            live_wfB_candidate = wfB_V
                        archive.append_snip(
                            ts_ns=rec_ts_ns, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                            channels_mask=ch_expr, sample_rate_hz=float(sr_hz),
                            wfA_V=wfA_V, wfB_V=wfB_V,
                            area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]),
                            area_B_Vs=float(areaB[r]), peak_B_V=float(peakB[r]),
                            baseline_A_V=float(baseA[r]), baseline_B_V=float(baseB[r]),
                        )
            else:
                A = raw.reshape(rpb, spr)
                areaA, peakA, baseA = reduce_u16(A, sr_hz, b0, b1, g0, g1, vppA)
                for r in range(rpb):
                    rec_g = global_rec + r
                    rec_ts_ns = int(ts_ns + r * record_dt_ns)
                    red_rows.append(dict(
                        session_id=sid, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                        timestamp_ns=rec_ts_ns, sample_rate_hz=float(sr_hz),
                        samples_per_record=spr, records_per_buffer=rpb,
                        channels_mask=ch_expr,
                        area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]), baseline_A_V=float(baseA[r]),
                        area_B_Vs=0.0, peak_B_V=0.0, baseline_B_V=0.0,
                    ))
                    if wf_enable and wf.want(rec_g, float(areaA[r]), float(peakA[r])):
                        wfA_V = _codes_to_volts_u16(A[r], vpp=vppA) if store_volts else A[r].astype(np.float32)
                        if preview_mode == "archive_match" and live_wfA_candidate is None:
                            live_wfA_candidate = wfA_V
                            live_wfB_candidate = None
                        archive.append_snip(
                            ts_ns=rec_ts_ns, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                            channels_mask=ch_expr, sample_rate_hz=float(sr_hz),
                            wfA_V=wfA_V, wfB_V=None,
                            area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]),
                            area_B_Vs=0.0, peak_B_V=0.0,
                            baseline_A_V=float(baseA[r]), baseline_B_V=0.0,
                        )

            # Per-buffer summaries for live GUI plotting
            try:
                if ch_count == 2:
                    notifier.update(
                        buffer_mean_area_A=float(np.mean(areaA)), buffer_mean_peak_A=float(np.mean(peakA)),
                        buffer_mean_area_B=float(np.mean(areaB)), buffer_mean_peak_B=float(np.mean(peakB)),
                    )
                    completed_buffers = buf_done + 1
                    if gui_every_buffers > 0 and (completed_buffers - last_gui_emit_buf) >= gui_every_buffers:
                        last_gui_emit_buf = completed_buffers
                        notifier.update(
                            last_capture=time.strftime('%Y-%m-%d %H:%M:%S'),
                            last_capture_unix=time.time(),
                            buffers=completed_buffers,
                        )
                        notifier.emit_now()
                else:
                    notifier.update(
                        buffer_mean_area_A=float(np.mean(areaA)), buffer_mean_peak_A=float(np.mean(peakA))
                    )
                    completed_buffers = buf_done + 1
                    if gui_every_buffers > 0 and (completed_buffers - last_gui_emit_buf) >= gui_every_buffers:
                        last_gui_emit_buf = completed_buffers
                        notifier.update(
                            last_capture=time.strftime('%Y-%m-%d %H:%M:%S'),
                            last_capture_unix=time.time(),
                            buffers=completed_buffers,
                        )
                        notifier.emit_now()
            except Exception:
                pass

            # Write representative waveforms to the live ring at a configurable cadence.
            if (buf_done % live_waveform_every_n) == 0:
                try:
                    if preview_mode == "archive_match" and live_wfA_candidate is not None:
                        wfA_live = live_wfA_candidate
                        wfB_live = live_wfB_candidate if (ch_count == 2 and show_channel_b_live) else None
                        chmask_live = 3 if (ch_count == 2 and show_channel_b_live and wfB_live is not None) else 1
                    else:
                        # Fallback for non-archived or record0 preview mode.
                        wfA_live = _codes_to_volts_u16(A[0], vpp=vppA)
                        wfB_live = None
                        chmask_live = 1
                        if ch_count == 2 and show_channel_b_live:
                            wfB_live = _codes_to_volts_u16(B[0], vpp=vppB)
                            chmask_live = 3
                    ring.write(wfA_live, wfB_live, buf_idx=buf_done, chmask=chmask_live)
                except Exception:
                    pass

            archive.append_reduced(red_rows, ts_ns)

            next_wait_idx += 1
            buf_done += 1
            global_rec += rpb

            now = time.time()
            if (now - last >= 1.0) or (dashboard_every_buffers > 0 and (buf_done - last_emit_buf) >= dashboard_every_buffers):
                rate = global_rec / max(now - t0, 1e-9)
                # status fields for GUI
                last_capture_unix = time.time()
                last_capture = time.strftime('%Y-%m-%d %H:%M:%S')

                # latest waveform for GUI (downsampled). Uses last computed wfA_V/wfB_V if available.
                latest_wf_A = None
                latest_wf_B = None
                try:
                    if 'wfA_V' in locals():
                        w = wfA_V
                        step = max(1, int(len(w) // 1200))
                        latest_wf_A = w[::step].astype(float).tolist()
                    if 'wfB_V' in locals() and wfB_V is not None:
                        w = wfB_V
                        step = max(1, int(len(w) // 1200))
                        latest_wf_B = w[::step].astype(float).tolist()
                except Exception:
                    latest_wf_A = None
                    latest_wf_B = None

                notifier.update(state="running", time=time.strftime("%Y-%m-%d %H:%M:%S"),
                               buffers=buf_done, records=global_rec, rate_hz=rate, last_capture=last_capture, last_capture_unix=last_capture_unix, latest_waveform_A=latest_wf_A, latest_waveform_B=latest_wf_B,
                               reduced_rows=getattr(archive, "_n_reduced", 0),
                               snips=getattr(archive, "_n_snips", 0),
                               last_buffer_ago_s=(time.time_ns()-last_buffer_ns)/1e9)
                notifier.maybe_emit()
                print(f"[CAPPY] buffers={buf_done} records={global_rec} rate={rate/1e3:.1f} kHz snips={archive._n_snips}")
                last = now
                last_emit_buf = buf_done

    finally:
        try:
            board.abortAsyncRead()
        except Exception:
            pass
        try:
            ring.close()
        except Exception:
            pass
        notifier.update(state='stopped', time=time.strftime('%Y-%m-%d %H:%M:%S'))
        notifier.maybe_emit()
        archive.finalize(ch_expr)

    return 0


# ---------------------------------------------------------------------------
# Quick Config  —  disposable ~10-buffer scout capture + waveform analysis
# ---------------------------------------------------------------------------
# Captures a handful of buffers with current board settings, analyses the raw
# ADC data to derive optimal trigger level, samples_per_record, and input
# range, then emits a single JSON line on stdout so the GUI can parse it.
#
# Design follows cappyarchive_db.py style:
#   • background thread owns all hardware access
#   • GUI thread only reads final JSON result
#   • compact numpy analysis, no pandas
#   • graceful cleanup in all paths
# ---------------------------------------------------------------------------

_QC_BUFFERS = 48          # large scout sample for stable zero-cross statistics
_QC_NOISE_SIGMA = 5.0     # noise floor = median ± N * MAD
_QC_HEADROOM = 1.20       # 20 % headroom on range selection
_QC_TAIL_PAD_FRAC = 0.25  # pad record length 25 % beyond the mean peak location
_QC_ZERO_GUARD_SIGMA = 3.0
_QC_ZERO_ALIGN_PRE_FRAC = 0.20
_QC_MIN_ANALYSIS_RECORDS = 24

# Input ranges ordered by Vpp (ascending) for auto-range selection
_INPUT_RANGE_VPP_ORDERED: List[Tuple[str, float]] = [
    ("PM_20_MV",  0.04), ("PM_40_MV",  0.08), ("PM_50_MV",  0.10),
    ("PM_80_MV",  0.16), ("PM_100_MV", 0.20), ("PM_200_MV", 0.40),
    ("PM_400_MV", 0.80), ("PM_500_MV", 1.00), ("PM_800_MV", 1.60),
    ("PM_1_V",    2.00), ("PM_2_V",    4.00), ("PM_4_V",    8.00),
]


def _qc_round_spr(n: int, lo: int = 64, hi: Optional[int] = None) -> int:
    out = max(int(lo), ((int(n) + 15) // 16) * 16)
    if hi is not None:
        out = min(out, int(hi))
    return out


def _qc_find_zero_crossings(volts: np.ndarray, baseline_v: float, sigma_v: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return (indices, slope_sign) for strong zero crossings in each record.

    A crossing is accepted only if the record spans both sides of 0 V and the local
    excursion around the crossing exceeds a sigma-based guard band so noise does not
    dominate the alignment.
    """
    n_recs, spr = volts.shape
    idxs = np.full((n_recs,), -1, dtype=np.int32)
    slopes = np.zeros((n_recs,), dtype=np.int8)
    if spr < 2:
        return idxs, slopes

    centered = volts - float(baseline_v)
    guard = max(float(_QC_ZERO_GUARD_SIGMA) * float(sigma_v), 1e-6)
    for i in range(n_recs):
        rec = centered[i]
        if float(np.min(rec)) >= -guard or float(np.max(rec)) <= guard:
            continue
        s0 = rec[:-1]
        s1 = rec[1:]
        cand = np.flatnonzero(((s0 <= 0.0) & (s1 > 0.0)) | ((s0 >= 0.0) & (s1 < 0.0)))
        if cand.size == 0:
            continue
        best_j = -1
        best_score = -1.0
        best_slope = 0
        for j in cand:
            a = float(rec[j])
            b = float(rec[j + 1])
            score = abs(a) + abs(b)
            if score < 2.0 * guard:
                continue
            slope = 1 if (b - a) >= 0.0 else -1
            if score > best_score:
                best_score = score
                best_j = int(j)
                best_slope = int(slope)
        if best_j >= 0:
            idxs[i] = best_j
            slopes[i] = best_slope
    return idxs, slopes


def _qc_analyse_buffers(
    bufs: List[np.ndarray],
    spr: int,
    rpb: int,
    sr_hz: float,
    vpp: float,
    ch_count: int,
) -> dict:
    """Analyse raw uint16 ADC buffers and return optimal settings as a dict.

    Quick Config now uses a larger scout capture and aligns records on the first
    robust 0 V crossing. The trigger is then set to 0 %FS and the record length is
    chosen from the mean peak position after that crossing.
    """
    records: List[np.ndarray] = []
    for raw in bufs:
        try:
            if ch_count == 2:
                both = raw.reshape(rpb, spr, 2)
                recs = both[:, :, 0]
            else:
                recs = raw.reshape(rpb, spr)
            records.append(recs)
        except Exception:
            continue

    if not records:
        return {"error": "no valid buffers captured"}

    all_recs = np.vstack(records)
    n_recs = int(all_recs.shape[0])
    volts = (all_recs.astype(np.float32) - 32768.0) * (float(vpp) / 65536.0)

    pre_win = min(max(32, spr // 8), spr)
    baselines = volts[:, :pre_win]
    baseline_per_rec = np.median(baselines, axis=1).astype(np.float32)
    baseline_v = float(np.median(baseline_per_rec))
    centered = volts - baseline_per_rec[:, None]

    mad = float(np.median(np.abs(centered[:, :pre_win])))
    sigma_est = 1.4826 * mad if mad > 0 else float(np.std(centered[:, :pre_win]))
    sigma_est = max(float(sigma_est), 1e-9)

    rec_maxes = np.max(centered, axis=1)
    rec_mins = np.min(centered, axis=1)
    peak_pos = float(np.percentile(rec_maxes, 95))
    peak_neg = float(np.percentile(rec_mins, 5))
    peak_abs = max(abs(peak_pos), abs(peak_neg))

    needed_vpp = 2.0 * peak_abs * _QC_HEADROOM
    best_range = "PM_4_V"
    best_vpp = 8.0
    for rname, rvpp in _INPUT_RANGE_VPP_ORDERED:
        if rvpp >= needed_vpp:
            best_range = rname
            best_vpp = rvpp
            break

    zc_idx, zc_slope = _qc_find_zero_crossings(volts, baseline_v, sigma_est)
    valid = np.flatnonzero(zc_idx >= 0)
    notes: List[str] = []

    crossing_mode = "zero-cross"
    if valid.size >= max(8, min(_QC_MIN_ANALYSIS_RECORDS, n_recs // 4 if n_recs >= 4 else 8)):
        pos_n = int(np.count_nonzero(zc_slope[valid] > 0))
        neg_n = int(np.count_nonzero(zc_slope[valid] < 0))
        dom_slope = 1 if pos_n >= neg_n else -1
        use = valid[zc_slope[valid] == dom_slope]
        if use.size < max(8, valid.size // 3):
            use = valid
            crossing_mode = "mixed zero-cross"
        else:
            crossing_mode = "rising zero-cross" if dom_slope > 0 else "falling zero-cross"

        cross_mean = float(np.mean(zc_idx[use]))
        cross_std = float(np.std(zc_idx[use])) if use.size > 1 else 0.0

        pre_pts = max(16, int(round(spr * _QC_ZERO_ALIGN_PRE_FRAC)))
        pre_pts = min(pre_pts, max(16, int(cross_mean) if cross_mean > 0 else 16))
        max_post = int(max(8, spr - 1 - np.max(zc_idx[use])))
        post_pts = max(8, max_post)

        if pre_pts + post_pts < 8:
            mean_peak_idx = float(cross_mean)
            opt_spr = spr
            active_samples = int(round(cross_mean))
        else:
            aligned = []
            kept_cross = []
            for ridx in use:
                ci = int(zc_idx[ridx])
                start = ci - pre_pts
                stop = ci + post_pts
                if start < 0 or stop > spr:
                    continue
                aligned.append(centered[int(ridx), start:stop])
                kept_cross.append(ci)
            if aligned:
                mean_wave = np.mean(np.asarray(aligned, dtype=np.float32), axis=0)
                zero_i = pre_pts
                peak_rel = int(np.argmax(np.abs(mean_wave[zero_i:]))) if zero_i < mean_wave.size else 0
                mean_peak_idx = float(np.mean(kept_cross)) + float(peak_rel)
                active_samples = int(round(mean_peak_idx))
                opt_spr = _qc_round_spr(int(np.ceil((mean_peak_idx + 1.0) * (1.0 + _QC_TAIL_PAD_FRAC))), hi=spr)
            else:
                mean_peak_idx = float(cross_mean)
                active_samples = int(round(cross_mean))
                opt_spr = _qc_round_spr(int(np.ceil((cross_mean + 1.0) * (1.0 + _QC_TAIL_PAD_FRAC))), hi=spr)

        trig_pct = 0.0
        notes.append(f"{crossing_mode}: {use.size}/{n_recs} records")
        notes.append(f"mean crossing={cross_mean:.1f}±{cross_std:.1f} samples")
        notes.append(f"mean peak @ {mean_peak_idx:.1f} samples")
    else:
        avg_wave = np.mean(np.abs(centered), axis=0)
        noise_thr = _QC_NOISE_SIGMA * sigma_est * 0.5
        active_mask = avg_wave > noise_thr
        active_idxs = np.flatnonzero(active_mask)
        if active_idxs.size > 0:
            last_active = int(active_idxs[-1])
            opt_spr = _qc_round_spr(int(np.ceil((last_active + 1) * (1.0 + _QC_TAIL_PAD_FRAC))), hi=spr)
            active_samples = last_active
        else:
            opt_spr = spr
            active_samples = 0
        trig_pct = 0.0
        notes.append(f"only {valid.size}/{n_recs} usable zero-crossing records; fell back to envelope sizing")

    if peak_abs < sigma_est * 3.0:
        notes.append("signal amplitude very close to noise; results may be unreliable")
    if valid.size == 0:
        notes.append("no robust 0 V crossings found")

    return {
        "trigger_level_pct": round(trig_pct, 2),
        "samples_per_record": int(opt_spr),
        "input_range": best_range,
        "input_range_vpp": best_vpp,
        "peak_v": round(peak_abs, 6),
        "noise_sigma_v": round(sigma_est, 8),
        "baseline_v": round(baseline_v, 6),
        "active_samples": int(active_samples),
        "total_records": n_recs,
        "zero_cross_records": int(valid.size),
        "analysis_note": "; ".join(notes) if notes else "OK",
    }


def run_quick_config(cfg_path: Path) -> int:
    """Run a disposable large-sample scout capture and print optimal settings JSON.

    Designed to be invoked as a subprocess by the GUI:
        python CAPPY_v1_3.py quick_config --config <yaml>

    Prints exactly one line of JSON prefixed with 'CAPPY_QC_RESULT ' on success.
    """
    if not ATS_AVAILABLE or ats is None:
        print(json.dumps({"error": "atsapi not available"}))
        return 2

    cfg = load_config(cfg_path)
    cfg, _, cfg_errors = validate_and_normalize_capture_cfg(cfg)
    if cfg_errors:
        print(json.dumps({"error": f"config errors: {'; '.join(cfg_errors)}"}))
        return 2

    acq  = cfg.get("acquisition", {})
    trig = cfg.get("trigger", {})

    pre  = int(acq.get("pre_trigger_samples", 0))
    post = int(acq.get("post_trigger_samples", 256))
    spr  = pre + post
    rpb  = int(acq.get("records_per_buffer", 128))
    bufN = max(4, _QC_BUFFERS + 2)  # allocate a few spare DMA buffers
    ch_expr = str(acq.get("channels_mask", "CHANNEL_A"))

    binfo = cfg.get("board", {}) if isinstance(cfg, dict) else {}
    systemId = int(binfo.get("system_id", 2))
    boardId  = int(binfo.get("board_id", 1))

    try:
        board = ats.Board(systemId=systemId, boardId=boardId)
        if not hasattr(board, 'handle') or board.handle is None:
            print(json.dumps({"error": "board handle is None"}))
            return 2
    except Exception as ex:
        print(json.dumps({"error": f"board init failed: {ex}"}))
        return 2

    try:
        sr_hz, vppA, vppB = configure_board(board, cfg)
    except Exception as ex:
        print(json.dumps({"error": f"configure_board failed: {ex}"}))
        return 2

    ch_mask  = channels_from_mask_expr(ch_expr)
    ch_count = infer_channel_count_from_mask(ch_mask)
    _, bps   = board.getChannelInfo()
    bpS      = (bps.value + 7) // 8
    stype    = ctypes.c_uint8 if bpS == 1 else ctypes.c_uint16
    bpBuf    = bpS * spr * rpb * ch_count

    buffers = [ats.DMABuffer(board.handle, stype, bpBuf) for _ in range(bufN)]
    board.setRecordSize(pre, post)

    recs_per_acq = rpb * _QC_BUFFERS  # finite acquisition
    adma_flags   = ats.ADMA_TRADITIONAL_MODE
    if bool(trig.get("external_startcapture", False)):
        adma_flags |= ats.ADMA_EXTERNAL_STARTCAPTURE

    board.beforeAsyncRead(ch_mask, -pre, spr, rpb, recs_per_acq, adma_flags)
    for b in buffers:
        board.postAsyncBuffer(b.addr, b.size_bytes)

    captured: List[np.ndarray] = []
    wt_ms = int(acq.get("wait_timeout_ms", 5000))
    # Use a generous timeout for the scout capture
    wt_ms = max(wt_ms, 5000)

    print(f"[QC] capturing {_QC_BUFFERS} buffers (spr={spr} rpb={rpb} ch={ch_count})…", flush=True)
    board.startCapture()

    try:
        for bi in range(_QC_BUFFERS):
            buf = buffers[bi % bufN]
            try:
                board.waitAsyncBufferComplete(buf.addr, timeout_ms=wt_ms)
            except Exception as ex:
                if _is_ats_dma_done(ex):
                    break
                print(f"[QC] timeout/error on buffer {bi}: {ex}", flush=True)
                break

            raw = np.array(buf.buffer, copy=True)
            captured.append(raw)
            board.postAsyncBuffer(buf.addr, buf.size_bytes)
            print(f"[QC] buffer {bi+1}/{_QC_BUFFERS} OK", flush=True)
    finally:
        try:
            board.abortAsyncRead()
        except Exception:
            pass

    if not captured:
        print(json.dumps({"error": "no buffers captured — check trigger or signal"}))
        return 1

    result = _qc_analyse_buffers(captured, spr, rpb, sr_hz, vppA, ch_count)
    result["scout_buffers"] = len(captured)
    result["sample_rate_hz"] = sr_hz
    result["current_vpp"] = vppA

    # Prefix so the GUI can reliably parse this line from mixed output
    print(f"CAPPY_QC_RESULT {json.dumps(result)}", flush=True)
    return 0


class ArchiveBrowser(ttk.Frame):
    def __init__(self, data_dir: Path, master=None):
        super().__init__(master)
        self._tz = datetime.now().astimezone().tzinfo
        self.data_dir = data_dir
        _ensure_dir(self.data_dir)
        self.captures = data_dir / "captures"
        _ensure_dir(self.captures)
        self.sessions = _lazy_pandas().DataFrame()
        self.snips = _lazy_pandas().DataFrame()
        self._snip_db_dir: Optional[Path] = None
        self._snips_view = _lazy_pandas().DataFrame()
        self._snips_total_count = 0
        self._snip_load_limit = _clamp_int(os.environ.get("CAPPY_ARCHIVE_SNIP_LIMIT", 200000), 1000, 2000000, 200000)
        self._sel_date = None
        self._sel_hour = None
        self._session_summary = tk.StringVar(value="No sessions loaded")
        self._build()
        self._refresh()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.BOTH, expand=True)
        filt = ttk.Frame(top)
        filt.pack(fill=tk.X)

        self.var_dir = tk.StringVar(value=str(self.data_dir))
        self.var_baseline_subtract = tk.BooleanVar(value=False)
        self.var_auto_trim_tail = tk.BooleanVar(value=True)
        
        pan = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        pan.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        left = ttk.Frame(pan, padding=6)
        right = ttk.Frame(pan, padding=6)
        self._right = right
        pan.add(left, weight=1)
        pan.add(right, weight=2)
        left.rowconfigure(6, weight=1)
        left.columnconfigure(0, weight=1)

        ttk.Label(left, text="Sessions").pack(anchor="w")
        sframe = ttk.Frame(left)
        sframe.pack(fill=tk.BOTH, expand=True)
        self.slist = ttk.Treeview(
            sframe,
            columns=("started", "session_id", "snips"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        self.slist.heading("started", text="Started")
        self.slist.heading("session_id", text="Session")
        self.slist.heading("snips", text="Snips")
        self.slist.column("started", width=140, anchor="w")
        self.slist.column("session_id", width=140, anchor="w")
        self.slist.column("snips", width=64, anchor="e")
        sscroll = ttk.Scrollbar(sframe, orient=tk.VERTICAL, command=self.slist.yview)
        self.slist.configure(yscrollcommand=sscroll.set)
        self.slist.grid(row=0, column=0, sticky="nsew")
        sscroll.grid(row=0, column=1, sticky="ns")
        sframe.columnconfigure(0, weight=1)
        sframe.rowconfigure(0, weight=1)
        self.slist.bind("<<TreeviewSelect>>", self._on_session)
        ttk.Label(left, textvariable=self._session_summary).pack(anchor="w", pady=(4, 0))
        ttk.Label(left, text="Date").pack(anchor="w", pady=(8,0))
        self.var_date = tk.StringVar(value="")
        self.cmb_date = ttk.Combobox(left, textvariable=self.var_date, state="readonly", width=18)
        self.cmb_date.pack(fill=tk.X, expand=False)
        self.cmb_date.bind("<<ComboboxSelected>>", self._on_date)
        hour_row = ttk.Frame(left)
        hour_row.pack(fill=tk.X, pady=(6,0))
        ttk.Label(hour_row, text="Minute").pack(side=tk.LEFT)
        self.var_seek = tk.StringVar(value="")
        seek_entry = ttk.Entry(hour_row, textvariable=self.var_seek, width=7)
        seek_entry.pack(side=tk.LEFT, padx=(8,4))
        ttk.Label(hour_row, text="MM:SS").pack(side=tk.LEFT)
        ttk.Button(hour_row, text="Go", command=self._seek_mmss).pack(side=tk.LEFT, padx=(6,0))
        # allow Enter in the seek box to trigger
        seek_entry.bind("<Return>", lambda _e: self._seek_mmss())

        self.hlist = tk.Listbox(left, height=8, exportselection=False)
        self.hlist.pack(fill=tk.X, expand=False)
        self.hlist.bind("<<ListboxSelect>>", self._on_hour)
        self.hlist.bind("<MouseWheel>", self._on_hour_wheel)

        ttk.Label(left, text="Waveform snippets").pack(anchor="w", pady=(8,0))
        wframe = ttk.Frame(left)
        wframe.pack(fill=tk.BOTH, expand=True)
        self.wlist = ttk.Treeview(
            wframe,
            columns=("time", "id", "buf", "rec", "global"),
            show="headings",
            height=14,
            selectmode="browse",
        )
        self.wlist.heading("time", text="Timestamp")
        self.wlist.heading("id", text="ID")
        self.wlist.heading("buf", text="Buf")
        self.wlist.heading("rec", text="Rec")
        self.wlist.heading("global", text="Global")
        self.wlist.column("time", width=125, anchor="w")
        self.wlist.column("id", width=52, anchor="e")
        self.wlist.column("buf", width=52, anchor="e")
        self.wlist.column("rec", width=52, anchor="e")
        self.wlist.column("global", width=72, anchor="e")
        wscroll = ttk.Scrollbar(wframe, orient=tk.VERTICAL, command=self.wlist.yview)
        self.wlist.configure(yscrollcommand=wscroll.set)
        self.wlist.grid(row=0, column=0, sticky="nsew")
        wscroll.grid(row=0, column=1, sticky="ns")
        wframe.columnconfigure(0, weight=1)
        wframe.rowconfigure(0, weight=1)
        self.wlist.bind("<<TreeviewSelect>>", self._on_snip)

        plot_opts = ttk.Frame(self._right)
        plot_opts.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(
            plot_opts,
            text="Baseline subtract (archive view)",
            variable=self.var_baseline_subtract,
            command=self._redraw_selected_snip,
        ).pack(side=tk.LEFT)
        ttk.Checkbutton(
            plot_opts,
            text="Auto trim flat tail",
            variable=self.var_auto_trim_tail,
            command=self._redraw_selected_snip,
        ).pack(side=tk.LEFT, padx=(10, 0))

        _, plt, _FigureCanvasTkAgg = _lazy_mpl()
        self.fig, (self.axA, self.axB, self.axI) = plt.subplots(3, 1, figsize=(7.5,6.8))
        self.fig.patch.set_facecolor(T_BG)
        self.axA.set_facecolor(T_SURFACE)
        self.axB.set_facecolor(T_SURFACE)
        self.axI.set_facecolor(T_SURFACE)
        self.fig.tight_layout(pad=1.0)

        # Canvas for matplotlib figure
        self.canvas = _FigureCanvasTkAgg(self.fig, master=self._right)

        # Navigation toolbar for zoom/pan/home/save
        self._nav_toolbar = None
        try:
            try:
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _NavTB
            except ImportError:
                from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as _NavTB
            toolbar_frame = tk.Frame(self._right, bg=T_SURFACE, height=36)
            toolbar_frame.pack(side=tk.TOP, fill=tk.X)
            toolbar_frame.pack_propagate(True)
            self._nav_toolbar = _NavTB(self.canvas, toolbar_frame)
            self._nav_toolbar.update()
        except Exception as e:
            print(f"[CAPPY] Toolbar init failed: {e}")

        # Pack canvas below toolbar
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Readout label below plots
        self._readout_var = tk.StringVar(value="")
        readout_lbl = tk.Label(self._right, textvariable=self._readout_var,
                               font=('Consolas', 10, 'bold'), fg=T_CYAN, bg=T_BG,
                               anchor='w', padx=6, pady=2)
        readout_lbl.pack(fill=tk.X)

        self.meta = tk.Text(self._right, height=10, bg=T_SURFACE, fg=T_TEXT, insertbackground=T_CYAN, font=('Consolas', 9))
        self.meta.pack(fill=tk.X, pady=(8,0))
        self.meta.configure(state=tk.DISABLED)

        # Hover state
        self._hover_annots = {}
        self._hover_lines = {}
        self._hover_sr = 1.0
        self._hover_ts_ns = 0
        self._hover_time_unit = "s"
        self._hover_vlines = {}
        self._init_hover_vlines()
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _init_hover_vlines(self):
        """Create/recreate crosshair vlines for all axes."""
        self._hover_vlines = {}
        try:
            self._hover_vlines[self.axA] = self.axA.axvline(0, color=NEON_PINK, alpha=0.4, linewidth=0.8, visible=False)
            self._hover_vlines[self.axB] = self.axB.axvline(0, color=NEON_GREEN, alpha=0.4, linewidth=0.8, visible=False)
            self._hover_vlines[self.axI] = self.axI.axvline(0, color='white', alpha=0.4, linewidth=0.8, visible=False)
        except Exception:
            pass

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        for iid in tree.get_children():
            tree.delete(iid)

    def _pick_dir(self):
        p = filedialog.askdirectory(title="Select data directory")
        if p:
            self.data_dir = Path(p)
            self.captures = self.data_dir / "captures"
            self.var_dir.set(p)
            self._refresh()

    def _iter_day_dirs(self):
        """Yield all YYYY-MM-DD directories under captures/YYYY/YYYY-MM/"""
        if not self.captures.exists():
            return
        for ydir in sorted(self.captures.iterdir()):
            if not ydir.is_dir():
                continue
            for mdir in sorted(ydir.iterdir()):
                if not mdir.is_dir():
                    continue
                for ddir in sorted(mdir.iterdir()):
                    if not ddir.is_dir():
                        continue
                    # day folder should be YYYY-MM-DD
                    try:
                        _ = datetime.strptime(ddir.name, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    yield ddir

    def _list_sessions(self, sd: Optional[date], ed: Optional[date]) -> _lazy_pandas().DataFrame:
        rows = []
        for ddir in self._iter_day_dirs() or []:
            try:
                d = datetime.strptime(ddir.name, "%Y-%m-%d").date()
            except ValueError:
                continue
            if sd and d < sd:
                continue
            if ed and d > ed:
                continue
            idx = ddir / "session_index.parquet"
            if idx.exists():
                try:
                    rows += _lazy_pandas().read_parquet(idx).to_dict("records")
                except Exception:
                    pass
        if not rows:
            cols = SESSION_INDEX_SCHEMA.names if SESSION_INDEX_SCHEMA is not None else SESSION_INDEX_COLUMNS
            return _lazy_pandas().DataFrame(columns=cols)
        return _lazy_pandas().DataFrame(rows).sort_values("first_timestamp_ns", ascending=False)

    def _refresh(self):
        try:
            self.sessions = self._list_sessions(None, None)
            self.sessions_all = self.sessions.copy()
            self._sessions_view = self.sessions.copy().reset_index(drop=True)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            self.sessions = _lazy_pandas().DataFrame()

        self._clear_tree(self.slist)
        self._clear_tree(self.wlist)

        if self.sessions.empty:
            self._session_summary.set("No sessions found")
            self.slist.insert("", tk.END, values=("-", "(no sessions)", "-"))
            return

        self._sessions_view = getattr(self, '_sessions_view', self.sessions).reset_index(drop=True)
        total_snips = 0
        for i, r in self._sessions_view.iterrows():
            t0 = datetime.fromtimestamp(int(r["first_timestamp_ns"]) / 1e9, tz=self._tz)
            snips = int(r.get("waveform_snips", 0))
            total_snips += snips
            self.slist.insert(
                "",
                tk.END,
                iid=f"session_{i}",
                values=(t0.strftime("%Y-%m-%d %H:%M:%S"), str(r["session_id"]), snips),
            )
        self._session_summary.set(f"Sessions: {len(self._sessions_view):,}   Total snips: {total_snips:,}")

        # If a session is already selected and snips are loaded, re-apply snip filter.
        if getattr(self, 'snips', _lazy_pandas().DataFrame()).empty is False:
            self._apply_hour_filter()

    def _sel_sid(self) -> Optional[str]:
        if self.sessions.empty:
            return None
        sel = self.slist.selection()
        if not sel:
            return None
        vals = self.slist.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return None
        sid = str(vals[1]).strip()
        if sid.startswith("("):
            return None
        return sid


    def _build_hour_index(self):
        """Add _date/_hour/_minute columns to self.snips (local time)."""
        if self.snips is None or self.snips.empty:
            return
        # Treat stored ns as UTC epoch, then convert to local timezone for display/grouping.
        dt = _lazy_pandas().to_datetime(self.snips["timestamp_ns"], unit="ns", utc=True).dt.tz_convert(self._tz)
        self.snips = self.snips.copy()
        self.snips["_date"] = dt.dt.strftime("%Y-%m-%d")
        self.snips["_hour"] = dt.dt.strftime("%H")
        self.snips["_minute"] = dt.dt.strftime("%H:%M")

    def _populate_hours(self):
        """Populate date combobox + hour listbox from currently loaded snips."""
        self.hlist.delete(0, tk.END)
        if self.snips is None or self.snips.empty:
            try:
                self.cmb_date["values"] = []
                self.var_date.set("")
            except Exception:
                pass
            return

        self._build_hour_index()

        # Dates available in this session's snips
        dates = sorted(self.snips["_date"].unique().tolist())
        # show newest first in UI
        dates = list(reversed(dates))

        try:
            self.cmb_date["values"] = dates
        except Exception:
            pass

        # Default to newest date unless user already selected one
        if self._sel_date is None or self._sel_date not in dates:
            self._sel_date = dates[0] if dates else None

        if self._sel_date is not None:
            self.var_date.set(str(self._sel_date))

        # Minute buckets for selected date
        sub = self.snips[self.snips["_date"] == self._sel_date] if self._sel_date is not None else self.snips
        counts = sub["_minute"].value_counts().sort_index()
        minutes = list(counts.index)

        for hm in minutes:
            self.hlist.insert(tk.END, f"{hm}  ({int(counts[hm])})")

        if self._sel_hour is None or self._sel_hour not in minutes:
            self._sel_hour = str(minutes[0]) if minutes else None

        if self._sel_hour is not None and self._sel_hour in minutes:
            idx = minutes.index(self._sel_hour)
            self.hlist.selection_clear(0, tk.END)
            self.hlist.selection_set(idx)
            self.hlist.see(idx)

    def _apply_hour_filter(self):
        """Render wlist using selected date/hour."""
        self._clear_tree(self.wlist)
        self._snips_view = _lazy_pandas().DataFrame()
        if self.snips is None or self.snips.empty:
            self.wlist.insert("", tk.END, values=("(no saved waveforms)", "-", "-", "-", "-"))
            return
        df = self.snips
        if "_date" in df.columns and self._sel_date is not None:
            df = df[df["_date"] == self._sel_date]
        if "_minute" in df.columns and self._sel_hour is not None:
            df = df[df["_minute"] == self._sel_hour]

        self._snips_view = df.sort_values("timestamp_ns", ascending=False)

        if self._snips_view.empty:
            self.wlist.insert("", tk.END, values=("(no waveforms in selected minute)", "-", "-", "-", "-"))
            return

        for _, r in self._snips_view.iterrows():
            ts_ns_val = int(r["timestamp_ns"])
            ts = datetime.fromtimestamp(ts_ns_val / 1e9, tz=self._tz)
            # Show full microsecond precision: HH:MM:SS.uuuuuu
            us_str = ts.strftime('%H:%M:%S') + f".{ts_ns_val % 1_000_000_000 // 1000:06d}"
            sid = int(r["id"])
            self.wlist.insert(
                "",
                tk.END,
                iid=f"snip_{sid}",
                values=(us_str, sid, int(r["buffer_index"]), int(r["record_in_buffer"]), int(r["record_global"])),
            )

    def _on_date(self, _=None):
        if self.snips is None or self.snips.empty:
            return
        d = (self.var_date.get() or "").strip()
        if not d:
            return
        self._sel_date = d
        # reset hour selection so we pick first available for that date
        self._sel_hour = None
        self._populate_hours()
        self._apply_hour_filter()

    def _on_hour(self, _=None):
        if self.snips is None or self.snips.empty:
            return
        sel = self.hlist.curselection()
        if not sel:
            return
        txt = self.hlist.get(sel[0])
        # 'HH:MM  (N)'
        self._sel_hour = txt.split()[0].strip()
        self._apply_hour_filter()


    def _seek_mmss(self):
        """Jump to the snip closest to MM:SS (or SS) in the selected date/minute."""
        if getattr(self, "_snips_view", _lazy_pandas().DataFrame()).empty:
            return
        txt = (self.var_seek.get() if hasattr(self, "var_seek") else "").strip()
        if not txt:
            return
        # Parse MM:SS or SS. With SS, keep currently selected minute.
        mm = 0
        ss = 0
        try:
            if ":" in txt:
                a, b = txt.split(":", 1)
                mm = int(a)
                ss = int(b)
            else:
                if self._sel_hour is not None and ":" in str(self._sel_hour):
                    mm = int(str(self._sel_hour).split(":", 1)[1])
                ss = int(txt)
        except Exception:
            messagebox.showerror("Invalid time", "Enter time as MM:SS (or SS).")
            return
        if ss < 0 or ss > 59 or mm < 0:
            messagebox.showerror("Invalid time", "Seconds must be 0–59.")
            return
        if self._sel_date is None or self._sel_hour is None:
            return
        try:
            # Build a local-time target and compare in epoch ns
            y, m, d = map(int, str(self._sel_date).split("-"))
            hh_txt = str(self._sel_hour).strip()
            if ":" in hh_txt:
                hh = int(hh_txt.split(":", 1)[0])
            else:
                hh = int(hh_txt)
            target_local = datetime(y, m, d, hh, mm, ss, tzinfo=self._tz)
            target_ns = int(target_local.timestamp() * 1e9)
        except Exception:
            return

        df = self._snips_view.copy()
        # Find nearest timestamp
        try:
            idx_min = (df["timestamp_ns"].astype("int64") - target_ns).abs().idxmin()
        except Exception:
            return
        # Position in the currently rendered order
        try:
            pos = df.reset_index().index[df.reset_index()["index"] == idx_min][0]
        except Exception:
            # fallback: brute force
            pos = 0
        items = self.wlist.get_children()
        if not items:
            return
        pos = max(0, min(len(items) - 1, int(pos)))
        target = items[pos]
        self.wlist.selection_set(target)
        self.wlist.focus(target)
        self.wlist.see(target)
        # trigger display
        try:
            self._on_snip()
        except Exception:
            pass

    def _on_hour_wheel(self, evt):
        # Mouse wheel scroll selects next/prev hour entry
        if self.hlist.size() == 0:
            return "break"
        cur = self.hlist.curselection()
        idx = int(cur[0]) if cur else 0
        idx = max(0, min(self.hlist.size() - 1, idx + (-1 if evt.delta > 0 else 1)))
        self.hlist.selection_clear(0, tk.END)
        self.hlist.selection_set(idx)
        self.hlist.see(idx)
        self._on_hour()
        return "break"

    def _on_session(self, _=None):
        sid = self._sel_sid()
        if not sid:
            return
        self.snips = _lazy_pandas().DataFrame()
        self._snip_db_dir = None
        self._snips_total_count = 0

        for ddir in self._iter_day_dirs() or []:
            idx_dir = ddir / "index"
            if not idx_dir.exists():
                continue
            # Try exact match first
            db = idx_dir / f"snips_{sid}.sqlite"
            if not db.exists():
                # Fallback: search for any sqlite file containing the session_id
                candidates = sorted(idx_dir.glob("snips_*.sqlite"))
                for c in candidates:
                    if sid in c.stem:
                        db = c
                        break
                else:
                    continue
            if not db.exists():
                continue
            conn = sqlite3.connect(db)
            try:
                try:
                    self._snips_total_count = int(conn.execute("SELECT COUNT(*) FROM snips WHERE session_id=?", (sid,)).fetchone()[0] or 0)
                except Exception:
                    self._snips_total_count = 0
                self.snips = _lazy_pandas().read_sql_query(
                    "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                    "file,offset_bytes,nbytes,"
                    "file_A,offset_A,nbytes_A,codec_A,file_B,offset_B,nbytes_B,codec_B,"
                    "area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V "
                    "FROM snips WHERE session_id=? ORDER BY timestamp_ns DESC LIMIT ?",
                    conn,
                    params=(sid, self._snip_load_limit),
                )
                # If query returned nothing, try without session_id filter (DB might only have one session)
                if self.snips.empty:
                    try:
                        self._snips_total_count = int(conn.execute("SELECT COUNT(*) FROM snips").fetchone()[0] or 0)
                    except Exception:
                        self._snips_total_count = 0
                    self.snips = _lazy_pandas().read_sql_query(
                        "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                        "file,offset_bytes,nbytes,"
                        "file_A,offset_A,nbytes_A,codec_A,file_B,offset_B,nbytes_B,codec_B,"
                        "area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V "
                        "FROM snips ORDER BY timestamp_ns DESC LIMIT ?",
                        conn,
                        params=(self._snip_load_limit,),
                    )
            except Exception:
                # Legacy DB schema (no channel-separated columns or baseline columns)
                try:
                    self.snips = _lazy_pandas().read_sql_query(
                        "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                        "file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V "
                        "FROM snips WHERE session_id=? ORDER BY timestamp_ns DESC LIMIT ?",
                        conn,
                        params=(sid, self._snip_load_limit),
                    )
                    if self.snips.empty:
                        self.snips = _lazy_pandas().read_sql_query(
                            "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                            "file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V "
                            "FROM snips ORDER BY timestamp_ns DESC LIMIT ?",
                            conn,
                            params=(self._snip_load_limit,),
                        )
                except Exception:
                    pass
            finally:
                conn.close()
            self._snip_db_dir = ddir
            break

        self._clear_tree(self.wlist)
        if self.snips.empty:
            self.wlist.insert("", tk.END, values=("(no saved waveforms)", "-", "-", "-", "-"))
            return
        # Populate hour list and apply hour filter
        self._populate_hours()
        self._apply_hour_filter()
        loaded = len(self.snips)
        total = int(self._snips_total_count or loaded)
        if total > loaded:
            self._set_meta(
                f"Snips loaded: {loaded:,} of {total:,} "
                f"(limited to {self._snip_load_limit:,}; set CAPPY_ARCHIVE_SNIP_LIMIT to change)"
            )
        else:
            self._set_meta(f"Snips loaded: {loaded:,}")

    def _redraw_selected_snip(self):
        try:
            self._on_snip()
        except Exception:
            pass

    def _on_snip(self, _=None):
        if self.snips.empty or self._snip_db_dir is None:
            return
        sel = self.wlist.selection()
        if not sel:
            return
        try:
            vals = self.wlist.item(sel[0], "values")
            if not vals or len(vals) < 2:
                return
            snip_id = int(vals[1])
        except Exception:
            return

        df = self._snips_view if (hasattr(self, '_snips_view') and not self._snips_view.empty) else self.snips
        row = df[df["id"] == snip_id]
        if row.empty:
            self._set_meta(f"Error: Could not find snip with id={snip_id}")
            return
        r = row.iloc[0]

        try:
            wa, wb = load_waveforms_from_row(r, self._snip_db_dir)
        except Exception as e:
            self._set_meta(f"Error loading waveform: {e}\nRow data: {r.to_dict()}")
            import traceback
            traceback.print_exc()
            return

        channels_mask_text = str(r.get("channels_mask", "CHANNEL_A") or "CHANNEL_A")
        row_has_b = bool(channels_from_mask_expr(channels_mask_text) & 0x2)
        if not row_has_b:
            wb = None

        sr = float(r.get("sample_rate_hz", np.nan))
        if not np.isfinite(sr) or sr <= 0:
            sr = 1.0

        # Baseline values are recorded during acquisition; fall back to first 64 samples for legacy rows.
        baseline_A = float(r.get("baseline_A_V", 0.0))
        baseline_B = float(r.get("baseline_B_V", 0.0))

        if baseline_A == 0.0 and len(wa) >= 64:
            baseline_A = float(np.mean(wa[:64]))
        if baseline_B == 0.0 and wb is not None and len(wb) >= 64:
            baseline_B = float(np.mean(wb[:64]))

        # Integrals should represent signal charge/area, so always use baseline-subtracted waveforms.
        wa_int = wa - baseline_A
        wb_int = (wb - baseline_B) if wb is not None else None

        do_baseline_subtract = bool(self.var_baseline_subtract.get()) if hasattr(self, "var_baseline_subtract") else False
        wa_plot = (wa - baseline_A) if do_baseline_subtract else wa
        wb_plot = ((wb - baseline_B) if do_baseline_subtract else wb) if wb is not None else None

        # Time axis with adaptive units (ns/us/ms/s)
        tvec, unit = _auto_time_axis(len(wa), sr)

        # Store hover context
        self._hover_ts_ns = int(r["timestamp_ns"])
        self._hover_sr = sr
        self._hover_time_unit = unit
        self._hover_lines = {}

        # Clear axes
        self.axA.clear()
        self.axB.clear()
        self.axI.clear()

        # Re-create crosshair lines (cleared by ax.clear())
        self._init_hover_vlines()
        self._hover_annots = {}  # will be recreated on next hover per-axis

        # Waveform A
        self.axA.plot(tvec, wa_plot, color=NEON_PINK, linewidth=1.5)
        self.axA.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
        self.axA.set_ylabel("A (V)", color=NEON_PINK)
        self.axA.tick_params(colors=NEON_PINK)
        self.axA.spines['left'].set_color(NEON_PINK)
        self.axA.spines['bottom'].set_color('white')
        self.axA.spines['top'].set_visible(False)
        self.axA.spines['right'].set_visible(False)
        self.axA.grid(True, alpha=0.15, color=NEON_PINK)
        self._hover_lines[self.axA] = (tvec, wa_plot, "Ch A", NEON_PINK)

        # Waveform B (if available)
        if wb_plot is not None:
            self.axB.plot(tvec, wb_plot, color=NEON_GREEN, linewidth=1.5)
            self.axB.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
            self.axB.set_ylabel("B (V)", color=NEON_GREEN)
            self.axB.tick_params(colors=NEON_GREEN)
            self.axB.spines['left'].set_color(NEON_GREEN)
            self.axB.spines['bottom'].set_color('white')
            self.axB.spines['top'].set_visible(False)
            self.axB.spines['right'].set_visible(False)
            self._hover_lines[self.axB] = (tvec, wb_plot, "Ch B", NEON_GREEN)
        else:
            self.axB.text(0.02, 0.5, "Channel B not captured in this snip", transform=self.axB.transAxes, color='white')
            self.axB.set_ylabel("B (V)", color='white')
            self.axB.tick_params(colors='white')
            self.axB.spines['left'].set_color('white')
            self.axB.spines['bottom'].set_color('white')
            self.axB.spines['top'].set_visible(False)
            self.axB.spines['right'].set_visible(False)
        self.axB.grid(True, alpha=0.15, color='white')

        # Cumulative integral (V·s)
        dt = 1.0 / sr
        intA = np.cumsum(np.asarray(wa_int, dtype=np.float64)) * dt
        self.axI.plot(tvec, intA, color=NEON_PINK, linewidth=1.5, label="∫A dt")
        self._hover_lines[self.axI] = (tvec, intA, "∫A (V·s)", NEON_PINK)
        if wb_int is not None:
            intB = np.cumsum(np.asarray(wb_int, dtype=np.float64)) * dt
            self.axI.plot(tvec, intB, color=NEON_GREEN, linewidth=1.5, linestyle="--", label="∫B dt")
        self.axI.set_ylabel("Integral (V·s)", color='white')
        self.axI.set_xlabel(f"Time ({unit})", color='white')
        self.axI.tick_params(colors='white')
        self.axI.spines['left'].set_color('white')
        self.axI.spines['bottom'].set_color('white')
        self.axI.spines['top'].set_visible(False)
        self.axI.spines['right'].set_visible(False)
        self.axI.grid(True, alpha=0.15, color='white')
        legend = self.axI.legend(loc="best")
        for text in legend.get_texts():
            text.set_color('white')

        auto_trim_tail = bool(self.var_auto_trim_tail.get()) if hasattr(self, "var_auto_trim_tail") else False
        if auto_trim_tail and len(tvec) > 8:
            trim_idx = _active_tail_index(wa_int)
            if wb_int is not None:
                trim_idx = max(trim_idx, _active_tail_index(wb_int))
            if trim_idx < (len(tvec) - 2):
                right = float(tvec[max(1, trim_idx)])
                left = float(tvec[0])
                self.axA.set_xlim(left=left, right=right)
                self.axB.set_xlim(left=left, right=right)
                self.axI.set_xlim(left=left, right=right)

        self.canvas.draw()

        # Reset the toolbar's home/zoom history so 'Home' button returns to this view
        if self._nav_toolbar is not None:
            try:
                self._nav_toolbar.update()
            except Exception:
                pass

        ts_ns_val = int(r["timestamp_ns"])
        ts = datetime.fromtimestamp(ts_ns_val / 1e9, tz=self._tz)
        # Keep nanosecond precision so closely spaced records remain distinguishable.
        us_str = ts.strftime('%Y-%m-%d %H:%M:%S') + f".{ts_ns_val % 1_000_000_000:09d}"
        a_vs = float(r.get("area_A_Vs", np.nan)) if "area_A_Vs" in r else float(r.get("area_A_Vs", np.nan))
        b_vs = float(r.get("area_B_Vs", np.nan)) if "area_B_Vs" in r else float(r.get("area_B_Vs", np.nan))
        self._set_meta(
            f"Timestamp: {us_str}\n"
            f"Session: {r.get('session_id','?')}\n"
            f"Buffer: {int(r.get('buffer_index',-1))}  Record: {int(r.get('record_in_buffer',-1))}  Global: {int(r.get('record_global',-1))}\n"
            f"Channels: {r.get('channels_mask','?')}  Sample rate: {sr:.6g} Hz\n"
            f"Display mode: {'baseline-subtracted' if do_baseline_subtract else 'raw volts'}\n"
            f"Integral mode: baseline-subtracted cumulative\n"
            f"Area A: {float(r.get('area_A_Vs',0.0)):.6g} V·s   Peak A: {float(r.get('peak_A_V',0.0)):.6g} V\n"
            f"Area B: {float(r.get('area_B_Vs',0.0)):.6g} V·s   Peak B: {float(r.get('peak_B_V',0.0)):.6g} V\n"
            f"Baseline A: {baseline_A:.6g} V   Baseline B: {baseline_B:.6g} V\n"
            f"Waveform points: {int(r.get('npts', len(wa)))}"
        )

    def _on_hover(self, event):
        """Show timestamp + voltage at cursor position on archive waveform plots."""
        # Don't interfere with zoom/pan toolbar actions
        if self._nav_toolbar is not None:
            try:
                mode = getattr(self._nav_toolbar, 'mode', '') or ''
                if mode != '':
                    return
            except Exception:
                pass

        # If mouse left the axes or no data loaded, hide everything
        if event.inaxes is None or not self._hover_lines:
            changed = False
            for vl in self._hover_vlines.values():
                try:
                    if vl.get_visible():
                        vl.set_visible(False)
                        changed = True
                except Exception:
                    pass
            for ann in getattr(self, '_hover_annots', {}).values():
                try:
                    if ann.get_visible():
                        ann.set_visible(False)
                        changed = True
                except Exception:
                    pass
            if changed:
                self._readout_var.set("")
                self.canvas.draw_idle()
            return

        ax = event.inaxes
        if ax not in self._hover_lines:
            return

        tvec, ydata, label, color = self._hover_lines[ax]
        if tvec is None or ydata is None or len(tvec) == 0:
            return

        x = event.xdata
        if x is None:
            return

        # Find nearest sample
        idx = int(np.clip(np.searchsorted(tvec, x), 0, len(tvec) - 1))
        if idx > 0 and idx < len(tvec) - 1:
            if abs(tvec[idx - 1] - x) < abs(tvec[idx] - x):
                idx -= 1

        t_val = float(tvec[idx])
        v_val = float(ydata[idx])

        # Compute absolute timestamp for this sample
        unit = self._hover_time_unit
        if unit == "ns":
            t_sec = t_val * 1e-9
        elif unit == "\u00b5s":
            t_sec = t_val * 1e-6
        elif unit == "ms":
            t_sec = t_val * 1e-3
        else:
            t_sec = t_val

        abs_ns = self._hover_ts_ns + int(t_sec * 1e9)
        abs_dt = datetime.fromtimestamp(abs_ns / 1e9, tz=self._tz)
        ns_part = abs_ns % 1_000_000_000
        ts_str = abs_dt.strftime('%H:%M:%S') + f".{ns_part:09d}"

        # Build annotation text: timestamp + voltage right at cursor
        annot_text = f"{ts_str}\n{v_val:.6g} V"

        # Update bottom readout bar
        self._readout_var.set(f"{label}  t = {t_val:.4g} {unit}   V = {v_val:.6g} V   @ {ts_str}")

        # Show crosshair on all axes
        for a, vl in self._hover_vlines.items():
            try:
                vl.set_xdata([t_val])
                vl.set_visible(True)
            except Exception:
                pass

        # Hide annotations on other axes, show on active axis
        if not hasattr(self, '_hover_annots'):
            self._hover_annots = {}

        for a, ann in self._hover_annots.items():
            try:
                if a != ax:
                    ann.set_visible(False)
            except Exception:
                pass

        if ax not in self._hover_annots:
            self._hover_annots[ax] = ax.annotate("", xy=(0, 0), xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e1e", ec=color, lw=1.5, alpha=0.95),
                color="white", fontsize=9, fontfamily="monospace", zorder=100, visible=False)

        ann = self._hover_annots[ax]
        ann.xy = (t_val, v_val)
        ann.set_text(annot_text)
        ann.get_bbox_patch().set_edgecolor(color)
        ann.set_visible(True)
        self.canvas.draw_idle()

    def _set_meta(self, s: str):
        self.meta.configure(state=tk.NORMAL)
        self.meta.delete("1.0", tk.END)
        self.meta.insert(tk.END, s)
        self.meta.configure(state=tk.DISABLED)

    # --- Legacy callback compatibility (older bindings may call these names) ---
    def _on_session_(self, *args, **kwargs):
        return self._on_session(*args, **kwargs)

    def on_session(self, *args, **kwargs):
        return self._on_session(*args, **kwargs)

    def on_session_(self, *args, **kwargs):
        return self._on_session(*args, **kwargs)

    def _on_snip_(self, *args, **kwargs):
        return self._on_snip(*args, **kwargs)

    def on_snip(self, *args, **kwargs):
        return self._on_snip(*args, **kwargs)

    def on_snip_(self, *args, **kwargs):
        return self._on_snip(*args, **kwargs)

class LiveDashboard(ttk.Frame):
    """
    Dashboard optimized for capture monitoring with enhanced live updates:
      - Color-coded stats bar with rate, captures, uptime, throughput, ring lag
      - Artist-reuse plots for smoother redraw (no ax.clear() per frame)
      - Channel A: Cyan waveform     Channel B: Gold waveform
      - Integration history: Teal
      - Adaptive FPS with backlog-aware catchup
    """
    def __init__(self, master, data_dir_var: tk.StringVar, on_status=None):
        super().__init__(master, padding=6)
        self.data_dir_var = data_dir_var
        self._on_status = on_status
        self.status_path: Optional[Path] = None
        self.ring_path: Optional[Path] = None
        self._ring_npts = 1024
        self._ring_rec_bytes = None
        self._ring_hdr_bytes = 32
        self._ring_last_seq = 0
        self._ring_play_seq = 0
        self._ring_nslots = 0
        self._ring_session_id: str = ""

        # rolling history (x in seconds since capture start, to-scale)
        self.t: list[float] = []
        self.areaA: list[float] = []
        self.areaB: list[float] = []

        # rolling waveform history (store last N downsampled waveforms)
        self._wfA_hist: list[np.ndarray] = []
        self._wfB_hist: list[np.ndarray] = []
        # rolling stream buffers for true scrolling (concatenate each buffer waveform)
        self._streamA = np.empty((0,), dtype=np.float32)
        self._streamB = np.empty((0,), dtype=np.float32)
        self._stream_window = 20000  # points shown in scrolling mode
        self._stream_window_s = 2.0  # seconds shown in scrolling mode
        self._latest_ring_unix: Optional[float] = None

        self._started_unix: Optional[float] = None
        self._last_seen_seq: int = 0
        self._tick_after_id = None
        self._is_destroyed = False
        self._redraw_in_progress = False
        self._auto_trim_scope_tail = True
        self._ui_fps = 8.0  # higher default for smoother updates
        self._min_redraw_interval_s = 1.0 / self._ui_fps
        self._next_redraw_monotonic = 0.0
        self._last_plot_cfg = None
        self._rate_history: list[float] = []  # track rate trend

        # ── Stats bar (color-coded cards) ───────────────────────────────
        stats = tk.Frame(self, bg=T_BG)
        stats.pack(fill=tk.X, pady=(0, 4))

        def mkcard(parent, label: str, color: str = T_TEXT_DIM, width: int = 0):
            """Create a compact color-coded stat card."""
            f = tk.Frame(parent, bg=T_SURFACE, padx=10, pady=4,
                         highlightbackground=T_BORDER, highlightthickness=1)
            f.pack(side=tk.LEFT, padx=(0, 3), fill=tk.Y)
            tk.Label(f, text=label, font=('Consolas', 7, 'bold'),
                     fg=color, bg=T_SURFACE, anchor='w').pack(anchor='w')
            v = tk.Label(f, text="—", font=('Consolas', 11),
                         fg=T_TEXT_BRIGHT, bg=T_SURFACE, anchor='w')
            v.pack(anchor='w')
            return v

        self.lbl_rate = mkcard(stats, "RATE (Hz)", T_CYAN)
        self.lbl_caps = mkcard(stats, "BUFFERS", T_TEXT_DIM)
        self.lbl_started = mkcard(stats, "STARTED", T_TEXT_DIM)
        self.lbl_last = mkcard(stats, "LAST CAPTURE", T_GREEN)
        self.lbl_peak = mkcard(stats, "PEAK (V)", T_GOLD)

        # Extra live-update stats
        stats2 = tk.Frame(self, bg=T_BG)
        stats2.pack(fill=tk.X, pady=(0, 4))
        self.lbl_uptime = mkcard(stats2, "UPTIME", T_GOLD)
        self.lbl_throughput = mkcard(stats2, "THROUGHPUT", T_CYAN)
        self.lbl_ring_lag = mkcard(stats2, "RING LAG", T_ORANGE)
        self.lbl_state = mkcard(stats2, "STATE", T_GREEN)
        self.lbl_disk = mkcard(stats2, "DISK (session)", T_MAGENTA)

        # ── Plots (artist reuse for smooth updates) ─────────────────────
        _, plt, _FigureCanvasTkAgg = _lazy_mpl()
        self.fig = plt.Figure(figsize=(8.2, 7.2))
        self.fig.patch.set_facecolor(T_BG)
        self.fig.subplots_adjust(left=0.10, right=0.97, top=0.97, bottom=0.06, hspace=0.28)

        # Scope-like layout: Channel A (top), Channel B (middle), Integration history (bottom)
        self.ax_wfA = self.fig.add_subplot(311)
        self.ax_wfA.set_facecolor(T_SURFACE)
        self.ax_wfB = self.fig.add_subplot(312)
        self.ax_wfB.set_facecolor(T_SURFACE)
        self.ax_int = self.fig.add_subplot(313)
        self.ax_int.set_facecolor(T_SURFACE)

        # Create persistent line artists (no ax.clear() needed)
        (self._lineA,) = self.ax_wfA.plot([], [], color=T_CYAN, linewidth=0.9, antialiased=False)
        (self._lineB,) = self.ax_wfB.plot([], [], color=T_GOLD, linewidth=0.9, antialiased=False)
        (self._lineIA,) = self.ax_int.plot([], [], color=T_GREEN, linewidth=1.2, antialiased=False, label="Mean integral A (V·s)")
        (self._lineIB,) = self.ax_int.plot([], [], color=T_GOLD, linewidth=1.2, linestyle="--", antialiased=False, label="Mean integral B (V·s)")

        # Style axes once (not on every redraw)
        for ax, ylabel, ycolor in [
            (self.ax_wfA, "A (V)", T_CYAN),
            (self.ax_wfB, "B (V)", T_GOLD),
            (self.ax_int, "Integral (V·s)", T_GREEN),
        ]:
            ax.set_ylabel(ylabel, color=ycolor, fontsize=9)
            ax.tick_params(colors=T_TEXT_DIM, labelsize=8)
            ax.spines["left"].set_color(ycolor)
            ax.spines["bottom"].set_color(T_BORDER)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            grid_color = ycolor if ax != self.ax_int else T_BORDER
            ax.grid(True, alpha=0.12, color=grid_color, linestyle="-", linewidth=0.5)

        self._artists_initialized = True

        # ── Trigger level "J pointer" (horizontal line + label) ─────────
        # Shows the trigger threshold on the Channel A waveform, like the
        # AlazarTech front-panel J arrow.  Updated every tick from status.
        self._trig_level_V: Optional[float] = None
        self._trig_lineA = self.ax_wfA.axhline(y=0, color=T_MAGENTA, linewidth=1.0,
                                                 linestyle='--', alpha=0.7, visible=False)
        self._trig_labelA = self.ax_wfA.text(0.01, 0, 'J', transform=self.ax_wfA.get_yaxis_transform(),
                                              color=T_MAGENTA, fontsize=10, fontweight='bold',
                                              va='center', ha='left', visible=False,
                                              bbox=dict(boxstyle='round,pad=0.15', facecolor=T_BG, edgecolor=T_MAGENTA, alpha=0.85))
        # Also show on Channel B if trigger source is Channel B
        self._trig_lineB = self.ax_wfB.axhline(y=0, color=T_MAGENTA, linewidth=1.0,
                                                 linestyle='--', alpha=0.7, visible=False)
        self._trig_labelB = self.ax_wfB.text(0.01, 0, 'J', transform=self.ax_wfB.get_yaxis_transform(),
                                              color=T_MAGENTA, fontsize=10, fontweight='bold',
                                              va='center', ha='left', visible=False,
                                              bbox=dict(boxstyle='round,pad=0.15', facecolor=T_BG, edgecolor=T_MAGENTA, alpha=0.85))

        self.canvas = _FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.meta = tk.Text(self, height=3, bg=T_SURFACE, fg=T_TEXT, insertbackground=T_CYAN,
                            font=('Consolas', 9), relief=tk.FLAT, bd=0, padx=8, pady=4)
        self.meta.pack(fill=tk.X, pady=(4, 0))
        self.meta.configure(state=tk.DISABLED)

        self.bind("<Destroy>", self._on_destroy, add="+")
        # Schedule periodic UI updates
        self._schedule_tick()

    def _on_destroy(self, evt=None):
        if evt is not None and getattr(evt, "widget", None) is not self:
            return
        self._is_destroyed = True
        aid = self._tick_after_id
        self._tick_after_id = None
        if aid is not None:
            try:
                self.after_cancel(aid)
            except Exception:
                pass

    def _schedule_tick(self):
        if self._is_destroyed:
            return
        try:
            if not bool(self.winfo_exists()):
                return
            tick_ms = max(16, int(round(1000.0 / max(1.0, float(self._ui_fps)))))
            self._tick_after_id = self.after(tick_ms, self._tick)
        except Exception:
            self._tick_after_id = None

    def _set_meta(self, s: str):
        if self._is_destroyed:
            return
        try:
            if not bool(self.winfo_exists()) or not bool(self.meta.winfo_exists()):
                return
            self.meta.configure(state=tk.NORMAL)
            self.meta.delete("1.0", tk.END)
            self.meta.insert(tk.END, s)
            self.meta.configure(state=tk.DISABLED)
        except Exception:
            return

    def update_trigger_pointer(self, level_pct: float, source: str, vpp_a: float = 2.0, vpp_b: float = 2.0):
        """Called by LauncherGUI/LiveControlPanel to update J pointer in real-time when slider moves."""
        try:
            trig_V_a = (level_pct / 100.0) * (vpp_a / 2.0)
            trig_V_b = (level_pct / 100.0) * (vpp_b / 2.0)
            self._trig_level_V = trig_V_a
            src = source.strip().upper()
            show_on_a = src in ("TRIG_CHAN_A", "TRIG_EXTERNAL", "EXTERNAL", "CHANNEL A", "")
            show_on_b = src in ("TRIG_CHAN_B", "CHANNEL B")

            self._trig_lineA.set_ydata([trig_V_a, trig_V_a])
            self._trig_lineA.set_visible(show_on_a)
            self._trig_labelA.set_position((0.01, trig_V_a))
            self._trig_labelA.set_text(f"J {level_pct:+.0f}%")
            self._trig_labelA.set_visible(show_on_a)

            self._trig_lineB.set_ydata([trig_V_b, trig_V_b])
            self._trig_lineB.set_visible(show_on_b)
            self._trig_labelB.set_position((0.01, trig_V_b))
            self._trig_labelB.set_text(f"J {level_pct:+.0f}%")
            self._trig_labelB.set_visible(show_on_b)

            self.canvas.draw_idle()
        except Exception:
            pass

    def _read_status(self) -> Optional[dict]:
        data_dir = Path(self.data_dir_var.get()).expanduser()
        p = data_dir / "status" / "cappy_status.json"
        self.status_path = p
        try:
            if not p.exists():
                return None
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _is_scope_mode(self, snap: dict) -> bool:
        """
        Use trigger-locked scope display for hardware-triggered runs.
        Keep scrolling stream mode for auto-trigger/noise-test captures.
        """
        timeout_ms = _to_int(snap.get("trigger_timeout_ms", 0), 0)
        if timeout_ms > 0:
            return False
        src = str(snap.get("trigger_source", "") or "").strip().upper()
        return src in {"TRIG_EXTERNAL", "TRIG_CHAN_A", "TRIG_CHAN_B"}

    def _append_point(self, snap: dict) -> bool:
        seq = snap.get("status_seq", None)
        if seq is None:
            return False
        try:
            seq = int(seq)
        except Exception:
            return False
        if seq <= self._last_seen_seq:
            return False
        self._last_seen_seq = seq

        su = snap.get("started_unix", None)
        tu = snap.get("status_unix", None)
        if su is not None:
            try:
                self._started_unix = float(su)
            except Exception:
                pass
        if self._started_unix is None and tu is not None:
            try:
                self._started_unix = float(tu)
            except Exception:
                pass
        if self._started_unix is None or tu is None:
            return False

        t = float(tu) - float(self._started_unix)
        self.t.append(t)
        self.areaA.append(float(snap.get("buffer_mean_area_A", 0.0)))
        self.areaB.append(float(snap.get("buffer_mean_area_B", 0.0)))

        while self.t and (self.t[-1] - self.t[0]) > 600.0:
            self.t.pop(0); self.areaA.pop(0); self.areaB.pop(0)
        return True


    def _open_ring_from_status(self, snap: dict):
        rp = snap.get("live_ring_path", None)
        if not rp:
            return
        sid = str(snap.get("session_id", "") or "")
        p = Path(str(rp)).expanduser()
        session_changed = bool(sid) and (sid != self._ring_session_id)
        if self.ring_path != p or session_changed:
            self.ring_path = p
            if sid:
                self._ring_session_id = sid
            # reset playback on ring change
            self._ring_last_seq = 0
            self._ring_play_seq = 0
            self._wfA_hist.clear()
            self._wfB_hist.clear()
            self._streamA = np.empty((0,), dtype=np.float32)
            self._streamB = np.empty((0,), dtype=np.float32)
            self._latest_ring_unix = None

    def _read_ring_batch(self, max_items: int) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[int], Optional[int]]]:
        """
        Read up to max_items records from the live ring in one file open.
        This avoids repeated open/close overhead in the UI loop.
        """
        out: List[Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[int], Optional[int]]] = []
        if self.ring_path is None or max_items <= 0:
            return out

        try:
            import struct
            with open(self.ring_path, "rb", buffering=0) as f:
                hdr = f.read(32)
                if len(hdr) != 32:
                    return out
                magic, _ver, nslots, npts, _rsv, write_seq = struct.unpack("<8sIIIIQ", hdr)
                if magic != LiveRingWriter.MAGIC:
                    return out
                self._ring_nslots = int(nslots)
                self._ring_npts = int(npts)
                rec_bytes = (8 + 8 + 8 + 4 + 4) + (self._ring_npts * 4) + (self._ring_npts * 4)
                self._ring_rec_bytes = rec_bytes
                self._ring_last_seq = int(write_seq)

                # If the writer restarted (seq counter wrapped back), reset reader state.
                if self._ring_play_seq > self._ring_last_seq:
                    self._ring_play_seq = 0
                    self._wfA_hist.clear()
                    self._wfB_hist.clear()
                    self._streamA = np.empty((0,), dtype=np.float32)
                    self._streamB = np.empty((0,), dtype=np.float32)

                # initialize play seq to most recent history window on first open
                if self._ring_play_seq == 0 and self._ring_last_seq > 0:
                    # start just behind the head so the plot fills quickly
                    self._ring_play_seq = max(1, self._ring_last_seq - 50)

                if self._ring_play_seq >= self._ring_last_seq:
                    return out

                n_to_read = min(int(max_items), int(self._ring_last_seq - self._ring_play_seq))
                for _ in range(max(0, n_to_read)):
                    seq = self._ring_play_seq + 1
                    slot = (seq - 1) % self._ring_nslots
                    off = self._ring_hdr_bytes + slot * rec_bytes
                    f.seek(off)
                    rec_hdr = f.read(8 + 8 + 8 + 4 + 4)
                    if len(rec_hdr) != (8 + 8 + 8 + 4 + 4):
                        break
                    r_seq, t_unix, _buf_idx, chmask, _r = struct.unpack("<QdQII", rec_hdr)
                    if int(r_seq) != int(seq):
                        # slot not yet written / overwritten; jump to current head
                        self._ring_play_seq = self._ring_last_seq
                        break

                    wfA = np.frombuffer(f.read(self._ring_npts * 4), dtype=np.float32).copy()
                    wfB = np.frombuffer(f.read(self._ring_npts * 4), dtype=np.float32).copy()
                    if (chmask & 2) == 0:
                        wfB = None
                    out.append((wfA, wfB, float(t_unix), int(chmask), int(seq)))
                    self._ring_play_seq = seq
        except Exception:
            return out
        return out

    def _redraw(self, snap: dict):
        """
        Redraw all plots using artist reuse (set_data) for smooth, flicker-free updates.
        Only relimits axes when data changes — no ax.clear() per frame.
        """
        if self._redraw_in_progress:
            return
        self._redraw_in_progress = True
        try:
            live_cfg = snap.get("live", {}) if isinstance(snap.get("live", {}), dict) else {}
            sr_hz = _to_float(snap.get("sample_rate_hz", 0.0), 0.0)
            spr = _to_int(snap.get("samples_per_record", 0), 0)
            ring_pts = _to_int(live_cfg.get("ring_points", self._ring_npts or 1), self._ring_npts or 1)
            downsample = 1.0
            if spr > 0 and ring_pts > 0:
                downsample = max(1.0, float(spr) / float(ring_pts))
            dt_ms = ((1e3 / sr_hz) * downsample) if sr_hz > 0 else 1.0
            scope_mode = self._is_scope_mode(snap)
            x_label = "Time from trigger (ms)" if scope_mode else "Time from latest (ms)"

            scope_right_ms = None
            if scope_mode and self._auto_trim_scope_tail:
                trim_idx = -1
                if self._streamA.size > 8:
                    trim_idx = max(trim_idx, _active_tail_index(self._streamA))
                if self._streamB.size > 8 and not np.all(np.isnan(self._streamB)):
                    trim_idx = max(trim_idx, _active_tail_index(self._streamB))
                if trim_idx >= 0:
                    scope_right_ms = dt_ms * float(max(1, trim_idx))

            # ── Channel A waveform ──────────────────────────────────────
            if self._streamA.size > 1:
                stream_a = self._streamA.copy()
                stream_a = stream_a - np.mean(stream_a)
                if scope_mode:
                    x_full = np.arange(stream_a.size, dtype=np.float64) * dt_ms
                else:
                    x_full = (np.arange(stream_a.size, dtype=np.float64) - (stream_a.size - 1)) * dt_ms
                if stream_a.size > 50000:
                    step = max(1, stream_a.size // 20000)
                    x_view = x_full[::step]
                    y_view = stream_a[::step]
                else:
                    x_view = x_full
                    y_view = stream_a
                self._lineA.set_data(x_view, y_view)
                self.ax_wfA.set_title("")
                # Update limits
                if y_view.size > 0:
                    ylo, yhi = float(np.nanmin(y_view)), float(np.nanmax(y_view))
                    pad = max(abs(yhi - ylo) * 0.08, 1e-9)
                    self.ax_wfA.set_ylim(ylo - pad, yhi + pad)
            else:
                self._lineA.set_data([], [])
                self.ax_wfA.set_title("Channel A: waiting for waveforms…", color=T_TEXT_DIM, fontsize=9)
            self.ax_wfA.set_xlabel(x_label, color=T_TEXT_DIM, fontsize=8)

            # ── Channel B waveform ──────────────────────────────────────
            show_channel_b = _to_bool(live_cfg.get("show_channel_b", False), False)
            has_b_channel = show_channel_b and bool(channels_from_mask_expr(str(snap.get("channels_mask", "CHANNEL_A"))) & 0x2)
            if has_b_channel and self._streamB.size > 1 and not np.all(np.isnan(self._streamB)):
                stream_b = self._streamB.copy()
                stream_b = stream_b - np.mean(stream_b)
                if scope_mode:
                    xb_full = np.arange(stream_b.size, dtype=np.float64) * dt_ms
                else:
                    xb_full = (np.arange(stream_b.size, dtype=np.float64) - (stream_b.size - 1)) * dt_ms
                if stream_b.size > 50000:
                    step = max(1, stream_b.size // 20000)
                    xb_view = xb_full[::step]
                    yb_view = stream_b[::step]
                else:
                    xb_view = xb_full
                    yb_view = stream_b
                self._lineB.set_data(xb_view, yb_view)
                self.ax_wfB.set_title("")
                if yb_view.size > 0:
                    ylo, yhi = float(np.nanmin(yb_view)), float(np.nanmax(yb_view))
                    pad = max(abs(yhi - ylo) * 0.08, 1e-9)
                    self.ax_wfB.set_ylim(ylo - pad, yhi + pad)
            else:
                self._lineB.set_data([], [])
                if not show_channel_b:
                    self.ax_wfB.set_title("Channel B: display disabled", color=T_TEXT_DIM, fontsize=9)
                elif has_b_channel:
                    self.ax_wfB.set_title("Channel B: waiting for waveforms…", color=T_TEXT_DIM, fontsize=9)
                else:
                    self.ax_wfB.set_title("Channel B: disabled in channels mask", color=T_TEXT_DIM, fontsize=9)
            self.ax_wfB.set_xlabel(x_label, color=T_TEXT_DIM, fontsize=8)

            # ── X-axis limits (scope vs stream) ─────────────────────────
            if scope_mode:
                try:
                    if self._streamA.size > 1:
                        right_a = dt_ms * float(max(1, self._streamA.size - 1))
                        if scope_right_ms is not None:
                            right_a = min(right_a, max(dt_ms, float(scope_right_ms)))
                        self.ax_wfA.set_xlim(left=0.0, right=right_a)
                    if has_b_channel and self._streamB.size > 1:
                        right_b = dt_ms * float(max(1, self._streamB.size - 1))
                        if scope_right_ms is not None:
                            right_b = min(right_b, max(dt_ms, float(scope_right_ms)))
                        self.ax_wfB.set_xlim(left=0.0, right=right_b)
                except Exception:
                    pass
            elif self._stream_window > 1:
                w_ms = dt_ms * float(max(1, self._stream_window - 1))
                try:
                    if self._streamA.size > 1:
                        self.ax_wfA.set_xlim(left=-w_ms, right=0.0)
                    if has_b_channel and self._streamB.size > 1:
                        self.ax_wfB.set_xlim(left=-w_ms, right=0.0)
                except Exception:
                    pass

            # ── Integration history ─────────────────────────────────────
            if self.t:
                if len(self.t) > 5000:
                    step = max(1, len(self.t) // 2000)
                    t_view = self.t[::step]
                    areaA_view = self.areaA[::step]
                    areaB_view = self.areaB[::step] if self.areaB else []
                else:
                    t_view = self.t
                    areaA_view = self.areaA
                    areaB_view = self.areaB
                self._lineIA.set_data(t_view, areaA_view)
                if has_b_channel and areaB_view and np.any(np.isfinite(np.asarray(areaB_view, dtype=float))) and np.any(np.abs(np.asarray(areaB_view, dtype=float)) > 0):
                    self._lineIB.set_data(t_view, areaB_view)
                    self._lineIB.set_visible(True)
                else:
                    self._lineIB.set_data([], [])
                    self._lineIB.set_visible(False)
                # Update legend only if needed
                if not hasattr(self, '_int_legend_drawn'):
                    legend = self.ax_int.legend(loc="best", fontsize=7, facecolor=T_SURFACE,
                                                 edgecolor=T_BORDER, labelcolor=T_TEXT)
                    self._int_legend_drawn = True
                # Update y limits
                try:
                    aa = np.asarray(areaA_view, dtype=float)
                    ylo, yhi = float(np.nanmin(aa)), float(np.nanmax(aa))
                    pad = max(abs(yhi - ylo) * 0.08, 1e-9)
                    self.ax_int.set_ylim(ylo - pad, yhi + pad)
                    if t_view and len(t_view) > 1:
                        self.ax_int.set_xlim(float(t_view[0]), float(t_view[-1]))
                except Exception:
                    pass
                self.ax_int.set_title("")
            else:
                self._lineIA.set_data([], [])
                self._lineIB.set_data([], [])
                self.ax_int.set_title("Integration: waiting for data…", color=T_TEXT_DIM, fontsize=9)
            self.ax_int.set_xlabel("Time (s)", color=T_TEXT_DIM, fontsize=8)

            self.canvas.draw_idle()
        finally:
            self._redraw_in_progress = False

    def _tick(self):
        self._tick_after_id = None
        if self._is_destroyed:
            return
        try:
            if not bool(self.winfo_exists()):
                return
        except Exception:
            return

        try:
            snap = self._read_status()
            if snap:
                live_cfg = snap.get('live', {}) if isinstance(snap.get('live', {}), dict) else {}
                show_channel_b = _to_bool(live_cfg.get("show_channel_b", False), False)
                if callable(self._on_status):
                    try:
                        self._on_status(snap)
                    except Exception:
                        pass
                # stats
                try:
                    rate_val = float(snap.get('rate_hz',0.0))
                    self.lbl_rate.configure(text=f"{rate_val:.3f}")
                    # Track rate trend for throughput calc
                    self._rate_history.append(rate_val)
                    if len(self._rate_history) > 30:
                        self._rate_history = self._rate_history[-30:]
                except Exception:
                    self.lbl_rate.configure(text=str(snap.get("rate_hz", "—")))
                self.lbl_caps.configure(text=str(snap.get("buffers", "—")))
                self.lbl_started.configure(text=str(snap.get("started", "—")))
                self.lbl_last.configure(text=str(snap.get("last_capture", snap.get("time", "—"))))

                # mean peak stat (A [+ B])
                pA = snap.get("buffer_mean_peak_A", None)
                pB = snap.get("buffer_mean_peak_B", None)
                try:
                    if pA is None:
                        self.lbl_peak.configure(text="—")
                    else:
                        if show_channel_b and pB is not None and float(pB) != 0.0:
                            self.lbl_peak.configure(text=f"A {float(pA):.6g}   B {float(pB):.6g}")
                        else:
                            self.lbl_peak.configure(text=f"{float(pA):.6g}")
                except Exception:
                    self.lbl_peak.configure(text=str(pA))

                try:
                    if 'stream_window_points' in live_cfg:
                        self._stream_window = int(live_cfg.get('stream_window_points', self._stream_window))
                    if 'stream_window_seconds' in live_cfg:
                        self._stream_window_s = float(live_cfg.get('stream_window_seconds', self._stream_window_s))
                    if 'ui_fps' in live_cfg:
                        fps = _clamp_float(live_cfg.get('ui_fps', self._ui_fps), 1.0, LIVE_UI_FPS_MAX, self._ui_fps)
                        self._ui_fps = float(fps)
                        self._min_redraw_interval_s = 1.0 / self._ui_fps
                except Exception:
                    pass

                self._open_ring_from_status(snap)
                status_changed = self._append_point(snap)
                # Drain a bounded number of waveforms per UI tick to avoid CPU spikes.
                try:
                    max_wf = int((snap.get('live', {}) or {}).get('max_waveforms_per_tick', 12)) if isinstance(snap.get('live', {}), dict) else 12
                except Exception:
                    max_wf = 12
                max_wf = min(max(1, int(max_wf)), LIVE_MAX_CATCHUP_WAVEFORMS_PER_TICK)
                # If capture outpaces UI, drain more records for this tick so plots catch up.
                backlog = max(0, int(self._ring_last_seq) - int(self._ring_play_seq))
                if backlog > (2 * max_wf):
                    max_wf = min(max_wf * 2, LIVE_MAX_CATCHUP_WAVEFORMS_PER_TICK)
                channels_mask_text = str(snap.get("channels_mask", "CHANNEL_A") or "CHANNEL_A")
                status_has_b = show_channel_b and bool(channels_from_mask_expr(channels_mask_text) & 0x2)
                scope_mode = self._is_scope_mode(snap)
                plot_cfg = (
                    bool(scope_mode),
                    bool(status_has_b),
                    str(channels_mask_text),
                    int(self._stream_window),
                    float(self._stream_window_s),
                    int(self._ring_npts),
                )
                cfg_changed = plot_cfg != self._last_plot_cfg
                self._last_plot_cfg = plot_cfg

                def _append_stream(values: np.ndarray, is_b: bool) -> None:
                    if values is None or values.size == 0:
                        return
                    vals = values.astype(np.float32, copy=False)
                    if scope_mode:
                        # Scope mode shows a trigger-locked record; window points control
                        # how much of that record is displayed from trigger time onward.
                        if self._stream_window > 0 and vals.size > self._stream_window:
                            vals = vals[: self._stream_window]
                        if is_b:
                            self._streamB = vals.copy()
                        else:
                            self._streamA = vals.copy()
                        return
                    if is_b:
                        self._streamB = np.concatenate([self._streamB, vals])
                        if self._streamB.size > self._stream_window:
                            self._streamB = self._streamB[-self._stream_window:]
                    else:
                        self._streamA = np.concatenate([self._streamA, vals])
                        if self._streamA.size > self._stream_window:
                            self._streamA = self._streamA[-self._stream_window:]

                saw_ring_record = False
                saw_b_record = False
                ring_records = self._read_ring_batch(max_wf)
                for wfA, wfB, wf_t_unix, _chmask, _seq in ring_records:
                    if wfA is None and wfB is None:
                        continue
                    saw_ring_record = True
                    if wf_t_unix is not None:
                        self._latest_ring_unix = float(wf_t_unix)
                    if wfA is not None:
                        try:
                            _append_stream(wfA, is_b=False)
                        except Exception:
                            pass
                    if wfB is not None and status_has_b:
                        try:
                            _append_stream(wfB, is_b=True)
                            saw_b_record = True
                        except Exception:
                            pass

                # Prevent stale Channel B traces when capture switches to A-only.
                b_cleared = False
                if not status_has_b:
                    if self._streamB.size:
                        self._streamB = np.empty((0,), dtype=np.float32)
                        b_cleared = True
                elif saw_ring_record and (not saw_b_record):
                    self._streamB = np.empty((0,), dtype=np.float32)
                    b_cleared = True

                # ── Update J trigger pointer ────────────────────────────
                try:
                    trig_src = str(snap.get("trigger_source", "") or "").strip().upper()
                    trig_code = _to_int(snap.get("trigger_level_code", 128), 128)
                    trig_pct = _level_code_to_trigger_pct(trig_code)
                    vpp_a = _to_float(snap.get("vpp_A", 2.0), 2.0)
                    vpp_b = _to_float(snap.get("vpp_B", 2.0), 2.0)
                    # Convert trigger percent to volts (baseline-subtracted)
                    # The trigger level is relative to mid-scale (0V in bipolar mode)
                    trig_V_a = (trig_pct / 100.0) * (vpp_a / 2.0)
                    trig_V_b = (trig_pct / 100.0) * (vpp_b / 2.0)
                    self._trig_level_V = trig_V_a

                    # Show J pointer on the relevant channel
                    show_on_a = trig_src in ("TRIG_CHAN_A", "TRIG_EXTERNAL", "")
                    show_on_b = trig_src == "TRIG_CHAN_B"

                    self._trig_lineA.set_ydata([trig_V_a, trig_V_a])
                    self._trig_lineA.set_visible(show_on_a and self._streamA.size > 1)
                    self._trig_labelA.set_position((0.01, trig_V_a))
                    self._trig_labelA.set_visible(show_on_a and self._streamA.size > 1)

                    self._trig_lineB.set_ydata([trig_V_b, trig_V_b])
                    self._trig_lineB.set_visible(show_on_b and self._streamB.size > 1)
                    self._trig_labelB.set_position((0.01, trig_V_b))
                    self._trig_labelB.set_visible(show_on_b and self._streamB.size > 1)
                except Exception:
                    pass

                now_mono = time.monotonic()
                if now_mono >= self._next_redraw_monotonic and (saw_ring_record or status_changed or cfg_changed or b_cleared):
                    self._redraw(snap)
                    self._next_redraw_monotonic = now_mono + self._min_redraw_interval_s

                latest_ring = "-"
                if self._latest_ring_unix is not None:
                    try:
                        latest_ring = datetime.fromtimestamp(float(self._latest_ring_unix)).strftime("%H:%M:%S.%f")[:-3]
                    except Exception:
                        latest_ring = str(self._latest_ring_unix)
                mode = "scope" if scope_mode else "stream"

                # ── Enhanced live stats ─────────────────────────────────
                # Uptime
                try:
                    if self._started_unix is not None:
                        elapsed = time.time() - float(self._started_unix)
                        h, rem = divmod(int(elapsed), 3600)
                        m, s = divmod(rem, 60)
                        self.lbl_uptime.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
                    else:
                        self.lbl_uptime.configure(text="—")
                except Exception:
                    self.lbl_uptime.configure(text="—")

                # Throughput (rolling average rate × records_per_buffer)
                try:
                    rpb = _to_int(snap.get("records_per_buffer", 0), 0)
                    avg_rate = np.mean(self._rate_history) if self._rate_history else 0.0
                    tput = avg_rate * max(1, rpb)
                    if tput > 1e6:
                        self.lbl_throughput.configure(text=f"{tput/1e6:.2f} Mrec/s")
                    elif tput > 1e3:
                        self.lbl_throughput.configure(text=f"{tput/1e3:.1f} krec/s")
                    else:
                        self.lbl_throughput.configure(text=f"{tput:.0f} rec/s")
                except Exception:
                    self.lbl_throughput.configure(text="—")

                # Ring lag (how far behind the UI reader is from the writer)
                try:
                    lag = max(0, int(self._ring_last_seq) - int(self._ring_play_seq))
                    lag_color = T_GREEN if lag < 10 else (T_ORANGE if lag < 100 else T_RED)
                    self.lbl_ring_lag.configure(text=f"{lag:,}", fg=lag_color)
                except Exception:
                    self.lbl_ring_lag.configure(text="—")

                # State with color coding
                try:
                    state = str(snap.get('state', '?') or '?')
                    state_color = T_GREEN if state == 'capturing' else (T_ORANGE if state == 'waiting' else T_TEXT_DIM)
                    self.lbl_state.configure(text=state, fg=state_color)
                except Exception:
                    self.lbl_state.configure(text="?")

                # Disk usage for this session
                try:
                    disk_bytes = _to_int(snap.get("disk_bytes", 0), 0)
                    if disk_bytes <= 0:
                        reduced = _to_int(snap.get("reduced_rows", 0), 0)
                        snips = _to_int(snap.get("snips", 0), 0)
                        self.lbl_disk.configure(text=f"{reduced:,}r {snips:,}s")
                    elif disk_bytes > 1024**3:
                        self.lbl_disk.configure(text=f"{disk_bytes/1024**3:.2f} GiB")
                    elif disk_bytes > 1024**2:
                        self.lbl_disk.configure(text=f"{disk_bytes/1024**2:.1f} MiB")
                    else:
                        self.lbl_disk.configure(text=f"{disk_bytes/1024:.0f} KiB")
                except Exception:
                    self.lbl_disk.configure(text="—")

                self._set_meta(
                    f"State: {snap.get('state','?')}    Last ring wf: {latest_ring}    "
                    f"Points A/B: {self._streamA.size}/{self._streamB.size} (mode={mode}, window={self._stream_window})    Status: {self.status_path}"
                )
        except tk.TclError:
            return
        finally:
            self._schedule_tick()

class _ProcLogPump:
    """
    Reads a subprocess' stdout in a background thread and pushes lines into a Queue.
    Prevents Tkinter freezes from blocking readline().
    """
    def __init__(self, proc):
        self.proc = proc
        self.q: "queue.Queue[str]" = queue.Queue()
        self._stop_evt = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        try:
            f = self.proc.stdout
            if f is None:
                return
            for line in f:
                if self._stop_evt.is_set():
                    break
                self.q.put(line.rstrip("\n"))
        except Exception as e:
            try:
                self.q.put(f"[GUI] log pump error: {e}")
            except Exception:
                pass

    def stop(self):
        self._stop_evt.set()

    def drain(self, max_lines: int = 200):
        out = []
        for _ in range(max_lines):
            try:
                out.append(self.q.get_nowait())
            except queue.Empty:
                break
        return out



class LiveControlPanel(ttk.Frame):
    """
    Scope-style capture control panel bound directly to YAML config keys.
    Exposes capture/trigger/channel/storage/live controls (excluding math windows).
    """
    def __init__(self, master, launcher, **kwargs):
        super().__init__(master, **kwargs)
        self.launcher = launcher
        self._loading = False
        self._msg_var = tk.StringVar(value="")
        self._out_var = launcher._live_out_dir if hasattr(launcher, "_live_out_dir") else tk.StringVar(value="Output: -")
        self._written_var = launcher._live_written if hasattr(launcher, "_live_written") else tk.StringVar(value="Written: -")
        self._flush_var = launcher._live_last_flush if hasattr(launcher, "_live_last_flush") else tk.StringVar(value="Last update: -")

        # Vars (GUI <-> YAML)
        self.var_clock_source = tk.StringVar(value="INTERNAL_CLOCK")
        self.var_rate_msps = tk.DoubleVar(value=250.0)
        self.var_channels_mask = tk.StringVar(value="CHANNEL_A|CHANNEL_B")

        self.var_pre = tk.IntVar(value=0)
        self.var_samples_per_record = tk.IntVar(value=256)
        self.var_post = tk.IntVar(value=256)
        self.var_rpb = tk.IntVar(value=128)
        self.var_bufN = tk.IntVar(value=16)
        self.var_bpa = tk.IntVar(value=0)
        self.var_wait_timeout_ms = tk.IntVar(value=1000)

        self.var_range_a = tk.StringVar(value="PM_1_V")
        self.var_coupling_a = tk.StringVar(value="DC")
        self.var_imp_a = tk.StringVar(value="50_OHM")
        self.var_range_b = tk.StringVar(value="PM_1_V")
        self.var_coupling_b = tk.StringVar(value="DC")
        self.var_imp_b = tk.StringVar(value="50_OHM")

        self.var_trig_source = tk.StringVar(value="External")
        self.var_trig_slope = tk.StringVar(value="Positive")
        self.var_trig_level_pct = tk.DoubleVar(value=0.0)
        self.var_trig_timeout_ms = tk.IntVar(value=0)
        self.var_timeout_pause_s = tk.DoubleVar(value=0.0)
        self.var_trig_mode = tk.StringVar(value="TRIG_ENGINE_OP_J")
        self.var_ext_range = tk.StringVar(value="ETR_5V")
        self.var_ext_coupling = tk.StringVar(value="DC_COUPLING")
        self.var_ext_startcapture = tk.BooleanVar(value=False)
        self.var_trigger_delay_us = tk.DoubleVar(value=0.0)
        self._trig_level_code_var = tk.StringVar(value="code=128")
        # K engine
        self.var_trig_sourceK = tk.StringVar(value="Disabled")
        self.var_trig_slopeK = tk.StringVar(value="Positive")
        self.var_trig_levelK_pct = tk.DoubleVar(value=0.0)
        self._trig_levelK_code_var = tk.StringVar(value="code=128")
        self.var_wf_every_n = tk.IntVar(value=16)
        self.var_live_waveform_every_n = tk.IntVar(value=1)
        self.var_wf_max_per_sec = tk.IntVar(value=120)

        self.var_rearm_s = tk.IntVar(value=300)
        self.var_rearm_cooldown_s = tk.IntVar(value=30)
        self.var_rearm_per_hour = tk.IntVar(value=12)
        self.var_runtime_profile = tk.StringVar(value="Balanced")

        self.var_flush_records = tk.IntVar(value=20000)
        self.var_flush_seconds = tk.DoubleVar(value=2.0)
        self.var_sqlite_commit = tk.IntVar(value=200)
        self.var_rotate_hours = tk.DoubleVar(value=24.0)

        self.var_ring_slots = tk.IntVar(value=4096)
        self.var_ring_points = tk.IntVar(value=4096)
        self.var_stream_pts = tk.IntVar(value=100000)
        self.var_stream_s = tk.DoubleVar(value=2.0)
        self.var_max_wf_tick = tk.IntVar(value=12)
        self.var_show_channel_b_live = tk.BooleanVar(value=False)
        self.var_preview_mode = tk.StringVar(value="archive_match")
        self._math_var = tk.StringVar(value="")
        self._harmonizing = False
        self._spin_flush_records = None
        self._spin_stream_pts = None
        self._stream_edit_source = "points"
        self.var_trig_level_pct.trace_add("write", self._sync_trigger_level_code)
        self.var_trig_source.trace_add("write", self._on_trigger_source_or_slope_change)
        self.var_trig_slope.trace_add("write", self._on_trigger_source_or_slope_change)

        # Layout: one vertically scrollable controls partition.
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        scroll_host = ttk.Frame(self, padding=(8, 8, 0, 8))
        scroll_host.grid(row=0, column=0, sticky="nsew")
        scroll_host.columnconfigure(0, weight=1)
        scroll_host.rowconfigure(0, weight=1)

        self._scroll_canvas = tk.Canvas(
            scroll_host,
            bg=T_BG,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
        )
        self._scroll_canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar_y = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL, command=self._scroll_canvas.yview)
        self._scrollbar_y.grid(row=0, column=1, sticky="ns")
        self._scroll_canvas.configure(yscrollcommand=self._scrollbar_y.set)

        self._controls_root = ttk.Frame(self._scroll_canvas, padding=(0, 0, 8, 0))
        self._controls_root.columnconfigure(0, weight=1)
        self._controls_root_window = self._scroll_canvas.create_window((0, 0), window=self._controls_root, anchor="nw")
        self._controls_root.bind("<Configure>", self._on_controls_root_configure, add="+")
        self._scroll_canvas.bind("<Configure>", self._on_controls_canvas_configure, add="+")
        self._wheel_bound = False
        self._bind_controls_mousewheel()

        acq = ttk.LabelFrame(self._controls_root, text="  ▸ Acquire", padding=8, style='Acquire.TLabelframe')
        acq.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        acq.columnconfigure(1, weight=1)
        self._add_combobox(acq, 0, "Clock source", self.var_clock_source, ["INTERNAL_CLOCK", "EXTERNAL_CLOCK_10MHZ_REF"])
        self._add_combobox(acq, 1, "Rate (MS/s)", self.var_rate_msps, [str(v) for v in SAMPLE_RATE_OPTIONS_MSPS], width=12)
        self._add_combobox(acq, 2, "Channels", self.var_channels_mask, CHANNEL_MASK_OPTIONS, width=18)
        self._add_spinbox(acq, 3, "Pre-trigger samples", self.var_pre, from_=0, to=7_999_984, increment=16)
        self._add_spinbox(acq, 4, "Samples/record", self.var_samples_per_record, from_=16, to=8_000_000, increment=16)
        self._add_readonly_value(acq, 5, "Post-trigger samples", self.var_post)
        self._add_spinbox(acq, 6, "Records/buffer", self.var_rpb, from_=1, to=100_000, increment=1)
        self._add_spinbox(acq, 7, "Buffers allocated", self.var_bufN, from_=2, to=4096, increment=1)
        self._add_spinbox(acq, 8, "Buffers/acquisition (0=run)", self.var_bpa, from_=0, to=2_000_000_000, increment=1)
        self._add_spinbox(acq, 9, "DMA wait timeout (ms)", self.var_wait_timeout_ms, from_=10, to=120_000, increment=10)

        vert = ttk.LabelFrame(self._controls_root, text="  ▸ Vertical", padding=8, style='Vertical.TLabelframe')
        vert.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        vert.columnconfigure(1, weight=1)
        ttk.Label(vert, text="Channel A").grid(row=0, column=0, sticky="w")
        self._add_combobox(vert, 1, "A range", self.var_range_a, INPUT_RANGE_OPTIONS)
        self._add_combobox(vert, 2, "A coupling", self.var_coupling_a, COUPLING_OPTIONS)
        self._add_combobox(vert, 3, "A impedance", self.var_imp_a, IMPEDANCE_OPTIONS)
        ttk.Separator(vert, orient=tk.HORIZONTAL).grid(row=4, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(vert, text="Channel B").grid(row=5, column=0, sticky="w")
        self._add_combobox(vert, 6, "B range", self.var_range_b, INPUT_RANGE_OPTIONS)
        self._add_combobox(vert, 7, "B coupling", self.var_coupling_b, COUPLING_OPTIONS)
        self._add_combobox(vert, 8, "B impedance", self.var_imp_b, IMPEDANCE_OPTIONS)

        trig = ttk.LabelFrame(self._controls_root, text="  ▸ Trigger", padding=8, style='Trigger.TLabelframe')
        trig.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        trig.columnconfigure(1, weight=1)
        self._add_combobox(trig, 0, "Mode", self.var_trig_mode, TRIGGER_MODE_OPTIONS, width=18)
        self._add_combobox(trig, 1, "Source", self.var_trig_source, list(TRIGGER_SOURCE_LABEL_TO_CONST.keys()), width=12)
        self._add_combobox(trig, 2, "Slope", self.var_trig_slope, list(TRIGGER_SLOPE_LABEL_TO_CONST.keys()), width=12)
        ttk.Label(trig, text="Level (%FS)").grid(row=3, column=0, sticky="w", pady=2)
        trig_level = ttk.Frame(trig)
        trig_level.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=2)
        trig_level.columnconfigure(0, weight=1)
        ttk.Scale(
            trig_level,
            from_=-100.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.var_trig_level_pct,
            command=lambda _v: self._sync_trigger_level_code(),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Entry(trig_level, textvariable=self.var_trig_level_pct, width=7).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(trig_level, textvariable=self._trig_level_code_var, width=10).grid(row=0, column=2, padx=(6, 0))

        trig_presets = ttk.Frame(trig)
        trig_presets.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 2))
        ttk.Label(trig_presets, text="Quick level").pack(side=tk.LEFT)
        for pct in TRIGGER_LEVEL_PRESETS_PCT:
            ttk.Button(
                trig_presets,
                text=f"{int(pct)}%",
                command=lambda p=pct: self._set_trigger_level_pct(p),
                width=5,
            ).pack(side=tk.LEFT, padx=(4, 0))

        self._add_spinbox(trig, 5, "Auto timeout (ms)", self.var_trig_timeout_ms, from_=0, to=2_147_483_647, increment=1)
        self._add_spinbox(trig, 6, "Timeout pause (s)", self.var_timeout_pause_s, from_=0.0, to=3600.0, increment=0.1, width=10)
        self._add_combobox(trig, 7, "External range", self.var_ext_range, EXT_TRIGGER_RANGE_OPTIONS)
        self._add_combobox(trig, 8, "External coupling", self.var_ext_coupling, ["DC_COUPLING", "AC_COUPLING"])
        self._add_spinbox(trig, 9, "Trigger delay (µs)", self.var_trigger_delay_us, from_=0.0, to=100000.0, increment=0.1, width=10)
        ttk.Checkbutton(trig, text="External start-capture", variable=self.var_ext_startcapture).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        # ── Engine K ────────────────────────────────────────────────────
        ttk.Separator(trig, orient=tk.HORIZONTAL).grid(row=11, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(trig, text="Engine K", font=('Consolas', 9, 'bold')).grid(row=12, column=0, sticky="w")
        self._add_combobox(trig, 13, "K Source", self.var_trig_sourceK, list(TRIGGER_SOURCEK_LABEL_TO_CONST.keys()), width=12)
        self._add_combobox(trig, 14, "K Slope", self.var_trig_slopeK, list(TRIGGER_SLOPE_LABEL_TO_CONST.keys()), width=12)
        ttk.Label(trig, text="K Level (%FS)").grid(row=15, column=0, sticky="w", pady=2)
        trigK_level = ttk.Frame(trig)
        trigK_level.grid(row=15, column=1, sticky="ew", padx=(8, 0), pady=2)
        trigK_level.columnconfigure(0, weight=1)
        ttk.Scale(
            trigK_level, from_=-100.0, to=100.0, orient=tk.HORIZONTAL,
            variable=self.var_trig_levelK_pct,
            command=lambda _v: self._sync_trigK_level_code(),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Entry(trigK_level, textvariable=self.var_trig_levelK_pct, width=7).grid(row=0, column=1, padx=(6, 0))
        ttk.Label(trigK_level, textvariable=self._trig_levelK_code_var, width=10).grid(row=0, column=2, padx=(6, 0))

        live = ttk.LabelFrame(self._controls_root, text="  ▸ Runtime / Storage / Live", padding=8, style='Runtime.TLabelframe')
        live.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        live.columnconfigure(1, weight=1)
        ttk.Label(live, text="Runtime profile").grid(row=0, column=0, sticky="w", pady=2)
        live_profile = ttk.Frame(live)
        live_profile.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=2)
        live_profile.columnconfigure(0, weight=1)
        ttk.Combobox(
            live_profile,
            textvariable=self.var_runtime_profile,
            values=RUNTIME_PROFILE_OPTIONS,
            state="readonly",
            width=16,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(live_profile, text="Apply", command=self._apply_runtime_profile, width=7).grid(row=0, column=1, padx=(6, 0))

        self._add_spinbox(live, 1, "Rearm if no trigger (s)", self.var_rearm_s, from_=0, to=86_400, increment=1)
        self._add_spinbox(live, 2, "Rearm cooldown (s)", self.var_rearm_cooldown_s, from_=0, to=86_400, increment=1)
        self._add_spinbox(live, 3, "Max rearms/hour", self.var_rearm_per_hour, from_=1, to=100_000, increment=1)
        self._spin_flush_records = self._add_spinbox(
            live, 4, "Flush every records", self.var_flush_records, from_=1, to=50_000_000, increment=128
        )
        self._add_spinbox(live, 5, "Flush every seconds", self.var_flush_seconds, from_=0.0, to=86_400.0, increment=0.1, width=10)
        self._add_spinbox(live, 6, "SQLite commit every snips", self.var_sqlite_commit, from_=1, to=10_000_000, increment=128)
        self._add_spinbox(live, 7, "Session rotate (hours)", self.var_rotate_hours, from_=0.0, to=720.0, increment=0.5, width=10)
        self._add_spinbox(live, 8, "Ring slots", self.var_ring_slots, from_=16, to=1_000_000, increment=16)
        self._add_spinbox(live, 9, "Ring points", self.var_ring_points, from_=32, to=65536, increment=32)
        self._spin_stream_pts = self._add_spinbox(
            live, 10, "Live window points", self.var_stream_pts, from_=256, to=5_000_000, increment=512
        )
        self._add_spinbox(live, 11, "Live window seconds", self.var_stream_s, from_=0.25, to=120.0, increment=0.1, width=10)
        self._add_spinbox(live, 12, "Max waveforms/tick", self.var_max_wf_tick, from_=1, to=2000, increment=1)
        self._add_spinbox(live, 13, "Save waveform every N", self.var_wf_every_n, from_=1, to=10_000_000, increment=1)
        ttk.Label(live, text="Live waveform every N buffers").grid(row=14, column=0, sticky="w", pady=2)
        wf_slider = ttk.Frame(live)
        wf_slider.grid(row=14, column=1, sticky="ew", padx=(8, 0), pady=2)
        wf_slider.columnconfigure(0, weight=1)
        tk.Scale(
            wf_slider,
            from_=1,
            to=WAVEFORM_EVERY_N_MAX,
            orient=tk.HORIZONTAL,
            resolution=1,
            showvalue=False,
            variable=self.var_live_waveform_every_n,
            command=lambda _v: self._harmonize_controls(),
            bg=T_BG,
            fg=T_TEXT,
            troughcolor=T_SURFACE,
            highlightthickness=0,
            activebackground=T_CYAN,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(wf_slider, textvariable=self.var_live_waveform_every_n, width=6).grid(row=0, column=1, padx=(6, 0))
        self._add_spinbox(live, 15, "Saved waveforms/sec cap", self.var_wf_max_per_sec, from_=1, to=50_000, increment=1)
        ttk.Checkbutton(live, text="Show Channel B live waveform", variable=self.var_show_channel_b_live).grid(
            row=16, column=0, columnspan=2, sticky="w", pady=(4, 2)
        )
        self._add_combobox(live, 17, "Preview mode", self.var_preview_mode, ["archive_match", "record0"], width=16)
        ttk.Label(live, textvariable=self._math_var).grid(row=18, column=0, columnspan=2, sticky="w", pady=(6, 0))

        disk = ttk.LabelFrame(self._controls_root, text="  ▸ Disk write status", padding=8, style='Disk.TLabelframe')
        disk.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(disk, textvariable=self._out_var).pack(anchor="w")
        ttk.Label(disk, textvariable=self._written_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(disk, textvariable=self._flush_var).pack(anchor="w", pady=(2, 0))

        # ── Quick Config  ─────────────────────────────────────────────
        # Scout-capture a large sample, align on 0 V crossings, auto-set optimal
        # trigger level, samples/record, and input range.
        qc = ttk.LabelFrame(self._controls_root, text="  ⚡ Quick Config", padding=8, style='Trigger.TLabelframe')
        qc.grid(row=5, column=0, sticky="ew", pady=(0, 8))
        qc.columnconfigure(1, weight=1)

        self._qc_status_var = tk.StringVar(value="Large scout capture → zero-cross trigger + mean-peak record sizing")
        self._qc_running    = False
        self._qc_proc       = None

        qc_top = ttk.Frame(qc)
        qc_top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._qc_btn = ttk.Button(
            qc_top, text="⚡ Quick Config",
            command=self._quick_config,
            style="Primary.TButton",
        )
        self._qc_btn.pack(side=tk.LEFT)
        self._qc_apply_btn = ttk.Button(
            qc_top, text="↩ Undo",
            command=self._qc_undo,
            state=tk.DISABLED,
        )
        self._qc_apply_btn.pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(qc, textvariable=self._qc_status_var,
                 font=("Consolas", 8), fg=T_TEAL, bg=T_BG,
                 anchor="w", wraplength=340, justify=tk.LEFT,
                 ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        # Result details (populated after analysis)
        self._qc_detail_var = tk.StringVar(value="")
        tk.Label(qc, textvariable=self._qc_detail_var,
                 font=("Consolas", 8), fg=T_TEXT_DIM, bg=T_BG,
                 anchor="w", wraplength=340, justify=tk.LEFT,
                 ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        # Stash for undo
        self._qc_prev_settings: Optional[dict] = None

        tools = ttk.Frame(self._controls_root, padding=(0, 0, 0, 8))
        tools.grid(row=6, column=0, sticky="ew")
        ttk.Button(tools, text="Save Controls To YAML", command=self.launcher._save_controls_to_yaml).pack(side=tk.LEFT)
        ttk.Button(tools, text="Reload Controls From YAML", command=self.launcher._load_controls_from_yaml).pack(side=tk.LEFT)
        ttk.Label(tools, textvariable=self._msg_var).pack(side=tk.LEFT, padx=(12, 0))
        self._wire_harmonizers()
        self._harmonize_controls()
        self._sync_trigger_level_code()

    def _add_spinbox(
        self,
        parent: ttk.Widget,
        row: int,
        label: str,
        var: tk.Variable,
        *,
        from_: float,
        to: float,
        increment: float,
        width: int = 12,
    ):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        sb = tk.Spinbox(
            parent,
            from_=from_,
            to=to,
            increment=increment,
            textvariable=var,
            width=width,
            bg=T_SURFACE,
            fg=T_TEXT,
            insertbackground=T_CYAN,
            buttonbackground=T_BTN,
            relief=tk.FLAT,
            highlightbackground=T_BORDER,
            highlightcolor=T_CYAN,
            highlightthickness=1,
            selectbackground=T_SEL_HI,
            selectforeground=T_TEXT_BRIGHT,
        )
        sb.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        sb.bind("<FocusOut>", lambda _e: self._harmonize_controls())
        sb.bind("<Return>", lambda _e: self._harmonize_controls())
        # Block scroll wheel from changing spinbox values
        sb.bind("<MouseWheel>", lambda e: "break")
        sb.bind("<Button-4>", lambda e: "break")
        sb.bind("<Button-5>", lambda e: "break")
        return sb

    def _add_readonly_value(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Label(parent, textvariable=var).grid(row=row, column=1, sticky="w", padx=(8, 0), pady=2)

    def _add_combobox(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable, values: List[str], width: int = 14) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)
        # Block scroll wheel from changing combobox selection
        cb.bind("<MouseWheel>", lambda e: "break")
        cb.bind("<Button-4>", lambda e: "break")
        cb.bind("<Button-5>", lambda e: "break")

    def _on_controls_root_configure(self, _evt=None) -> None:
        try:
            self._scroll_canvas.configure(scrollregion=self._scroll_canvas.bbox("all"))
        except Exception:
            pass

    def _on_controls_canvas_configure(self, evt=None) -> None:
        try:
            width = int(getattr(evt, "width", 0) or self._scroll_canvas.winfo_width())
            if width > 0:
                self._scroll_canvas.itemconfigure(self._controls_root_window, width=width)
        except Exception:
            pass

    def _bind_controls_mousewheel(self) -> None:
        if self._wheel_bound:
            return
        self._wheel_bound = True
        self.bind_all("<MouseWheel>", self._on_controls_mousewheel, add="+")
        self.bind_all("<Button-4>", self._on_controls_mousewheel, add="+")
        self.bind_all("<Button-5>", self._on_controls_mousewheel, add="+")

    def _is_descendant(self, widget: tk.Misc, ancestor: tk.Misc) -> bool:
        w = widget
        while w is not None:
            if w == ancestor:
                return True
            parent_name = getattr(w, "winfo_parent", lambda: "")()
            if not parent_name:
                return False
            try:
                w = w.nametowidget(parent_name)
            except Exception:
                return False
        return False

    def _on_controls_mousewheel(self, evt) -> str:
        try:
            target = getattr(evt, "widget", None)
            if target is None:
                return ""
            if not (self._is_descendant(target, self._controls_root) or self._is_descendant(target, self._scroll_canvas)):
                return ""
            if getattr(evt, "num", None) == 4:
                self._scroll_canvas.yview_scroll(-1, "units")
                return "break"
            if getattr(evt, "num", None) == 5:
                self._scroll_canvas.yview_scroll(1, "units")
                return "break"
            delta = int(getattr(evt, "delta", 0))
            if delta == 0:
                return "break"
            steps = int(-delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            self._scroll_canvas.yview_scroll(steps, "units")
            return "break"
        except Exception:
            return "break"

    # ── Quick Config methods ──────────────────────────────────────────
    # Launches a subprocess that captures a large disposable scout sample, analyses
    # the waveform characteristics, and returns optimal settings as JSON.
    # GUI polls the subprocess and applies results when ready.

    def _quick_config(self) -> None:
        """Launch a scout capture in a subprocess and poll for results."""
        if self._qc_running:
            self._qc_status_var.set("Quick Config already running…")
            return
        if self.launcher.proc is not None:
            self._qc_status_var.set("⚠ Stop the active capture first.")
            return

        # Save current settings so we can undo later
        self._qc_prev_settings = {
            "trigger_level_pct":    self._safe_float(self.var_trig_level_pct, 0.0),
            "samples_per_record":   self._safe_int(self.var_samples_per_record, 256),
            "range_a":              self._var_text(self.var_range_a, "PM_1_V"),
            "range_b":              self._var_text(self.var_range_b, "PM_1_V"),
        }

        # Build a temporary run config from current panel settings
        cfg_path = Path(self.launcher.var_config.get()).expanduser()
        if not cfg_path.exists():
            self._qc_status_var.set("⚠ Config YAML not found")
            return

        try:
            base_cfg = load_config(cfg_path)
            run_cfg  = self.apply_to_cfg(dict(base_cfg))
            # Force a finite, quick acquisition
            acq = run_cfg.setdefault("acquisition", {})
            trig = run_cfg.setdefault("trigger", {})
            runtime = run_cfg.setdefault("runtime", {})
            acq["buffers_per_acquisition"] = _QC_BUFFERS
            # Scout captures should not hang on a missing external trigger or a
            # stale threshold. Use a short auto-trigger fallback and do not
            # require external start-capture for this disposable run.
            trig["timeout_ms"] = max(
                _to_int(trig.get("timeout_ms", 0), 0),
                _to_int(runtime.get("autotrigger_timeout_ms", 10), 10),
                10,
            )
            trig["allow_autotrigger_with_external"] = True
            trig["external_startcapture"] = False
            run_cfg, _, errs = validate_and_normalize_capture_cfg(run_cfg)
            if errs:
                self._qc_status_var.set(f"⚠ Config error: {errs[0]}")
                return
        except Exception as ex:
            self._qc_status_var.set(f"⚠ {ex}")
            return

        qc_cfg_path = cfg_path.with_suffix(cfg_path.suffix + ".qc.yaml")
        try:
            _atomic_write_text(qc_cfg_path, yaml.safe_dump(run_cfg, sort_keys=False))
        except Exception as ex:
            self._qc_status_var.set(f"⚠ write failed: {ex}")
            return

        cmd = [sys.executable, str(self.launcher.script_path),
               "quick_config", "--config", str(qc_cfg_path)]

        try:
            self._qc_proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                cwd=str(self.launcher.script_path.parent),
            )
        except Exception as ex:
            self._qc_status_var.set(f"⚠ launch failed: {ex}")
            return

        self._qc_running = True
        self._qc_btn.configure(state=tk.DISABLED)
        self._qc_status_var.set("⏳ Capturing scout buffers…")
        self._qc_detail_var.set("")
        self._qc_lines: List[str] = []
        self.after(150, self._qc_poll)

    def _qc_poll(self) -> None:
        """Poll the scout subprocess; when done, parse JSON and apply."""
        if self._qc_proc is None:
            self._qc_running = False
            self._qc_btn.configure(state=tk.NORMAL)
            return

        # Read available lines
        try:
            while True:
                ln = self._qc_proc.stdout.readline()
                if not ln:
                    break
                ln = ln.strip()
                if ln:
                    self._qc_lines.append(ln)
                    # Update status with progress lines
                    if ln.startswith("[QC]"):
                        self._qc_status_var.set(ln)
        except Exception:
            pass

        rc = self._qc_proc.poll()
        if rc is None:
            # Still running
            self.after(200, self._qc_poll)
            return

        # Process finished
        self._qc_running = False
        self._qc_btn.configure(state=tk.NORMAL)
        self._qc_proc = None

        # Find the CAPPY_QC_RESULT line
        result = None
        fallback_error = None
        for ln in self._qc_lines:
            if ln.startswith("CAPPY_QC_RESULT "):
                try:
                    result = json.loads(ln[len("CAPPY_QC_RESULT "):])
                except Exception:
                    pass
                continue
            if not ln.startswith("{"):
                continue
            try:
                payload = json.loads(ln)
            except Exception:
                continue
            if isinstance(payload, dict) and "error" in payload:
                fallback_error = payload

        if result is None:
            result = fallback_error

        if result is None:
            self._qc_status_var.set(
                f"⚠ Quick Config exited without a result{'' if rc == 0 else f' (code {rc})'}"
            )
            self._qc_detail_var.set("\n".join(self._qc_lines[-5:]))
            return

        if "error" in result:
            self._qc_status_var.set(f"⚠ {result['error']}")
            self._qc_detail_var.set("\n".join(self._qc_lines[-5:]))
            return

        self._qc_apply(result)

    def _qc_apply(self, result: dict) -> None:
        """Apply Quick Config analysis results to the GUI controls."""
        trig_pct = float(result.get("trigger_level_pct", 0.0))
        opt_spr  = int(result.get("samples_per_record", 256))
        opt_range = str(result.get("input_range", "PM_1_V"))
        peak_v    = float(result.get("peak_v", 0.0))
        noise_v   = float(result.get("noise_sigma_v", 0.0))
        note      = str(result.get("analysis_note", ""))
        n_bufs    = int(result.get("scout_buffers", 0))
        n_recs    = int(result.get("total_records", 0))
        active    = int(result.get("active_samples", 0))

        # Apply to controls
        self.var_trig_level_pct.set(round(trig_pct, 1))
        self.var_samples_per_record.set(opt_spr)
        self.var_range_a.set(opt_range)
        # Sync dependent controls
        self._sync_trigger_level_code()
        self._harmonize_controls()

        # Status feedback
        self._qc_status_var.set(
            f"✓ Applied: trigger={trig_pct:+.1f}%  spr={opt_spr}  range={opt_range}"
        )
        self._qc_detail_var.set(
            f"peak={peak_v:.4g} V  noise σ={noise_v:.2g} V  "
            f"active={active} pts  {n_bufs} bufs / {n_recs} recs  {note}"
        )
        self._qc_apply_btn.configure(state=tk.NORMAL)
        self._msg_var.set("Quick Config applied. Review settings before capture.")

    def _qc_undo(self) -> None:
        """Restore the settings that were active before Quick Config."""
        prev = self._qc_prev_settings
        if prev is None:
            return
        self.var_trig_level_pct.set(float(prev.get("trigger_level_pct", 0.0)))
        self.var_samples_per_record.set(int(prev.get("samples_per_record", 256)))
        self.var_range_a.set(str(prev.get("range_a", "PM_1_V")))
        self.var_range_b.set(str(prev.get("range_b", "PM_1_V")))
        self._sync_trigger_level_code()
        self._harmonize_controls()
        self._qc_status_var.set("↩ Settings restored to pre-Quick Config values.")
        self._qc_detail_var.set("")
        self._qc_apply_btn.configure(state=tk.DISABLED)
        self._qc_prev_settings = None
        self._msg_var.set("Quick Config undone.")

    def _set_trigger_level_pct(self, pct: float) -> None:
        src = self._var_text(self.var_trig_source, "External")
        slope = self._var_text(self.var_trig_slope, "Positive")
        safe_pct = float(pct)
        if src == "External":
            if slope == "Positive" and safe_pct < 0.0:
                safe_pct = 0.0
            elif slope == "Negative" and safe_pct > 0.0:
                safe_pct = 0.0
        self.var_trig_level_pct.set(float(safe_pct))
        self._sync_trigger_level_code()

    def _on_trigger_source_or_slope_change(self, *_args) -> None:
        if self._loading or self._harmonizing:
            return
        src = self._var_text(self.var_trig_source, "External")
        slope = self._var_text(self.var_trig_slope, "Positive")
        pct = _clamp_float(self._safe_float(self.var_trig_level_pct, 0.0), -100.0, 100.0, 0.0)
        if src == "External":
            if (slope == "Positive" and pct < 0.0) or (slope == "Negative" and pct > 0.0):
                self.var_trig_level_pct.set(0.0)
                self._msg_var.set("External trigger level set to 0% (code=128) for reliable edge crossing.")
        self._sync_trigger_level_code()

    def _sync_trigger_level_code(self, *_args) -> None:
        pct = _clamp_float(self._safe_float(self.var_trig_level_pct, 0.0), -100.0, 100.0, 0.0)
        code = _trigger_pct_to_level_code(pct, 0.0)
        self._trig_level_code_var.set(f"code={code}")
        # Update the J trigger pointer on the live dashboard in real-time
        try:
            dashboard = getattr(self.launcher, 'dashboard', None)
            if dashboard is not None and hasattr(dashboard, 'update_trigger_pointer'):
                src_label = self._var_text(self.var_trig_source, "External")
                src_const = TRIGGER_SOURCE_LABEL_TO_CONST.get(src_label, "TRIG_EXTERNAL")
                range_a = self._var_text(self.var_range_a, "PM_1_V")
                range_b = self._var_text(self.var_range_b, "PM_1_V")
                vpp_a = _range_name_to_vpp(range_a, 2.0)
                vpp_b = _range_name_to_vpp(range_b, 2.0)
                dashboard.update_trigger_pointer(pct, src_const, vpp_a, vpp_b)
        except Exception:
            pass

    def _sync_trigK_level_code(self, *_args) -> None:
        pct = _clamp_float(self._safe_float(self.var_trig_levelK_pct, 0.0), -100.0, 100.0, 0.0)
        code = _trigger_pct_to_level_code(pct, 0.0)
        self._trig_levelK_code_var.set(f"code={code}")

    def _round_up_to_step(self, value: int, step: int, min_value: int = 0) -> int:
        st = max(1, int(step))
        v = max(int(min_value), int(value))
        return ((v + st - 1) // st) * st

    def _round_nearest_step(self, value: int, step: int, min_value: int = 0) -> int:
        st = max(1, int(step))
        v = max(int(min_value), int(value))
        return int(round(float(v) / float(st))) * st

    def _wire_harmonizers(self) -> None:
        self.var_pre.trace_add("write", self._on_harmonize_var)
        self.var_samples_per_record.trace_add("write", self._on_harmonize_var)
        self.var_rpb.trace_add("write", self._on_harmonize_var)
        self.var_flush_records.trace_add("write", self._on_harmonize_var)
        self.var_wf_every_n.trace_add("write", self._on_harmonize_var)
        self.var_live_waveform_every_n.trace_add("write", self._on_harmonize_var)
        self.var_wf_max_per_sec.trace_add("write", self._on_harmonize_var)
        self.var_sqlite_commit.trace_add("write", self._on_harmonize_var)
        self.var_ring_points.trace_add("write", self._on_harmonize_var)
        self.var_rate_msps.trace_add("write", self._on_harmonize_var)
        self.var_stream_pts.trace_add("write", self._on_stream_pts_change)
        self.var_stream_s.trace_add("write", self._on_stream_seconds_change)

    def _on_stream_pts_change(self, *_args) -> None:
        if self._loading or self._harmonizing:
            return
        self._stream_edit_source = "points"
        self._harmonize_controls()

    def _on_stream_seconds_change(self, *_args) -> None:
        if self._loading or self._harmonizing:
            return
        self._stream_edit_source = "seconds"
        self._harmonize_controls()

    def _on_harmonize_var(self, *_args) -> None:
        self._harmonize_controls()

    def _harmonize_controls(self) -> None:
        if self._loading or self._harmonizing:
            return
        self._harmonizing = True
        try:
            max_spr = 8_000_000
            pre = _clamp_int(self._safe_int(self.var_pre, 0), 0, max_spr - 16, 0)
            pre = self._round_nearest_step(pre, 16, 0)

            spr = _clamp_int(self._safe_int(self.var_samples_per_record, 256), 16, max_spr, 256)
            spr = self._round_nearest_step(spr, 16, 16)
            spr = max(spr, pre + 16)
            if spr > max_spr:
                spr = max_spr
                pre = min(pre, spr - 16)

            post = max(16, spr - pre)

            rpb = _clamp_int(self._safe_int(self.var_rpb, 128), 1, 100_000, 128)
            flush = _clamp_int(self._safe_int(self.var_flush_records, 20000), 1, 50_000_000, 20000)
            flush = self._round_up_to_step(flush, rpb, rpb)
            flush = min(flush, 50_000_000)
            wf_every_n = _clamp_int(self._safe_int(self.var_wf_every_n, 16), 1, 10_000_000, 16)
            live_waveform_every_n = _clamp_int(self._safe_int(self.var_live_waveform_every_n, 1), 1, WAVEFORM_EVERY_N_MAX, 1)
            wf_max_per_sec = _clamp_int(self._safe_int(self.var_wf_max_per_sec, 120), 1, 50_000, 120)

            sqlite_commit = _clamp_int(self._safe_int(self.var_sqlite_commit, 200), 1, 10_000_000, 200)
            sqlite_commit = self._round_up_to_step(sqlite_commit, rpb, rpb)
            sqlite_commit = min(sqlite_commit, 10_000_000)

            ring_points = _clamp_int(self._safe_int(self.var_ring_points, 4096), 32, 65536, 4096)
            ring_points = self._round_nearest_step(ring_points, 32, 32)

            stream_pts = _clamp_int(self._safe_int(self.var_stream_pts, 100000), 256, 5_000_000, 100000)
            stream_s = _clamp_float(self._safe_float(self.var_stream_s, 2.0), 0.25, 120.0, 2.0)
            sr_hz = max(1.0, self._safe_float(self.var_rate_msps, 250.0) * 1e6)
            record_s = float(spr) / float(sr_hz)

            if self._stream_edit_source == "seconds":
                recs = max(1, int(round(stream_s / max(record_s, 1e-12))))
                stream_pts = recs * ring_points
            else:
                stream_pts = self._round_up_to_step(stream_pts, ring_points, ring_points)

            stream_pts = _clamp_int(stream_pts, 256, 5_000_000, max(256, ring_points))
            stream_pts = self._round_up_to_step(stream_pts, ring_points, ring_points)
            if stream_pts > 5_000_000:
                stream_pts = max(ring_points, (5_000_000 // ring_points) * ring_points)
            window_records = max(1, int(stream_pts // max(1, ring_points)))
            stream_s = _clamp_float(window_records * record_s, 0.25, 120.0, 2.0)

            self.var_pre.set(int(pre))
            self.var_samples_per_record.set(int(spr))
            self.var_post.set(int(post))
            self.var_rpb.set(int(rpb))
            self.var_flush_records.set(int(flush))
            self.var_wf_every_n.set(int(wf_every_n))
            self.var_live_waveform_every_n.set(int(live_waveform_every_n))
            self.var_wf_max_per_sec.set(int(wf_max_per_sec))
            self.var_sqlite_commit.set(int(sqlite_commit))
            self.var_ring_points.set(int(ring_points))
            self.var_stream_pts.set(int(stream_pts))
            self.var_stream_s.set(float(stream_s))

            if self._spin_flush_records is not None:
                try:
                    self._spin_flush_records.configure(increment=max(1, int(rpb)))
                except Exception:
                    pass
            if self._spin_stream_pts is not None:
                try:
                    self._spin_stream_pts.configure(increment=max(1, int(ring_points)))
                except Exception:
                    pass

            record_us = record_s * 1e6
            buffer_ms = record_s * float(rpb) * 1e3
            delay_us = self._safe_float(self.var_trigger_delay_us, 0.0)
            delay_str = f", trig delay={delay_us:.1f} µs" if delay_us > 0 else ""
            self._math_var.set(
                f"Math: post = samples - pre ({post}); flush/commit step = records_per_buffer ({rpb}); "
                f"record={record_us:.3f} us, buffer={buffer_ms:.3f} ms, live window={stream_s:.3f} s, "
                f"live every {live_waveform_every_n} buffers, snips every {wf_every_n} rec (cap {wf_max_per_sec}/s){delay_str}."
            )
        finally:
            self._harmonizing = False

    def _guess_runtime_profile(self) -> str:
        curr = dict(
            rearm_if_no_trigger_s=self._safe_int(self.var_rearm_s, 300),
            rearm_cooldown_s=self._safe_int(self.var_rearm_cooldown_s, 30),
            max_rearms_per_hour=self._safe_int(self.var_rearm_per_hour, 12),
            flush_every_records=self._safe_int(self.var_flush_records, 20000),
            flush_every_seconds=self._safe_float(self.var_flush_seconds, 2.0),
            sqlite_commit_every_snips=self._safe_int(self.var_sqlite_commit, 200),
            stream_window_points=self._safe_int(self.var_stream_pts, 100000),
            stream_window_seconds=self._safe_float(self.var_stream_s, 2.0),
            max_waveforms_per_tick=self._safe_int(self.var_max_wf_tick, 12),
        )
        for name, preset in RUNTIME_PROFILE_PRESETS.items():
            match = True
            for key, want in preset.items():
                got = curr.get(key)
                if isinstance(want, float):
                    if abs(float(got) - float(want)) > 1e-9:
                        match = False
                        break
                else:
                    if int(got) != int(want):
                        match = False
                        break
            if match:
                return name
        return "Custom"

    def _apply_runtime_profile(self) -> None:
        name = str(self.var_runtime_profile.get() or "").strip()
        preset = RUNTIME_PROFILE_PRESETS.get(name)
        if not preset:
            self._msg_var.set("Runtime profile is custom.")
            return
        self.var_rearm_s.set(int(preset["rearm_if_no_trigger_s"]))
        self.var_rearm_cooldown_s.set(int(preset["rearm_cooldown_s"]))
        self.var_rearm_per_hour.set(int(preset["max_rearms_per_hour"]))
        self.var_flush_records.set(int(preset["flush_every_records"]))
        self.var_flush_seconds.set(float(preset["flush_every_seconds"]))
        self.var_sqlite_commit.set(int(preset["sqlite_commit_every_snips"]))
        self.var_stream_pts.set(int(preset["stream_window_points"]))
        self.var_stream_s.set(float(preset["stream_window_seconds"]))
        self.var_max_wf_tick.set(int(preset["max_waveforms_per_tick"]))
        self._stream_edit_source = "points"
        self._harmonize_controls()
        self._msg_var.set(f"Applied runtime profile: {name}.")

    def _var_text(self, var: tk.Variable, default: str = "") -> str:
        try:
            name = getattr(var, "_name", None)
            if name:
                return str(self.tk.globalgetvar(name))
        except Exception:
            pass
        try:
            return str(var.get())
        except Exception:
            return str(default)

    def _safe_int(self, var: tk.Variable, default: int) -> int:
        return _to_int(self._var_text(var, str(default)), default)

    def _safe_float(self, var: tk.Variable, default: float) -> float:
        return _to_float(self._var_text(var, str(default)), default)

    def load_from_cfg(self, cfg: Dict[str, Any]) -> None:
        self._loading = True
        try:
            c = cfg.get("clock", {}) or {}
            acq = cfg.get("acquisition", {}) or {}
            ch = cfg.get("channels", {}) or {}
            trig = cfg.get("trigger", {}) or {}
            waves = cfg.get("waveforms", {}) or {}
            rt = cfg.get("runtime", {}) or {}
            storage = cfg.get("storage", {}) or {}
            live = cfg.get("live", {}) or {}

            self.var_clock_source.set(str(c.get("source", "INTERNAL_CLOCK")))
            self.var_rate_msps.set(_to_float(c.get("sample_rate_msps", 250.0), 250.0))

            mask_expr = str(acq.get("channels_mask", "CHANNEL_A|CHANNEL_B"))
            mask = channels_from_mask_expr(mask_expr)
            self.var_channels_mask.set(channels_mask_to_str(mask))
            pre = _to_int(acq.get("pre_trigger_samples", 0), 0)
            post = _to_int(acq.get("post_trigger_samples", 256), 256)
            spr = _to_int(acq.get("samples_per_record", pre + post), pre + post)
            if spr < (pre + 16):
                spr = pre + 16
            self.var_pre.set(pre)
            self.var_samples_per_record.set(spr)
            self.var_post.set(max(16, spr - pre))
            self.var_rpb.set(_to_int(acq.get("records_per_buffer", 128), 128))
            self.var_bufN.set(_to_int(acq.get("buffers_allocated", 16), 16))
            self.var_bpa.set(_to_int(acq.get("buffers_per_acquisition", 0), 0))
            self.var_wait_timeout_ms.set(_to_int(acq.get("wait_timeout_ms", 1000), 1000))

            a = ch.get("A", {}) or {}
            b = ch.get("B", {}) or {}
            self.var_range_a.set(str(a.get("range", "PM_1_V")).replace("INPUT_RANGE_", ""))
            self.var_coupling_a.set(str(a.get("coupling", "DC")).replace("_COUPLING", ""))
            self.var_imp_a.set(str(a.get("impedance", "50_OHM")))
            self.var_range_b.set(str(b.get("range", "PM_1_V")).replace("INPUT_RANGE_", ""))
            self.var_coupling_b.set(str(b.get("coupling", "DC")).replace("_COUPLING", ""))
            self.var_imp_b.set(str(b.get("impedance", "50_OHM")))

            src = str(trig.get("sourceJ", "TRIG_EXTERNAL"))
            slope = str(trig.get("slopeJ", "TRIGGER_SLOPE_POSITIVE"))
            level_code = _clamp_int(trig.get("levelJ", 128), 0, 255, 128)
            self.var_trig_source.set(TRIGGER_SOURCE_CONST_TO_LABEL.get(src, "External"))
            self.var_trig_slope.set(TRIGGER_SLOPE_CONST_TO_LABEL.get(slope, "Positive"))
            self.var_trig_level_pct.set(round(_level_code_to_trigger_pct(level_code, 128), 3))
            self.var_trig_timeout_ms.set(_to_int(trig.get("timeout_ms", 0), 0))
            self.var_timeout_pause_s.set(_to_float(trig.get("timeout_pause_s", 0.0), 0.0))
            self.var_trig_mode.set(str(trig.get("operation", "TRIG_ENGINE_OP_J")))
            self.var_ext_range.set(str(trig.get("ext_range", "ETR_5V")))
            self.var_ext_coupling.set(str(trig.get("ext_coupling", "DC_COUPLING")))
            self.var_ext_startcapture.set(bool(trig.get("external_startcapture", False)))
            self.var_trigger_delay_us.set(float(trig.get("trigger_delay_us", 0.0) or 0.0))
            # K engine
            srcK = str(trig.get("sourceK", "TRIG_DISABLE") or "TRIG_DISABLE")
            slopeK = str(trig.get("slopeK", "TRIGGER_SLOPE_POSITIVE") or "TRIGGER_SLOPE_POSITIVE")
            levelK_code = _to_int(trig.get("levelK", 128), 128)
            self.var_trig_sourceK.set(TRIGGER_SOURCEK_CONST_TO_LABEL.get(srcK, "Disabled"))
            self.var_trig_slopeK.set(TRIGGER_SLOPE_CONST_TO_LABEL.get(slopeK, "Positive"))
            self.var_trig_levelK_pct.set(round(_level_code_to_trigger_pct(levelK_code, 128), 3))
            self._sync_trigK_level_code()
            self._sync_trigger_level_code()
            self.var_wf_every_n.set(_to_int(waves.get("every_n", 16), 16))
            self.var_wf_max_per_sec.set(_to_int(waves.get("max_waveforms_per_sec", 120), 120))

            self.var_rearm_s.set(_to_int(rt.get("rearm_if_no_trigger_s", 300), 300))
            self.var_rearm_cooldown_s.set(_to_int(rt.get("rearm_cooldown_s", 30), 30))
            self.var_rearm_per_hour.set(_to_int(rt.get("max_rearms_per_hour", 12), 12))
            rt_profile = str(rt.get("profile", "") or "").strip()
            rt_profile_norm = {name.lower(): name for name in RUNTIME_PROFILE_PRESETS}.get(rt_profile.lower(), "")
            if rt_profile_norm:
                self.var_runtime_profile.set(rt_profile_norm)
            else:
                self.var_runtime_profile.set(self._guess_runtime_profile())

            self.var_flush_records.set(_to_int(storage.get("flush_every_records", 20000), 20000))
            self.var_flush_seconds.set(_to_float(storage.get("flush_every_seconds", 2.0), 2.0))
            self.var_sqlite_commit.set(_to_int(storage.get("sqlite_commit_every_snips", 200), 200))
            self.var_rotate_hours.set(_to_float(storage.get("session_rotate_hours", 24.0), 24.0))

            self.var_ring_slots.set(_to_int(live.get("ring_slots", 4096), 4096))
            self.var_ring_points.set(_to_int(live.get("ring_points", 4096), 512))
            self.var_live_waveform_every_n.set(_to_int(live.get("waveform_every_n_buffers", 1), 1))
            self.var_stream_pts.set(_to_int(live.get("stream_window_points", 100000), 100000))
            self.var_stream_s.set(_to_float(live.get("stream_window_seconds", 2.0), 2.0))
            self.var_max_wf_tick.set(_to_int(live.get("max_waveforms_per_tick", 12), 12))
            self.var_show_channel_b_live.set(_to_bool(live.get("show_channel_b", False), False))
            self.var_preview_mode.set(str(live.get("preview_mode", "archive_match")))
            if not rt_profile_norm:
                self.var_runtime_profile.set(self._guess_runtime_profile())
            self._stream_edit_source = "points"
            self._msg_var.set("Controls loaded from YAML.")
        finally:
            self._loading = False
        self._harmonize_controls()

    def apply_to_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        self._harmonize_controls()
        out = dict(cfg)
        clock = out.setdefault("clock", {}) or {}
        acq = out.setdefault("acquisition", {}) or {}
        channels = out.setdefault("channels", {}) or {}
        trig = out.setdefault("trigger", {}) or {}
        waves = out.setdefault("waveforms", {}) or {}
        runtime = out.setdefault("runtime", {}) or {}
        storage = out.setdefault("storage", {}) or {}
        live = out.setdefault("live", {}) or {}

        out["clock"] = clock
        out["acquisition"] = acq
        out["channels"] = channels
        out["trigger"] = trig
        out["waveforms"] = waves
        out["runtime"] = runtime
        out["storage"] = storage
        out["live"] = live

        local_warns: List[str] = []

        clock["source"] = self._var_text(self.var_clock_source, "INTERNAL_CLOCK") or "INTERNAL_CLOCK"
        clock["sample_rate_msps"] = self._safe_float(self.var_rate_msps, 250.0)

        acq["channels_mask"] = self._var_text(self.var_channels_mask, "CHANNEL_A") or "CHANNEL_A"
        pre = max(0, self._safe_int(self.var_pre, 0))
        post_user = self._safe_int(self.var_post, 256)
        spr_default = pre + max(16, post_user)
        spr_target = max(16, self._safe_int(self.var_samples_per_record, spr_default))
        if spr_target < (pre + 16):
            local_warns.append(
                f"samples_per_record={spr_target} is too small for pre={pre}; using {pre + 16}."
            )
            spr_target = pre + 16
        post = max(16, spr_target - pre)

        acq["pre_trigger_samples"] = pre
        acq["post_trigger_samples"] = post
        acq["samples_per_record"] = spr_target
        acq["records_per_buffer"] = self._safe_int(self.var_rpb, 128)
        acq["buffers_allocated"] = self._safe_int(self.var_bufN, 16)
        acq["buffers_per_acquisition"] = self._safe_int(self.var_bpa, 0)
        acq["wait_timeout_ms"] = self._safe_int(self.var_wait_timeout_ms, 1000)

        # Keep visible fields internally consistent so users see what will be applied.
        self.var_samples_per_record.set(int(spr_target))
        self.var_post.set(int(post))

        chA = channels.setdefault("A", {}) or {}
        chB = channels.setdefault("B", {}) or {}
        channels["A"] = chA
        channels["B"] = chB
        chA["range"] = self._var_text(self.var_range_a, "PM_1_V") or "PM_1_V"
        chA["coupling"] = self._var_text(self.var_coupling_a, "DC") or "DC"
        chA["impedance"] = self._var_text(self.var_imp_a, "50_OHM") or "50_OHM"
        chB["range"] = self._var_text(self.var_range_b, "PM_1_V") or "PM_1_V"
        chB["coupling"] = self._var_text(self.var_coupling_b, "DC") or "DC"
        chB["impedance"] = self._var_text(self.var_imp_b, "50_OHM") or "50_OHM"

        trig["operation"] = self._var_text(self.var_trig_mode, "TRIG_ENGINE_OP_J") or "TRIG_ENGINE_OP_J"
        trig["engine1"] = "TRIG_ENGINE_J"
        trig["engine2"] = "TRIG_ENGINE_K"
        trig["sourceJ"] = TRIGGER_SOURCE_LABEL_TO_CONST.get(self._var_text(self.var_trig_source, "External"), "TRIG_EXTERNAL")
        trig["slopeJ"] = TRIGGER_SLOPE_LABEL_TO_CONST.get(self._var_text(self.var_trig_slope, "Positive"), "TRIGGER_SLOPE_POSITIVE")
        trig["levelJ"] = _trigger_pct_to_level_code(self._safe_float(self.var_trig_level_pct, 0.0), 0.0)
        self._sync_trigger_level_code()
        trig["sourceK"] = TRIGGER_SOURCEK_LABEL_TO_CONST.get(self._var_text(self.var_trig_sourceK, "Disabled"), "TRIG_DISABLE")
        trig["slopeK"] = TRIGGER_SLOPE_LABEL_TO_CONST.get(self._var_text(self.var_trig_slopeK, "Positive"), "TRIGGER_SLOPE_POSITIVE")
        trig["levelK"] = _trigger_pct_to_level_code(self._safe_float(self.var_trig_levelK_pct, 0.0), 0.0)
        trig["timeout_ms"] = self._safe_int(self.var_trig_timeout_ms, 0)
        trig["timeout_pause_s"] = self._safe_float(self.var_timeout_pause_s, 0.0)
        trig["ext_range"] = self._var_text(self.var_ext_range, "ETR_5V") or "ETR_5V"
        trig["ext_coupling"] = self._var_text(self.var_ext_coupling, "DC_COUPLING") or "DC_COUPLING"
        trig["external_startcapture"] = bool(self.var_ext_startcapture.get())
        trig["trigger_delay_us"] = self._safe_float(self.var_trigger_delay_us, 0.0)
        waves["every_n"] = self._safe_int(self.var_wf_every_n, 16)
        waves["max_waveforms_per_sec"] = self._safe_int(self.var_wf_max_per_sec, 120)

        runtime["rearm_if_no_trigger_s"] = self._safe_int(self.var_rearm_s, 300)
        runtime["rearm_cooldown_s"] = self._safe_int(self.var_rearm_cooldown_s, 30)
        runtime["max_rearms_per_hour"] = self._safe_int(self.var_rearm_per_hour, 12)
        runtime["profile"] = self._var_text(self.var_runtime_profile, "Custom") or "Custom"

        storage["data_dir"] = str(self.launcher.var_data_dir.get() or "dataFile")
        storage["flush_every_records"] = self._safe_int(self.var_flush_records, 20000)
        storage["flush_every_seconds"] = self._safe_float(self.var_flush_seconds, 2.0)
        storage["sqlite_commit_every_snips"] = self._safe_int(self.var_sqlite_commit, 200)
        storage["session_rotate_hours"] = self._safe_float(self.var_rotate_hours, 24.0)

        live["ring_slots"] = self._safe_int(self.var_ring_slots, 4096)
        live["ring_points"] = self._safe_int(self.var_ring_points, 512)
        live["waveform_every_n_buffers"] = _clamp_int(
            self._safe_int(self.var_live_waveform_every_n, 1), 1, WAVEFORM_EVERY_N_MAX, 1
        )
        live["stream_window_points"] = self._safe_int(self.var_stream_pts, 100000)
        live["stream_window_seconds"] = self._safe_float(self.var_stream_s, 2.0)
        live["max_waveforms_per_tick"] = self._safe_int(self.var_max_wf_tick, 12)
        live["show_channel_b"] = bool(self.var_show_channel_b_live.get())
        live["preview_mode"] = self._var_text(self.var_preview_mode, "archive_match") or "archive_match"

        norm, warns, errs = validate_and_normalize_capture_cfg(out)
        all_warns = list(local_warns) + list(warns)
        if errs:
            self._msg_var.set("Invalid settings; fix highlighted values.")
        elif all_warns:
            self._msg_var.set(f"Applied with {len(all_warns)} warning(s).")
        else:
            self._msg_var.set("Settings OK.")
        return norm

    def update_from_event(self, evt: dict):
        # Kept for compatibility with launcher callback; no-op for this panel.
        return


class LauncherGUI(tk.Tk):
    """Simple launcher: Start Capture / Browse Archive / Open YAML."""
    def __init__(self, script_path: Path):
        super().__init__()
        self.script_path = script_path
        self._state_path = self.script_path.parent / ".cappy_v1_3_gui_state.json"
        self.proc = None
        self.pump = None
        self._kill_after_id = None
        self._poll_after_id = None
        self._close_after_id = None
        self._is_closing = False

        # ── Sleek dark theme ──────────────────────────────────────────────
        self.tk_setPalette(
            background=T_BG, foreground=T_TEXT,
            activeBackground=T_BTN_HI, activeForeground=T_TEXT_BRIGHT,
        )
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame',             background=T_BG)
        style.configure('TLabel',             background=T_BG, foreground=T_TEXT)
        style.configure('TButton',            background=T_BTN, foreground=T_TEXT,
                         borderwidth=1, relief='flat', padding=(10, 4))
        style.map('TButton',                  background=[('active', T_BTN_HI)])
        style.configure('TLabelframe',        background=T_BG, foreground=T_TEXT,
                         bordercolor=T_BORDER, relief='flat', borderwidth=1)
        style.configure('TLabelframe.Label',  background=T_BG, foreground=T_CYAN)
        style.configure('TCheckbutton',       background=T_BG, foreground=T_TEXT)
        style.map('TCheckbutton',             background=[('active', T_BG)],
                   foreground=[('active', T_TEXT_BRIGHT)])
        style.configure('TNotebook',          background=T_BG, borderwidth=0, tabmargins=(2, 4, 2, 0))
        style.configure('TNotebook.Tab',      background=T_BTN, foreground=T_TEXT_DIM,
                         padding=[20, 8], font=('Consolas', 9))
        style.map('TNotebook.Tab',            background=[('selected', T_SURFACE2)],
                   foreground=[('selected', T_CYAN)])
        style.configure('TEntry',             fieldbackground=T_SURFACE, background=T_SURFACE,
                         foreground=T_TEXT, borderwidth=1, relief='flat',
                         insertcolor=T_CYAN)
        style.configure('TCombobox',          fieldbackground=T_SURFACE, background=T_SURFACE,
                         foreground=T_TEXT, arrowcolor=T_TEXT_DIM)
        style.map(
            'TCombobox',
            fieldbackground=[('readonly', T_SURFACE)],
            foreground=[('readonly', T_TEXT)],
            selectforeground=[('readonly', T_TEXT_BRIGHT)],
            selectbackground=[('readonly', T_SEL)],
        )
        style.configure('Treeview',           background=T_SURFACE, fieldbackground=T_SURFACE,
                         foreground=T_TEXT, borderwidth=0, rowheight=24,
                         font=('Consolas', 9))
        style.configure('Treeview.Heading',   background=T_BORDER, foreground=T_CYAN,
                         font=('Consolas', 9, 'bold'))
        style.map('Treeview',                 background=[('selected', T_SEL_HI)],
                   foreground=[('selected', T_TEXT_BRIGHT)])
        style.configure('TScrollbar',         background=T_BORDER, troughcolor=T_BG,
                         arrowcolor=T_TEXT_DIM)
        style.configure('TPanedwindow',       background=T_BG)
        style.configure('TSeparator',         background=T_BORDER)
        style.configure('TScale',             background=T_BG, troughcolor=T_SURFACE)
        # Colored section label-frames
        style.configure('Acquire.TLabelframe.Label', foreground=T_CYAN)
        style.configure('Vertical.TLabelframe.Label', foreground=T_TEAL)
        style.configure('Trigger.TLabelframe.Label', foreground=T_MAGENTA)
        style.configure('Runtime.TLabelframe.Label', foreground=T_GREEN)
        style.configure('Disk.TLabelframe.Label', foreground=T_ORANGE)
        # Primary action button — subtle dark with cyan text, not a bright block
        style.configure('Primary.TButton',    background=T_BTN, foreground=T_CYAN,
                         font=('Consolas', 10, 'bold'), padding=(14, 6),
                         borderwidth=1, relief='flat')
        style.map('Primary.TButton',          background=[('active', T_BTN_HI)],
                   foreground=[('active', T_TEXT_BRIGHT)])
        # Stop button — subtle dark with red text
        style.configure('Stop.TButton',       background=T_BTN, foreground=T_RED,
                         font=('Consolas', 10, 'bold'), padding=(14, 6),
                         borderwidth=1, relief='flat')
        style.map('Stop.TButton',             background=[('active', T_BTN_HI)],
                   foreground=[('active', '#ff1744')])

        self.var_config = tk.StringVar(value="CAPPY_v1_3.yaml")
        self.var_data_dir = tk.StringVar(value=_preferred_data_dir("dataFile"))
        # Live written-to-disk status (updated from flush events)
        self._live_out_dir = tk.StringVar(value="Output: -")
        self._live_written = tk.StringVar(value="Written: -")
        self._live_last_flush = tk.StringVar(value="Last flush: -")
        self._disk_scan_interval_s = 10.0
        self._last_disk_scan_unix = 0.0
        self._last_disk_size_bytes = 0
        self._admin_unlocked = False
        self._noise_trigger_on = False

        self._restore_gui_state()

        # Auto-create default config so you never need to run `init` in a terminal.
        cfgp = Path(self.var_config.get())
        if not cfgp.exists():
            try:
                _atomic_write_text(cfgp, DEFAULT_YAML)
            except Exception as ex:
                messagebox.showerror('CAPPY', f'Failed to create default config {cfgp}: {ex}')

        # ── Top bar — admin-only (config YAML / data dir / browse) ────
        self._top_bar = ttk.Frame(self, padding=8)
        # NOT packed yet — hidden until admin unlock

        ttk.Label(self._top_bar, text="Config YAML:").pack(side=tk.LEFT)
        ttk.Entry(self._top_bar, textvariable=self.var_config, width=52).pack(side=tk.LEFT, padx=(6,8))
        ttk.Button(self._top_bar, text="Browse…", command=self._pick_yaml).pack(side=tk.LEFT)

        ttk.Label(self._top_bar, text="Data dir:").pack(side=tk.LEFT, padx=(16,4))
        ttk.Entry(self._top_bar, textvariable=self.var_data_dir, width=24).pack(side=tk.LEFT)

        ttk.Button(self._top_bar, text="Open YAML", command=self._open_yaml).pack(side=tk.LEFT, padx=(16,6))
        ttk.Button(self._top_bar, text="Browse Archive", command=self._browse).pack(side=tk.LEFT)

        # ── Control bar — always visible ──────────────────────────────
        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill=tk.X)

        self.btn = ttk.Button(ctrl, text="Start Capture", command=self._toggle, style='Primary.TButton')
        self.btn.pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Stop", command=self._stop, style='Stop.TButton').pack(side=tk.LEFT, padx=(8,0))
        self.lbl = ttk.Label(ctrl, text="State: idle", foreground=T_TEXT_DIM, font=('Consolas', 10))
        self.lbl.pack(side=tk.LEFT, padx=(16,0))

        # ── Noise trigger toggle ──────────────────────────────────────
        self._noise_btn_var = tk.StringVar(value="Noise Trigger: OFF")
        self._noise_btn = ttk.Button(ctrl, textvariable=self._noise_btn_var,
                                     command=self._toggle_noise_trigger)
        self._noise_btn.pack(side=tk.LEFT, padx=(20, 0))

        # ── Admin lock button (far right) ─────────────────────────────
        self._lock_btn_var = tk.StringVar(value="Unlock Admin")
        ttk.Button(ctrl, textvariable=self._lock_btn_var,
                   command=self._admin_gate).pack(side=tk.RIGHT)

        # ── Main content area ─────────────────────────────────────────
        self._content = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self._content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left = ttk.Frame(self._content)
        self._right_frame = ttk.Frame(self._content)
        self._content.add(left, weight=3)
        self._content.add(self._right_frame, weight=2)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self._right_frame.columnconfigure(0, weight=1)
        self._right_frame.rowconfigure(0, weight=1)

        tabs = ttk.Notebook(left)
        tabs.grid(row=0, column=0, sticky="nsew")

        self.dashboard = LiveDashboard(tabs, self.var_data_dir, on_status=self._on_status_snapshot)
        tabs.add(self.dashboard, text="Overview")

        logbox = ttk.Frame(tabs, padding=6)
        tabs.add(logbox, text="Log")
        self.log = tk.Text(logbox, wrap="word", state=tk.DISABLED, bg=T_SURFACE, fg=T_GREEN, insertbackground=T_CYAN, font=('Consolas', 9), relief=tk.FLAT, bd=0, padx=6, pady=4)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.live_panel = LiveControlPanel(self._right_frame, self)
        self.live_panel.grid(row=0, column=0, sticky="nsew")

        self._load_controls_from_yaml()

        # ── Start in DUMMY mode — hide admin surfaces ─────────────────
        self._hide_admin_panels()

        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._schedule_poll()

    # ── Dummy / Admin mode ──────────────────────────────────────────────
    def _admin_gate(self):
        """Toggle between dummy and admin mode.  Requires password to unlock."""
        if self._admin_unlocked:
            # Lock back to dummy
            self._admin_unlocked = False
            self._hide_admin_panels()
            self._lock_btn_var.set("Unlock Admin")
            self._append("[CAPPY] Admin controls locked.")
            return

        dlg = tk.Toplevel(self)
        dlg.title("Admin Unlock")
        dlg.geometry("310x120")
        dlg.configure(bg=T_BG)
        dlg.resizable(False, False)
        dlg.transient(self)
        dlg.grab_set()
        tk.Label(dlg, text="Enter admin password:", bg=T_BG, fg=T_TEXT,
                 font=('Consolas', 10)).pack(pady=(16, 6))
        pw = tk.StringVar()
        ent = tk.Entry(dlg, textvariable=pw, show="*", width=22,
                       bg=T_SURFACE, fg=T_TEXT, insertbackground=T_CYAN,
                       font=('Consolas', 10))
        ent.pack()
        ent.focus_set()
        def _try(_e=None):
            if pw.get() == ADMIN_PASSWORD:
                dlg.destroy()
                self._admin_unlocked = True
                self._show_admin_panels()
                self._lock_btn_var.set("Lock Admin")
                self._append("[CAPPY] Admin controls unlocked.")
            else:
                messagebox.showerror("CAPPY", "Wrong password.", parent=dlg)
        ent.bind("<Return>", _try)
        ttk.Button(dlg, text="OK", command=_try).pack(pady=8)

    def _hide_admin_panels(self):
        """Hide admin-only surfaces (config bar + right-side controls panel)."""
        self._top_bar.pack_forget()
        try:
            self._content.forget(self._right_frame)
        except Exception:
            pass

    def _show_admin_panels(self):
        """Reveal admin surfaces."""
        # Re-pack top bar just above the content pane
        self._top_bar.pack(fill=tk.X, before=self._content)
        try:
            self._content.add(self._right_frame, weight=2)
        except Exception:
            pass

    def _toggle_noise_trigger(self):
        """Toggle noise-trigger mode — sets runtime.noise_test + auto-timeout."""
        self._noise_trigger_on = not self._noise_trigger_on
        if self._noise_trigger_on:
            self._noise_btn_var.set("Noise Trigger: ON")
            # Push noise-trigger settings into the live panel so _start picks them up
            if hasattr(self, "live_panel") and self.live_panel is not None:
                try:
                    self.live_panel.var_trig_timeout_ms.set(
                        max(self.live_panel._safe_int(self.live_panel.var_trig_timeout_ms, 0), 10)
                    )
                except Exception:
                    pass
            self._append("[CAPPY] Noise trigger ON — board will auto-trigger on ambient noise.")
        else:
            self._noise_btn_var.set("Noise Trigger: OFF")
            if hasattr(self, "live_panel") and self.live_panel is not None:
                try:
                    self.live_panel.var_trig_timeout_ms.set(0)
                except Exception:
                    pass
            self._append("[CAPPY] Noise trigger OFF — reverting to normal trigger mode.")

    def _restore_gui_state(self) -> None:
        try:
            p = self._state_path
            if not p.exists():
                return
            st = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(st, dict):
                return
            cfg = str(st.get("config_path", "") or "").strip()
            ddir = str(st.get("data_dir", "") or "").strip()
            geom = str(st.get("geometry", "") or "").strip()
            if cfg:
                self.var_config.set(cfg)
            if ddir:
                self.var_data_dir.set(ddir)
            if geom:
                try:
                    self.geometry(geom)
                except Exception:
                    pass
        except Exception:
            pass

    def _save_gui_state(self) -> None:
        try:
            st = {
                "config_path": str(self.var_config.get() or ""),
                "data_dir": str(self.var_data_dir.get() or ""),
                "geometry": str(self.geometry() or ""),
                "saved_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            _atomic_write_text(self._state_path, json.dumps(st, indent=2))
        except Exception:
            pass

    def _on_destroy(self, evt=None):
        if evt is not None and getattr(evt, "widget", None) is not self:
            return
        self._is_closing = True
        self._cancel_after_callback("_poll_after_id")
        self._cancel_after_callback("_kill_after_id")
        self._cancel_after_callback("_close_after_id")

    def _cancel_after_callback(self, attr_name: str) -> None:
        aid = getattr(self, attr_name, None)
        setattr(self, attr_name, None)
        if aid is not None:
            try:
                self.after_cancel(aid)
            except Exception:
                pass

    def _schedule_poll(self) -> None:
        if self._is_closing:
            return
        try:
            if not bool(self.winfo_exists()):
                return
            self._poll_after_id = self.after(80, self._poll)
        except Exception:
            self._poll_after_id = None

    def _append(self, s: str):
        if self._is_closing:
            return
        try:
            if not bool(self.winfo_exists()) or not bool(self.log.winfo_exists()):
                return
            self.log.configure(state=tk.NORMAL)
            self.log.insert(tk.END, s + "\n")
            self.log.see(tk.END)
            self.log.configure(state=tk.DISABLED)
        except Exception:
            return

    def _capture_disk_size(self, data_dir: Path, session_id: str = "") -> int:
        """
        Estimate on-disk bytes for capture output with throttled scans.
        Prefer current session's day folder when session_id is available.
        """
        now = time.time()
        if (now - float(self._last_disk_scan_unix)) < float(self._disk_scan_interval_s):
            return int(self._last_disk_size_bytes)

        base = data_dir / "captures"
        scan_root = base
        sid = str(session_id or "").strip()
        if len(sid) >= 8 and sid[:8].isdigit():
            ymd = sid[:8]
            candidate = base / ymd[:4] / f"{ymd[:4]}-{ymd[4:6]}" / f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"
            if candidate.exists():
                scan_root = candidate

        total = 0
        try:
            if scan_root.exists():
                for root, _dirs, files in os.walk(scan_root):
                    for fn in files:
                        fp = Path(root) / fn
                        try:
                            total += int(fp.stat().st_size)
                        except Exception:
                            pass
        except Exception:
            return int(self._last_disk_size_bytes)

        self._last_disk_scan_unix = now
        self._last_disk_size_bytes = max(int(total), int(self._last_disk_size_bytes))
        return int(self._last_disk_size_bytes)


    def _handle_line(self, ln: str):
        # Flush events emitted by patched capture: "CAPPY_EVENT {json}"
        if not ln:
            return
        ln = ln.strip()
        if ln.startswith("CAPPY_EVENT "):
            try:
                evt = json.loads(ln[len("CAPPY_EVENT "):])
            except Exception:
                return
            if isinstance(evt, dict) and evt.get("type") == "flush":
                w = evt.get("written") or {}
                w_bytes = _to_int(w.get('bytes', 0), 0)
                self._live_written.set(
                    f"Written: {w_bytes:,} bytes ({_format_size_gib(w_bytes)}), {w.get('records',0)} records, {w.get('samples',0)} samples"
                )
                self._last_disk_size_bytes = max(self._last_disk_size_bytes, w_bytes)
                ts = evt.get("ts_iso") or ""
                if not ts and isinstance(evt.get("ts_ns"), int):
                    ts = _ns_to_iso(int(evt["ts_ns"]))
                self._live_last_flush.set(f"Last flush: {ts or '-'}")
                out_dir = evt.get("out_dir") or "-"
                self._live_out_dir.set(f"Output: {out_dir}")
                try:
                    if hasattr(self, "live_panel") and self.live_panel is not None:
                        self.live_panel.update_from_event(evt)
                except Exception:
                    pass

    def _on_status_snapshot(self, snap: dict):
        data_dir_path = Path(str(snap.get("data_dir", self.var_data_dir.get()) or self.var_data_dir.get())).expanduser()
        try:
            data_dir = str(data_dir_path)
            sid = str(snap.get("session_id", "") or "")
            if sid:
                self._live_out_dir.set(f"Output: {data_dir}/captures (session {sid})")
            else:
                self._live_out_dir.set(f"Output: {data_dir}/captures")
        except Exception:
            pass
        try:
            reduced = int(snap.get("reduced_rows", 0))
            snips = int(snap.get("snips", 0))
            sid = str(snap.get("session_id", "") or "")
            disk_est = self._capture_disk_size(data_dir_path, sid)
            snap_disk = _to_int(snap.get("disk_bytes", -1), -1)
            disk_bytes = int(max(disk_est, snap_disk if snap_disk >= 0 else 0))
            self._live_written.set(
                f"Written: reduced_rows={reduced:,}   snips={snips:,}   disk={_format_size_gib(disk_bytes)}"
            )
        except Exception:
            pass
        try:
            ts = str(snap.get("last_capture", snap.get("time", "")) or "")
            state = str(snap.get("state", "?") or "?")
            self._live_last_flush.set(f"Last update: {ts or '-'}   state={state}")
        except Exception:
            pass

    def _load_controls_from_yaml(self):
        cfg_path = Path(self.var_config.get()).expanduser()
        try:
            cfg = load_config(cfg_path)
            if hasattr(self, "live_panel") and self.live_panel is not None:
                self.live_panel.load_from_cfg(cfg)
            storage = cfg.get("storage", {}) if isinstance(cfg, dict) else {}
            if isinstance(storage, dict) and storage.get("data_dir"):
                self.var_data_dir.set(str(storage.get("data_dir")))
            self._append(f"[CAPPY] Loaded controls from {cfg_path}")
        except Exception as ex:
            self._append(f"[CAPPY] Could not load controls from {cfg_path}: {ex}")

    def _save_controls_to_yaml(self):
        cfg_path = Path(self.var_config.get()).expanduser()
        try:
            base_cfg = load_config(cfg_path) if cfg_path.exists() else yaml.safe_load(DEFAULT_YAML)
            if not isinstance(base_cfg, dict):
                base_cfg = {}
            out_cfg = dict(base_cfg)
            if hasattr(self, "live_panel") and self.live_panel is not None:
                out_cfg = self.live_panel.apply_to_cfg(out_cfg)
            out_cfg, warns, errs = validate_and_normalize_capture_cfg(out_cfg)
            if errs:
                messagebox.showerror("CAPPY", "Cannot save config:\n- " + "\n- ".join(errs))
                return
            # Backup current config before overwriting
            if cfg_path.exists():
                bak = cfg_path.with_suffix(cfg_path.suffix + ".bak")
                try:
                    import shutil as _shutil_bak
                    _shutil_bak.copy2(str(cfg_path), str(bak))
                except Exception:
                    pass
            header = f"# CAPPY v1.3 config — saved {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            body = yaml.safe_dump(out_cfg, sort_keys=False, default_flow_style=False,
                                  allow_unicode=True, width=120)
            _atomic_write_text(cfg_path, header + body)
            self._append(f"[CAPPY] Saved controls to {cfg_path}")
            for w in warns:
                self._append(f"[CAPPY] Config warning: {w}")
            try:
                if hasattr(self, "live_panel") and self.live_panel is not None:
                    self.live_panel._msg_var.set("Saved to YAML.")
            except Exception:
                pass
        except Exception as ex:
            messagebox.showerror("CAPPY", f"Failed to save config:\n{cfg_path}\n{ex}")

    def _pick_yaml(self):
        p = filedialog.askopenfilename(title="Select YAML", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if p:
            self.var_config.set(p)
            self._load_controls_from_yaml()

    def _open_yaml(self):
        p = Path(self.var_config.get()).expanduser()
        if not p.exists():
            messagebox.showerror("Missing", f"Config not found:\n{p}")
            return
        try:
            import os, subprocess
            if os.name == "posix":
                subprocess.Popen(["xdg-open", str(p)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                os.startfile(str(p))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _browse(self):
        try:
            data_path = Path(self.var_data_dir.get())
            _ensure_dir(data_path)
            win = tk.Toplevel(self)
            # Prefer the optimized archive DB browser when available
            try:
                from cappy_archive_db import ArchiveDB, run_archive_db, C_BG
                win.title('CAPPY.ARCH v1.3')
                win.configure(bg=C_BG)
                win.geometry("1440x920")
                win.minsize(900, 600)
                app = ArchiveDB(data_path, master=win)
                app.pack(fill=tk.BOTH, expand=True)
                win.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), win.destroy()))
            except ImportError:
                win.title('CAPPY Archive')
                app = ArchiveBrowser(data_path, master=win)
                app.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror('CAPPY', str(e))

    def _toggle(self):
        if self.proc is None:
            self._start()
        else:
            self._stop()

    def _start(self):
        import subprocess
        if self.proc is not None:
            return
        cfg_path = Path(self.var_config.get()).expanduser()
        if not cfg_path.exists():
            messagebox.showerror("Missing", f"Config not found:{cfg_path}")
            return

        # Build a run-time config (do not overwrite the user's YAML)
        run_cfg_path = cfg_path.with_suffix(cfg_path.suffix + ".run.yaml")
        try:
            base_cfg = load_config(cfg_path)
        except Exception as ex:
            messagebox.showerror("CAPPY", f"Failed to read config:\n{cfg_path}\n{ex}")
            return

        try:
            run_cfg = dict(base_cfg)
            if hasattr(self, "live_panel") and self.live_panel is not None:
                run_cfg = self.live_panel.apply_to_cfg(run_cfg)
            # ── Noise trigger override ────────────────────────────────
            if getattr(self, "_noise_trigger_on", False):
                rt = run_cfg.setdefault("runtime", {}) or {}
                rt["noise_test"] = True
                run_cfg["runtime"] = rt
                trig = run_cfg.setdefault("trigger", {}) or {}
                trig["timeout_ms"] = max(_to_int(trig.get("timeout_ms", 0), 0), 10)
                run_cfg["trigger"] = trig
                self._append("[CAPPY] Noise trigger injected into run config (noise_test=true, timeout_ms≥10).")
            try:
                st = run_cfg.setdefault("storage", {}) or {}
                dd = str(st.get("data_dir", "") or "").strip()
                if dd and not os.path.isabs(dd):
                    st["data_dir"] = _preferred_data_dir(dd)
                    run_cfg["storage"] = st
            except Exception:
                pass
            run_cfg, warns, errs = validate_and_normalize_capture_cfg(run_cfg)
            if errs:
                messagebox.showerror("CAPPY", "Invalid run configuration:\n- " + "\n- ".join(errs))
                return
            for w in warns:
                self._append(f"[CAPPY] Config warning: {w}")

            run_yaml = yaml.safe_dump(run_cfg, sort_keys=False, default_flow_style=False)
            _atomic_write_text(run_cfg_path, run_yaml)

            sourceJ = str((run_cfg.get("trigger", {}) or {}).get("sourceJ", "?"))
            self._append(f"[CAPPY] Run config: {run_cfg_path} (sourceJ={sourceJ})")
            try:
                dd = str((run_cfg.get("storage", {}) or {}).get("data_dir", self.var_data_dir.get()))
                if dd:
                    self.var_data_dir.set(dd)
            except Exception:
                pass
        except Exception as ex:
            messagebox.showerror("CAPPY", f"Failed to write run config:\n{run_cfg_path}\n{ex}")
            return

        # Optional maintenance only (disabled by default for fast starts).
        # Set env CAPPY_CLEAR_PYCACHE=1 if you explicitly want a cleanup pass.
        if str(os.environ.get("CAPPY_CLEAR_PYCACHE", "0")).strip() in {"1", "true", "TRUE", "yes", "YES"}:
            try:
                removed = clear_pycache(self.script_path.parent)
                self._append(f"[CAPPY] Cleared pycache (removed ~{removed} items)")
            except Exception:
                pass

        cmd = [sys.executable, str(self.script_path), "capture", "--config", str(run_cfg_path)]
        self._append("RUN: " + " ".join(cmd))

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.script_path.parent),
            )
            self.pump = _ProcLogPump(self.proc)
        except Exception as e:
            self.proc = None
            self.pump = None
            messagebox.showerror("CAPPY", f"Failed to start capture:\n{e}")
            return

        self.lbl.config(text="State: capturing", foreground=T_GREEN)
        self.btn.config(text="Stop Capture")

    def _stop(self):
        if self.proc is None:
            return
        try:
            try:
                self.proc.send_signal(signal.SIGINT)
                self._append("[GUI] sent SIGINT")
            except Exception:
                self.proc.terminate()
                self._append("[GUI] sent terminate")
        except Exception as e:
            self._append("[GUI] stop failed: " + str(e))

        self._cancel_after_callback("_kill_after_id")
        if self._kill_after_id is None:
            self._kill_after_id = self.after(2500, self._kill_if_running)

    def _kill_if_running(self):
        self._kill_after_id = None
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.kill()
                self._append("[GUI] forced kill")
        except Exception as e:
            self._append("[GUI] kill failed: " + str(e))

    def _on_close(self):
        self._is_closing = True
        self._cancel_after_callback("_poll_after_id")
        self._cancel_after_callback("_close_after_id")
        try:
            if hasattr(self, "live_panel") and self.live_panel is not None:
                self._save_controls_to_yaml()
        except Exception:
            pass
        self._save_gui_state()
        try:
            if self.proc is not None and self.proc.poll() is None:
                self._stop()
                self._close_after_id = self.after(600, self.destroy)
                return
        except Exception:
            pass
        self.destroy()


    def _poll(self):
        self._poll_after_id = None
        if self._is_closing:
            return
        try:
            if not bool(self.winfo_exists()):
                return
        except Exception:
            return

        if self.pump is not None:
            for ln in self.pump.drain(max_lines=400):
                self._append(ln)
                self._handle_line(ln)

        if self.proc is not None:
            rc = self.proc.poll()
            if rc is not None:
                try:
                    if self.pump is not None:
                        for ln in self.pump.drain(max_lines=10000):
                            self._append(ln)
                            self._handle_line(ln)
                        self.pump.stop()
                except Exception:
                    pass
                self._append(f"[GUI] DAQ exited rc={rc}")
                self.proc = None
                self.pump = None
                self.lbl.config(text="State: idle", foreground=T_TEXT_DIM)
                self.btn.config(text="Start Capture")

        self._schedule_poll()


def run_browse(data_dir: Path) -> int:
    _ensure_dir(data_dir)
    root = tk.Tk()
    root.title('CAPPY Archive')
    app = ArchiveBrowser(data_dir, master=root)
    app.pack(fill=tk.BOTH, expand=True)
    root.mainloop()
    return 0

def main() -> int:
    argv = sys.argv[1:]
    if not argv:
        argv = ["gui"]
    if argv and argv[0] not in {"init","capture","browse","gui","quick_config"}:
        ap = argparse.ArgumentParser(prog="CAPPY_v1_3 (legacy)")
        ap.add_argument("--config", default="CAPPY_v1_3.yaml")
        args = ap.parse_args(argv)
        return run_capture(Path(args.config))

    ap = argparse.ArgumentParser(prog="CAPPY_v1_3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    cap = sub.add_parser("capture")
    cap.add_argument("--config", default="CAPPY_v1_3.yaml")
    br = sub.add_parser("browse")
    br.add_argument("--data_dir", default="dataFile")
    sub.add_parser("gui")
    qc = sub.add_parser("quick_config")
    qc.add_argument("--config", default="CAPPY_v1_3.yaml")

    args = ap.parse_args(argv)

    if args.cmd == "init":
        p = Path("CAPPY_v1_3.yaml")
        if not p.exists():
            _atomic_write_text(p, DEFAULT_YAML)
        print("[CAPPY] init done")
        return 0
    if args.cmd == "capture":
        return run_capture(Path(args.config))
    if args.cmd == "quick_config":
        return run_quick_config(Path(args.config))
    if args.cmd == "browse":
        return run_browse(Path(args.data_dir))
    if args.cmd == "gui":
        LauncherGUI(Path(__file__).resolve()).mainloop()
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
