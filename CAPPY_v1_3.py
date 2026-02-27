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


# Define neon colors
NEON_PINK = "#FF00EE"
NEON_GREEN = "#26FF00"

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
TRIGGER_SLOPE_LABEL_TO_CONST = {
    "Positive": "TRIGGER_SLOPE_POSITIVE",
    "Negative": "TRIGGER_SLOPE_NEGATIVE",
}
TRIGGER_SLOPE_CONST_TO_LABEL = {v: k for k, v in TRIGGER_SLOPE_LABEL_TO_CONST.items()}
TRIGGER_MODE_OPTIONS = ["TRIG_ENGINE_OP_J", "TRIG_ENGINE_OP_J_OR_K", "TRIG_ENGINE_OP_J_AND_K"]
EXT_TRIGGER_RANGE_OPTIONS = ["ETR_5V", "ETR_1V"]


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
# Proton beam default: 250 MS/s
# Use *samples* where possible
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
  timeout_ms: 0                   # 0 = wait forever (unless runtime.noise_test=true)
  timeout_pause_s: 0.0            # if >0 and no triggers arrive for this long, pause then rearm

  external_startcapture: false

timing:
  bunch_spacing_samples: 424      # adjust to your bunch spacing

acquisition:
  channels_mask: CHANNEL_A|CHANNEL_B
  pre_trigger_samples: 0
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
  every_n: 1
  threshold_integral_Vs: 0.0
  threshold_peak_V: 0.0
  max_waveforms_per_sec: 500
  store_volts: true

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
  heartbeat_seconds: 1
  interval_minutes: 120

runtime:
  noise_test: false
  autotrigger_timeout_ms: 10
  rearm_if_no_trigger_s: 300
  rearm_cooldown_s: 30
  max_rearms_per_hour: 12

live:
  ring_slots: 4096
  ring_points: 512
  stream_window_points: 20000
  stream_window_seconds: 2.0
  max_waveforms_per_tick: 20
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
        ("bunch_spacing_samples", _pa0.int32()),
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
    tmp = path.with_suffix(path.suffix + ".tmp")
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

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

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

            try:
                self._fh.seek(off)
                self._fh.write(rec_hdr)
                self._fh.write(a.tobytes(order="C"))
                self._fh.write(b.tobytes(order="C"))
                # also update header write_seq for reader to know latest
                self._fh.seek(24)  # write_seq offset in header
                self._fh.write(struct.pack("<Q", seq))
            except Exception:
                pass


class StatusNotifier:
    """Heartbeat file + periodic status email, non-blocking."""
    def __init__(self, cfg: dict, data_dir: Path):
        self.cfg = cfg
        self.data_dir = data_dir
        self.notify = (cfg.get("notify", {}) or {})
        self.hb_seconds = float(self.notify.get("heartbeat_seconds", 1.0))
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
                to_addr = str(self.notify.get("to_email", "")).strip()
                if to_addr:
                    with self._lock:
                        snap = dict(self._latest)
                    send_status_email(cfg=self.cfg, snap=snap)
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



class WaveBinSqliteStore:
    """
    Store waveform snippets as:
      - raw float32 channel-separated data appended to time-rolled .bin files inside hourly folders
      - SQLite index (WAL) at day_dir/index/snips_<session>.sqlite pointing to file+offset per channel

    Directory layout (under captures/<YYYY>/<YYYY-MM>/<YYYY-MM-DD>/):
      - index/snips_<session>.sqlite
      - <HH:00>/waveforms/A_snips_<session>_<YYYYMMDD_HHMM>.bin
      - <HH:00>/waveforms/B_snips_<session>_<YYYYMMDD_HHMM>.bin
    """
    def __init__(self, day_dir: Path, session_id: str, rollover_minutes: int, commit_every: int):
        self.day_dir = day_dir
        self.session_id = session_id
        self.rollover_minutes = max(1, int(rollover_minutes))
        self.commit_every = max(1, int(commit_every))
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
              file_B TEXT,
              offset_B INTEGER,
              nbytes_B INTEGER,

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
                "file_B": "ALTER TABLE snips ADD COLUMN file_B TEXT",
                "offset_B": "ALTER TABLE snips ADD COLUMN offset_B INTEGER",
                "nbytes_B": "ALTER TABLE snips ADD COLUMN nbytes_B INTEGER",
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
        self._fhA = open(pathA, "ab", buffering=0)
        self._fhB = open(pathB, "ab", buffering=0)
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
        payloadA = wfA.tobytes(order="C")

        offA = self._fhA.tell()
        self._fhA.write(payloadA)

        if wfB_V is not None:
            wfB = wfB_V.astype(np.float32, copy=False)
            payloadB = wfB.tobytes(order="C")
            offB = self._fhB.tell()
            self._fhB.write(payloadB)
            n_channels = 2
        else:
            payloadB = b""
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
            "file,offset_bytes,nbytes,file_A,offset_A,nbytes_A,file_B,offset_B,nbytes_B,area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.session_id, int(ts_ns), int(buffer_index), int(record_in_buffer), int(record_global),
             str(channels_mask), float(sample_rate_hz), int(wfA.shape[0]), int(n_channels),
             file_legacy, int(off_legacy), int(nbytes_legacy),
             fileA, int(offA), int(len(payloadA)),
             (fileB if wfB_V is not None else None),
             (int(offB) if wfB_V is not None else None),
             (int(len(payloadB)) if wfB_V is not None else None),
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

        if isinstance(fileA, str) and fileA and _lazy_pandas().notna(offA) and _lazy_pandas().notna(nbytesA):
            binA = day_dir / str(fileA)
            with open(binA, "rb") as fh:
                fh.seek(int(offA))
                payloadA = fh.read(int(nbytesA))
            a = np.frombuffer(payloadA, dtype=np.float32)[:n_samples]

            fileB = row.get("file_B", None)
            offB = row.get("offset_B", None)
            nbytesB = row.get("nbytes_B", None)
            if isinstance(fileB, str) and fileB and _lazy_pandas().notna(offB) and _lazy_pandas().notna(nbytesB):
                binB = day_dir / str(fileB)
                with open(binB, "rb") as fh:
                    fh.seek(int(offB))
                    payloadB = fh.read(int(nbytesB))
                b = np.frombuffer(payloadB, dtype=np.float32)[:n_samples]
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
        arr = np.frombuffer(payload, dtype=np.float32)
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

    if isinstance(fileA, str) and fileA and _lazy_pandas().notna(offA) and _lazy_pandas().notna(nbytesA):
        binA = day_dir / str(fileA)
        with open(binA, "rb") as fh:
            fh.seek(int(offA))
            payloadA = fh.read(int(nbytesA))
        a = np.frombuffer(payloadA, dtype=np.float32)[:n_samples]

        fileB = row.get("file_B", None)
        offB = row.get("offset_B", None)
        nbytesB = row.get("nbytes_B", None)
        if isinstance(fileB, str) and fileB and _lazy_pandas().notna(offB) and _lazy_pandas().notna(nbytesB):
            binB = day_dir / str(fileB)
            with open(binB, "rb") as fh:
                fh.seek(int(offB))
                payloadB = fh.read(int(nbytesB))
            b = np.frombuffer(payloadB, dtype=np.float32)[:n_samples]
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
    arr = np.frombuffer(payload, dtype=np.float32)
    if n_channels == 2:
        return arr[:n_samples], arr[n_samples:2*n_samples]
    return arr[:n_samples], None



class CappyArchive:

    def __init__(self, data_dir: Path, rollover_minutes: int, flush_every_records: int,
                 session_rotate_hours: float, sqlite_commit_every_snips: int, flush_every_seconds: float = 10.0):
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
        self.wave_store = WaveBinSqliteStore(self.day_dir, sid, self.rollover_minutes, self.sqlite_commit_every_snips)

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
    timing = out.setdefault("timing", {}) or {}
    trig = out.setdefault("trigger", {}) or {}
    runtime = out.setdefault("runtime", {}) or {}
    storage = out.setdefault("storage", {}) or {}
    live = out.setdefault("live", {}) or {}

    out["clock"] = clock
    out["acquisition"] = acq
    out["timing"] = timing
    out["trigger"] = trig
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

    pre = _clamp_int(acq.get("pre_trigger_samples", 0), 0, 8_000_000, 0)
    post = _clamp_int(acq.get("post_trigger_samples", 256), 16, 8_000_000, 256)
    spr = pre + post
    if spr <= 0:
        errors.append("pre_trigger_samples + post_trigger_samples must be > 0.")
    acq["pre_trigger_samples"] = pre
    acq["post_trigger_samples"] = post

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

    bunch = _clamp_int(timing.get("bunch_spacing_samples", spr), 1, 8_000_000, max(1, spr))
    if bunch < spr:
        warnings.append(
            f"timing.bunch_spacing_samples={bunch} is < samples_per_record={spr}; using {spr}."
        )
        bunch = spr
    timing["bunch_spacing_samples"] = bunch

    levelJ = _clamp_int(trig.get("levelJ", 128), 0, 255, 128)
    levelK = _clamp_int(trig.get("levelK", 128), 0, 255, 128)
    trig["levelJ"] = levelJ
    trig["levelK"] = levelK
    trig["timeout_ms"] = _clamp_int(trig.get("timeout_ms", 0), 0, 2_147_483_647, 0)
    trig["timeout_pause_s"] = _clamp_float(trig.get("timeout_pause_s", 0.0), 0.0, 3600.0, 0.0)
    if _to_int(trig.get("timeout_ms", 0), 0) > 0:
        warnings.append(
            "trigger.timeout_ms > 0 enables auto-trigger mode; this can create non-hardware trigger records."
        )

    runtime["rearm_if_no_trigger_s"] = _clamp_int(runtime.get("rearm_if_no_trigger_s", 300), 0, 86_400, 300)
    runtime["rearm_cooldown_s"] = _clamp_int(runtime.get("rearm_cooldown_s", 30), 0, 86_400, 30)
    runtime["max_rearms_per_hour"] = _clamp_int(runtime.get("max_rearms_per_hour", 12), 1, 100_000, 12)

    storage["flush_every_records"] = _clamp_int(storage.get("flush_every_records", 20000), 1, 50_000_000, 20000)
    storage["flush_every_seconds"] = _clamp_float(storage.get("flush_every_seconds", 2.0), 0.0, 86_400.0, 2.0)
    storage["sqlite_commit_every_snips"] = _clamp_int(
        storage.get("sqlite_commit_every_snips", 200), 1, 10_000_000, 200
    )

    live["ring_slots"] = _clamp_int(live.get("ring_slots", 4096), 16, 1_000_000, 4096)
    live["ring_points"] = _clamp_int(live.get("ring_points", 512), 32, 16384, 512)
    live["stream_window_points"] = _clamp_int(live.get("stream_window_points", 20000), 256, 5_000_000, 20000)
    live["stream_window_seconds"] = _clamp_float(live.get("stream_window_seconds", 2.0), 0.25, 120.0, 2.0)
    live["max_waveforms_per_tick"] = _clamp_int(live.get("max_waveforms_per_tick", 20), 1, 2000, 20)
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


def get_board_temperatures_c(board) -> Dict[str, Optional[float]]:
    """
    Read FPGA/ADC temperatures if available.
    
    Notes:
      - Temperature reading is not available in all atsapi Python wrapper versions
      - The ATS-9352 board supports temperature sensing in firmware, but the Python
        API may not expose it depending on the wrapper version
      - This function gracefully returns None if temperature is unavailable
    
    Returns:
        Dict with 'fpga' and 'adc' temperature in Celsius (or None if unavailable),
        and 'diag' with diagnostic information
    """
    import ctypes, struct, math

    out: Dict[str, Optional[float]] = {"fpga": None, "adc": None}

    # Strategy 1: Try high-level board methods (if exposed by wrapper)
    for method_name in ['getBoardTemperature', 'getTemperature', 'get_temperature']:
        if hasattr(board, method_name):
            try:
                method = getattr(board, method_name)
                temp = float(method())
                if -40.0 <= temp <= 150.0:
                    out["adc"] = temp  # Usually returns ADC temp
                    out["diag"] = {"method": method_name, "value": temp, "status": "success"}
                    return out
            except Exception as e:
                # Method exists but failed - record it
                out["diag"] = {"method": method_name, "error": str(e), "status": "failed"}
                continue

    # Strategy 2: Try ctypes-based low-level API (if AlazarGetParameter exists)
    if hasattr(ats, 'AlazarGetParameter'):
        try:
            # Parameter IDs from ATS-SDK docs
            GET_FPGA_TEMPERATURE = getattr(ats, "GET_FPGA_TEMPERATURE", 0x10000080)
            GET_ADC_TEMPERATURE  = getattr(ats, "GET_ADC_TEMPERATURE",  0x10000104)
            API_SUCCESS_CODES = [0, 512, 200]
            
            def _get_handle():
                for name in ("handle", "boardHandle", "_handle", "_board_handle", "hBoard", "_hBoard"):
                    h = getattr(board, name, None)
                    if h is not None:
                        return h
                return None

            def _decode_long_as_temp(val_long: int) -> Optional[float]:
                # Try as integer
                if -40 <= val_long <= 150:
                    return float(val_long)
                # Try low 32 bits as float
                try:
                    u32 = val_long & 0xFFFFFFFF
                    f = struct.unpack("<f", struct.pack("<I", u32))[0]
                    if math.isfinite(f) and (-40.0 <= f <= 150.0):
                        return float(f)
                except:
                    pass
                # Try high 32 bits as float (ATS-9352)
                try:
                    u32_high = (val_long >> 32) & 0xFFFFFFFF
                    f = struct.unpack("<f", struct.pack("<I", u32_high))[0]
                    if math.isfinite(f) and (-40.0 <= f <= 150.0):
                        return float(f)
                except:
                    pass
                return None

            h = _get_handle()
            if h is not None:
                # Try to get ADC temperature (channel 1)
                ret_long = ctypes.c_long(0)
                try:
                    rc = ats.AlazarGetParameter(h, 1, GET_ADC_TEMPERATURE, ctypes.byref(ret_long))
                    if rc in API_SUCCESS_CODES:
                        temp = _decode_long_as_temp(int(ret_long.value))
                        if temp is not None:
                            out["adc"] = temp
                            out["diag"] = {"method": "AlazarGetParameter", "param": "ADC_TEMP", "value": temp, "status": "success"}
                            return out
                except Exception as e:
                    pass  # Silently continue if this fails
        except Exception:
            pass  # Low-level API not available or failed

    # Temperature reading not available
    out["diag"] = {
        "status": "unavailable",
        "message": "Temperature reading not supported by this atsapi wrapper version. "
                   "The ATS-9352 board has temperature sensors, but your Python wrapper "
                   "does not expose getBoardTemperature(), getTemperature(), or AlazarGetParameter(). "
                   "This is normal for some atsapi versions - temperature monitoring is optional."
    }
    
    return out
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

    ch_cfg = cfg.get("channels", {}) or {}
    for nm, mask in [("A", ats.CHANNEL_A), ("B", ats.CHANNEL_B)]:
        if nm in ch_cfg:
            cc = ch_cfg[nm] or {}
            coupling_name = str(cc.get('coupling', 'DC'))
            if not coupling_name.endswith('_COUPLING'):
                coupling_name = coupling_name + '_COUPLING'
            coupling = ats_const(coupling_name)
            rng_name = str(cc.get('range', 'PM_1_V'))
            rng = ats_const('INPUT_RANGE_', rng_name)
            if nm == 'A':
                vpp_A = _range_name_to_vpp(rng_name, default_vpp=vpp_default)
            elif nm == 'B':
                vpp_B = _range_name_to_vpp(rng_name, default_vpp=vpp_default)
            imp = ats_const("IMPEDANCE_", str(cc.get("impedance", "50_OHM")))
            board.inputControlEx(mask, coupling, rng, imp)

    t = cfg.get("trigger", {}) or {}
    operation = ats_const(str(t.get("operation", "TRIG_ENGINE_OP_J")))
    engine1 = ats_const(str(t.get("engine1", "TRIG_ENGINE_J")))
    engine2 = ats_const(str(t.get("engine2", "TRIG_ENGINE_K")))

    board.setTriggerOperation(
        operation,
        engine1,
        ats_const(str(t.get("sourceJ", "TRIG_EXTERNAL"))),
        ats_const(str(t.get("slopeJ", "TRIGGER_SLOPE_POSITIVE"))),
        int(t.get("levelJ", 128)),
        engine2,
        ats_const(str(t.get("sourceK", "TRIG_DISABLE"))),
        ats_const(str(t.get("slopeK", "TRIGGER_SLOPE_POSITIVE"))),
        int(t.get("levelK", 128)),
    )
    ext_coupling_name = str(t.get("ext_coupling", "DC_COUPLING"))
    if not ext_coupling_name.endswith("_COUPLING"):
        ext_coupling_name = ext_coupling_name + "_COUPLING"
    board.setExternalTrigger(
        ats_const(ext_coupling_name),
        ats_const(str(t.get("ext_range", "ETR_5V")))
    )

    board.setTriggerDelay(int(t.get("delay_samples", 0)))

    timeout_ms = int(t.get("timeout_ms", 0))
    rt = cfg.get("runtime", {}) or {}
    if bool(rt.get("noise_test", False)) and timeout_ms == 0:
        timeout_ms = int(rt.get("autotrigger_timeout_ms", 10))
        print(f"[CAPPY] noise_test enabled -> using trigger.timeout_ms={timeout_ms} for auto-trigger noise captures")
    board.setTriggerTimeOut(timeout_ms)

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

def run_capture(cfg_path: Path) -> int:
    if not ATS_AVAILABLE or ats is None:
        print("[CAPPY] atsapi not available on this machine.")
        return 2
    if REDUCED_SCHEMA is None or SESSION_INDEX_SCHEMA is None:
        print("[CAPPY] pyarrow is required for capture/archive writing. Please install pyarrow.")
        return 2

    cfg = load_config(cfg_path)
    cfg, cfg_warnings, cfg_errors = validate_and_normalize_capture_cfg(cfg)
    if cfg_errors:
        for e in cfg_errors:
            print(f"[CAPPY] Config error: {e}")
        return 2
    for w in cfg_warnings:
        print(f"[CAPPY] Config warning: {w}")

    acq = cfg.get("acquisition", {})
    timing = cfg.get("timing", {}) or {}
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
    bunch_spacing = int(timing.get("bunch_spacing_samples", spr))

    b0, b1 = map(int, integ.get("baseline_window_samples", [0, min(64, spr)]))
    g0, g1 = map(int, integ.get("integral_window_samples", [min(64, spr), min(128, spr)]))

    if spr > bunch_spacing:
        raise ValueError("samples_per_record must be <= bunch_spacing_samples")

    wf_enable = bool(waves.get("enable", True))
    wf = WfPolicy(
        mode=str(waves.get("mode", "every_n")),
        every_n=int(waves.get("every_n", 20000)),
        thr_area=float(waves.get("threshold_integral_Vs", 0.0)),
        thr_peak=float(waves.get("threshold_peak_V", 0.0)),
        max_per_sec=int(waves.get("max_waveforms_per_sec", 50)),
    )
    store_volts = bool(waves.get("store_volts", True))

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
    )
    sid = archive.start(tag=str(storage.get("session_tag", "")).strip(), channels_mask=ch_expr)
    notifier = StatusNotifier(cfg, Path(str(storage.get('data_dir', 'dataFile'))))
    # Live waveform ring (writes one downsampled waveform per buffer)
    live_cfg = (cfg.get('live', {}) or {})
    ring_nslots = int(live_cfg.get('ring_slots', 4096))
    ring_npts = int(live_cfg.get('ring_points', 512))
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
        live_ring_path=str(ring_path),
        live=dict(
            ring_slots=int(live_cfg.get('ring_slots', ring_nslots)),
            ring_points=int(live_cfg.get('ring_points', ring_npts)),
            stream_window_points=int(live_cfg.get('stream_window_points', 20000)),
            stream_window_seconds=float(live_cfg.get('stream_window_seconds', 2.0)),
            max_waveforms_per_tick=int(live_cfg.get('max_waveforms_per_tick', 20)),
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

    print(f"[CAPPY] Running session {sid}. " + ("Press <enter> to stop." if (sys.stdin is not None and hasattr(sys.stdin,"isatty") and sys.stdin.isatty()) else "Use Stop (GUI) or Ctrl+C to stop."))
    buf_done = 0
    next_wait_idx = 0
    global_rec = 0
    t0 = time.time()
    last = t0
    dashboard_every_buffers = int((cfg.get('notify', {}) or {}).get('dashboard_every_buffers', 1000))
    gui_every_buffers = int((cfg.get('notify', {}) or {}).get('gui_every_buffers', 50))
    last_emit_buf = 0
    last_gui_emit_buf = 0
    timeout_count = 0
    last_buffer_ns = time.time_ns()
    live_cfg = cfg.get('live', {}) or {}
    rt = cfg.get('runtime', {}) or {}
    rearm_if_no_trigger_s = int(rt.get('rearm_if_no_trigger_s', 300))
    rearm_cooldown_s = int(rt.get('rearm_cooldown_s', 30))
    max_rearms_per_hour = int(rt.get('max_rearms_per_hour', 12))
    timeout_pause_s = float(trig.get("timeout_pause_s", 0.0) or 0.0)
    preview_mode = str(live_cfg.get("preview_mode", "archive_match")).strip().lower()
    if preview_mode not in {"archive_match", "record0"}:
        preview_mode = "archive_match"
    rearm_times: List[float] = []

    try:
        def _do_rearm(force: bool = False):
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

            print(f"[CAPPY] Rearming acquisition (no completed buffers for {rearm_if_no_trigger_s}s). rearms_last_hour={len(rearm_times)}")
            notifier.update(state='rearming', time=time.strftime('%Y-%m-%d %H:%M:%S'), timeouts=timeout_count)
            notifier.maybe_emit()

            try:
                board.abortAsyncRead()
            except Exception:
                pass

            board.beforeAsyncRead(ch_mask, -pre, spr, rpb, recordsPerAcq, adma_flags)
            for b in buffers:
                board.postAsyncBuffer(b.addr, b.size_bytes)
            board.startCapture()

            timeout_count = 0
            last_buffer_ns = time.time_ns()
            next_wait_idx = 0
            notifier.update(state='running')
            notifier.maybe_emit()

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
            _do_rearm(force=True)

        board.startCapture()
        while buf_done < buf_target and not _should_stop():
            buf = buffers[next_wait_idx % len(buffers)]
            try:
                board.waitAsyncBufferComplete(buf.addr, wait_timeout_ms)
            except Exception as ex:
                if "ApiBufferNotReady" in str(ex):
                    print("[CAPPY] Warning: ApiBufferNotReady while waiting for DMA buffer; resetting DMA queue.")
                    _do_rearm(force=True)
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
                        _do_rearm()
                    continue
                if "ApiWaitCanceled" in str(ex) or "ApiWaitCancelled" in str(ex):
                    # Normal stop path (SIGINT / abortAsyncRead)
                    break
                raise

            timeout_count = 0
            last_buffer_ns = time.time_ns()
            ts_ns = last_buffer_ns
            # Per-record time offset: each record in the buffer is spaced by bunch_spacing samples
            record_dt_ns = int(round(float(bunch_spacing) / float(sr_hz) * 1e9))

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
                if "ApiBufferOverflow" in msg:
                    print("[CAPPY] Warning: ApiBufferOverflow while recycling DMA buffer; rearming capture.")
                    _do_rearm(force=True)
                    continue
                if "ApiWaitCanceled" in msg or "ApiWaitCancelled" in msg:
                    break
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
                        channels_mask=ch_expr, bunch_spacing_samples=bunch_spacing,
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
                        channels_mask=ch_expr, bunch_spacing_samples=bunch_spacing,
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
                    if gui_every_buffers > 0 and (buf_done - last_gui_emit_buf) >= gui_every_buffers:
                        last_gui_emit_buf = buf_done
                        try:
                            temps = get_board_temperatures_c(board)
                            bt = temps.get('adc') if temps.get('adc') is not None else temps.get('fpga')
                        except Exception:
                            bt = None
                        notifier.update(last_capture=time.strftime('%Y-%m-%d %H:%M:%S'), last_capture_unix=time.time(), buffers=buf_done, board_temp_c=bt, board_temp_diag=(temps.get('diag') if isinstance(temps, dict) else None))
                        notifier.emit_now()
                else:
                    notifier.update(
                        buffer_mean_area_A=float(np.mean(areaA)), buffer_mean_peak_A=float(np.mean(peakA))
                    )
                    if gui_every_buffers > 0 and (buf_done - last_gui_emit_buf) >= gui_every_buffers:
                        last_gui_emit_buf = buf_done
                        try:
                            temps = get_board_temperatures_c(board)
                            bt = temps.get('adc') if temps.get('adc') is not None else temps.get('fpga')
                        except Exception:
                            bt = None
                        notifier.update(last_capture=time.strftime('%Y-%m-%d %H:%M:%S'), last_capture_unix=time.time(), buffers=buf_done, board_temp_c=bt, board_temp_diag=(temps.get('diag') if isinstance(temps, dict) else None))
                        notifier.emit_now()
            except Exception:
                pass

            # Write EVERY buffer's representative waveform into the live ring for smooth GUI playback
            try:
                if preview_mode == "archive_match" and live_wfA_candidate is not None:
                    wfA_live = live_wfA_candidate
                    wfB_live = live_wfB_candidate if ch_count == 2 else None
                    chmask_live = 3 if (ch_count == 2 and wfB_live is not None) else 1
                else:
                    # Fallback for non-archived or record0 preview mode.
                    wfA_live = _codes_to_volts_u16(A[0], vpp=vppA)
                    wfB_live = None
                    chmask_live = 1
                    if ch_count == 2:
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
                # board temperature (best-effort; may be unavailable depending on atsapi)
                board_temp_c = None
                try:
                    # Some atsapi wrappers expose getBoardTemperature / getTemperature
                    if hasattr(board, 'getBoardTemperature'):
                        board_temp_c = float(board.getBoardTemperature())
                    elif hasattr(board, 'getTemperature'):
                        board_temp_c = float(board.getTemperature())
                except Exception:
                    board_temp_c = None

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
                               buffers=buf_done, records=global_rec, rate_hz=rate, last_capture=last_capture, last_capture_unix=last_capture_unix, board_temp_c=board_temp_c, latest_waveform_A=latest_wf_A, latest_waveform_B=latest_wf_B,
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
        self._sel_date = None
        self._sel_hour = None
        self._build()
        self._refresh()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.BOTH, expand=True)
        filt = ttk.Frame(top)
        filt.pack(fill=tk.X)

        self.var_dir = tk.StringVar(value=str(self.data_dir))
        
        pan = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        pan.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        left = ttk.Frame(pan, padding=6)
        right = ttk.Frame(pan, padding=6)
        self._right = right
        pan.add(left, weight=1)
        pan.add(right, weight=2)

        ttk.Label(left, text="Sessions").pack(anchor="w")
        self.slist = tk.Listbox(left)
        self.slist.pack(fill=tk.BOTH, expand=True)
        self.slist.bind("<<ListboxSelect>>", self._on_session)
        ttk.Label(left, text="Date").pack(anchor="w", pady=(8,0))
        self.var_date = tk.StringVar(value="")
        self.cmb_date = ttk.Combobox(left, textvariable=self.var_date, state="readonly", width=18)
        self.cmb_date.pack(fill=tk.X, expand=False)
        self.cmb_date.bind("<<ComboboxSelected>>", self._on_date)
        hour_row = ttk.Frame(left)
        hour_row.pack(fill=tk.X, pady=(6,0))
        ttk.Label(hour_row, text="Hour").pack(side=tk.LEFT)
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
        self.wlist = tk.Listbox(left, exportselection=False)
        self.wlist.pack(fill=tk.BOTH, expand=True)
        self.wlist.bind("<<ListboxSelect>>", self._on_snip)

        _, plt, _FigureCanvasTkAgg = _lazy_mpl()
        self.fig, (self.axA, self.axB, self.axI) = plt.subplots(3, 1, figsize=(7.5,6.8))
        self.fig.patch.set_facecolor('#1e1e1e')
        self.axA.set_facecolor('#2d2d2d')
        self.axB.set_facecolor('#2d2d2d')
        self.axI.set_facecolor('#2d2d2d')
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
            toolbar_frame = tk.Frame(self._right, bg='#3d3d3d', height=36)
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
                               font=('Courier', 10, 'bold'), fg=NEON_PINK, bg='#1e1e1e',
                               anchor='w', padx=6, pady=2)
        readout_lbl.pack(fill=tk.X)

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

        self.meta = tk.Text(self._right, height=11, bg='#2d2d2d', fg='white', insertbackground='white', font=('Courier', 9))
        self.meta.pack(fill=tk.X, pady=(8,0))
        self.meta.configure(state=tk.DISABLED)

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

        self.slist.delete(0, tk.END)
        self.wlist.delete(0, tk.END)

        if self.sessions.empty:
            self.slist.insert(tk.END, "(no sessions)")
            return

        self._sessions_view = getattr(self, '_sessions_view', self.sessions).reset_index(drop=True)
        for _, r in self._sessions_view.iterrows():
            t0 = datetime.fromtimestamp(int(r["first_timestamp_ns"]) / 1e9, tz=self._tz)
            self.slist.insert(tk.END, f"{t0.strftime('%Y-%m-%d %H:%M:%S')}  {r['session_id']}  snips={int(r['waveform_snips'])}")

        # If a session is already selected and snips are loaded, re-apply snip filter.
        if getattr(self, 'snips', _lazy_pandas().DataFrame()).empty is False:
            self._apply_hour_filter()

    def _sel_sid(self) -> Optional[str]:
        if self.sessions.empty:
            return None
        sel = self.slist.curselection()
        if not sel:
            return None
        view = getattr(self, '_sessions_view', self.sessions)
        return str(view.iloc[sel[0]]["session_id"]).strip()


    def _build_hour_index(self):
        """Add _date/_hour columns to self.snips (local time)."""
        if self.snips is None or self.snips.empty:
            return
        # Treat stored ns as UTC epoch, then convert to local timezone for display/grouping.
        dt = _lazy_pandas().to_datetime(self.snips["timestamp_ns"], unit="ns", utc=True).dt.tz_convert(self._tz)
        self.snips = self.snips.copy()
        self.snips["_date"] = dt.dt.strftime("%Y-%m-%d")
        self.snips["_hour"] = dt.dt.strftime("%H")

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

        # Hours for selected date
        sub = self.snips[self.snips["_date"] == self._sel_date] if self._sel_date is not None else self.snips
        counts = sub["_hour"].value_counts().sort_index()
        hours = list(counts.index)

        for hh in hours:
            self.hlist.insert(tk.END, f"{hh}:00  ({int(counts[hh])})")

        if self._sel_hour is None or self._sel_hour not in hours:
            self._sel_hour = str(hours[0]) if hours else None

        if self._sel_hour is not None and self._sel_hour in hours:
            idx = hours.index(self._sel_hour)
            self.hlist.selection_clear(0, tk.END)
            self.hlist.selection_set(idx)
            self.hlist.see(idx)

    def _apply_hour_filter(self):
        """Render wlist using selected date/hour."""
        self.wlist.delete(0, tk.END)
        self._snips_view = _lazy_pandas().DataFrame()
        if self.snips is None or self.snips.empty:
            self.wlist.insert(tk.END, "(no saved waveforms)")
            return
        df = self.snips
        if "_date" in df.columns and self._sel_date is not None:
            df = df[df["_date"] == self._sel_date]
        if "_hour" in df.columns and self._sel_hour is not None:
            df = df[df["_hour"] == self._sel_hour]

        self._snips_view = df.sort_values("timestamp_ns", ascending=False)

        if self._snips_view.empty:
            self.wlist.insert(tk.END, "(no saved waveforms in this hour)")
            return

        for _, r in self._snips_view.iterrows():
            ts_ns_val = int(r["timestamp_ns"])
            ts = datetime.fromtimestamp(ts_ns_val / 1e9, tz=self._tz)
            # Show full microsecond precision: HH:MM:SS.uuuuuu
            us_str = ts.strftime('%H:%M:%S') + f".{ts_ns_val % 1_000_000_000 // 1000:06d}"
            self.wlist.insert(tk.END, f"{us_str}  id={int(r['id'])}  buf={int(r['buffer_index'])}  rec={int(r['record_in_buffer'])}  g={int(r['record_global'])}")

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
        # 'HH:00  (N)'
        self._sel_hour = txt.split(":")[0].strip()
        self._apply_hour_filter()


    def _seek_mmss(self):
        """Jump to the snip closest to MM:SS within the currently selected date/hour."""
        if getattr(self, "_snips_view", _lazy_pandas().DataFrame()).empty:
            return
        txt = (self.var_seek.get() if hasattr(self, "var_seek") else "").strip()
        if not txt:
            return
        # Parse MM:SS (allow SS or M:SS)
        mm = 0
        ss = 0
        try:
            if ":" in txt:
                a, b = txt.split(":", 1)
                mm = int(a)
                ss = int(b)
            else:
                ss = int(txt)
        except Exception:
            messagebox.showerror("Invalid time", "Enter time as MM:SS (e.g., 12:34).")
            return
        if ss < 0 or ss > 59 or mm < 0:
            messagebox.showerror("Invalid time", "Seconds must be 0–59.")
            return
        if self._sel_date is None or self._sel_hour is None:
            return
        try:
            # Build a local-time target and compare in epoch ns
            y, m, d = map(int, str(self._sel_date).split("-"))
            hh = int(self._sel_hour)
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
        self.wlist.selection_clear(0, tk.END)
        self.wlist.selection_set(pos)
        self.wlist.see(pos)
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
                self.snips = _lazy_pandas().read_sql_query(
                    "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                    "file,offset_bytes,nbytes,"
                    "file_A,offset_A,nbytes_A,file_B,offset_B,nbytes_B,"
                    "area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V "
                    "FROM snips WHERE session_id=? ORDER BY timestamp_ns DESC LIMIT 50000",
                    conn,
                    params=(sid,),
                )
                # If query returned nothing, try without session_id filter (DB might only have one session)
                if self.snips.empty:
                    self.snips = _lazy_pandas().read_sql_query(
                        "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                        "file,offset_bytes,nbytes,"
                        "file_A,offset_A,nbytes_A,file_B,offset_B,nbytes_B,"
                        "area_A_Vs,peak_A_V,baseline_A_V,area_B_Vs,peak_B_V,baseline_B_V "
                        "FROM snips ORDER BY timestamp_ns DESC LIMIT 50000",
                        conn,
                    )
            except Exception:
                # Legacy DB schema (no channel-separated columns or baseline columns)
                try:
                    self.snips = _lazy_pandas().read_sql_query(
                        "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                        "file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V "
                        "FROM snips WHERE session_id=? ORDER BY timestamp_ns DESC LIMIT 50000",
                        conn,
                        params=(sid,),
                    )
                    if self.snips.empty:
                        self.snips = _lazy_pandas().read_sql_query(
                            "SELECT id,session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,"
                            "file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V "
                            "FROM snips ORDER BY timestamp_ns DESC LIMIT 50000",
                            conn,
                        )
                except Exception:
                    pass
            finally:
                conn.close()
            self._snip_db_dir = ddir
            break

        self.wlist.delete(0, tk.END)
        if self.snips.empty:
            self.wlist.insert(tk.END, "(no saved waveforms)")
            return
        # Populate hour list and apply hour filter
        self._populate_hours()
        self._apply_hour_filter()
        self._set_meta(f"Snips loaded: {len(self.snips):,}")

    def _on_snip(self, _=None):
        if self.snips.empty or self._snip_db_dir is None:
            return
        sel = self.wlist.curselection()
        if not sel:
            return
        # listbox entry format: "HH:MM:SS  id=123  buf=..."
        try:
            txt = str(self.wlist.get(sel[0]))
            # Extract the id value from "id=123"
            for part in txt.split():
                if part.startswith("id="):
                    snip_id = int(part.split("=")[1])
                    break
            else:
                return
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

        sr = float(r.get("sample_rate_hz", np.nan))
        if not np.isfinite(sr) or sr <= 0:
            sr = 1.0

        # Get baseline values (mean voltage in baseline window, already computed during acquisition)
        baseline_A = float(r.get("baseline_A_V", 0.0))
        baseline_B = float(r.get("baseline_B_V", 0.0))
        
        # Fallback: If baseline values are missing (old data), calculate from waveform
        # Using first 64 samples as baseline window (matching default config)
        if baseline_A == 0.0 and len(wa) >= 64:
            baseline_A = float(np.mean(wa[:64]))
        if baseline_B == 0.0 and wb is not None and len(wb) >= 64:
            baseline_B = float(np.mean(wb[:64]))
        
        # Apply baseline subtraction to waveforms for display
        wa_bs = wa - baseline_A
        wb_bs = wb - baseline_B if wb is not None else None

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

        # Waveform A (baseline-subtracted)
        self.axA.plot(tvec, wa_bs, color=NEON_PINK, linewidth=1.5)
        self.axA.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
        self.axA.set_ylabel("A (V)", color=NEON_PINK)
        self.axA.tick_params(colors=NEON_PINK)
        self.axA.spines['left'].set_color(NEON_PINK)
        self.axA.spines['bottom'].set_color('white')
        self.axA.spines['top'].set_visible(False)
        self.axA.spines['right'].set_visible(False)
        self.axA.grid(True, alpha=0.15, color=NEON_PINK)
        self._hover_lines[self.axA] = (tvec, wa_bs, "Ch A", NEON_PINK)

        # Waveform B (baseline-subtracted if available)
        if wb_bs is not None:
            self.axB.plot(tvec, wb_bs, color=NEON_GREEN, linewidth=1.5)
            self.axB.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
            self.axB.set_ylabel("B (V)", color=NEON_GREEN)
            self.axB.tick_params(colors=NEON_GREEN)
            self.axB.spines['left'].set_color(NEON_GREEN)
            self.axB.spines['bottom'].set_color('white')
            self.axB.spines['top'].set_visible(False)
            self.axB.spines['right'].set_visible(False)
            self._hover_lines[self.axB] = (tvec, wb_bs, "Ch B", NEON_GREEN)
        else:
            self.axB.text(0.02, 0.5, "Channel B not captured in this snip", transform=self.axB.transAxes, color='white')
            self.axB.set_ylabel("B (V)", color='white')
            self.axB.tick_params(colors='white')
            self.axB.spines['left'].set_color('white')
            self.axB.spines['bottom'].set_color('white')
            self.axB.spines['top'].set_visible(False)
            self.axB.spines['right'].set_visible(False)
        self.axB.grid(True, alpha=0.15, color='white')

        # Cumulative integral (V·s) - using baseline-subtracted waveforms
        dt = 1.0 / sr
        intA = np.cumsum(np.asarray(wa_bs, dtype=np.float64)) * dt
        self.axI.plot(tvec, intA, color=NEON_PINK, linewidth=1.5, label="∫A dt")
        self._hover_lines[self.axI] = (tvec, intA, "∫A (V·s)", NEON_PINK)
        if wb_bs is not None:
            intB = np.cumsum(np.asarray(wb_bs, dtype=np.float64)) * dt
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

        self.canvas.draw()

        # Reset the toolbar's home/zoom history so 'Home' button returns to this view
        if self._nav_toolbar is not None:
            try:
                self._nav_toolbar.update()
            except Exception:
                pass

        ts_ns_val = int(r["timestamp_ns"])
        ts = datetime.fromtimestamp(ts_ns_val / 1e9, tz=self._tz)
        us_str = ts.strftime('%Y-%m-%d %H:%M:%S') + f".{ts_ns_val % 1_000_000_000 // 1000:06d}"
        a_vs = float(r.get("area_A_Vs", np.nan)) if "area_A_Vs" in r else float(r.get("area_A_Vs", np.nan))
        b_vs = float(r.get("area_B_Vs", np.nan)) if "area_B_Vs" in r else float(r.get("area_B_Vs", np.nan))
        self._set_meta(
            f"Timestamp: {us_str}\n"
            f"Session: {r.get('session_id','?')}\n"
            f"Buffer: {int(r.get('buffer_index',-1))}  Record: {int(r.get('record_in_buffer',-1))}  Global: {int(r.get('record_global',-1))}\n"
            f"Channels: {r.get('channels_mask','?')}  Sample rate: {sr:.6g} Hz\n"
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
        us_part = abs_ns % 1_000_000_000 // 1000
        ts_str = abs_dt.strftime('%H:%M:%S') + f".{us_part:06d}"

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
    Dashboard optimized for capture monitoring:
      - Top stats: rate, captures, started, last capture, mean peak, board temperature
      - Plots: mean integral history + latest waveform
    Colors:
      - Channel A: integral RED, waveform BLUE
      - Channel B (if enabled): PINK (dashed)
    """
    def __init__(self, master, data_dir_var: tk.StringVar):
        super().__init__(master, padding=6)
        self.data_dir_var = data_dir_var
        self.status_path: Optional[Path] = None
        self.ring_path: Optional[Path] = None
        self._ring_npts = 1024
        self._ring_rec_bytes = None
        self._ring_hdr_bytes = 32
        self._ring_last_seq = 0
        self._ring_play_seq = 0
        self._ring_nslots = 0

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
        self._streamAT = np.empty((0,), dtype=np.float64)
        self._streamBT = np.empty((0,), dtype=np.float64)
        self._stream_window = 20000  # points shown in scrolling mode
        self._stream_window_s = 2.0  # seconds shown in scrolling mode
        self._latest_ring_unix: Optional[float] = None

        self._started_unix: Optional[float] = None
        self._last_seen_seq: int = 0

        # --- top stats ---
        stats = ttk.Frame(self)
        stats.pack(fill=tk.X, pady=(0, 6))

        def mkrow(label: str):
            f = ttk.Frame(stats)
            f.pack(side=tk.LEFT, padx=(0, 14))
            ttk.Label(f, text=label).pack(anchor="w")
            v = ttk.Label(f, text="—")
            v.pack(anchor="w")
            return v

        self.lbl_rate = mkrow("Rate (Hz)")
        self.lbl_caps = mkrow("Captures (buffers)")
        self.lbl_started = mkrow("Started")
        self.lbl_last = mkrow("Last capture")
        self.lbl_peak = mkrow("Mean peak (V)")
        self.lbl_temp = mkrow("Board temp (°C)")

        # --- plots ---
        _, plt, _FigureCanvasTkAgg = _lazy_mpl()
        self.fig = plt.Figure(figsize=(8.2, 7.2))
        self.fig.patch.set_facecolor('#1e1e1e')

        # Scope-like layout: Channel A (top), Channel B (middle), Integration history (bottom)
        self.ax_wfA = self.fig.add_subplot(311)
        self.ax_wfA.set_facecolor('#2d2d2d')
        self.ax_wfB = self.fig.add_subplot(312)
        self.ax_wfB.set_facecolor('#2d2d2d')
        self.ax_int = self.fig.add_subplot(313)
        self.ax_int.set_facecolor('#2d2d2d')

        self.canvas = _FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.meta = tk.Text(self, height=3, bg='#2d2d2d', fg='white', insertbackground='white', font=('Courier', 9))
        self.meta.pack(fill=tk.X, pady=(6, 0))
        self.meta.configure(state=tk.DISABLED)

        # Schedule periodic UI updates
        if hasattr(self, "_tick"):
            self.after(100, self._tick)
        else:
            # Fallback to avoid startup crash if _tick is missing due to editing/merge issues
            self.after(100, lambda: None)

    def _set_meta(self, s: str):
        self.meta.configure(state=tk.NORMAL)
        self.meta.delete("1.0", tk.END)
        self.meta.insert(tk.END, s)
        self.meta.configure(state=tk.DISABLED)

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

    def _append_point(self, snap: dict):
        seq = snap.get("status_seq", None)
        if seq is None:
            return
        try:
            seq = int(seq)
        except Exception:
            return
        if seq <= self._last_seen_seq:
            return
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
            return

        t = float(tu) - float(self._started_unix)
        self.t.append(t)
        self.areaA.append(float(snap.get("buffer_mean_area_A", 0.0)))
        self.areaB.append(float(snap.get("buffer_mean_area_B", 0.0)))

        while self.t and (self.t[-1] - self.t[0]) > 600.0:
            self.t.pop(0); self.areaA.pop(0); self.areaB.pop(0)


    def _open_ring_from_status(self, snap: dict):
        rp = snap.get("live_ring_path", None)
        if not rp:
            return
        p = Path(str(rp)).expanduser()
        if self.ring_path != p:
            self.ring_path = p
            # reset playback on ring change
            self._ring_last_seq = 0
            self._ring_play_seq = 0
            self._wfA_hist.clear()
            self._wfB_hist.clear()
            self._streamA = np.empty((0,), dtype=np.float32)
            self._streamB = np.empty((0,), dtype=np.float32)
            self._streamAT = np.empty((0,), dtype=np.float64)
            self._streamBT = np.empty((0,), dtype=np.float64)
            self._latest_ring_unix = None

    def _read_ring_next(self):
        """
        Read the next available waveform record from the live ring.
        Returns (wfA, wfB, t_unix, chmask, seq) or all-None if nothing new.
        """
        if self.ring_path is None:
            return None, None, None, None, None

        try:
            import struct
            with open(self.ring_path, "rb", buffering=0) as f:
                hdr = f.read(32)
                if len(hdr) != 32:
                    return None, None, None, None, None
                magic, ver, nslots, npts, _rsv, write_seq = struct.unpack("<8sIIIIQ", hdr)
                if magic != LiveRingWriter.MAGIC:
                    return None, None, None, None, None
                self._ring_nslots = int(nslots)
                self._ring_npts = int(npts)
                rec_bytes = (8 + 8 + 8 + 4 + 4) + (self._ring_npts * 4) + (self._ring_npts * 4)
                self._ring_rec_bytes = rec_bytes
                self._ring_last_seq = int(write_seq)

                # initialize play seq to most recent history window on first open
                if self._ring_play_seq == 0 and self._ring_last_seq > 0:
                    # start just behind the head so the plot fills quickly
                    self._ring_play_seq = max(1, self._ring_last_seq - 50)

                if self._ring_play_seq >= self._ring_last_seq:
                    return None, None, None, None, None

                # read next record
                self._ring_play_seq += 1
                seq = self._ring_play_seq
                slot = (seq - 1) % self._ring_nslots
                off = self._ring_hdr_bytes + slot * rec_bytes
                f.seek(off)
                rec_hdr = f.read(8 + 8 + 8 + 4 + 4)
                if len(rec_hdr) != (8 + 8 + 8 + 4 + 4):
                    return None, None, None, None, None
                r_seq, t_unix, buf_idx, chmask, _r = struct.unpack("<QdQII", rec_hdr)
                if int(r_seq) != int(seq):
                    # slot not yet written / overwritten; jump to last
                    self._ring_play_seq = self._ring_last_seq
                    return None, None, None, None, None

                wfA = np.frombuffer(f.read(self._ring_npts * 4), dtype=np.float32).copy()
                wfB = np.frombuffer(f.read(self._ring_npts * 4), dtype=np.float32).copy()
                if (chmask & 2) == 0:
                    wfB = None
                return wfA, wfB, float(t_unix), int(chmask), int(seq)
        except Exception:
            return None, None, None, None, None

    def _redraw(self, snap: dict):
        """
        Redraw all plots with optimizations for smooth scrolling.
        Uses smart axis limits and efficient plotting for better performance.
        """
        # Clear axes efficiently
        self.ax_wfA.clear()
        self.ax_wfB.clear()
        self.ax_int.clear()

        # --- Channel A waveform (top) - Optimized rendering ---
        if self._streamA.size > 1:
            # Baseline-subtract: remove DC offset using rolling mean of the stream
            stream_a = self._streamA.copy()
            bl_a = np.mean(stream_a)
            stream_a = stream_a - bl_a

            if self._streamAT.size == stream_a.size and self._streamAT.size > 1:
                x_full = (self._streamAT - self._streamAT[-1]) * 1e3
                x_label = "Time from latest (ms)"
            else:
                x_full = np.arange(stream_a.size, dtype=np.float64)
                x_label = "Rolling samples"

            # Use downsampled view if stream is very large for better performance
            if stream_a.size > 50000:
                # Decimate for display only - use every Nth point
                step = max(1, stream_a.size // 20000)
                x_view = x_full[::step]
                y_view = stream_a[::step]
            else:
                x_view = x_full
                y_view = stream_a
            
            # Plot with reduced antialiasing for speed
            line_a, = self.ax_wfA.plot(x_view, y_view, color=NEON_PINK, linewidth=0.8, antialiased=True, rasterized=True)
            self.ax_wfA.set_ylabel("A (V)", color=NEON_PINK)
            self.ax_wfA.set_xlabel(x_label, color='white')
            self.ax_wfA.tick_params(colors='white')
            self.ax_wfA.spines['left'].set_color(NEON_PINK)
            self.ax_wfA.spines['bottom'].set_color('white')
            self.ax_wfA.spines['top'].set_visible(False)
            self.ax_wfA.spines['right'].set_visible(False)
            self.ax_wfA.grid(True, alpha=0.15, color=NEON_PINK, linestyle='-', linewidth=0.5)
        else:
            self.ax_wfA.set_title("Channel A: waiting for waveforms…", color='white')
            self.ax_wfA.set_ylabel("A (V)", color=NEON_PINK)
            self.ax_wfA.set_xlabel("Time from latest (ms)", color='white')
            self.ax_wfA.tick_params(colors='white')
            self.ax_wfA.spines['left'].set_color(NEON_PINK)
            self.ax_wfA.spines['bottom'].set_color('white')
            self.ax_wfA.spines['top'].set_visible(False)
            self.ax_wfA.spines['right'].set_visible(False)
            self.ax_wfA.grid(True, alpha=0.15, color=NEON_PINK, linestyle='-', linewidth=0.5)

        # --- Channel B waveform (middle) - Optimized rendering ---
        if self._streamB.size > 1 and not np.all(np.isnan(self._streamB)):
            # Baseline-subtract: remove DC offset using rolling mean of the stream
            stream_b = self._streamB.copy()
            bl_b = np.mean(stream_b)
            stream_b = stream_b - bl_b

            if self._streamBT.size == stream_b.size and self._streamBT.size > 1:
                xb_full = (self._streamBT - self._streamBT[-1]) * 1e3
                xb_label = "Time from latest (ms)"
            else:
                xb_full = np.arange(stream_b.size, dtype=np.float64)
                xb_label = "Rolling samples"

            # Use downsampled view if stream is very large
            if stream_b.size > 50000:
                step = max(1, stream_b.size // 20000)
                xb_view = xb_full[::step]
                yb_view = stream_b[::step]
            else:
                xb_view = xb_full
                yb_view = stream_b
            
            line_b, = self.ax_wfB.plot(xb_view, yb_view, color=NEON_GREEN, linewidth=0.8, antialiased=True, rasterized=True)
            self.ax_wfB.set_ylabel("B (V)", color=NEON_GREEN)
            self.ax_wfB.set_xlabel(xb_label, color='white')
            self.ax_wfB.tick_params(colors='white')
            self.ax_wfB.spines['left'].set_color(NEON_GREEN)
            self.ax_wfB.spines['bottom'].set_color('white')
            self.ax_wfB.spines['top'].set_visible(False)
            self.ax_wfB.spines['right'].set_visible(False)
            self.ax_wfB.grid(True, alpha=0.15, color=NEON_GREEN, linestyle='-', linewidth=0.5)
        else:
            self.ax_wfB.set_title("Channel B: waiting for waveforms…", color='white')
            self.ax_wfB.set_ylabel("B (V)", color=NEON_GREEN)
            self.ax_wfB.set_xlabel("Time from latest (ms)", color='white')
            self.ax_wfB.tick_params(colors='white')
            self.ax_wfB.spines['left'].set_color(NEON_GREEN)
            self.ax_wfB.spines['bottom'].set_color('white')
            self.ax_wfB.spines['top'].set_visible(False)
            self.ax_wfB.spines['right'].set_visible(False)
            self.ax_wfB.grid(True, alpha=0.15, color=NEON_GREEN, linestyle='-', linewidth=0.5)

        if self._stream_window_s > 0:
            w_ms = float(self._stream_window_s) * 1e3
            try:
                if self._streamAT.size == self._streamA.size and self._streamA.size > 1:
                    self.ax_wfA.set_xlim(left=-w_ms, right=0.0)
                if self._streamBT.size == self._streamB.size and self._streamB.size > 1:
                    self.ax_wfB.set_xlim(left=-w_ms, right=0.0)
            except Exception:
                pass

        # --- Integration history strip (bottom) ---
        if self.t:
            # Downsample integration history if very long for better performance
            if len(self.t) > 5000:
                step = max(1, len(self.t) // 2000)
                t_view = self.t[::step]
                areaA_view = self.areaA[::step]
                areaB_view = self.areaB[::step] if self.areaB else []
            else:
                t_view = self.t
                areaA_view = self.areaA
                areaB_view = self.areaB
            
            self.ax_int.plot(t_view, areaA_view, color=NEON_PINK, linewidth=1.2, label="Mean integral A (V·s)", antialiased=True, rasterized=True)
            if areaB_view and np.any(np.isfinite(np.asarray(areaB_view, dtype=float))) and np.any(np.abs(np.asarray(areaB_view, dtype=float)) > 0):
                self.ax_int.plot(t_view, areaB_view, color=NEON_GREEN, linewidth=1.2, linestyle="--", label="Mean integral B (V·s)", antialiased=True, rasterized=True)
            self.ax_int.set_ylabel("Integral (V·s)", color='white')
            self.ax_int.set_xlabel("Time (s)", color='white')
            self.ax_int.tick_params(colors='white')
            self.ax_int.spines['left'].set_color('white')
            self.ax_int.spines['bottom'].set_color('white')
            self.ax_int.spines['top'].set_visible(False)
            self.ax_int.spines['right'].set_visible(False)
            self.ax_int.grid(True, alpha=0.15, color='white', linestyle='-', linewidth=0.5)
            legend = self.ax_int.legend(loc="best")
            for text in legend.get_texts():
                text.set_color('white')
        else:
            self.ax_int.set_title("Integration: waiting for data…", color='white')
            self.ax_int.set_ylabel("Integral (V·s)", color='white')
            self.ax_int.set_xlabel("Time (s)", color='white')
            self.ax_int.tick_params(colors='white')
            self.ax_int.spines['left'].set_color('white')
            self.ax_int.spines['bottom'].set_color('white')
            self.ax_int.spines['top'].set_visible(False)
            self.ax_int.spines['right'].set_visible(False)
            self.ax_int.grid(True, alpha=0.15, color='white', linestyle='-', linewidth=0.5)

        # Use tight layout for better spacing
        try:
            self.fig.tight_layout(pad=0.5)
        except Exception:
            pass
        
        # Efficient canvas draw with flush
        self.canvas.draw_idle()  # Use draw_idle() instead of draw() for better performance
        self.canvas.flush_events()

    def _tick(self):
        snap = self._read_status()
        if snap:
            # stats
            try:
                self.lbl_rate.configure(text=f"{float(snap.get('rate_hz',0.0)):.3f}")
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
                    if pB is not None and float(pB) != 0.0:
                        self.lbl_peak.configure(text=f"A {float(pA):.6g}   B {float(pB):.6g}")
                    else:
                        self.lbl_peak.configure(text=f"{float(pA):.6g}")
            except Exception:
                self.lbl_peak.configure(text=str(pA))

            try:
                live_cfg = snap.get('live', {}) if isinstance(snap.get('live', {}), dict) else {}
                if 'stream_window_points' in live_cfg:
                    self._stream_window = int(live_cfg.get('stream_window_points', self._stream_window))
                if 'stream_window_seconds' in live_cfg:
                    self._stream_window_s = float(live_cfg.get('stream_window_seconds', self._stream_window_s))
            except Exception:
                pass

            bt = snap.get("board_temp_c", None)
            self.lbl_temp.configure(text="—" if bt is None else f"{float(bt):.2f}")

            self._open_ring_from_status(snap)
            self._append_point(snap)
            # Drain multiple waveforms per UI tick so you can see EVERY buffer even if UI refresh is slower.
            try:
                max_wf = int((snap.get('live', {}) or {}).get('max_waveforms_per_tick', 20)) if isinstance(snap.get('live', {}), dict) else 20
            except Exception:
                max_wf = 20
            sr_hz = _to_float(snap.get("sample_rate_hz", 0.0), 0.0)
            spr = _to_int(snap.get("samples_per_record", 0), 0)
            est_record_s = (float(spr) / float(sr_hz)) if (sr_hz > 0 and spr > 0) else 0.0

            def _append_stream(values: np.ndarray, t_unix: float, is_b: bool) -> None:
                if values is None or values.size == 0:
                    return
                n = int(values.size)
                if est_record_s > 0:
                    t0 = float(t_unix) - est_record_s
                    tvec = np.linspace(t0, float(t_unix), num=n, endpoint=False, dtype=np.float64)
                else:
                    dt = 1.0 / float(sr_hz) if sr_hz > 0 else 1e-6
                    t0 = float(t_unix) - (n * dt)
                    tvec = t0 + (np.arange(n, dtype=np.float64) * dt)
                if is_b:
                    self._streamB = np.concatenate([self._streamB, values.astype(np.float32, copy=False)])
                    self._streamBT = np.concatenate([self._streamBT, tvec])
                    if self._stream_window_s > 0 and self._streamBT.size:
                        cut = self._streamBT[-1] - float(self._stream_window_s)
                        i0 = int(np.searchsorted(self._streamBT, cut, side="left"))
                        if i0 > 0:
                            self._streamBT = self._streamBT[i0:]
                            self._streamB = self._streamB[i0:]
                    if self._streamB.size > self._stream_window:
                        self._streamB = self._streamB[-self._stream_window:]
                        self._streamBT = self._streamBT[-self._stream_window:]
                else:
                    self._streamA = np.concatenate([self._streamA, values.astype(np.float32, copy=False)])
                    self._streamAT = np.concatenate([self._streamAT, tvec])
                    if self._stream_window_s > 0 and self._streamAT.size:
                        cut = self._streamAT[-1] - float(self._stream_window_s)
                        i0 = int(np.searchsorted(self._streamAT, cut, side="left"))
                        if i0 > 0:
                            self._streamAT = self._streamAT[i0:]
                            self._streamA = self._streamA[i0:]
                    if self._streamA.size > self._stream_window:
                        self._streamA = self._streamA[-self._stream_window:]
                        self._streamAT = self._streamAT[-self._stream_window:]

            for _ in range(max_wf):
                wfA, wfB, wf_t_unix, _chmask, _seq = self._read_ring_next()
                if wfA is None and wfB is None:
                    break
                if wf_t_unix is not None:
                    self._latest_ring_unix = float(wf_t_unix)
                if wfA is not None:
                    try:
                        _append_stream(wfA, float(wf_t_unix or time.time()), is_b=False)
                    except Exception:
                        pass
                if wfB is not None:
                    try:
                        _append_stream(wfB, float(wf_t_unix or time.time()), is_b=True)
                    except Exception:
                        pass
            self._redraw(snap)

            latest_ring = "-"
            if self._latest_ring_unix is not None:
                try:
                    latest_ring = datetime.fromtimestamp(float(self._latest_ring_unix)).strftime("%H:%M:%S.%f")[:-3]
                except Exception:
                    latest_ring = str(self._latest_ring_unix)
            self._set_meta(
                f"State: {snap.get('state','?')}    Last ring wf: {latest_ring}    Status: {self.status_path}"
            )
        # Schedule periodic UI updates
        if hasattr(self, "_tick"):
            self.after(100, self._tick)
        else:
            # Fallback to avoid startup crash if _tick is missing due to editing/merge issues
            self.after(100, lambda: None)

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

        # Vars (GUI <-> YAML)
        self.var_clock_source = tk.StringVar(value="INTERNAL_CLOCK")
        self.var_rate_msps = tk.DoubleVar(value=250.0)
        self.var_channels_mask = tk.StringVar(value="CHANNEL_A|CHANNEL_B")

        self.var_pre = tk.IntVar(value=0)
        self.var_post = tk.IntVar(value=256)
        self.var_bunch = tk.IntVar(value=424)
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
        self.var_trig_level_pct = tk.DoubleVar(value=50.0)
        self.var_trig_timeout_ms = tk.IntVar(value=0)
        self.var_timeout_pause_s = tk.DoubleVar(value=0.0)
        self.var_trig_mode = tk.StringVar(value="TRIG_ENGINE_OP_J")
        self.var_ext_range = tk.StringVar(value="ETR_5V")
        self.var_ext_coupling = tk.StringVar(value="DC_COUPLING")
        self.var_ext_startcapture = tk.BooleanVar(value=False)

        self.var_rearm_s = tk.IntVar(value=300)
        self.var_rearm_cooldown_s = tk.IntVar(value=30)
        self.var_rearm_per_hour = tk.IntVar(value=12)

        self.var_flush_records = tk.IntVar(value=20000)
        self.var_flush_seconds = tk.DoubleVar(value=2.0)
        self.var_sqlite_commit = tk.IntVar(value=200)
        self.var_rotate_hours = tk.DoubleVar(value=24.0)

        self.var_ring_slots = tk.IntVar(value=4096)
        self.var_ring_points = tk.IntVar(value=512)
        self.var_stream_pts = tk.IntVar(value=20000)
        self.var_stream_s = tk.DoubleVar(value=2.0)
        self.var_max_wf_tick = tk.IntVar(value=20)
        self.var_preview_mode = tk.StringVar(value="archive_match")

        # Layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        left.columnconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        acq = ttk.LabelFrame(left, text="Acquire", padding=8)
        acq.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        acq.columnconfigure(1, weight=1)
        self._add_combobox(acq, 0, "Clock source", self.var_clock_source, ["INTERNAL_CLOCK", "EXTERNAL_CLOCK_10MHZ_REF"])
        self._add_combobox(acq, 1, "Rate (MS/s)", self.var_rate_msps, [str(v) for v in SAMPLE_RATE_OPTIONS_MSPS], width=12)
        self._add_combobox(acq, 2, "Channels", self.var_channels_mask, CHANNEL_MASK_OPTIONS, width=18)
        self._add_entry(acq, 3, "Pre-trigger samples", self.var_pre)
        self._add_entry(acq, 4, "Post-trigger samples", self.var_post)
        self._add_entry(acq, 5, "Bunch spacing", self.var_bunch)
        self._add_entry(acq, 6, "Records/buffer", self.var_rpb)
        self._add_entry(acq, 7, "Buffers allocated", self.var_bufN)
        self._add_entry(acq, 8, "Buffers/acquisition (0=run)", self.var_bpa)
        self._add_entry(acq, 9, "DMA wait timeout (ms)", self.var_wait_timeout_ms)

        vert = ttk.LabelFrame(left, text="Vertical", padding=8)
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

        trig = ttk.LabelFrame(right, text="Trigger", padding=8)
        trig.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        trig.columnconfigure(1, weight=1)
        self._add_combobox(trig, 0, "Mode", self.var_trig_mode, TRIGGER_MODE_OPTIONS, width=18)
        self._add_combobox(trig, 1, "Source", self.var_trig_source, list(TRIGGER_SOURCE_LABEL_TO_CONST.keys()), width=12)
        self._add_combobox(trig, 2, "Slope", self.var_trig_slope, list(TRIGGER_SLOPE_LABEL_TO_CONST.keys()), width=12)
        self._add_entry(trig, 3, "Level (%)", self.var_trig_level_pct)
        self._add_entry(trig, 4, "Auto timeout (ms)", self.var_trig_timeout_ms)
        self._add_entry(trig, 5, "Timeout pause (s)", self.var_timeout_pause_s)
        self._add_combobox(trig, 6, "External range", self.var_ext_range, EXT_TRIGGER_RANGE_OPTIONS)
        self._add_combobox(trig, 7, "External coupling", self.var_ext_coupling, ["DC_COUPLING", "AC_COUPLING"])
        ttk.Checkbutton(trig, text="External start-capture", variable=self.var_ext_startcapture).grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        live = ttk.LabelFrame(right, text="Runtime / Storage / Live", padding=8)
        live.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        live.columnconfigure(1, weight=1)
        self._add_entry(live, 0, "Rearm if no trigger (s)", self.var_rearm_s)
        self._add_entry(live, 1, "Rearm cooldown (s)", self.var_rearm_cooldown_s)
        self._add_entry(live, 2, "Max rearms/hour", self.var_rearm_per_hour)
        self._add_entry(live, 3, "Flush every records", self.var_flush_records)
        self._add_entry(live, 4, "Flush every seconds", self.var_flush_seconds)
        self._add_entry(live, 5, "SQLite commit every snips", self.var_sqlite_commit)
        self._add_entry(live, 6, "Session rotate (hours)", self.var_rotate_hours)
        self._add_entry(live, 7, "Ring slots", self.var_ring_slots)
        self._add_entry(live, 8, "Ring points", self.var_ring_points)
        self._add_entry(live, 9, "Live window points", self.var_stream_pts)
        self._add_entry(live, 10, "Live window seconds", self.var_stream_s)
        self._add_entry(live, 11, "Max waveforms/tick", self.var_max_wf_tick)
        self._add_combobox(live, 12, "Preview mode", self.var_preview_mode, ["archive_match", "record0"], width=16)

        tools = ttk.Frame(self, padding=(8, 0))
        tools.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Button(tools, text="Reload Controls From YAML", command=self.launcher._load_controls_from_yaml).pack(side=tk.LEFT)
        ttk.Label(tools, textvariable=self._msg_var).pack(side=tk.LEFT, padx=(12, 0))

    def _add_entry(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable, width: int = 12) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

    def _add_combobox(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable, values: List[str], width: int = 14) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width)
        cb.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=2)

    def load_from_cfg(self, cfg: Dict[str, Any]) -> None:
        self._loading = True
        try:
            c = cfg.get("clock", {}) or {}
            acq = cfg.get("acquisition", {}) or {}
            timing = cfg.get("timing", {}) or {}
            ch = cfg.get("channels", {}) or {}
            trig = cfg.get("trigger", {}) or {}
            rt = cfg.get("runtime", {}) or {}
            storage = cfg.get("storage", {}) or {}
            live = cfg.get("live", {}) or {}

            self.var_clock_source.set(str(c.get("source", "INTERNAL_CLOCK")))
            self.var_rate_msps.set(_to_float(c.get("sample_rate_msps", 250.0), 250.0))

            mask_expr = str(acq.get("channels_mask", "CHANNEL_A|CHANNEL_B"))
            mask = channels_from_mask_expr(mask_expr)
            self.var_channels_mask.set(channels_mask_to_str(mask))
            self.var_pre.set(_to_int(acq.get("pre_trigger_samples", 0), 0))
            self.var_post.set(_to_int(acq.get("post_trigger_samples", 256), 256))
            self.var_bunch.set(_to_int(timing.get("bunch_spacing_samples", 424), 424))
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
            self.var_trig_level_pct.set((float(level_code) / 255.0) * 100.0)
            self.var_trig_timeout_ms.set(_to_int(trig.get("timeout_ms", 0), 0))
            self.var_timeout_pause_s.set(_to_float(trig.get("timeout_pause_s", 0.0), 0.0))
            self.var_trig_mode.set(str(trig.get("operation", "TRIG_ENGINE_OP_J")))
            self.var_ext_range.set(str(trig.get("ext_range", "ETR_5V")))
            self.var_ext_coupling.set(str(trig.get("ext_coupling", "DC_COUPLING")))
            self.var_ext_startcapture.set(bool(trig.get("external_startcapture", False)))

            self.var_rearm_s.set(_to_int(rt.get("rearm_if_no_trigger_s", 300), 300))
            self.var_rearm_cooldown_s.set(_to_int(rt.get("rearm_cooldown_s", 30), 30))
            self.var_rearm_per_hour.set(_to_int(rt.get("max_rearms_per_hour", 12), 12))

            self.var_flush_records.set(_to_int(storage.get("flush_every_records", 20000), 20000))
            self.var_flush_seconds.set(_to_float(storage.get("flush_every_seconds", 2.0), 2.0))
            self.var_sqlite_commit.set(_to_int(storage.get("sqlite_commit_every_snips", 200), 200))
            self.var_rotate_hours.set(_to_float(storage.get("session_rotate_hours", 24.0), 24.0))

            self.var_ring_slots.set(_to_int(live.get("ring_slots", 4096), 4096))
            self.var_ring_points.set(_to_int(live.get("ring_points", 512), 512))
            self.var_stream_pts.set(_to_int(live.get("stream_window_points", 20000), 20000))
            self.var_stream_s.set(_to_float(live.get("stream_window_seconds", 2.0), 2.0))
            self.var_max_wf_tick.set(_to_int(live.get("max_waveforms_per_tick", 20), 20))
            self.var_preview_mode.set(str(live.get("preview_mode", "archive_match")))
            self._msg_var.set("Controls loaded from YAML.")
        finally:
            self._loading = False

    def apply_to_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(cfg)
        clock = out.setdefault("clock", {}) or {}
        acq = out.setdefault("acquisition", {}) or {}
        timing = out.setdefault("timing", {}) or {}
        channels = out.setdefault("channels", {}) or {}
        trig = out.setdefault("trigger", {}) or {}
        runtime = out.setdefault("runtime", {}) or {}
        storage = out.setdefault("storage", {}) or {}
        live = out.setdefault("live", {}) or {}

        out["clock"] = clock
        out["acquisition"] = acq
        out["timing"] = timing
        out["channels"] = channels
        out["trigger"] = trig
        out["runtime"] = runtime
        out["storage"] = storage
        out["live"] = live

        clock["source"] = str(self.var_clock_source.get() or "INTERNAL_CLOCK")
        clock["sample_rate_msps"] = _to_float(self.var_rate_msps.get(), 250.0)

        acq["channels_mask"] = str(self.var_channels_mask.get() or "CHANNEL_A")
        acq["pre_trigger_samples"] = _to_int(self.var_pre.get(), 0)
        acq["post_trigger_samples"] = _to_int(self.var_post.get(), 256)
        acq["records_per_buffer"] = _to_int(self.var_rpb.get(), 128)
        acq["buffers_allocated"] = _to_int(self.var_bufN.get(), 16)
        acq["buffers_per_acquisition"] = _to_int(self.var_bpa.get(), 0)
        acq["wait_timeout_ms"] = _to_int(self.var_wait_timeout_ms.get(), 1000)

        timing["bunch_spacing_samples"] = _to_int(self.var_bunch.get(), 424)

        chA = channels.setdefault("A", {}) or {}
        chB = channels.setdefault("B", {}) or {}
        channels["A"] = chA
        channels["B"] = chB
        chA["range"] = str(self.var_range_a.get() or "PM_1_V")
        chA["coupling"] = str(self.var_coupling_a.get() or "DC")
        chA["impedance"] = str(self.var_imp_a.get() or "50_OHM")
        chB["range"] = str(self.var_range_b.get() or "PM_1_V")
        chB["coupling"] = str(self.var_coupling_b.get() or "DC")
        chB["impedance"] = str(self.var_imp_b.get() or "50_OHM")

        trig["operation"] = str(self.var_trig_mode.get() or "TRIG_ENGINE_OP_J")
        trig["engine1"] = "TRIG_ENGINE_J"
        trig["engine2"] = "TRIG_ENGINE_K"
        trig["sourceJ"] = TRIGGER_SOURCE_LABEL_TO_CONST.get(self.var_trig_source.get(), "TRIG_EXTERNAL")
        trig["slopeJ"] = TRIGGER_SLOPE_LABEL_TO_CONST.get(self.var_trig_slope.get(), "TRIGGER_SLOPE_POSITIVE")
        trig["levelJ"] = int(round(_clamp_float(self.var_trig_level_pct.get(), 0.0, 100.0, 50.0) * 255.0 / 100.0))
        trig["sourceK"] = str(trig.get("sourceK", "TRIG_DISABLE") or "TRIG_DISABLE")
        trig["slopeK"] = str(trig.get("slopeK", "TRIGGER_SLOPE_POSITIVE") or "TRIGGER_SLOPE_POSITIVE")
        trig["levelK"] = _clamp_int(trig.get("levelK", 128), 0, 255, 128)
        trig["timeout_ms"] = _to_int(self.var_trig_timeout_ms.get(), 0)
        trig["timeout_pause_s"] = _to_float(self.var_timeout_pause_s.get(), 0.0)
        trig["ext_range"] = str(self.var_ext_range.get() or "ETR_5V")
        trig["ext_coupling"] = str(self.var_ext_coupling.get() or "DC_COUPLING")
        trig["external_startcapture"] = bool(self.var_ext_startcapture.get())

        runtime["rearm_if_no_trigger_s"] = _to_int(self.var_rearm_s.get(), 300)
        runtime["rearm_cooldown_s"] = _to_int(self.var_rearm_cooldown_s.get(), 30)
        runtime["max_rearms_per_hour"] = _to_int(self.var_rearm_per_hour.get(), 12)

        storage["data_dir"] = str(self.launcher.var_data_dir.get() or "dataFile")
        storage["flush_every_records"] = _to_int(self.var_flush_records.get(), 20000)
        storage["flush_every_seconds"] = _to_float(self.var_flush_seconds.get(), 2.0)
        storage["sqlite_commit_every_snips"] = _to_int(self.var_sqlite_commit.get(), 200)
        storage["session_rotate_hours"] = _to_float(self.var_rotate_hours.get(), 24.0)

        live["ring_slots"] = _to_int(self.var_ring_slots.get(), 4096)
        live["ring_points"] = _to_int(self.var_ring_points.get(), 512)
        live["stream_window_points"] = _to_int(self.var_stream_pts.get(), 20000)
        live["stream_window_seconds"] = _to_float(self.var_stream_s.get(), 2.0)
        live["max_waveforms_per_tick"] = _to_int(self.var_max_wf_tick.get(), 20)
        live["preview_mode"] = str(self.var_preview_mode.get() or "archive_match")

        norm, warns, errs = validate_and_normalize_capture_cfg(out)
        if errs:
            self._msg_var.set("Invalid settings; fix highlighted values.")
        elif warns:
            self._msg_var.set(f"Applied with {len(warns)} warning(s).")
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
        self.proc = None
        self.pump = None
        self._kill_after_id = None

        # Apply dark mode theme
        self.tk_setPalette(background='#2d2d2d', foreground='white', activeBackground='#404040', activeForeground='white')
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='white')
        style.configure('TButton', background='#404040', foreground='white', borderwidth=1)
        style.map('TButton', background=[('active', '#505050')])
        style.configure('TNotebook', background='#2d2d2d', borderwidth=0)
        style.configure('TNotebook.Tab', background='#404040', foreground='white', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#505050')])
        style.configure('TEntry', fieldbackground='#3d3d3d', background='#3d3d3d', foreground='white', borderwidth=1)
        style.configure('TCombobox', fieldbackground='#3d3d3d', background='#3d3d3d', foreground='white')

        self.var_config = tk.StringVar(value="CAPPY_v1_3.yaml")
        self.var_data_dir = tk.StringVar(value="dataFile")
        # Live written-to-disk status (updated from flush events)
        self._live_out_dir = tk.StringVar(value="Output: -")
        self._live_written = tk.StringVar(value="Written: -")
        self._live_last_flush = tk.StringVar(value="Last flush: -")

        # Auto-create default config so you never need to run `init` in a terminal.
        cfgp = Path(self.var_config.get())
        if not cfgp.exists():
            try:
                _atomic_write_text(cfgp, DEFAULT_YAML)
            except Exception as ex:
                messagebox.showerror('CAPPY', f'Failed to create default config {cfgp}: {ex}')

        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Config YAML:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.var_config, width=52).pack(side=tk.LEFT, padx=(6,8))
        ttk.Button(top, text="Browse…", command=self._pick_yaml).pack(side=tk.LEFT)

        ttk.Label(top, text="Data dir:").pack(side=tk.LEFT, padx=(16,4))
        ttk.Entry(top, textvariable=self.var_data_dir, width=24).pack(side=tk.LEFT)

        ttk.Button(top, text="Open YAML", command=self._open_yaml).pack(side=tk.LEFT, padx=(16,6))
        ttk.Button(top, text="Browse Archive", command=self._browse).pack(side=tk.LEFT)

        ctrl = ttk.Frame(self, padding=8)
        ctrl.pack(fill=tk.X)

        self.btn = ttk.Button(ctrl, text="Start Capture", command=self._toggle)
        self.btn.pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Stop", command=self._stop).pack(side=tk.LEFT, padx=(8,0))
        self.lbl = ttk.Label(ctrl, text="State: idle")
        self.lbl.pack(side=tk.LEFT, padx=(16,0))

        # Tabs (Overview + Log)
        tabs = ttk.Notebook(self)
        tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.dashboard = LiveDashboard(tabs, self.var_data_dir)
        tabs.add(self.dashboard, text="Overview")

        self.live_panel = LiveControlPanel(tabs, self)
        tabs.add(self.live_panel, text="Control")

        logbox = ttk.Frame(tabs, padding=6)
        tabs.add(logbox, text="Log")
        self.log = tk.Text(logbox, wrap="word", state=tk.DISABLED, bg='#2d2d2d', fg='#00ff41', insertbackground='white', font=('Courier', 9))
        self.log.pack(fill=tk.BOTH, expand=True)

        self._load_controls_from_yaml()
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self.after(100, self._poll)

    def _append(self, s: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, s + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)


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
                self._live_written.set(f"Written: {w.get('bytes',0)} bytes, {w.get('records',0)} records, {w.get('samples',0)} samples")
                ts = evt.get("ts_iso") or ""
                if not ts and isinstance(evt.get("ts_ns"), int):
                    ts = ns_to_iso(int(evt["ts_ns"]))
                self._live_last_flush.set(f"Last flush: {ts or '-'}")
                out_dir = evt.get("out_dir") or "-"
                self._live_out_dir.set(f"Output: {out_dir}")
                try:
                    if hasattr(self, "live_panel") and self.live_panel is not None:
                        self.live_panel.update_from_event(evt)
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

        # Clear Python bytecode cache to avoid stale imports and reduce startup churn to avoid stale imports and reduce startup churn
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

        self.lbl.config(text="State: capturing")
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
        try:
            if self.proc is not None and self.proc.poll() is None:
                self._stop()
                self.after(600, self.destroy)
                return
        except Exception:
            pass
        self.destroy()


    def _poll(self):
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
                self.lbl.config(text="State: idle")
                self.btn.config(text="Start Capture")

        self.after(100, self._poll)


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
    if argv and argv[0] not in {"init","capture","browse","gui"}:
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

    args = ap.parse_args(argv)

    if args.cmd == "init":
        p = Path("CAPPY_v1_3.yaml")
        if not p.exists():
            _atomic_write_text(p, DEFAULT_YAML)
        print("[CAPPY] init done")
        return 0
    if args.cmd == "capture":
        return run_capture(Path(args.config))
    if args.cmd == "browse":
        return run_browse(Path(args.data_dir))
    if args.cmd == "gui":
        LauncherGUI(Path(__file__).resolve()).mainloop()
        return 0
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
