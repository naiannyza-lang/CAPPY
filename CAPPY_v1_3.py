#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import sqlite3
import sys
import time
import signal
import threading
import json
import os
import smtplib
import subprocess
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
import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# GUI + plotting (optional)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

clock:
  source: INTERNAL_CLOCK
  sample_rate_msps: 250.0
  edge: CLOCK_EDGE_RISING

channels:
  A:
    coupling: DC
    range: PM_1_V          # maps to INPUT_RANGE_PM_1_V (old-script style)
    impedance: 50_OHM
  # B:
  #   coupling: DC
  #   range: PM_1_V
  #   impedance: 50_OHM

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

  external_startcapture: false

timing:
  bunch_spacing_samples: 424      # adjust to your bunch spacing

acquisition:
  channels_mask: CHANNEL_A        # CHANNEL_A or CHANNEL_A|CHANNEL_B
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
  every_n: 20000
  threshold_integral_Vs: 0.0
  threshold_peak_V: 0.0
  max_waveforms_per_sec: 50
  store_volts: true

storage:
  data_dir: dataFile
  session_tag: ""
  rollover_minutes: 60
  session_rotate_hours: 24
  flush_every_records: 200000
  sqlite_commit_every_snips: 2000

notify:
  enabled: false
  to: "user@example.com"
  from: "cappy@localhost"
  method: "sendmail"
  sendmail_path: "/usr/sbin/sendmail"
  subject_prefix: "[CAPPY]"
  heartbeat_seconds: 60
  interval_minutes: 120

runtime:
  noise_test: false
  autotrigger_timeout_ms: 10
"""

REDUCED_SCHEMA = pa.schema([
    ("session_id", pa.string()),
    ("buffer_index", pa.int32()),
    ("record_in_buffer", pa.int32()),
    ("record_global", pa.int64()),
    ("timestamp_ns", pa.int64()),
    ("sample_rate_hz", pa.float64()),
    ("samples_per_record", pa.int32()),
    ("records_per_buffer", pa.int32()),
    ("channels_mask", pa.string()),
    ("bunch_spacing_samples", pa.int32()),
    ("area_A_Vs", pa.float64()),
    ("peak_A_V", pa.float64()),
    ("baseline_A_V", pa.float64()),
    ("area_B_Vs", pa.float64()),
    ("peak_B_V", pa.float64()),
    ("baseline_B_V", pa.float64()),
])

SESSION_INDEX_SCHEMA = pa.schema([
    ("session_id", pa.string()),
    ("date", pa.string()),
    ("first_timestamp_ns", pa.int64()),
    ("last_timestamp_ns", pa.int64()),
    ("reduced_rows", pa.int64()),
    ("waveform_snips", pa.int64()),
    ("channels_mask", pa.string()),
])

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)

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

class StatusNotifier:
    """Heartbeat file + periodic status email, non-blocking."""
    def __init__(self, cfg: dict, data_dir: Path):
        self.cfg = cfg
        self.data_dir = data_dir
        self.notify = (cfg.get("notify", {}) or {})
        self.hb_seconds = int(self.notify.get("heartbeat_seconds", 60))
        self.email_seconds = int(self.notify.get("interval_minutes", 120)) * 60
        self._last_hb = 0.0
        self._last_email = 0.0
        self._lock = threading.Lock()
        self._latest: dict = {}

    def update(self, **kw) -> None:
        with self._lock:
            self._latest.update(kw)

    def maybe_emit(self) -> None:
        now = time.time()
        if now - self._last_hb >= self.hb_seconds:
            self._last_hb = now
            with self._lock:
                snap = dict(self._latest)
            try:
                _write_json_atomic(self.data_dir / "status" / "cappy_status.json", snap)
            except Exception as e:
                pass

        if bool(self.notify.get("enabled", False)) and (now - self._last_email >= self.email_seconds):
            self._last_email = now
            with self._lock:
                snap = dict(self._latest)
            prefix = str(self.notify.get("subject_prefix", "[CAPPY]")).strip() or "[CAPPY]"
            subject = f"{prefix} status {snap.get('session_id','')}"
            body = "\n".join(f"{k}: {v}" for k, v in sorted(snap.items()))
            threading.Thread(target=_send_status_email, args=(self.cfg, subject, body), daemon=True).start()

def channels_from_mask_expr(expr: str) -> int:
    if not ATS_AVAILABLE or ats is None:
        raise RuntimeError("atsapi not available.")
    v = 0
    for part in expr.split("|"):
        part = part.strip()
        if part:
            v |= ats_const(part)
    return v

def infer_channel_count(mask: int) -> int:
    if not ATS_AVAILABLE or ats is None:
        raise RuntimeError("atsapi not available.")
    # ats.channels exists in atsapi
    return sum(1 for c in ats.channels if (c & mask == c))  # type: ignore[attr-defined]

class ParquetRollingWriter:
    def __init__(self, out_dir: Path, prefix: str, schema: pa.Schema, rollover_minutes: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.schema = schema
        self.rollover_minutes = max(1, int(rollover_minutes))
        _ensure_dir(out_dir)
        self._writer: Optional[pq.ParquetWriter] = None
        self._open_key: Optional[str] = None

    def _minute_key(self, ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9).strftime("%Y%m%d_%H%M")

    def _open_new(self, key: str) -> None:
        path = self.out_dir / f"{self.prefix}_{key}.parquet"
        self._writer = pq.ParquetWriter(path, self.schema, compression="snappy", use_dictionary=True)
        self._open_key = key

    def _maybe_roll(self, ts_ns: int) -> None:
        key = self._minute_key(ts_ns)
        if self._open_key is None:
            self._open_new(key)
            return
        t0 = datetime.strptime(self._open_key, "%Y%m%d_%H%M")
        t1 = datetime.strptime(key, "%Y%m%d_%H%M")
        if (t1 - t0).total_seconds() >= 60 * self.rollover_minutes:
            self.close()
            self._open_new(key)

    def write_rows(self, rows: List[Dict[str, Any]], ts_ns: int) -> int:
        if not rows:
            return 0
        self._maybe_roll(ts_ns)
        assert self._writer is not None
        df = pd.DataFrame(rows)
        for name in self.schema.names:
            if name not in df.columns:
                df[name] = np.nan
        df = df[self.schema.names]
        tbl = pa.Table.from_pandas(df, schema=self.schema, preserve_index=False)
        self._writer.write_table(tbl)
        return int(tbl.num_rows)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

class WaveBinSqliteStore:
    def __init__(self, out_dir: Path, session_id: str, rollover_minutes: int, commit_every: int):
        self.out_dir = out_dir
        self.session_id = session_id
        self.rollover_minutes = max(1, int(rollover_minutes))
        self.commit_every = max(1, int(commit_every))
        _ensure_dir(out_dir)
        self.db_path = out_dir / f"snips_{session_id}.sqlite"
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
              n_channels INTEGER,
              file TEXT,
              offset_bytes INTEGER,
              nbytes INTEGER,
              area_A_Vs REAL,
              peak_A_V REAL,
              area_B_Vs REAL,
              peak_B_V REAL
            );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_snips_session_time ON snips(session_id, timestamp_ns);")
        self.conn.commit()
        self._bin_fh = None
        self._bin_key = None
        self._rows_since_commit = 0

    def _minute_key(self, ts_ns: int) -> str:
        return datetime.fromtimestamp(ts_ns / 1e9).strftime("%Y%m%d_%H%M")

    def _open_bin(self, key: str):
        path = self.out_dir / f"snips_{self.session_id}_{key}.bin"
        self._bin_fh = open(path, "ab", buffering=0)
        self._bin_key = key

    def _maybe_roll(self, ts_ns: int):
        key = self._minute_key(ts_ns)
        if self._bin_key is None:
            self._open_bin(key)
            return
        t0 = datetime.strptime(self._bin_key, "%Y%m%d_%H%M")
        t1 = datetime.strptime(key, "%Y%m%d_%H%M")
        if (t1 - t0).total_seconds() >= 60 * self.rollover_minutes:
            self.close_bin()
            self._open_bin(key)

    def append(self, *, ts_ns: int, buffer_index: int, record_in_buffer: int, record_global: int,
               channels_mask: str, sample_rate_hz: float, wfA_V: np.ndarray, wfB_V: Optional[np.ndarray],
               area_A_Vs: float, peak_A_V: float, area_B_Vs: float, peak_B_V: float):
        self._maybe_roll(ts_ns)
        assert self._bin_fh is not None and self._bin_key is not None
        wfA = wfA_V.astype(np.float32, copy=False)
        if wfB_V is None:
            payload = wfA.tobytes(order="C")
            n_channels = 1
        else:
            wfB = wfB_V.astype(np.float32, copy=False)
            payload = wfA.tobytes(order="C") + wfB.tobytes(order="C")
            n_channels = 2
        offset = self._bin_fh.tell()
        self._bin_fh.write(payload)
        file_name = f"snips_{self.session_id}_{self._bin_key}.bin"
        self.conn.execute(
            "INSERT INTO snips(session_id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.session_id, int(ts_ns), int(buffer_index), int(record_in_buffer), int(record_global),
             str(channels_mask), float(sample_rate_hz), int(wfA.shape[0]), int(n_channels),
             file_name, int(offset), int(len(payload)),
             float(area_A_Vs), float(peak_A_V), float(area_B_Vs), float(peak_B_V))
        )
        self._rows_since_commit += 1
        if self._rows_since_commit >= self.commit_every:
            self.conn.commit()
            self._rows_since_commit = 0

    def close_bin(self):
        if self._bin_fh is not None:
            try:
                self._bin_fh.close()
            finally:
                self._bin_fh = None
                self._bin_key = None

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

    def load_waveforms(self, row: pd.Series, base_dir: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        bin_path = base_dir / str(row["file"])
        offset = int(row["offset_bytes"])
        nbytes = int(row["nbytes"])
        n_samples = int(row["n_samples"])
        n_channels = int(row["n_channels"])
        with open(bin_path, "rb") as fh:
            fh.seek(offset)
            payload = fh.read(nbytes)
        arr = np.frombuffer(payload, dtype=np.float32)
        if n_channels == 2:
            return arr[:n_samples], arr[n_samples:2*n_samples]
        return arr[:n_samples], None

class CappyArchive:
    def __init__(self, data_dir: Path, rollover_minutes: int, flush_every_records: int,
                 session_rotate_hours: float, sqlite_commit_every_snips: int):
        self.data_dir = data_dir
        self.captures = data_dir / "captures"
        _ensure_dir(self.captures)
        self.rollover_minutes = int(rollover_minutes)
        self.flush_every_records = int(flush_every_records)
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
        day = date.today().strftime("%Y-%m-%d")
        self.day_dir = self.captures / day
        red_dir = self.day_dir / "reduced"
        wf_dir = self.day_dir / "waveforms"
        _ensure_dir(red_dir)
        _ensure_dir(wf_dir)
        self.reduced_writer = ParquetRollingWriter(red_dir, f"reduced_{sid}", REDUCED_SCHEMA, self.rollover_minutes)
        self.wave_store = WaveBinSqliteStore(wf_dir, sid, self.rollover_minutes, self.sqlite_commit_every_snips)
        _atomic_write_text(self.day_dir / f"session_{sid}.txt", f"CAPPY v1.3 session {sid}\nchannels={channels_mask}\n")
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
            df = pd.read_parquet(idx)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df = df.drop_duplicates(subset=["session_id"], keep="last")
        pq.write_table(pa.Table.from_pandas(df, schema=SESSION_INDEX_SCHEMA, preserve_index=False), idx, compression="snappy")
        print(f"[CAPPY] Finalized session {self.session_id} reduced={self._n_reduced} snips={self._n_snips}")

def reduce_u16(raw: np.ndarray, sr_hz: float, b0: int, b1: int, g0: int, g1: int, vpp: float):
    baseline = raw[:, b0:b1].mean(axis=1, dtype=np.float32)
    gate_sum = raw[:, g0:g1].sum(axis=1, dtype=np.uint64).astype(np.float64)
    sum_counts = gate_sum - baseline.astype(np.float64) * float(g1 - g0)
    scale = vpp / 65535.0
    area = (sum_counts * scale) * (1.0 / sr_hz)
    peak = (raw[:, g0:g1].max(axis=1).astype(np.float32) - baseline).astype(np.float64) * scale
    baseline_V = (baseline.astype(np.float64) * scale) - (vpp * 0.5)
    return area, peak, baseline_V

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

def channels_mask_to_str(mask: int) -> str:
    parts = []
    if mask & ats.CHANNEL_A:
        parts.append("CHANNEL_A")
    if mask & ats.CHANNEL_B:
        parts.append("CHANNEL_B")
    return "|".join(parts) if parts else "0"

def configure_board(board: Any, cfg: Dict[str, Any]) -> Tuple[float, float]:
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

    vpp = 4.0

    ch_cfg = cfg.get("channels", {}) or {}
    for nm, mask in [("A", ats.CHANNEL_A), ("B", ats.CHANNEL_B)]:
        if nm in ch_cfg:
            cc = ch_cfg[nm] or {}
            coupling_name = str(cc.get('coupling', 'DC'))
            if not coupling_name.endswith('_COUPLING'):
                coupling_name = coupling_name + '_COUPLING'
            coupling = ats_const(coupling_name)
            rng = ats_const("INPUT_RANGE_", str(cc.get("range", "PM_1_V")))
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

    return sr_hz, vpp

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

    cfg = load_config(cfg_path)

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

    board = ats.Board(systemId=2, boardId=1)
    sr_hz, vpp = configure_board(board, cfg)

    ch_mask = channels_from_mask_expr(ch_expr)
    ch_count = infer_channel_count(ch_mask)

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
    )
    sid = archive.start(tag=str(storage.get("session_tag", "")).strip(), channels_mask=ch_expr)
    notifier = StatusNotifier(cfg, Path(str(storage.get('data_dir', 'dataFile'))))
    notifier.update(session_id=sid, state='running', started=time.strftime('%Y-%m-%d %H:%M:%S'), data_dir=str(storage.get('data_dir','dataFile')), channels_mask=ch_expr, sample_rate_hz=sr_hz, samples_per_record=spr, records_per_buffer=rpb)
    notifier.maybe_emit()

    adma_flags = ats.ADMA_TRADITIONAL_MODE
    if bool(trig.get("external_startcapture", False)):
        adma_flags |= ats.ADMA_EXTERNAL_STARTCAPTURE

    board.beforeAsyncRead(ch_mask, -pre, spr, rpb, recordsPerAcq, adma_flags)

    for b in buffers:
        board.postAsyncBuffer(b.addr, b.size_bytes)

    print(f"[CAPPY] Running session {sid}. " + ("Press <enter> to stop." if (sys.stdin is not None and hasattr(sys.stdin,"isatty") and sys.stdin.isatty()) else "Use Stop (GUI) or Ctrl+C to stop."))
    buf_done = 0
    global_rec = 0
    t0 = time.time()
    last = t0
    timeout_count = 0
    last_buffer_ns = time.time_ns()
    rt = cfg.get('runtime', {}) or {}
    rearm_if_no_trigger_s = int(rt.get('rearm_if_no_trigger_s', 300))
    rearm_cooldown_s = int(rt.get('rearm_cooldown_s', 30))
    max_rearms_per_hour = int(rt.get('max_rearms_per_hour', 12))
    rearm_times: List[float] = []

    try:
        def _do_rearm():
            nonlocal timeout_count, last_buffer_ns
            now = time.time()
            # keep only last hour
            while rearm_times and (now - rearm_times[0]) > 3600:
                rearm_times.pop(0)
            if rearm_times and (now - rearm_times[-1]) < rearm_cooldown_s:
                return
            if len(rearm_times) >= max_rearms_per_hour:
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
            notifier.update(state='running')
            notifier.maybe_emit()

        board.startCapture()
        while buf_done < buf_target and not _should_stop():
            buf = buffers[buf_done % len(buffers)]
            try:
                board.waitAsyncBufferComplete(buf.addr, wait_timeout_ms)
            except Exception as ex:
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
                    if rearm_if_no_trigger_s > 0 and ago_s >= float(rearm_if_no_trigger_s):
                        _do_rearm()
                    continue
                raise

            timeout_count = 0
            last_buffer_ns = time.time_ns()
            ts_ns = last_buffer_ns

            if archive.should_rotate():
                archive.finalize(ch_expr)
                sid = archive.start(tag=str(storage.get("session_tag", "")).strip(), channels_mask=ch_expr)

            raw = buf.buffer
            if raw.dtype != np.uint16:
                raw = raw.astype(np.uint16, copy=False)

            red_rows: List[Dict[str, Any]] = []

            if ch_count == 2:
                A = raw[0::2].reshape(rpb, spr)
                B = raw[1::2].reshape(rpb, spr)
                areaA, peakA, baseA = reduce_u16(A, sr_hz, b0, b1, g0, g1, vpp)
                areaB, peakB, baseB = reduce_u16(B, sr_hz, b0, b1, g0, g1, vpp)
                for r in range(rpb):
                    rec_g = global_rec + r
                    red_rows.append(dict(
                        session_id=sid, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                        timestamp_ns=int(ts_ns), sample_rate_hz=float(sr_hz),
                        samples_per_record=spr, records_per_buffer=rpb,
                        channels_mask=ch_expr, bunch_spacing_samples=bunch_spacing,
                        area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]), baseline_A_V=float(baseA[r]),
                        area_B_Vs=float(areaB[r]), peak_B_V=float(peakB[r]), baseline_B_V=float(baseB[r]),
                    ))
                    if wf_enable and wf.want(rec_g, float(areaA[r]), float(peakA[r])):
                        wfA_V = vpp * (A[r].astype(np.float32) / 65535.0 - 0.5) if store_volts else A[r].astype(np.float32)
                        wfB_V = vpp * (B[r].astype(np.float32) / 65535.0 - 0.5) if store_volts else B[r].astype(np.float32)
                        archive.append_snip(
                            ts_ns=ts_ns, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                            channels_mask=ch_expr, sample_rate_hz=float(sr_hz),
                            wfA_V=wfA_V, wfB_V=wfB_V,
                            area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]),
                            area_B_Vs=float(areaB[r]), peak_B_V=float(peakB[r]),
                        )
            else:
                A = raw.reshape(rpb, spr)
                areaA, peakA, baseA = reduce_u16(A, sr_hz, b0, b1, g0, g1, vpp)
                for r in range(rpb):
                    rec_g = global_rec + r
                    red_rows.append(dict(
                        session_id=sid, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                        timestamp_ns=int(ts_ns), sample_rate_hz=float(sr_hz),
                        samples_per_record=spr, records_per_buffer=rpb,
                        channels_mask=ch_expr, bunch_spacing_samples=bunch_spacing,
                        area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]), baseline_A_V=float(baseA[r]),
                        area_B_Vs=0.0, peak_B_V=0.0, baseline_B_V=0.0,
                    ))
                    if wf_enable and wf.want(rec_g, float(areaA[r]), float(peakA[r])):
                        wfA_V = vpp * (A[r].astype(np.float32) / 65535.0 - 0.5) if store_volts else A[r].astype(np.float32)
                        archive.append_snip(
                            ts_ns=ts_ns, buffer_index=buf_done, record_in_buffer=r, record_global=rec_g,
                            channels_mask=ch_expr, sample_rate_hz=float(sr_hz),
                            wfA_V=wfA_V, wfB_V=None,
                            area_A_Vs=float(areaA[r]), peak_A_V=float(peakA[r]),
                            area_B_Vs=0.0, peak_B_V=0.0,
                        )

            archive.append_reduced(red_rows, ts_ns)
            board.postAsyncBuffer(buf.addr, buf.size_bytes)

            buf_done += 1
            global_rec += rpb

            now = time.time()
            if now - last >= 1.0:
                rate = global_rec / max(now - t0, 1e-9)
                notifier.update(state="running", time=time.strftime("%Y-%m-%d %H:%M:%S"),
                               buffers=buf_done, records=global_rec, rate_hz=rate,
                               reduced_rows=getattr(archive, "_n_reduced", 0),
                               snips=getattr(archive, "_n_snips", 0),
                               last_buffer_ago_s=(time.time_ns()-last_buffer_ns)/1e9)
                notifier.maybe_emit()
                print(f"[CAPPY] buffers={buf_done} records={global_rec} rate={rate/1e3:.1f} kHz snips={archive._n_snips}")
                last = now

    finally:
        try:
            board.abortAsyncRead()
        except Exception:
            pass
        notifier.update(state='stopped', time=time.strftime('%Y-%m-%d %H:%M:%S'))
        notifier.maybe_emit()
        archive.finalize(ch_expr)

    return 0

class ArchiveBrowser(tk.Tk):
    def __init__(self, data_dir: Path):
        super().__init__()
        self.title("CAPPY v1.3 Archive Browser")
        self.geometry("1250x740")
        self.data_dir = data_dir
        self.captures = data_dir / "captures"
        self.sessions = pd.DataFrame()
        self.snips = pd.DataFrame()
        self._snip_db_dir: Optional[Path] = None
        self._build()
        self._refresh()

    def _build(self):
        top = ttk.Frame(self, padding=8)
        top.pack(fill=tk.BOTH, expand=True)
        filt = ttk.Frame(top)
        filt.pack(fill=tk.X)

        self.var_dir = tk.StringVar(value=str(self.data_dir))
        self.var_start = tk.StringVar(value="")
        self.var_end = tk.StringVar(value="")

        ttk.Label(filt, text="Data dir:").pack(side=tk.LEFT)
        ttk.Entry(filt, textvariable=self.var_dir, width=42).pack(side=tk.LEFT, padx=(4,8))
        ttk.Button(filt, text="Browse…", command=self._pick_dir).pack(side=tk.LEFT)

        ttk.Label(filt, text="Start (YYYY-MM-DD):").pack(side=tk.LEFT, padx=(16,4))
        ttk.Entry(filt, textvariable=self.var_start, width=12).pack(side=tk.LEFT)
        ttk.Label(filt, text="End:").pack(side=tk.LEFT, padx=(8,4))
        ttk.Entry(filt, textvariable=self.var_end, width=12).pack(side=tk.LEFT)
        ttk.Button(filt, text="Refresh", command=self._refresh).pack(side=tk.LEFT, padx=(12,0))

        pan = ttk.PanedWindow(top, orient=tk.HORIZONTAL)
        pan.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        left = ttk.Frame(pan, padding=6)
        right = ttk.Frame(pan, padding=6)
        pan.add(left, weight=1)
        pan.add(right, weight=2)

        ttk.Label(left, text="Sessions").pack(anchor="w")
        self.slist = tk.Listbox(left)
        self.slist.pack(fill=tk.BOTH, expand=True)
        self.slist.bind("<<ListboxSelect>>", self._on_session)

        ttk.Label(left, text="Waveform snippets (timestamped)").pack(anchor="w", pady=(8,0))
        self.wlist = tk.Listbox(left)
        self.wlist.pack(fill=tk.BOTH, expand=True)
        self.wlist.bind("<<ListboxSelect>>", self._on_snip)

        self.fig, self.ax = plt.subplots(figsize=(7.5,4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.meta = tk.Text(right, height=11)
        self.meta.pack(fill=tk.X, pady=(8,0))
        self.meta.configure(state=tk.DISABLED)

    def _pick_dir(self):
        p = filedialog.askdirectory(title="Select data directory")
        if p:
            self.data_dir = Path(p)
            self.captures = self.data_dir / "captures"
            self.var_dir.set(p)
            self._refresh()

    def _list_sessions(self, sd: Optional[date], ed: Optional[date]) -> pd.DataFrame:
        rows = []
        if not self.captures.exists():
            return pd.DataFrame(columns=SESSION_INDEX_SCHEMA.names)
        for ddir in sorted(self.captures.iterdir()):
            if not ddir.is_dir():
                continue
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
                    rows += pd.read_parquet(idx).to_dict("records")
                except Exception:
                    pass
        if not rows:
            return pd.DataFrame(columns=SESSION_INDEX_SCHEMA.names)
        return pd.DataFrame(rows).sort_values("first_timestamp_ns", ascending=False)

    def _refresh(self):
        try:
            sd = _parse_date(self.var_start.get().strip()) if self.var_start.get().strip() else None
            ed = _parse_date(self.var_end.get().strip()) if self.var_end.get().strip() else None
            self.sessions = self._list_sessions(sd, ed)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            self.sessions = pd.DataFrame()

        self.slist.delete(0, tk.END)
        self.wlist.delete(0, tk.END)

        if self.sessions.empty:
            self.slist.insert(tk.END, "(no sessions)")
            return

        for _, r in self.sessions.iterrows():
            t0 = datetime.fromtimestamp(int(r["first_timestamp_ns"]) / 1e9)
            self.slist.insert(tk.END, f"{t0.strftime('%Y-%m-%d %H:%M:%S')}  {r['session_id']}  snips={int(r['waveform_snips'])}")

    def _sel_sid(self) -> Optional[str]:
        if self.sessions.empty:
            return None
        sel = self.slist.curselection()
        if not sel:
            return None
        return str(self.sessions.iloc[sel[0]]["session_id"])

    def _on_session(self, _=None):
        sid = self._sel_sid()
        if not sid:
            return
        self.snips = pd.DataFrame()
        self._snip_db_dir = None

        for ddir in self.captures.iterdir():
            if not ddir.is_dir():
                continue
            wf_dir = ddir / "waveforms"
            db = wf_dir / f"snips_{sid}.sqlite"
            if db.exists():
                conn = sqlite3.connect(db)
                self.snips = pd.read_sql_query(
                    "SELECT id,timestamp_ns,buffer_index,record_in_buffer,record_global,channels_mask,sample_rate_hz,n_samples,n_channels,file,offset_bytes,nbytes,area_A_Vs,peak_A_V,area_B_Vs,peak_B_V "
                    "FROM snips WHERE session_id=? ORDER BY timestamp_ns DESC LIMIT 50000",
                    conn,
                    params=(sid,),
                )
                conn.close()
                self._snip_db_dir = wf_dir
                break

        self.wlist.delete(0, tk.END)
        if self.snips.empty:
            self.wlist.insert(tk.END, "(no saved waveforms)")
            return

        for _, r in self.snips.iterrows():
            ts = datetime.fromtimestamp(int(r["timestamp_ns"]) / 1e9)
            self.wlist.insert(tk.END, f"{int(r['id'])}  {ts.strftime('%Y-%m-%d %H:%M:%S')}  buf={int(r['buffer_index'])} rec={int(r['record_in_buffer'])} g={int(r['record_global'])}")
        self._set_meta(f"Snips loaded: {len(self.snips):,}")

    def _on_snip(self, _=None):
        if self.snips.empty or self._snip_db_dir is None:
            return
        sel = self.wlist.curselection()
        if not sel:
            return
        snip_id = int(self.wlist.get(sel[0]).split()[0])
        row = self.snips[self.snips["id"] == snip_id]
        if row.empty:
            return
        r = row.iloc[0]
        store = WaveBinSqliteStore(self._snip_db_dir, "tmp", rollover_minutes=60, commit_every=1000)
        try:
            wa, wb = store.load_waveforms(r, self._snip_db_dir)
        finally:
            store.close()

        sr = float(r["sample_rate_hz"])
        tvec = np.arange(len(wa)) / sr

        self.ax.clear()
        self.ax.plot(tvec, wa, label="A")
        if wb is not None:
            self.ax.plot(tvec, wb, label="B")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Volts")
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw()

        ts = datetime.fromtimestamp(int(r["timestamp_ns"]) / 1e9)
        self._set_meta(
            f"Timestamp: {ts.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"buf={int(r['buffer_index'])} rec={int(r['record_in_buffer'])} global={int(r['record_global'])}\n"
            f"area_A_Vs={float(r['area_A_Vs'])} peak_A_V={float(r['peak_A_V'])}\n"
            f"samples={int(r['n_samples'])} channels={int(r['n_channels'])}"
        )

    def _set_meta(self, s: str):
        self.meta.configure(state=tk.NORMAL)
        self.meta.delete("1.0", tk.END)
        self.meta.insert(tk.END, s)
        self.meta.configure(state=tk.DISABLED)

class LauncherGUI(tk.Tk):
    """Simple launcher: Start Capture / Browse Archive / Open YAML."""
    def __init__(self, script_path: Path):
        super().__init__()
        self.title("CAPPY v1.3 Launcher")
        self.geometry("1000x650")
        self.script_path = script_path
        self.proc = None

        self.var_config = tk.StringVar(value="CAPPY_v1_3.yaml")
        self.var_data_dir = tk.StringVar(value="dataFile")

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

        logbox = ttk.LabelFrame(self, text="Log", padding=6)
        logbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log = tk.Text(logbox, wrap="word", state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

        self.after(200, self._poll)

    def _append(self, s: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, s + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _pick_yaml(self):
        p = filedialog.askopenfilename(title="Select YAML", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")])
        if p:
            self.var_config.set(p)

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
        ArchiveBrowser(Path(self.var_data_dir.get())).mainloop()

    def _toggle(self):
        if self.proc is None:
            self._start()
        else:
            self._stop()

    def _start(self):
        import subprocess
        cmd = [sys.executable, str(self.script_path), "capture", "--config", self.var_config.get()]
        self._append("RUN: " + " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            cwd=str(self.script_path.parent)
        )
        self.lbl.config(text="State: capturing")
        self.btn.config(text="Stop Capture")

    def _stop(self):
        if self.proc is None:
            return
        try:
            self.proc.terminate()
            self._append("[GUI] sent terminate")
        except Exception as e:
            self._append("[GUI] terminate failed: " + str(e))

    def _poll(self):
        if self.proc is not None and self.proc.stdout is not None:
            try:
                # non-blocking-ish: read what is available
                line = self.proc.stdout.readline()
                if line:
                    self._append(line.rstrip())
            except Exception:
                pass
            rc = self.proc.poll()
            if rc is not None:
                # drain
                try:
                    rest = self.proc.stdout.read()
                    if rest:
                        for ln in rest.splitlines():
                            self._append(ln)
                except Exception:
                    pass
                self._append(f"[GUI] DAQ exited rc={rc}")
                self.proc = None
                self.lbl.config(text="State: idle")
                self.btn.config(text="Start Capture")
        self.after(200, self._poll)

def run_browse(data_dir: Path) -> int:
    app = ArchiveBrowser(data_dir)
    app.mainloop()
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
