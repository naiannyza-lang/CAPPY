#!/usr/bin/env python3
"""
CAPPY Scope Rewrite (archive-first, flush-driven live)

Run:
  GUI:
    python3 CAPPY_scope_rewrite.py

  Capture (launched by GUI, but can run manually):
    python3 CAPPY_scope_rewrite.py capture --config CAPPY.yaml

Key guarantees:
- Archive-first: RAW + SQLite index are written/committed first.
- Live updates only at flush boundaries (writer emits JSON flush events).
- Trigger timeout => PAUSE capture (clean stop, flush, exit rc=0).
- Backpressure => PAUSE capture (no silent drops).
- Safe AutoDMA sizing: ensures bytes/buffer >= min_dma_buffer_bytes.
"""

from __future__ import annotations

import os, sys, time, json, math, signal, queue, sqlite3, threading, subprocess, datetime as dt
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

try:
    import yaml
except Exception as e:
    print("Missing dependency: pyyaml (pip install pyyaml)", file=sys.stderr)
    raise

# -----------------------------
# ATS API lazy import
# -----------------------------
def import_atsapi():
    try:
        import atsapi
        return atsapi
    except Exception:
        pass

    candidates = [
        "/usr/local/AlazarTech/samples/Samples_Python/Library/atsapi",
        "/usr/local/AlazarTech/ats-sdk/samples/Samples_Python/Library/atsapi",
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            try:
                import atsapi
                return atsapi
            except Exception:
                continue
    raise ImportError("Cannot import atsapi. Install ATS-SDK python samples or add atsapi to PYTHONPATH.")

# -----------------------------
# Helpers
# -----------------------------
def now_ns() -> int:
    return time.time_ns()

def ns_to_iso(ns: int) -> str:
    # full ns precision
    s = ns // 1_000_000_000
    n = ns % 1_000_000_000
    return dt.datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S") + f".{n:09d}"

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

# -----------------------------
# Configuration
# -----------------------------
DEFAULT_YAML = {
    "board": {"system_id": 1, "board_id": 1},
    "channels": {
        "A": {"enabled": True, "input_range": "INPUT_RANGE_PM_1_V"},
        "B": {"enabled": False, "input_range": "INPUT_RANGE_PM_1_V"},
    },
    "acquisition": {
        "sample_rate": "SAMPLE_RATE_250MSPS",
        "samples_per_record": 8192,
        "records_per_buffer": 64,                 # auto-increased if too small
        "min_dma_buffer_bytes": 4 * 1024 * 1024,  # 4MB default
        "max_writer_queue_buffers": 64,
    },
    "trigger": {
        "sourceJ": "TRIG_EXTERNAL",   # TRIG_EXTERNAL or TRIG_CHAN_A
        "slope": "TRIG_SLOPE_POS",    # TRIG_SLOPE_POS or TRIG_SLOPE_NEG
        "level_mV": 100.0,
        "timeout_ms": 0,              # if >0 and no data for timeout => PAUSE
    },
    "storage": {
        "base_dir": "dataFile/captures",
        "flush_every_samples": 2_000_000,   # flush+commit every N samples (0 disables)
        "flush_every_seconds": 2.0,         # flush+commit every T seconds (0 disables)
        "durable_fsync": False,             # fsync is safer but slower
        "preview_max_points": 2000,         # preview decimation for live (flush events)
    },
}

def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        # create default
        path.write_text(yaml.safe_dump(DEFAULT_YAML, sort_keys=False))
    cfg = yaml.safe_load(path.read_text()) or {}
    # deep-merge defaults
    def merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out
    return merge(DEFAULT_YAML, cfg)

# -----------------------------
# SQLite index store + raw writer
# -----------------------------
class ArchiveWriter:
    """
    Writes RAW binary (append) + SQLite index.
    Emits flush events for UI/live updates only at flush boundaries.
    """

    def __init__(self, base_dir: Path, session_id: str, preview_max_points: int):
        self.base_dir = base_dir
        self.session_id = session_id
        self.preview_max_points = preview_max_points

        # directory hierarchy: YYYY/YYYY-MM/YYYY-MM-DD/HH:MM
        t = dt.datetime.now()
        out_dir = base_dir / f"{t.year:04d}" / f"{t.year:04d}-{t.month:02d}" / f"{t.year:04d}-{t.month:02d}-{t.day:02d}" / f"{t.hour:02d}:{t.minute:02d}"
        ensure_dir(out_dir)
        self.out_dir = out_dir

        self.raw_path = out_dir / f"raw_{session_id}.bin"
        self.db_path = out_dir / f"index_{session_id}.sqlite"

        self._raw = open(self.raw_path, "ab", buffering=0)
        self._db = sqlite3.connect(str(self.db_path), isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL;")
        self._db.execute("PRAGMA synchronous=NORMAL;")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS records(
              rec_id INTEGER PRIMARY KEY,
              ts_ns INTEGER NOT NULL,
              chan_mask INTEGER NOT NULL,
              sample_rate_hz REAL NOT NULL,
              samples_per_record INTEGER NOT NULL,
              bytes_per_sample INTEGER NOT NULL,
              file_offset INTEGER NOT NULL,
              nbytes INTEGER NOT NULL
            );
        """)
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_records_ts ON records(ts_ns);")

        self.bytes_written = 0
        self.records_written = 0
        self.samples_written = 0
        self.last_write_ns = 0
        self.last_flush_ns = 0

        self._lock = threading.Lock()

    def close(self):
        try:
            self._raw.close()
        except Exception:
            pass
        try:
            self._db.close()
        except Exception:
            pass

    def append_records(self,
                       ts0_ns: int,
                       chan_mask: int,
                       sample_rate_hz: float,
                       samples_per_record: int,
                       bytes_per_sample: int,
                       interleaved_bytes: bytes,
                       n_records: int) -> Tuple[int, int]:
        """
        Append raw bytes then insert sqlite rows.
        Returns (file_offset, nbytes).
        """
        with self._lock:
            file_offset = self._raw.tell()
            nbytes = len(interleaved_bytes)
            self._raw.write(interleaved_bytes)

            # Insert rows for each record. We store ts0 for the buffer and compute per-record ts.
            # We assume contiguous records in time: ts = ts0 + rec * (samples_per_record / sample_rate) seconds.
            dt_ns = int(samples_per_record * 1e9 / sample_rate_hz)
            # one transaction for the batch
            self._db.execute("BEGIN;")
            rec_base = self.records_written
            for i in range(n_records):
                ts_ns = ts0_ns + i * dt_ns
                rec_id = rec_base + i
                offset_i = file_offset + i * (nbytes // n_records) if n_records > 0 else file_offset
                self._db.execute(
                    "INSERT INTO records(rec_id, ts_ns, chan_mask, sample_rate_hz, samples_per_record, bytes_per_sample, file_offset, nbytes) VALUES(?,?,?,?,?,?,?,?);",
                    (rec_id, ts_ns, chan_mask, sample_rate_hz, samples_per_record, bytes_per_sample, offset_i, (nbytes // n_records) if n_records > 0 else nbytes)
                )
            self._db.execute("COMMIT;")

            self.bytes_written += nbytes
            self.records_written += n_records
            self.samples_written += n_records * samples_per_record
            self.last_write_ns = now_ns()

            return file_offset, nbytes

    def flush_raw_and_index(self, durable_fsync: bool):
        with self._lock:
            self._raw.flush()
            self._db.commit()
            if durable_fsync:
                try:
                    os.fsync(self._raw.fileno())
                except Exception:
                    pass
                # sqlite WAL durability is more complex; fsync the db file best-effort
                try:
                    os.fsync(open(self.db_path, "rb").fileno())
                except Exception:
                    pass
            self.last_flush_ns = now_ns()

    def make_preview(self,
                     sample_rate_hz: float,
                     samples_per_record: int,
                     chan_mask: int,
                     bytes_per_sample: int,
                     last_record_bytes: bytes) -> Dict[str, Any]:
        """
        Create a lightweight decimated preview from *one* record (latest committed).
        Assumes channel-interleaved record with enabled channels in order A,B.
        Output is JSON-serializable.
        """
        # Interpret int16 for now (common). Extend if using other widths.
        import struct
        if bytes_per_sample != 2:
            return {"ok": False, "reason": f"bytes_per_sample={bytes_per_sample} not supported in preview"}
        n_ch = 0
        if chan_mask & 0x1: n_ch += 1  # A
        if chan_mask & 0x2: n_ch += 1  # B
        if n_ch == 0:
            return {"ok": False, "reason": "no channels enabled"}

        total_samples = samples_per_record
        # data are interleaved per-sample across channels: [A0,B0,A1,B1,...] if both enabled
        # Convert to ints
        n_int16 = len(last_record_bytes) // 2
        vals = struct.unpack("<" + "h" * n_int16, last_record_bytes)

        # stride to max points
        stride = max(1, total_samples // max(1, self.preview_max_points))
        # Build per-channel arrays
        a, b = [], []
        for i in range(0, total_samples, stride):
            base = i * n_ch
            if chan_mask & 0x1:
                a.append(vals[base + 0])
            if (chan_mask & 0x2) and n_ch == 2:
                b.append(vals[base + 1])
        # x axis in seconds relative within record
        xs = [(i * stride) / sample_rate_hz for i in range(len(a) if a else len(b))]

        return {
            "ok": True,
            "xs": xs,
            "a": a,
            "b": b,
            "chan_mask": chan_mask,
            "sample_rate_hz": sample_rate_hz,
            "samples_per_record": samples_per_record,
            "bytes_per_sample": bytes_per_sample,
        }

# -----------------------------
# Capture engine
# -----------------------------
@dataclass
class CaptureState:
    stop: bool = False
    paused_reason: Optional[str] = None
    error: Optional[str] = None

def resolve_const(atsapi, name: str) -> int:
    if not hasattr(atsapi, name):
        raise ValueError(f"atsapi missing constant {name}")
    return int(getattr(atsapi, name))

def get_enabled_mask(cfg: Dict[str, Any]) -> int:
    mask = 0
    if cfg["channels"]["A"].get("enabled", True):
        mask |= 0x1
    if cfg["channels"]["B"].get("enabled", False):
        mask |= 0x2
    return mask

def bytes_per_sample_from_board(bits_per_sample: int) -> int:
    # ATS boards commonly return 8/12/14/16 bits, stored in 16-bit words in DMA.
    return 2

def run_capture(cfg_path: Path) -> int:
    cfg = load_config(cfg_path)
    atsapi = import_atsapi()

    board_sys = int(cfg["board"].get("system_id", 1))
    board_id = int(cfg["board"].get("board_id", 1))
    session_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(cfg["storage"].get("base_dir", "dataFile/captures"))
    ensure_dir(base_dir)

    preview_max_points = int(cfg["storage"].get("preview_max_points", 2000))
    writer = ArchiveWriter(base_dir, session_id, preview_max_points)

    # Print run header for GUI
    print(f"[CAPPY] Started session {session_id} in {writer.out_dir}", flush=True)

    # --- configure board ---
    try:
        board = atsapi.Board(board_sys, board_id)
    except Exception as e:
        print(f"[CAPPY] ERROR: cannot open board system={board_sys} board={board_id}: {e}", flush=True)
        writer.close()
        return 2

    chan_mask = get_enabled_mask(cfg)
    if chan_mask == 0:
        print("[CAPPY] ERROR: no channels enabled.", flush=True)
        writer.close()
        return 2

    # sample rate constant
    sr_name = cfg["acquisition"]["sample_rate"]
    try:
        sr_const = resolve_const(atsapi, sr_name)
    except Exception as e:
        print(f"[CAPPY] ERROR: invalid sample_rate {sr_name}: {e}", flush=True)
        writer.close()
        return 2

    samples_per_record = int(cfg["acquisition"].get("samples_per_record", 8192))
    records_per_buffer = int(cfg["acquisition"].get("records_per_buffer", 64))
    min_dma_bytes = int(cfg["acquisition"].get("min_dma_buffer_bytes", 4 * 1024 * 1024))
    max_writer_q = int(cfg["acquisition"].get("max_writer_queue_buffers", 64))

    trig = cfg.get("trigger", {})
    trig_source = trig.get("sourceJ", "TRIG_EXTERNAL")
    trig_slope = trig.get("slope", "TRIG_SLOPE_POS")
    trig_level_mv = float(trig.get("level_mV", 100.0))
    trig_timeout_ms = int(trig.get("timeout_ms", 0))

    flush_every_samples = int(cfg["storage"].get("flush_every_samples", 0))
    flush_every_seconds = float(cfg["storage"].get("flush_every_seconds", 0.0))
    durable_fsync = bool(cfg["storage"].get("durable_fsync", False))

    # configure board channels/ranges
    # This rewrite keeps configuration minimal: input range per channel.
    # Extend as needed for coupling/impedance if your hardware supports it.
    try:
        # channel input control
        def set_ch(ch_name: str, ats_ch_const_name: str):
            ch_cfg = cfg["channels"][ch_name]
            if not ch_cfg.get("enabled", True):
                return
            rng = ch_cfg.get("input_range", "INPUT_RANGE_PM_1_V")
            rng_const = resolve_const(atsapi, rng)
            # coupling/impedance are board-specific; use DC + 1M by default
            board.inputControl(getattr(atsapi, ats_ch_const_name), atsapi.DC_COUPLING, rng_const, atsapi.IMPEDANCE_1M_OHM)

        if chan_mask & 0x1:
            set_ch("A", "CHANNEL_A")
        if chan_mask & 0x2:
            set_ch("B", "CHANNEL_B")

        # set sample rate (board-specific clocking)
        board.setCaptureClock(atsapi.INTERNAL_CLOCK, sr_const, atsapi.CLOCK_EDGE_RISING, 0)

        # trigger settings
        src_const = resolve_const(atsapi, trig_source)
        slope_const = resolve_const(atsapi, trig_slope)

        # Convert mV to trigger code. ATS API expects "triggerLevel" 0..255.
        # We'll map linearly around mid-scale; exact mapping depends on range/board.
        # This is a pragmatic approximation; tune for your setup.
        # 128 is ~0V. 255 is +fullscale. 0 is -fullscale.
        trig_code = int(128 + (trig_level_mv / 1000.0) * 127)
        trig_code = clamp_int(trig_code, 0, 255)

        # J trigger engine configuration (single engine)
        board.setTriggerOperation(
            atsapi.TRIG_ENGINE_OP_J,
            atsapi.TRIG_ENGINE_J, src_const, slope_const, trig_code,
            atsapi.TRIG_ENGINE_K, atsapi.TRIG_DISABLE, atsapi.TRIG_SLOPE_POSITIVE, 128
        )
        board.setExternalTrigger(atsapi.DC_COUPLING, atsapi.ETR_5V)
        board.setTriggerDelay(0)
        board.setTriggerTimeOut(0)  # we enforce timeout in software for "PAUSE", not auto-trigger.

        # record size
        board.setRecordSize(0, samples_per_record)
        # number of records per acquisition: infinite streaming via AutoDMA, but API requires setting
        board.setRecordCount(0x7FFFFFFF)

    except Exception as e:
        print(f"[CAPPY] ERROR: board configuration failed: {e}", flush=True)
        writer.close()
        return 3

    # Determine actual sample rate Hz from sr_name if possible (for timestamps)
    # Map a few common constants. Extend as needed.
    SR_HZ_MAP = {
        "SAMPLE_RATE_250MSPS": 250_000_000.0,
        "SAMPLE_RATE_125MSPS": 125_000_000.0,
        "SAMPLE_RATE_100MSPS": 100_000_000.0,
        "SAMPLE_RATE_50MSPS": 50_000_000.0,
        "SAMPLE_RATE_10MSPS": 10_000_000.0,
        "SAMPLE_RATE_1GSPS": 1_000_000_000.0,
    }
    sample_rate_hz = SR_HZ_MAP.get(sr_name, 250_000_000.0)

    # --- AutoDMA sizing ---
    bits = int(board.getChannelInfo()[1]) if hasattr(board, "getChannelInfo") else 16
    bps = bytes_per_sample_from_board(bits)
    n_ch = (1 if (chan_mask & 0x1) else 0) + (1 if (chan_mask & 0x2) else 0)
    bytes_per_record = samples_per_record * n_ch * bps

    # Ensure buffer >= min_dma_bytes by increasing records_per_buffer.
    if bytes_per_record <= 0:
        print("[CAPPY] ERROR: invalid bytes_per_record.", flush=True)
        writer.close()
        return 3

    if records_per_buffer * bytes_per_record < min_dma_bytes:
        records_per_buffer = int(math.ceil(min_dma_bytes / bytes_per_record))

    # clamp to something sane
    records_per_buffer = clamp_int(records_per_buffer, 1, 4096)
    bytes_per_buffer = records_per_buffer * bytes_per_record

    print(f"[CAPPY] DMA: samples/rec={samples_per_record} recs/buf={records_per_buffer} bytes/buf={bytes_per_buffer} (min={min_dma_bytes})", flush=True)
    print(f"[CAPPY] Trigger: source={trig_source} slope={trig_slope} level_code={trig_code} timeout_ms={trig_timeout_ms}", flush=True)

    # Allocate DMA buffers
    try:
        buffers = [atsapi.DMABuffer(board.handle, atsapi.U16, bytes_per_buffer) for _ in range(4)]
        for b in buffers:
            board.postAsyncBuffer(b.addr, b.size_bytes)
        board.startCapture()
    except Exception as e:
        print(f"[CAPPY] ERROR: DMA setup failed: {e}", flush=True)
        writer.close()
        return 4

    # --- threads and queues ---
    writer_q: "queue.Queue[Tuple[int, bytes, int]]" = queue.Queue(maxsize=max_writer_q)
    # Each item: (ts0_ns, raw_bytes, n_records)

    state = CaptureState()
    state_lock = threading.Lock()

    def pause(reason: str):
        with state_lock:
            state.paused_reason = reason
            state.stop = True

    # Writer thread: raw+index commits, flush policy, emits flush events (JSON line)
    def writer_thread():
        nonlocal writer
        samples_since_flush = 0
        last_flush_t = time.time()
        last_preview = None

        while True:
            try:
                item = writer_q.get(timeout=0.2)
            except queue.Empty:
                with state_lock:
                    if state.stop:
                        break
                # timed flush even when no new data? Only if something was written.
                if flush_every_seconds and writer.records_written > 0:
                    if (time.time() - last_flush_t) >= flush_every_seconds:
                        writer.flush_raw_and_index(durable_fsync)
                        last_flush_t = time.time()
                        evt = {
                            "type": "flush",
                            "reason": "time",
                            "session_id": session_id,
                            "out_dir": str(writer.out_dir),
                            "ts_ns": writer.last_flush_ns,
                            "ts_iso": ns_to_iso(writer.last_flush_ns),
                            "written": {
                                "bytes": writer.bytes_written,
                                "records": writer.records_written,
                                "samples": writer.samples_written,
                            },
                            "preview": last_preview,
                        }
                        print("CAPPY_EVENT " + json.dumps(evt), flush=True)
                continue

            ts0_ns, raw_bytes, n_records = item
            try:
                # Append raw + sqlite index
                writer.append_records(ts0_ns, chan_mask, sample_rate_hz, samples_per_record, bps, raw_bytes, n_records)

                # build preview from the last record of this chunk (cheap)
                rec_bytes = raw_bytes[-bytes_per_record:] if len(raw_bytes) >= bytes_per_record else raw_bytes
                last_preview = writer.make_preview(sample_rate_hz, samples_per_record, chan_mask, bps, rec_bytes)

                samples_since_flush += n_records * samples_per_record

                do_flush = False
                reason = None
                if flush_every_samples and samples_since_flush >= flush_every_samples:
                    do_flush = True
                    reason = "samples"
                if flush_every_seconds and (time.time() - last_flush_t) >= flush_every_seconds:
                    do_flush = True
                    reason = "time"

                if do_flush:
                    writer.flush_raw_and_index(durable_fsync)
                    samples_since_flush = 0
                    last_flush_t = time.time()
                    evt = {
                        "type": "flush",
                        "reason": reason,
                        "session_id": session_id,
                        "out_dir": str(writer.out_dir),
                        "ts_ns": writer.last_flush_ns,
                        "ts_iso": ns_to_iso(writer.last_flush_ns),
                        "written": {
                            "bytes": writer.bytes_written,
                            "records": writer.records_written,
                            "samples": writer.samples_written,
                        },
                        "preview": last_preview,
                    }
                    # Live updates ONLY on flush boundary:
                    print("CAPPY_EVENT " + json.dumps(evt), flush=True)

            except Exception as e:
                with state_lock:
                    state.error = f"writer error: {e}"
                    state.stop = True
                break
            finally:
                writer_q.task_done()

    wt = threading.Thread(target=writer_thread, name="WriterThread", daemon=True)
    wt.start()

    # Acquisition loop: wait DMA, enqueue to writer, enforce timeout/backpressure pause
    last_data_t = time.time()
    session_start_ns = now_ns()
    rec_dt_ns = int(samples_per_record * 1e9 / sample_rate_hz)

    # For absolute timestamps: timestamp = session_start_ns + global_record_index * rec_dt_ns
    global_rec = 0

    try:
        while True:
            with state_lock:
                if state.stop:
                    break

            # backpressure pause
            if writer_q.qsize() >= max_writer_q - 2:
                pause("backpressure")
                break

            # software trigger timeout: if no completed buffers for timeout_ms, pause
            if trig_timeout_ms > 0 and (time.time() - last_data_t) * 1000.0 >= trig_timeout_ms:
                pause("trigger_timeout")
                break

            # Wait for buffer complete
            buf = buffers[global_rec % len(buffers)]
            try:
                board.waitAsyncBufferComplete(buf.addr, timeout_ms=200)  # short timeout to allow checks
            except Exception:
                # no buffer yet; loop to timeout checks
                continue

            last_data_t = time.time()

            # Copy bytes out (very important: do minimal work here)
            raw = bytes(buf.buffer)  # DMABuffer exposes underlying memoryview
            n_records = records_per_buffer

            # Compute ts0 for this buffer batch
            ts0_ns = session_start_ns + global_rec * rec_dt_ns

            # Enqueue for writer (block briefly; if blocked too long, pause)
            try:
                writer_q.put((ts0_ns, raw, n_records), timeout=0.2)
            except queue.Full:
                pause("backpressure")
                break

            # repost buffer ASAP
            board.postAsyncBuffer(buf.addr, buf.size_bytes)
            global_rec += n_records

    except KeyboardInterrupt:
        pause("sigint")
    finally:
        try:
            board.abortAsyncRead()
        except Exception:
            pass
        try:
            board.stopCapture()
        except Exception:
            pass

    # tell writer to stop and finish queued items
    with state_lock:
        reason = state.paused_reason
        err = state.error
        state.stop = True

    # drain: wait for writer queue to finish (bounded)
    try:
        writer_q.join()
    except Exception:
        pass

    # final flush to commit everything
    try:
        writer.flush_raw_and_index(durable_fsync)
        evt = {
            "type": "flush",
            "reason": "final",
            "session_id": session_id,
            "out_dir": str(writer.out_dir),
            "ts_ns": writer.last_flush_ns,
            "ts_iso": ns_to_iso(writer.last_flush_ns),
            "written": {
                "bytes": writer.bytes_written,
                "records": writer.records_written,
                "samples": writer.samples_written,
            },
            "preview": None,
        }
        print("CAPPY_EVENT " + json.dumps(evt), flush=True)
    except Exception:
        pass

    writer.close()

    if err:
        print(f"[CAPPY] ERROR: {err}", flush=True)
        return 1
    if reason:
        print(f"[CAPPY] PAUSED: {reason}", flush=True)
        return 0

    print("[CAPPY] Stopped.", flush=True)
    return 0

# -----------------------------
# GUI (flush-driven live)
# -----------------------------
def run_gui(default_cfg_path: Path) -> int:
    import tkinter as tk
    from tkinter import ttk, messagebox

    # Lazy matplotlib imports
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    cfg = load_config(default_cfg_path)

    root = tk.Tk()
    root.title("CAPPY Scope (Archive-First)")
    root.geometry("1280x780")

    # Theme (simple dark)
    root.configure(bg="#111111")
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TFrame", background="#111111")
    style.configure("TLabel", background="#111111", foreground="#e5e5e5")
    style.configure("TButton", padding=6)
    style.configure("TCombobox", fieldbackground="#222222", background="#222222", foreground="#e5e5e5")
    style.configure("TEntry", fieldbackground="#222222", foreground="#e5e5e5")

    main = ttk.Frame(root)
    main.pack(fill="both", expand=True)

    left = ttk.Frame(main, width=380)
    left.pack(side="left", fill="y")

    right = ttk.Frame(main)
    right.pack(side="right", fill="both", expand=True)

    # Plot
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title("Live (updates on flush)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("ADC counts")
    lineA, = ax.plot([], [], label="A")
    lineB, = ax.plot([], [], label="B")
    ax.legend(loc="upper right")

    canvas = FigureCanvasTkAgg(fig, master=right)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Status
    status_var = tk.StringVar(value="Idle")
    written_var = tk.StringVar(value="Written: 0 bytes, 0 records, 0 samples")
    last_flush_var = tk.StringVar(value="Last flush: -")
    out_dir_var = tk.StringVar(value="Output: -")

    # Controls vars
    trig_src_var = tk.StringVar(value=cfg["trigger"]["sourceJ"])
    trig_timeout_var = tk.IntVar(value=int(cfg["trigger"].get("timeout_ms", 0)))
    level_var = tk.DoubleVar(value=float(cfg["trigger"].get("level_mV", 100.0)))
    slope_var = tk.StringVar(value=cfg["trigger"].get("slope", "TRIG_SLOPE_POS"))

    sr_var = tk.StringVar(value=cfg["acquisition"]["sample_rate"])
    spr_var = tk.IntVar(value=int(cfg["acquisition"]["samples_per_record"]))
    rpb_var = tk.IntVar(value=int(cfg["acquisition"]["records_per_buffer"]))
    minbuf_var = tk.IntVar(value=int(cfg["acquisition"].get("min_dma_buffer_bytes", 4*1024*1024)))

    flush_samples_var = tk.IntVar(value=int(cfg["storage"].get("flush_every_samples", 0)))
    flush_seconds_var = tk.DoubleVar(value=float(cfg["storage"].get("flush_every_seconds", 0.0)))
    fsync_var = tk.BooleanVar(value=bool(cfg["storage"].get("durable_fsync", False)))

    chA_en = tk.BooleanVar(value=bool(cfg["channels"]["A"].get("enabled", True)))
    chB_en = tk.BooleanVar(value=bool(cfg["channels"]["B"].get("enabled", False)))
    chA_rng = tk.StringVar(value=cfg["channels"]["A"].get("input_range", "INPUT_RANGE_PM_1_V"))
    chB_rng = tk.StringVar(value=cfg["channels"]["B"].get("input_range", "INPUT_RANGE_PM_1_V"))

    proc: Optional[subprocess.Popen] = None
    proc_thread: Optional[threading.Thread] = None
    stop_read = threading.Event()

    def build_section(title: str) -> ttk.LabelFrame:
        lf = ttk.LabelFrame(left, text=title)
        lf.pack(fill="x", padx=10, pady=8)
        return lf

    # Acquire/Recorder section
    sec_acq = build_section("Acquire / Recorder")
    ttk.Label(sec_acq, textvariable=status_var).pack(anchor="w", padx=8, pady=(6,2))
    ttk.Label(sec_acq, textvariable=written_var).pack(anchor="w", padx=8, pady=2)
    ttk.Label(sec_acq, textvariable=last_flush_var).pack(anchor="w", padx=8, pady=2)
    ttk.Label(sec_acq, textvariable=out_dir_var).pack(anchor="w", padx=8, pady=(2,6))

    row = ttk.Frame(sec_acq); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Flush every samples").pack(side="left")
    ttk.Entry(row, textvariable=flush_samples_var, width=10).pack(side="right")

    row = ttk.Frame(sec_acq); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Flush every seconds").pack(side="left")
    ttk.Entry(row, textvariable=flush_seconds_var, width=10).pack(side="right")

    row = ttk.Frame(sec_acq); row.pack(fill="x", padx=8, pady=4)
    ttk.Checkbutton(row, text="Durable fsync", variable=fsync_var).pack(side="left")

    # Vertical section
    sec_vert = build_section("Vertical")
    row = ttk.Frame(sec_vert); row.pack(fill="x", padx=8, pady=4)
    ttk.Checkbutton(row, text="Enable Ch A", variable=chA_en).pack(side="left")
    ttk.Combobox(row, textvariable=chA_rng, width=22, values=[
        "INPUT_RANGE_PM_200_MV","INPUT_RANGE_PM_400_MV","INPUT_RANGE_PM_1_V","INPUT_RANGE_PM_2_V","INPUT_RANGE_PM_4_V"
    ]).pack(side="right")

    row = ttk.Frame(sec_vert); row.pack(fill="x", padx=8, pady=4)
    ttk.Checkbutton(row, text="Enable Ch B", variable=chB_en).pack(side="left")
    ttk.Combobox(row, textvariable=chB_rng, width=22, values=[
        "INPUT_RANGE_PM_200_MV","INPUT_RANGE_PM_400_MV","INPUT_RANGE_PM_1_V","INPUT_RANGE_PM_2_V","INPUT_RANGE_PM_4_V"
    ]).pack(side="right")

    # Trigger section
    sec_trig = build_section("Trigger")
    row = ttk.Frame(sec_trig); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Source").pack(side="left")
    ttk.Combobox(row, textvariable=trig_src_var, width=22, values=["TRIG_EXTERNAL","TRIG_CHAN_A"]).pack(side="right")

    row = ttk.Frame(sec_trig); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Slope").pack(side="left")
    ttk.Combobox(row, textvariable=slope_var, width=22, values=["TRIG_SLOPE_POS","TRIG_SLOPE_NEG"]).pack(side="right")

    row = ttk.Frame(sec_trig); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Level (mV)").pack(side="left")
    ttk.Entry(row, textvariable=level_var, width=10).pack(side="right")

    row = ttk.Frame(sec_trig); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Timeout ms (pause)").pack(side="left")
    ttk.Entry(row, textvariable=trig_timeout_var, width=10).pack(side="right")

    # Acquire section (rates/record)
    sec_rate = build_section("Acquire Settings")
    row = ttk.Frame(sec_rate); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Sample rate").pack(side="left")
    ttk.Combobox(row, textvariable=sr_var, width=22, values=[
        "SAMPLE_RATE_250MSPS","SAMPLE_RATE_125MSPS","SAMPLE_RATE_100MSPS","SAMPLE_RATE_50MSPS","SAMPLE_RATE_10MSPS"
    ]).pack(side="right")

    row = ttk.Frame(sec_rate); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Samples/record").pack(side="left")
    ttk.Entry(row, textvariable=spr_var, width=10).pack(side="right")

    row = ttk.Frame(sec_rate); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Records/buffer").pack(side="left")
    ttk.Entry(row, textvariable=rpb_var, width=10).pack(side="right")

    row = ttk.Frame(sec_rate); row.pack(fill="x", padx=8, pady=4)
    ttk.Label(row, text="Min DMA bytes").pack(side="left")
    ttk.Entry(row, textvariable=minbuf_var, width=10).pack(side="right")

        # Buttons
    btn_row = ttk.Frame(left); btn_row.pack(fill="x", padx=10, pady=10)

    def make_run_config() -> Path:
        # Build config dict from UI vars; write to *.run.yaml
        new_cfg = load_config(default_cfg_path)  # base merge
        new_cfg["trigger"]["sourceJ"] = trig_src_var.get().strip()
        new_cfg["trigger"]["slope"] = slope_var.get().strip()
        new_cfg["trigger"]["level_mV"] = float(level_var.get())
        new_cfg["trigger"]["timeout_ms"] = int(trig_timeout_var.get())

        new_cfg["acquisition"]["sample_rate"] = sr_var.get().strip()
        new_cfg["acquisition"]["samples_per_record"] = int(spr_var.get())
        new_cfg["acquisition"]["records_per_buffer"] = int(rpb_var.get())
        new_cfg["acquisition"]["min_dma_buffer_bytes"] = int(minbuf_var.get())

        new_cfg["storage"]["flush_every_samples"] = int(flush_samples_var.get())
        new_cfg["storage"]["flush_every_seconds"] = float(flush_seconds_var.get())
        new_cfg["storage"]["durable_fsync"] = bool(fsync_var.get())

        new_cfg["channels"]["A"]["enabled"] = bool(chA_en.get())
        new_cfg["channels"]["B"]["enabled"] = bool(chB_en.get())
        new_cfg["channels"]["A"]["input_range"] = chA_rng.get().strip()
        new_cfg["channels"]["B"]["input_range"] = chB_rng.get().strip()

        run_path = default_cfg_path.with_suffix(default_cfg_path.suffix + ".run.yaml")
        run_path.write_text(yaml.safe_dump(new_cfg, sort_keys=False))
        return run_path

    def update_plot_from_preview(preview: Dict[str, Any]):
        if not preview or not preview.get("ok", False):
            return
        xs = preview.get("xs", [])
        a = preview.get("a", [])
        b = preview.get("b", [])
        lineA.set_data(xs, a if a else [])
        lineB.set_data(xs, b if b else [])
        ax.relim()
        ax.autoscale_view()
        canvas.draw_idle()

    def reader_loop(p: subprocess.Popen):
        nonlocal proc
        while not stop_read.is_set():
            line = p.stdout.readline() if p.stdout else ""
            if not line:
                break
            line = line.strip()
            # Flush event line:
            if line.startswith("CAPPY_EVENT "):
                try:
                    evt = json.loads(line[len("CAPPY_EVENT "):])
                    if evt.get("type") == "flush":
                        w = evt.get("written", {})
                        written_var.set(f"Written: {w.get('bytes',0)} bytes, {w.get('records',0)} records, {w.get('samples',0)} samples")
                        last_flush_var.set(f"Last flush: {evt.get('ts_iso','-')}")
                        out_dir_var.set(f"Output: {evt.get('out_dir','-')}")
                        # Live updates only on flush:
                        update_plot_from_preview(evt.get("preview"))
                        status_var.set("Capturing (committed)")
                except Exception:
                    pass
            else:
                # regular logs
                if line.startswith("[CAPPY] PAUSED:"):
                    status_var.set(line)
                elif line.startswith("[CAPPY] ERROR:"):
                    status_var.set(line)
                elif line.startswith("[CAPPY] Started session"):
                    status_var.set("Armed / Capturing")
                # you can optionally display logs somewhere; kept simple here.

        rc = p.poll()
        if rc is None:
            rc = p.wait()
        status_var.set(f"Stopped (rc={rc})")
        proc = None

    def start_capture():
        nonlocal proc, proc_thread
        if proc is not None:
            messagebox.showinfo("CAPPY", "Capture already running.")
            return
        try:
            run_cfg = make_run_config()
        except Exception as e:
            messagebox.showerror("CAPPY", f"Config error: {e}")
            return

        # launch capture subprocess
        cmd = [sys.executable, str(Path(__file__).resolve()), "capture", "--config", str(run_cfg)]
        stop_read.clear()
        status_var.set("Starting…")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception as e:
            proc = None
            messagebox.showerror("CAPPY", f"Failed to start capture: {e}")
            return

        proc_thread = threading.Thread(target=reader_loop, args=(proc,), daemon=True)
        proc_thread.start()

    def stop_capture():
        nonlocal proc
        if proc is None:
            return
        status_var.set("Stopping…")
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass

    ttk.Button(btn_row, text="Start", command=start_capture).pack(side="left", expand=True, fill="x", padx=(0,6))
    ttk.Button(btn_row, text="Stop", command=stop_capture).pack(side="right", expand=True, fill="x", padx=(6,0))

    def on_close():
        stop_read.set()
        try:
            if proc is not None:
                proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    return 0

# -----------------------------
# CLI entry
# -----------------------------
def main(argv: List[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("gui")

    cap = sub.add_parser("capture")
    cap.add_argument("--config", required=True)

    ap.add_argument("--config", default="CAPPY.yaml", help="GUI default config path")

    args = ap.parse_args(argv)

    cfg_path = Path(args.config)

    if args.cmd == "capture":
        return run_capture(Path(args.config))
    # default to GUI
    return run_gui(cfg_path)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
