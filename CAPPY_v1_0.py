#!/usr/bin/env python3
"""
cappy_archive_db.py  -  CAPPY.ARCH  v2
=======================================
Hierarchical lazy-load archive browser. Handles tens of millions of snips
with near-zero CPU usage because the DB does ALL the work.

Navigation model (left panel, top to bottom):
  Session  ->  Day  ->  Hour  ->  Minute  ->  Snips for that minute only

Each level is a single cheap SQLite GROUP BY on integer-divided timestamp_ns.
No row iteration in Python until the user drills to a specific minute.
A minute window typically holds a few hundred to a few thousand snips --
that entire set lives in a single compact numpy structured array (~few KB).

Memory budget:
  - Sessions / days / hours / minutes:  tiny aggregate rows only
  - Snip index:                         one minute at a time, numpy packed dtype
  - Waveforms:                          LRU cache, 256 entries max

CPU budget is low so we need near zero baseline.
  - All SQLite on a background thread (priority queue, stale results discarded)
  - GUI thread only does Tk widget inserts (PAGE_SIZE rows max)
  - np.argsort for column sort (sub-ms for thousands of rows)
  - Matplotlib line.set_data() reuse -- no ax.clear()

Acquisition stays fast because nothing in this module runs during capture;
it is loaded only when the archive window is open.

Optional fast-decode: if cappy_native.so is present alongside this module (eventually),
cappy_batch_decode is used to decompress a minute's waveforms in one C call
(fallback to Python zlib if not available).
"""
from __future__ import annotations
import os, queue, sqlite3, struct, threading, zlib
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# failed: cappy_native.so — C-accelerated batch decode.future update pending...
# Falls back to pure-Python zlib silently if the .so is absent.

_NATIVE_ARCH = None
try:
    import ctypes as _ct_arch
    _so_path = Path(__file__).parent / "cappy_native.so"
    if _so_path.exists():
        _lib = _ct_arch.CDLL(str(_so_path))
        _lib.cappy_batch_decode.restype  = _ct_arch.c_int
        _lib.cappy_batch_decode.argtypes = [
            _ct_arch.POINTER(_ct_arch.c_uint8),
            _ct_arch.c_int,
            _ct_arch.c_int,
            _ct_arch.c_int,
            _ct_arch.POINTER(_ct_arch.c_float),
        ]
        _NATIVE_ARCH = _lib
except Exception:
    _NATIVE_ARCH = None


# Lazy matplotlib — deferred until the archive window opens so the
# acquisition path pays zero import cost at module load time.

_MPL: Optional[tuple] = None


def _lazy_mpl() -> tuple:
    global _MPL
    if _MPL is not None:
        return _MPL
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    try:
        plt.style.use("dark_background")
    except Exception:
        pass
    _MPL = (matplotlib, plt, FigureCanvasTkAgg)
    return _MPL

# Colour palette — matches CAPPY v1.3 main GUI theme

C_BG     = "#0b0e14"
C_PANEL  = "#111620"
C_PANEL2 = "#151b28"
C_BORDER = "#1e2636"
C_BDR2   = "#2a3550"
C_SEL    = "#0d2240"
C_SEL2   = "#143360"
C_CYAN   = "#00d4ff"
C_GOLD   = "#ffb700"
C_GREEN  = "#00e676"
C_TEAL   = "#1de9b6"
C_MAGENTA = "#e040fb"
C_TEXT   = "#c8d4e6"
C_TEXT2  = "#6e7f99"
C_DIM    = "#3a4560"
C_DIM2   = "#4e5f7a"
FONT     = ("Consolas", 9)
FONT_SM  = ("Consolas", 8)
FONT_B   = ("Consolas", 9, "bold")


# Waveform codec 
_CWZ1_MAGIC = b"CWZ1"
_CWZ1_FMT   = "<4sBBIf"
_CWZ1_SIZE  = struct.calcsize(_CWZ1_FMT)

def _decode_payload(raw: bytes, n: int) -> np.ndarray:
    """Decode waveform payload. Handles:
      - CWZ1 (delta_i16_zlib) from v1.0 and v1.3
      - f32_zlib (plain zlib-compressed float32) from v1.3
      - raw float32 (no compression)
    """
    if len(raw) >= _CWZ1_SIZE and raw[:4] == _CWZ1_MAGIC:
        try:
            _, _v, _q, nq, sc = struct.unpack(_CWZ1_FMT, raw[:_CWZ1_SIZE])
            d  = np.frombuffer(zlib.decompress(raw[_CWZ1_SIZE:]),
                               dtype=np.int16, count=int(nq))
            out = np.cumsum(d.astype(np.int32), dtype=np.int32).astype(np.float32)
            out *= np.float32(sc)
            return out[:n] if n > 0 else out
        except Exception:
            pass
    # Try f32_zlib: detect zlib magic bytes (0x78xx)
    if len(raw) >= 2 and raw[0:1] == b'\x78' and raw[1:2] in (b'\x01', b'\x5e', b'\x9c', b'\xda'):
        try:
            data = zlib.decompress(raw)
            arr = np.frombuffer(data, dtype=np.float32)
            return arr[:n] if n > 0 else arr
        except Exception:
            pass
    return np.frombuffer(raw, dtype=np.float32)[:n]


# LRU waveform cache

class _WaveCache:
    def __init__(self, maxsize: int = 256):
        self._d: OrderedDict = OrderedDict()
        self._m = maxsize

    def get(self, k: int):
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        return None

    def put(self, k: int, wa: np.ndarray, wb):
        self._d[k] = (wa, wb)
        self._d.move_to_end(k)
        if len(self._d) > self._m:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()

# DB worker thread
# Owns ALL sqlite connections. GUI never touches sqlite.
# Priority queue: 0 = user-visible (wave load), 1 = navigation, 2 = background

class _Req:
    __slots__ = ("kind", "payload", "token", "cb")
    def __init__(self, kind, payload, token, cb):
        self.kind = kind; self.payload = payload
        self.token = token; self.cb = cb

class _DBWorker(threading.Thread):
    # Integer divisors for time bucketing (nanoseconds)
    _DAY  = 86_400_000_000_000
    _HOUR =  3_600_000_000_000
    _MIN  =     60_000_000_000

    def __init__(self):
        super().__init__(daemon=True, name="cappy-db")
        self._q:  queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._cc: Dict[str, sqlite3.Connection] = {}
        self._stop = threading.Event()

    def submit(self, priority: int, kind: str, payload: Any,
               token: int, cb) -> None:
        self._seq += 1
        self._q.put((priority, self._seq, _Req(kind, payload, token, cb)))

    def stop(self):
        self._stop.set()
        self._q.put((999, 0, None))

    def _conn(self, path: str) -> sqlite3.Connection:
        if path not in self._cc:
            c = sqlite3.connect(
                f"file:{path}?mode=ro", uri=True,
                check_same_thread=False,
            )
            # Tune for read-heavy workloads on large DBs
            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=OFF;")
            c.execute("PRAGMA cache_size=-65536;")     # 64 MiB page cache
            c.execute("PRAGMA mmap_size=1073741824;")  # 1 GiB mmap
            c.execute("PRAGMA temp_store=MEMORY;")
            c.row_factory = sqlite3.Row
            self._cc[path] = c
        return self._cc[path]

    def run(self):
        while not self._stop.is_set():
            try:
                _, _, req = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            if req is None:
                break
            try:
                self._dispatch(req)
            except Exception as ex:
                try:
                    req.cb({"error": str(ex), "token": req.token})
                except Exception:
                    pass

    def _dispatch(self, req: _Req):
        k, p, tok, cb = req.kind, req.payload, req.token, req.cb

        #list_sessions: one cheap GROUP BY per DB file 
        if k == "list_sessions":
            rows = []
            for db in p:
                try:
                    for r in self._conn(db).execute(
                        "SELECT session_id,"
                        " MIN(timestamp_ns) AS t0,"
                        " MAX(timestamp_ns) AS t1,"
                        " COUNT(*) AS n"
                        " FROM snips"
                        " GROUP BY session_id"
                        " ORDER BY t0 DESC"
                    ):
                        rows.append({
                            "session_id": r["session_id"],
                            "t0": r["t0"], "t1": r["t1"],
                            "n": r["n"], "db_path": db,
                        })
                except Exception:
                    pass
            cb({"token": tok, "rows": rows})

        #  list_days: GROUP BY (timestamp_ns / DAY) 
        elif k == "list_days":
            db, sid = p["db_path"], p["session_id"]
            try:
                rows = [dict(r) for r in self._conn(db).execute(
                    "SELECT (timestamp_ns / ?) AS bucket,"
                    " COUNT(*) AS n,"
                    " MIN(timestamp_ns) AS t0"
                    " FROM snips"
                    " WHERE session_id = ?"
                    " GROUP BY bucket ORDER BY bucket",
                    (self._DAY, sid),
                ).fetchall()]
            except Exception:
                rows = []
            cb({"token": tok, "rows": rows})

        #list_hours: GROUP BY hour within a day bucket 
        elif k == "list_hours":
            db, sid, day = p["db_path"], p["session_id"], p["bucket"]
            t0 = day * self._DAY; t1 = t0 + self._DAY
            try:
                rows = [dict(r) for r in self._conn(db).execute(
                    "SELECT (timestamp_ns / ?) AS bucket,"
                    " COUNT(*) AS n"
                    " FROM snips"
                    " WHERE session_id = ?"
                    "   AND timestamp_ns >= ? AND timestamp_ns < ?"
                    " GROUP BY bucket ORDER BY bucket",
                    (self._HOUR, sid, t0, t1),
                ).fetchall()]
            except Exception:
                rows = []
            cb({"token": tok, "rows": rows})

        #list_minutes: GROUP BY minute within an hour bucket 
        elif k == "list_minutes":
            db, sid, hr = p["db_path"], p["session_id"], p["bucket"]
            t0 = hr * self._HOUR; t1 = t0 + self._HOUR
            try:
                rows = [dict(r) for r in self._conn(db).execute(
                    "SELECT (timestamp_ns / ?) AS bucket,"
                    " COUNT(*) AS n"
                    " FROM snips"
                    " WHERE session_id = ?"
                    "   AND timestamp_ns >= ? AND timestamp_ns < ?"
                    " GROUP BY bucket ORDER BY bucket",
                    (self._MIN, sid, t0, t1),
                ).fetchall()]
            except Exception:
                rows = []
            cb({"token": tok, "rows": rows})

        # load_minute: fetch lightweight index for ONE minute only 
        # This is the ONLY query that returns individual rows.
        # A minute at 1 MHz with 1-record-per-shot = at most ~60k rows.
      
        elif k == "load_minute":
            db, sid, mb = p["db_path"], p["session_id"], p["bucket"]
            t0 = mb * self._MIN; t1 = t0 + self._MIN
            COLS = (
                "id, timestamp_ns, buffer_index, record_in_buffer,"
                " record_global, area_A_Vs, peak_A_V,"
                " area_B_Vs, peak_B_V, sample_rate_hz, n_samples"
            )
            try:
                rows = self._conn(db).execute(
                    f"SELECT {COLS} FROM snips"
                    f" WHERE session_id = ?"
                    f"   AND timestamp_ns >= ? AND timestamp_ns < ?"
                    f" ORDER BY timestamp_ns DESC",
                    (sid, t0, t1),
                ).fetchall()
            except Exception:
                rows = []
            cb({"token": tok, "rows": rows})

        # load_wave: single waveform by id 
        elif k == "load_wave":
            db, sid_, dd = p["db_path"], p["snip_id"], Path(p["day_dir"])
            try:
                row = self._conn(db).execute(
                    "SELECT n_samples, n_channels,"
                    " file, offset_bytes, nbytes,"
                    " file_A, offset_A, nbytes_A,"
                    " file_B, offset_B, nbytes_B,"
                    " baseline_A_V, baseline_B_V, channels_mask"
                    " FROM snips WHERE id = ?",
                    (sid_,),
                ).fetchone()
            except Exception as ex:
                cb({"token": tok, "error": str(ex), "snip_id": sid_})
                return
            if row is None:
                cb({"token": tok, "error": "snip not found", "snip_id": sid_})
                return

            n = int(row["n_samples"] or 0)
            wa = wb = None

            def _rb(rel, off, nb):
                with open(dd / str(rel), "rb") as fh:
                    fh.seek(int(off))
                    return fh.read(int(nb))

            try:
                if row["file_A"]:
                    wa = _decode_payload(
                        _rb(row["file_A"], row["offset_A"], row["nbytes_A"]), n)
                    if row["file_B"]:
                        wb = _decode_payload(
                            _rb(row["file_B"], row["offset_B"], row["nbytes_B"]), n)
                elif row["file"]:
                    arr = np.frombuffer(
                        _rb(row["file"], row["offset_bytes"], row["nbytes"]),
                        dtype=np.float32)
                    wa = arr[:n]
                    if int(row["n_channels"] or 1) == 2:
                        wb = arr[n:2 * n]
            except Exception as ex:
                cb({"token": tok, "error": str(ex), "snip_id": sid_})
                return

            cb({
                "token": tok, "snip_id": sid_,
                "wa": wa, "wb": wb,
                "baseline_A": float(row["baseline_A_V"] or 0.0),
                "baseline_B": float(row["baseline_B_V"] or 0.0),
                "channels_mask": str(row["channels_mask"] or "CHANNEL_A"),
            })


# Numpy snip dtype  (compact packed; ~80 bytes per row)
_SNIP_DT = np.dtype([
    ("id",               "i8"),
    ("timestamp_ns",     "i8"),
    ("buffer_index",     "i8"),
    ("record_in_buffer", "i4"),
    ("record_global",    "i8"),
    ("area_A_Vs",        "f8"),
    ("peak_A_V",         "f8"),
    ("area_B_Vs",        "f8"),
    ("peak_B_V",         "f8"),
    ("sample_rate_hz",   "f8"),
    ("n_samples",        "i4"),
])

def _rows_to_arr(rows) -> np.ndarray:
    n = len(rows)
    a = np.empty(n, dtype=_SNIP_DT)
    for i, r in enumerate(rows):
        a["id"][i]               = int(r["id"] or 0)
        a["timestamp_ns"][i]     = int(r["timestamp_ns"] or 0)
        a["buffer_index"][i]     = int(r["buffer_index"] or 0)
        a["record_in_buffer"][i] = int(r["record_in_buffer"] or 0)
        a["record_global"][i]    = int(r["record_global"] or 0)
        a["area_A_Vs"][i]        = float(r["area_A_Vs"] or 0.0)
        a["peak_A_V"][i]         = float(r["peak_A_V"] or 0.0)
        a["area_B_Vs"][i]        = float(r["area_B_Vs"] or 0.0)
        a["peak_B_V"][i]         = float(r["peak_B_V"] or 0.0)
        a["sample_rate_hz"][i]   = float(r["sample_rate_hz"] or 0.0)
        a["n_samples"][i]        = int(r["n_samples"] or 0)
    return a

# Virtual Treeview -- numpy backing, PAGE_SIZE rows in Tk at all times

PAGE_SIZE = 200

class _VirtualTree:
    COLS  = ("time",  "id",  "buf", "rec", "global", "areaA",      "peakA")
    HDRS  = ("Timestamp","ID","Buf","Rec","Global","Area A (V·s)","Peak A (V)")
    WIDS  = (140, 68, 52, 44, 76, 100, 84)
    FIELD = {
        "time":   "timestamp_ns",
        "id":     "id",
        "buf":    "buffer_index",
        "rec":    "record_in_buffer",
        "global": "record_global",
        "areaA":  "area_A_Vs",
        "peakA":  "peak_A_V",
    }

    def __init__(self, parent, on_select):
        self._on_select = on_select
        self._idx: Optional[np.ndarray] = None
        self._pg  = 0
        self._sf  = "timestamp_ns"
        self._sa  = False          # ascending?
        self._tz  = datetime.now().astimezone().tzinfo

        frame = tk.Frame(parent, bg=C_BG)
        frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            frame, columns=self.COLS, show="headings",
            selectmode="browse", height=18,
        )
        for c, h, w in zip(self.COLS, self.HDRS, self.WIDS):
            self.tree.heading(c, text=h, command=lambda cc=c: self._sort(cc))
            self.tree.column(c, width=w,
                             anchor="w" if c == "time" else "e",
                             stretch=False)

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._yview)
        self.tree.configure(yscrollcommand=self._yscroll)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self._vsb = vsb
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.tree.bind("<<TreeviewSelect>>", self._on_sel)

        self._sv = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._sv,
                 font=FONT_SM, fg=C_DIM2, bg=C_BG, anchor="w",
                 ).pack(fill=tk.X, padx=4, pady=(1, 0))

    # public
    def load(self, rows: list):
        if not rows:
            self._idx = None; self._pg = 0
            self._render(); self._sv.set("No snips in this window.")
            return
        arr = _rows_to_arr(rows)
        # default: newest first
        self._idx = arr[np.argsort(arr["timestamp_ns"])[::-1]]
        self._pg = 0
        self._render()
        self._sv.set(f"{len(rows):,} snips  (this minute)")

    def clear(self):
        self._idx = None; self._pg = 0
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        self._sv.set("")

    def seek_to_ns(self, tgt: int) -> bool:
        if self._idx is None or len(self._idx) == 0:
            return False
        idx = int(np.argmin(np.abs(self._idx["timestamp_ns"] - tgt)))
        self._pg = max(0, idx - PAGE_SIZE // 2)
        self._render()
        iid = f"r_{idx}"
        if iid in self.tree.get_children():
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)
            self._on_sel()
        return True

    def get_by_id(self, snip_id: int):
        if self._idx is None:
            return None
        m = np.where(self._idx["id"] == snip_id)[0]
        return self._idx[m[0]] if len(m) else None

    # private
    def _render(self):
        t = self.tree
        for iid in t.get_children():
            t.delete(iid)
        if self._idx is None or len(self._idx) == 0:
            t.insert("", tk.END, values=("—",) * len(self.COLS))
            return

        a   = self._idx
        n   = len(a)
        ps  = max(0, min(self._pg, n - 1))
        pe  = min(ps + PAGE_SIZE, n)
        tz  = self._tz
        ins = t.insert

        ts_a = a["timestamp_ns"][ps:pe]
        id_a = a["id"][ps:pe]
        bu_a = a["buffer_index"][ps:pe]
        re_a = a["record_in_buffer"][ps:pe]
        gl_a = a["record_global"][ps:pe]
        aA_a = a["area_A_Vs"][ps:pe]
        pA_a = a["peak_A_V"][ps:pe]

        for i in range(pe - ps):
            tsn = int(ts_a[i])
            dt_ = datetime.fromtimestamp(tsn / 1e9, tz=tz)
            us  = tsn % 1_000_000_000 // 1_000
            ts  = dt_.strftime("%H:%M:%S") + f".{us:06d}"
            tag = ("alt",) if (ps + i) % 2 else ()
            ins("", tk.END, iid=f"r_{ps+i}",
                values=(ts, int(id_a[i]), int(bu_a[i]),
                        int(re_a[i]), int(gl_a[i]),
                        f"{float(aA_a[i]):.4g}", f"{float(pA_a[i]):.4g}"),
                tags=tag)
        t.tag_configure("alt", background=C_PANEL2)

        lo = ps / n if n else 0.0
        hi = min(pe / n, 1.0) if n else 1.0
        self._vsb.set(lo, hi)

    def _yview(self, *args):
        if not args or self._idx is None:
            return
        n = len(self._idx); cmd = args[0]
        if cmd == "moveto":
            np_ = int(float(args[1]) * n)
        elif cmd == "scroll":
            step = PAGE_SIZE // 4 if args[2] == "pages" else 10
            np_  = self._pg + int(args[1]) * step
        else:
            return
        np_ = max(0, min(n - 1, np_))
        if np_ != self._pg:
            self._pg = np_; self._render()

    def _yscroll(self, lo, hi):
        self._vsb.set(lo, hi)

    def _on_sel(self, _=None):
        sel = self.tree.selection()
        if not sel:
            return
        try:
            ri = int(sel[0].split("_")[1])
        except Exception:
            return
        if self._idx is None or ri >= len(self._idx):
            return
        r = self._idx[ri]
        self._on_select(int(r["id"]), ri)

    def _sort(self, col: str):
        if self._idx is None:
            return
        f = self.FIELD.get(col, "timestamp_ns")
        if self._sf == f:
            self._sa = not self._sa
        else:
            self._sf = f; self._sa = True
        o = np.argsort(self._idx[f], stable=True)
        if not self._sa:
            o = o[::-1]
        self._idx = self._idx[o]
        self._pg = 0; self._render()

# Waveform plot panel   (artist reuse - never ax.clear())

class _WavePlot:
    def __init__(self, parent):
        _, plt, FC = _lazy_mpl()
        self.fig = plt.Figure(figsize=(7.5, 6.2), facecolor=C_BG)
        self.fig.subplots_adjust(left=0.13, right=0.97,
                                 top=0.97, bottom=0.09, hspace=0.38)
        self.axA = self.fig.add_subplot(311)
        self.axB = self.fig.add_subplot(312)
        self.axI = self.fig.add_subplot(313)
        self._style_axes(self.axA, self.axB, self.axI)

        # Main waveform lines
        (self._lA,)  = self.axA.plot([], [], color=C_CYAN,  lw=1.2, zorder=3)
        (self._lB,)  = self.axB.plot([], [], color=C_GOLD,  lw=1.2, zorder=3)
        (self._lIA,) = self.axI.plot([], [], color=C_GREEN, lw=1.2, label="integral A", zorder=3)
        (self._lIB,) = self.axI.plot([], [], color=C_GOLD,  lw=1.2,
                                      linestyle="--", label="integral B", zorder=3)

        # Ombre fill polygons (updated in plot())
        self._fillA = self.axA.fill_between([], [], alpha=0)
        self._fillB = self.axB.fill_between([], [], alpha=0)
        self._fillI = self.axI.fill_between([], [], alpha=0)

        # Dot markers for pinned point
        (self._dotA,) = self.axA.plot([], [], 'o', color=C_MAGENTA, ms=7, mew=1.5,
                                       mec='white', zorder=10, visible=False)
        (self._dotB,) = self.axB.plot([], [], 'o', color=C_MAGENTA, ms=7, mew=1.5,
                                       mec='white', zorder=10, visible=False)

        self.axA.set_ylabel("Ch A (V)",  color=C_CYAN,  fontsize=8)
        self.axB.set_ylabel("Ch B (V)",  color=C_GOLD,  fontsize=8)
        self.axI.set_ylabel("∫ (V·s)",   color=C_GREEN, fontsize=8)
        self.axI.set_xlabel("Time",      color=C_TEXT2, fontsize=8)

        # Crosshair vlines
        self._vlA = self.axA.axvline(0, color=C_CYAN,  alpha=0.3, lw=0.8, visible=False)
        self._vlB = self.axB.axvline(0, color=C_GOLD,  alpha=0.3, lw=0.8, visible=False)
        self._vlI = self.axI.axvline(0, color=C_TEXT2, alpha=0.3, lw=0.8, visible=False)

        # Store last-plotted data for pop-out
        self._last_wa   = None
        self._last_wb   = None
        self._last_sr   = 1.0
        self._last_tsns = 0
        self._last_bA   = 0.0
        self._last_bB   = 0.0

        self.canvas = FC(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NTB
            tbf = tk.Frame(parent, bg=C_PANEL, height=28)
            tbf.pack(fill=tk.X, side=tk.BOTTOM)
            self._tb = NTB(self.canvas, tbf)
            self._tb.update()
        except Exception:
            self._tb = None

        self._tv   = None
        self._wap  = None
        self._wbp  = None
        self._iA   = None
        self._unit = "s"
        self._tsns = 0
        self._tz   = datetime.now().astimezone().tzinfo

        # Readout label (hover)
        self._rdv = tk.StringVar(value="")
        self._rdv_lbl = tk.Label(parent, textvariable=self._rdv,
                 font=("Consolas", 9), fg=C_CYAN, bg=C_BG,
                 anchor="w", padx=6)
        self._rdv_lbl.pack(fill=tk.X, side=tk.BOTTOM)

        # Draggy mode state
        self._draggy_active = False
        self._drag_p1: Optional[float] = None
        self._drag_spans: list = []
        self._drag_vlines: list = []
        self._draggy_result_var = tk.StringVar(value="")
        self._draggy_result_lbl = tk.Label(
            parent, textvariable=self._draggy_result_var,
            font=("Consolas", 9, "bold"), fg=C_MAGENTA, bg=C_BG,
            anchor="w", padx=6)
        self._draggy_result_lbl.pack(fill=tk.X, side=tk.BOTTOM)

        self.canvas.mpl_connect("motion_notify_event", self._hover)
        self.canvas.mpl_connect("button_press_event",  self._on_click)

    @staticmethod
    def _style_axes(*axes):
        """Apply consistent dark styling to a set of axes."""
        for ax in axes:
            ax.set_facecolor(C_PANEL)
            ax.tick_params(colors=C_DIM2, labelsize=8, which='both', direction='in')
            ax.tick_params(which='minor', length=2)
            ax.tick_params(which='major', length=4)
            for sp in ax.spines.values():
                sp.set_color(C_BORDER)
            ax.grid(True, which='major', color=C_DIM, alpha=0.3, linewidth=0.6)
            ax.grid(True, which='minor', color=C_DIM, alpha=0.10, linewidth=0.4)
            ax.minorticks_on()

    @staticmethod
    def _clear_collections(ax) -> None:
        """Remove all PolyCollection children (fill_between patches) from an axis."""
        for coll in list(ax.collections):
            try:
                coll.remove()
            except Exception:
                pass

    @staticmethod
    def _ombre(ax, tv, data, color, alpha_top: float = 0.22) -> None:
        """Two-pass fill_between to fake a vertical ombre under a waveform.

        Positive excursions get full alpha; negative lobes get half, so
        the shading stays visually anchored to zero without an explicit
        baseline offset.
        """
        baseline = np.zeros_like(data)
        kw = dict(color=color, zorder=1, linewidth=0)
        ax.fill_between(tv, data, baseline, where=(data >= baseline),
                        alpha=alpha_top * 0.9, **kw)
        ax.fill_between(tv, data, baseline, where=(data < baseline),
                        alpha=alpha_top * 0.5, **kw)

    #  draggy modeee

    def toggle_draggy(self) -> bool:
        """Flip draggy mode on/off. Returns the new active state."""
        self._draggy_active = not self._draggy_active
        if not self._draggy_active:
            self._drag_p1 = None
            self._clear_drag_artists()
            self._draggy_result_var.set("")
            if self._tv is not None and len(self._tv) > 1:
                x0, x1 = float(self._tv[0]), float(self._tv[-1])
                for ax in (self.axA, self.axB, self.axI):
                    ax.set_xlim(x0, x1)
            self.canvas.draw_idle()
        return self._draggy_active

    def _clear_drag_artists(self) -> None:
        """Remove all draggy selection overlays from every axis."""
        for artist in (*self._drag_spans, *self._drag_vlines):
            try:
                artist.remove()
            except Exception:
                pass
        self._drag_spans.clear()
        self._drag_vlines.clear()

    def _on_click(self, ev) -> None:  # mpl button_press_event
        if ev.inaxes is None or self._tv is None or self._wap is None:
            return
        if ev.button != 1:
            return
        x = ev.xdata
        if x is None:
            return

        # Draggy Modee
        if self._draggy_active:
            if self._drag_p1 is None:
                self._drag_p1 = x
                self._clear_drag_artists()
                self._draggy_result_var.set(
                    f"✦ DRAGGY  P1 = {x:.4g} {self._unit}  →  click second point")
                for ax in (self.axA, self.axB, self.axI):
                    vl = ax.axvline(x, color=C_MAGENTA, lw=1.4, linestyle="--", alpha=0.8)
                    self._drag_vlines.append(vl)
                self.canvas.draw_idle()
            else:
                self._draggy_commit(x)
            return

        #  Normal click: pin dot marker
        tv  = self._tv
        idx = int(np.clip(np.searchsorted(tv, x), 0, len(tv) - 1))
        t_  = float(tv[idx])
        if idx < len(self._wap):
            self._dotA.set_data([t_], [float(self._wap[idx])])
            self._dotA.set_visible(True)
        try:
            lbx, lby = self._lB.get_data()
            if lby is not None and hasattr(lby, '__len__') and len(lby) > 0 and idx < len(lby):
                self._dotB.set_data([t_], [float(lby[idx])])
                self._dotB.set_visible(True)
        except Exception:
            pass
        self.canvas.draw_idle()


    def _draggy_commit(self, x2: float) -> None:
        """Finalize a draggy selection: draw overlays, zoom, compute integrals."""
        x1, x2 = sorted([self._drag_p1, x2])
        self._drag_p1 = None
        self._clear_drag_artists()

        tv   = self._tv
        unit = self._unit
        scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3}.get(unit, 1.0)

        i1 = int(np.clip(np.searchsorted(tv, x1), 0, len(tv) - 1))
        i2 = int(np.clip(np.searchsorted(tv, x2), 0, len(tv) - 1))
        if i2 <= i1:
            i2 = min(i1 + 1, len(tv) - 1)

        t1_f, t2_f = float(tv[i1]), float(tv[i2])
        span = t2_f - t1_f
        pad  = span * 0.05 if span > 0 else 1e-9

        for ax in (self.axA, self.axB, self.axI):
            sp = ax.axvspan(t1_f, t2_f, color=C_MAGENTA, alpha=0.10, zorder=0)
            self._drag_spans.append(sp)
            for xb in (t1_f, t2_f):
                vl = ax.axvline(xb, color=C_MAGENTA, lw=1.0, linestyle="--", alpha=0.55)
                self._drag_vlines.append(vl)
            ax.set_xlim(t1_f - pad, t2_f + pad)

        # dt in seconds, estimated from the time axis
        dt_s = scale * (float(tv[1]) - float(tv[0])) if len(tv) > 1 else 1.0

        def _integral_stats(seg: np.ndarray):
            cum  = np.cumsum(seg.astype(np.float64)) * dt_s
            return float(np.mean(cum)), float(cum[-1]) if len(cum) else 0.0

        mean_iA, total_iA = _integral_stats(self._wap[i1:i2 + 1])
        result = (
            f"◈ {t1_f:.4g}–{t2_f:.4g} {unit}"
            f"  n={i2 - i1}"
            f"  ∫A μ={mean_iA:.5g} V·s  Σ={total_iA:.5g} V·s"
        )

        try:
            _, lby = self._lB.get_data()
            if lby is not None and len(lby) > i2:
                mean_iB, _ = _integral_stats(np.asarray(lby[i1:i2 + 1]))
                result += f"  ∫B μ={mean_iB:.5g} V·s"
        except Exception:
            pass

        self._draggy_result_var.set(result)
        self.canvas.draw_idle()

    def pop_out(self, title: str = "cappy · waveform compare") -> Optional[tk.Toplevel]:
        """Detached compare window — full feature parity with main window."""
        if self._last_wa is None:
            return None
        _, plt, FC = _lazy_mpl()

        win = tk.Toplevel()
        win.title(title)
        win.configure(bg=C_BG)
        win.geometry("760x640")
        win.minsize(520, 420)

        # ── figure ──────────────────────────────────────────────────────────
        fig = plt.Figure(figsize=(7.2, 5.6), facecolor=C_BG)
        fig.subplots_adjust(left=0.13, right=0.97, top=0.93, bottom=0.09, hspace=0.38)
        axA = fig.add_subplot(311)
        axB = fig.add_subplot(312)
        axI = fig.add_subplot(313)
        self._style_axes(axA, axB, axI)

        wa, wb = self._last_wa, self._last_wb
        sr, bA, bB = self._last_sr, self._last_bA, self._last_bB
        tv, unit = self._tax(len(wa), sr)
        wap = wa - bA
        dt  = 1.0 / sr

        lA,  = axA.plot(tv, wap, color=C_CYAN, lw=1.2, zorder=3)
        self._ombre(axA, tv, wap, C_CYAN, alpha_top=0.18)
        axA.set_ylabel("Ch A (V)", color=C_CYAN,  fontsize=8)
        axA.set_xlabel(f"Time ({unit})", color=C_TEXT2, fontsize=8)

        lB = None
        if wb is not None:
            wbp = wb - bB
            lB, = axB.plot(tv, wbp, color=C_GOLD, lw=1.2, zorder=3)
            self._ombre(axB, tv, wbp, C_GOLD, alpha_top=0.15)
            axB.set_ylabel("Ch B (V)", color=C_GOLD, fontsize=8)
        else:
            axB.set_visible(False)
        axB.set_xlabel(f"Time ({unit})", color=C_TEXT2, fontsize=8)

        iA = np.cumsum((wa - bA).astype(np.float64)) * dt
        lIA, = axI.plot(tv, iA, color=C_GREEN, lw=1.2, label="integral A", zorder=3)
        self._ombre(axI, tv, iA, C_GREEN, alpha_top=0.14)
        if wb is not None:
            iB = np.cumsum((wb - bB).astype(np.float64)) * dt
            axI.plot(tv, iB, color=C_GOLD, lw=1.2, linestyle="--", label="integral B")
        axI.set_ylabel("∫ (V·s)", color=C_GREEN, fontsize=8)
        axI.set_xlabel(f"Time ({unit})", color=C_TEXT2, fontsize=8)
        axI.legend(loc="best", fontsize=7, facecolor=C_PANEL,
                   edgecolor=C_BORDER, labelcolor=C_TEXT)

        # Auto-trim
        if len(tv) > 8:
            trim = self._tail(wa - bA)
            if wb is not None: trim = max(trim, self._tail(wb - bB))
            if trim < len(tv) - 2:
                right = float(tv[max(1, trim)])
                for ax in (axA, axB, axI):
                    ax.set_xlim(left=float(tv[0]), right=right)

        # Timestamp title
        ts_dt = datetime.fromtimestamp(self._last_tsns / 1e9, tz=self._tz)
        fig.suptitle(ts_dt.strftime("%Y-%m-%d  %H:%M:%S.%f"),
                     color=C_CYAN, fontsize=9, y=0.99)

        canvas = FC(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        try:
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NTB
            tbf = tk.Frame(win, bg=C_PANEL, height=28)
            tbf.pack(fill=tk.X, side=tk.BOTTOM)
            NTB(canvas, tbf).update()
        except Exception:
            pass

        #  state for this pop-out
        po_state = {
            "draggy_active": False,
            "drag_p1":       None,
            "drag_spans":    [],
            "drag_vlines":   [],
            "tv":            tv,
            "wap":           wap,
            "unit":          unit,
        }

        # crosshair vlines
        vlA = axA.axvline(0, color=C_CYAN,  alpha=0.3, lw=0.8, visible=False)
        vlB = axB.axvline(0, color=C_GOLD,  alpha=0.3, lw=0.8, visible=False)
        vlI = axI.axvline(0, color=C_TEXT2, alpha=0.3, lw=0.8, visible=False)

        # dot markers
        dotA, = axA.plot([], [], 'o', color=C_MAGENTA, ms=7, mew=1.5,
                          mec='white', zorder=10, visible=False)
        dotB, = axB.plot([], [], 'o', color=C_MAGENTA, ms=7, mew=1.5,
                          mec='white', zorder=10, visible=False) if wb is not None else (None,)

        # bottom bar
        bar = tk.Frame(win, bg=C_BG)
        bar.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0, 2))

        draggy_var = tk.BooleanVar(value=False)
        draggy_btn_txt = tk.StringVar(value="⬡  Draggy Mode")
        draggy_result  = tk.StringVar(value="")
        rdv            = tk.StringVar(value="")

        def _toggle_po_draggy():
            po_state["draggy_active"] = not po_state["draggy_active"]
            po_state["drag_p1"] = None
            _clear_po_drag()
            draggy_result.set("")
            if po_state["draggy_active"]:
                draggy_btn_txt.set("✦  Draggy Mode  ON")
            else:
                draggy_btn_txt.set("⬡  Draggy Mode")
                # restore full xlim
                x0, x1_ = float(tv[0]), float(tv[-1])
                for ax in (axA, axB, axI):
                    ax.set_xlim(x0, x1_)
                canvas.draw_idle()

        def _reset_po_zoom():
            po_state["drag_p1"] = None
            _clear_po_drag()
            draggy_result.set("")
            x0, x1_ = float(tv[0]), float(tv[-1])
            for ax in (axA, axB, axI): ax.set_xlim(x0, x1_)
            canvas.draw_idle()

        def _clear_po_drag():
            for p in po_state["drag_spans"]:
                try: p.remove()
                except Exception: pass
            po_state["drag_spans"].clear()
            for l in po_state["drag_vlines"]:
                try: l.remove()
                except Exception: pass
            po_state["drag_vlines"].clear()

        ttk.Button(bar, textvariable=draggy_btn_txt,
                   command=_toggle_po_draggy, style="TButton").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="↺ Reset Zoom",
                   command=_reset_po_zoom,   style="TButton").pack(side=tk.LEFT, padx=(0, 6))

        tk.Label(bar, textvariable=draggy_result,
                 font=("Consolas", 9, "bold"), fg=C_MAGENTA, bg=C_BG,
                 anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Label(win, textvariable=rdv,
                 font=("Consolas", 9), fg=C_CYAN, bg=C_BG,
                 anchor="w", padx=6).pack(fill=tk.X, side=tk.BOTTOM)

        # event handler
        def _po_hover(ev):
            if ev.inaxes is None or po_state["tv"] is None:
                for vl in (vlA, vlB, vlI): vl.set_visible(False)
                canvas.draw_idle(); return
            x = ev.xdata
            if x is None: return
            tv_ = po_state["tv"]; unit_ = po_state["unit"]
            idx = int(np.clip(np.searchsorted(tv_, x), 0, len(tv_) - 1))
            t_  = float(tv_[idx])
            for vl in (vlA, vlB, vlI):
                vl.set_xdata([t_]); vl.set_visible(True)
            scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3}.get(unit_, 1.0)
            ans   = self._last_tsns + int(t_ * scale * 1e9)
            ts    = (datetime.fromtimestamp(ans / 1e9, tz=self._tz)
                     .strftime("%H:%M:%S") + f".{ans % 1_000_000_000:09d}")
            wap_  = po_state["wap"]
            if ev.inaxes == axA:
                v = float(wap_[idx]) if idx < len(wap_) else 0.0
                rdv.set(f"Ch A   t={t_:.4g} {unit_}   V={v:.6g} V   @ {ts}")
            elif ev.inaxes == axB and lB is not None:
                try:
                    _, lby = lB.get_data()
                    v = float(lby[idx]) if lby is not None and idx < len(lby) else 0.0
                except Exception: v = 0.0
                rdv.set(f"Ch B   t={t_:.4g} {unit_}   V={v:.6g} V   @ {ts}")
            elif ev.inaxes == axI:
                try:
                    _, liy = lIA.get_data()
                    v = float(liy[idx]) if liy is not None and idx < len(liy) else 0.0
                except Exception: v = 0.0
                rdv.set(f"∫A     t={t_:.4g} {unit_}   I={v:.6g} V·s   @ {ts}")
            canvas.draw_idle()

        def _po_click(ev):
            if ev.inaxes is None or ev.button != 1: return
            x = ev.xdata
            if x is None: return
            tv_ = po_state["tv"]; unit_ = po_state["unit"]

            if po_state["draggy_active"]:
                if po_state["drag_p1"] is None:
                    po_state["drag_p1"] = x
                    _clear_po_drag()
                    draggy_result.set(f"✦ DRAGGY  P1={x:.4g} {unit_}  →  click P2")
                    for ax in (axA, axB, axI):
                        vl = ax.axvline(x, color=C_MAGENTA, lw=1.4, linestyle="--", alpha=0.8)
                        po_state["drag_vlines"].append(vl)
                    canvas.draw_idle()
                else:
                    x1, x2 = sorted([po_state["drag_p1"], x])
                    po_state["drag_p1"] = None
                    _clear_po_drag()
                    i1 = int(np.clip(np.searchsorted(tv_, x1), 0, len(tv_)-1))
                    i2 = int(np.clip(np.searchsorted(tv_, x2), 0, len(tv_)-1))
                    if i2 <= i1: i2 = min(i1+1, len(tv_)-1)
                    for ax in (axA, axB, axI):
                        sp = ax.axvspan(float(tv_[i1]), float(tv_[i2]),
                                        color=C_MAGENTA, alpha=0.10, zorder=0)
                        po_state["drag_spans"].append(sp)
                        for xb in (float(tv_[i1]), float(tv_[i2])):
                            vl = ax.axvline(xb, color=C_MAGENTA, lw=1.0,
                                            linestyle="--", alpha=0.55)
                            po_state["drag_vlines"].append(vl)
                    span = float(tv_[i2]) - float(tv_[i1])
                    pad  = span * 0.05 if span > 0 else 1e-9
                    for ax in (axA, axB, axI):
                        ax.set_xlim(float(tv_[i1])-pad, float(tv_[i2])+pad)
                    scale = {"ns":1e-9,"us":1e-6,"ms":1e-3}.get(unit_,1.0)
                    sr_est = 1.0/(scale*(float(tv_[1])-float(tv_[0]))) if len(tv_)>1 else 1.0
                    dt_s   = 1.0/sr_est
                    wap_   = po_state["wap"]
                    seg_a  = wap_[i1:i2+1].astype(np.float64)
                    iA_seg = np.cumsum(seg_a)*dt_s
                    mean_iA  = float(np.mean(iA_seg))
                    total_iA = float(iA_seg[-1]) if len(iA_seg) else 0.0
                    res = (f"✦ {float(tv_[i1]):.4g}–{float(tv_[i2]):.4g} {unit_}"
                           f"  ({i2-i1} pts)"
                           f"  ∫A mean={mean_iA:.5g} V·s"
                           f"  total={total_iA:.5g} V·s")
                    if lB is not None:
                        try:
                            _, lby = lB.get_data()
                            if lby is not None and len(lby) > i2:
                                seg_b  = np.asarray(lby[i1:i2+1], dtype=np.float64)
                                iB_seg = np.cumsum(seg_b)*dt_s
                                res += f"  ∫B mean={float(np.mean(iB_seg)):.5g} V·s"
                        except Exception: pass
                    draggy_result.set(res)
                    canvas.draw_idle()
                return

            # Normal click — pin dot
            idx = int(np.clip(np.searchsorted(tv_, x), 0, len(tv_)-1))
            t_  = float(tv_[idx])
            wap_ = po_state["wap"]
            if idx < len(wap_):
                dotA.set_data([t_], [float(wap_[idx])])
                dotA.set_visible(True)
            if dotB is not None and lB is not None:
                try:
                    _, lby = lB.get_data()
                    if lby is not None and idx < len(lby):
                        dotB.set_data([t_], [float(lby[idx])])
                        dotB.set_visible(True)
                except Exception: pass
            canvas.draw_idle()

        canvas.mpl_connect("motion_notify_event", _po_hover)
        canvas.mpl_connect("button_press_event",  _po_click)
        canvas.draw()
        return win

    @staticmethod
    def _tax(n: int, sr: float) -> tuple[np.ndarray, str]:
        """Build a time axis of length *n* at sample rate *sr* Hz.

        Returns (tv, unit) where unit is the SI prefix that keeps the
        largest value comfortably ≥ 1 (ns / us / ms / s).
        """
        t  = np.arange(n, dtype=np.float64) / sr
        mx = float(t[-1]) if t.size else 0.0
        if mx < 1e-6: return t * 1e9,  "ns"
        if mx < 1e-3: return t * 1e6,  "us"
        if mx < 1.0:  return t * 1e3,  "ms"
        return t, "s"

    def plot(self, wa: np.ndarray, wb: Optional[np.ndarray],
             sr: float, tsns: int,
             bA: float = 0.0, bB: float = 0.0) -> None:
        tv, unit = self._tax(len(wa), sr)
        self._tv = tv; self._unit = unit; self._tsns = tsns
        wap = wa - bA
        wbp = (wb - bB) if wb is not None else None
        dt  = 1.0 / sr
        self._wap = wap
        self._wbp = wbp

        # Store for pop-out compare
        self._last_wa = wa.copy()
        self._last_wb = wb.copy() if wb is not None else None
        self._last_sr = sr
        self._last_tsns = tsns
        self._last_bA = bA
        self._last_bB = bB

        # Clear dot markers and draggy on new waveform
        self._dotA.set_data([], []); self._dotA.set_visible(False)
        self._dotB.set_data([], []); self._dotB.set_visible(False)
        self._drag_p1 = None
        self._clear_drag_artists()
        self._draggy_result_var.set("")

        def _slim(v):
            lo, hi = float(v.min()), float(v.max())
            pad = max(abs(hi - lo) * 0.06, 1e-12)
            return lo - pad, hi + pad

        self._lA.set_data(tv, wap)
        self.axA.set_xlim(tv[0], tv[-1]); self.axA.set_ylim(*_slim(wap))
        # ombre under Ch A  (remove stale fills before redrawing)
        self._clear_collections(self.axA)
        self._ombre(self.axA, tv, wap, C_CYAN, alpha_top=0.18)

        if wbp is not None:
            self._lB.set_data(tv, wbp)
            self.axB.set_xlim(tv[0], tv[-1]); self.axB.set_ylim(*_slim(wbp))
            self.axB.set_visible(True)
            self._clear_collections(self.axB)
            self._ombre(self.axB, tv, wbp, C_GOLD, alpha_top=0.15)
        else:
            self._lB.set_data([], []); self.axB.set_visible(False)

        wai = wa - bA
        wbi = (wb - bB) if wb is not None else None
        iA  = np.cumsum(wai.astype(np.float64)) * dt
        self._iA = iA
        self._lIA.set_data(tv, iA)
        self._clear_collections(self.axI)
        self._ombre(self.axI, tv, iA, C_GREEN, alpha_top=0.14)

        if wbi is not None:
            iB = np.cumsum(wbi.astype(np.float64)) * dt
            self._lIB.set_data(tv, iB); self._lIB.set_visible(True)
            self.axI.set_ylim(*_slim(np.concatenate([iA, iB])))
        else:
            self._lIB.set_data([], []); self._lIB.set_visible(False)
            self.axI.set_ylim(*_slim(iA))
        self.axI.set_xlim(tv[0], tv[-1])
        self.axI.set_xlabel(f"Time ({unit})", color=C_TEXT2, fontsize=8)

        # auto-trim flat signal tail
        if len(tv) > 8:
            trim = self._tail(wai)
            if wbi is not None:
                trim = max(trim, self._tail(wbi))
            if trim < len(tv) - 2:
                right = float(tv[max(1, trim)])
                for ax in (self.axA, self.axB, self.axI):
                    ax.set_xlim(left=float(tv[0]), right=right)

        self.canvas.draw_idle()
        if self._tb:
            try: self._tb.update()
            except Exception: pass

    @staticmethod
    def _tail(a: np.ndarray) -> int:
        a = np.asarray(a, dtype=np.float64).ravel(); n = len(a)
        if n <= 8: return n - 1
        tail = a[int(n * .8):]
        base = float(np.median(tail)); dev = np.abs(a - base)
        mad  = float(np.median(np.abs(tail - base)))
        thr  = max(5.0 * 1.4826 * mad,
                   0.02 * float(np.percentile(dev, 99)), 1e-6)
        idx  = np.flatnonzero(dev > thr)
        return min(n - 1, int(idx[-1]) + max(4, n // 100)) if idx.size else n - 1

    def _hover(self, ev) -> None:  # mpl motion_notify_event
        if ev.inaxes is None or self._tv is None:
            for vl in (self._vlA, self._vlB, self._vlI):
                vl.set_visible(False)
            self.canvas.draw_idle(); return
        x = ev.xdata
        if x is None: return
        tv  = self._tv
        idx = int(np.clip(np.searchsorted(tv, x), 0, len(tv) - 1))
        t_  = float(tv[idx])
        for vl in (self._vlA, self._vlB, self._vlI):
            vl.set_xdata([t_]); vl.set_visible(True)
        unit  = self._unit
        scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3}.get(unit, 1.0)
        ans   = self._tsns + int(t_ * scale * 1e9)
        ts    = (datetime.fromtimestamp(ans / 1e9, tz=self._tz)
                 .strftime("%H:%M:%S") + f".{ans % 1_000_000_000:09d}")

        # Color readout based on which axis is hovered
        if ev.inaxes == self.axA:
            wa = self._wap
            v  = float(wa[idx]) if wa is not None and idx < len(wa) else 0.0
            self._rdv.set(f"Ch A   t = {t_:.4g} {unit}   V = {v:.6g} V   @ {ts}")
            self._rdv_lbl.configure(fg=C_CYAN)
        elif ev.inaxes == self.axB:
            try:
                _, lby = self._lB.get_data()
                v = float(lby[idx]) if lby is not None and hasattr(lby, '__len__') and idx < len(lby) else 0.0
            except Exception:
                v = 0.0
            self._rdv.set(f"Ch B   t = {t_:.4g} {unit}   V = {v:.6g} V   @ {ts}")
            self._rdv_lbl.configure(fg=C_GOLD)
        elif ev.inaxes == self.axI:
            try:
                _, liy = self._lIA.get_data()
                v = float(liy[idx]) if liy is not None and hasattr(liy, '__len__') and idx < len(liy) else 0.0
            except Exception:
                v = 0.0
            self._rdv.set(f"∫A     t = {t_:.4g} {unit}   I = {v:.6g} V·s   @ {ts}")
            self._rdv_lbl.configure(fg=C_GREEN)
        else:
            wa = self._wap
            v  = float(wa[idx]) if wa is not None and idx < len(wa) else 0.0
            self._rdv.set(f"t = {t_:.4g} {unit}   V = {v:.6g} V   @ {ts}")
            self._rdv_lbl.configure(fg=C_TEXT)
        self.canvas.draw_idle()

# ---------------------------------------------------------------------------
# Time bucket constants  (nanoseconds)
# ---------------------------------------------------------------------------
_NS_PER_DAY  = 86_400_000_000_000
_NS_PER_HOUR =  3_600_000_000_000
_NS_PER_MIN  =     60_000_000_000

_UNIT_SCALE: Dict[str, float] = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}


# ---------------------------------------------------------------------------
# ArchiveDB  —  root widget
#
# Navigation hierarchy (left panel):
#   Session → Day → Hour → Minute → Snips
#
# Each drill level is a single GROUP BY query dispatched to _DBWorker.
# The GUI thread only ever holds PAGE_SIZE Treeview rows at a time.
# ---------------------------------------------------------------------------
class ArchiveDB(ttk.Frame):

    #lifecycle

    def __init__(self, data_dir: Path, master=None) -> None:
        super().__init__(master)
        self._tz       = datetime.now().astimezone().tzinfo
        self.data_dir  = Path(data_dir)
        self.captures  = self.data_dir / "captures"

        self._worker   = _DBWorker()
        self._worker.start()
        self._cache    = _WaveCache(256)
        self._tok_val  = 0

        # session → (db_path, day_dir)
        self._db_map:    Dict[str, tuple]  = {}
        self._cur_sid:   Optional[str]     = None
        self._cur_db:    Optional[str]     = None
        self._cur_daydir: Optional[str]    = None
        self._cur_snip:  Optional[int]     = None
        self._popout_count = 0
        self._draggy_on    = False

        self._apply_theme()
        self._build()
        self.after(80, self._refresh_sessions)

    def destroy(self) -> None:
        try:
            self._worker.stop()
        except Exception:
            pass
        super().destroy()

    def _tok(self) -> int:
        """Monotonic request token — stale callbacks self-discard on mismatch."""
        self._tok_val += 1
        return self._tok_val

    # theme

    def _apply_theme(self) -> None:
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except Exception:
            pass

        s.configure("Dark.TFrame",      background=C_BG)
        s.configure("Treeview",         background=C_PANEL, foreground=C_TEXT,
                    fieldbackground=C_PANEL, rowheight=22, font=FONT)
        s.configure("Treeview.Heading", background=C_BORDER, foreground=C_CYAN,
                    font=FONT_B)
        s.map("Treeview",
              background=[("selected", C_SEL2)],
              foreground=[("selected", C_CYAN)])

        s.configure("TScrollbar", background=C_BORDER, troughcolor=C_BG,
                    arrowcolor=C_DIM)

        _btn_common = dict(borderwidth=1, relief="flat")
        s.configure("TButton",    background=C_BDR2, foreground=C_TEXT,
                    font=FONT,   padding=(8, 3),  **_btn_common)
        s.configure("Hi.TButton", background=C_BDR2, foreground=C_CYAN,
                    font=FONT_B, padding=(10, 4), **_btn_common)
        s.map("TButton",    background=[("active", C_SEL)])
        s.map("Hi.TButton", background=[("active", C_SEL)],
              foreground=[("active", C_TEXT)])

    # layout

    def _build(self) -> None:
        self.configure(style="Dark.TFrame")
        self._build_topbar()
        self._build_main_pane()

    def _build_topbar(self) -> None:
        bar = tk.Frame(self, bg=C_PANEL, bd=0)
        bar.pack(fill=tk.X, side=tk.TOP)

        tk.Label(bar, text="CAPPY.ARCH", font=("Consolas", 12, "bold"),
                 fg=C_CYAN, bg=C_PANEL, padx=12, pady=5).pack(side=tk.LEFT)

        self._dir_var = tk.StringVar(value=str(self.data_dir))
        tk.Entry(bar, textvariable=self._dir_var,
                 bg="#06080d", fg=C_TEXT2, insertbackground=C_TEXT,
                 font=FONT, relief=tk.FLAT, bd=4,
                 ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        ttk.Button(bar, text="Browse…",  command=self._pick_dir,
                   style="TButton").pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="↺ Reload", command=self._refresh_sessions,
                   style="Hi.TButton").pack(side=tk.LEFT, padx=(2, 10))

    def _build_main_pane(self) -> None:
        pane  = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        left  = tk.Frame(pane, bg=C_BG)
        right = tk.Frame(pane, bg=C_BG)
        pane.add(left, weight=1)
        pane.add(right, weight=2)

        self._build_nav(left)
        self._build_wave_panel(right)

    def _build_wave_panel(self, parent: tk.Frame) -> None:
        """Right side: waveform plot + action bar + metadata text."""
        self._wave_plot = _WavePlot(parent)

        # action bar (packed bottom-up, so define before meta)
        bar = tk.Frame(parent, bg=C_BG)
        bar.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(2, 0))

        ttk.Button(bar, text="⎋ Compare",
                   command=self._pop_out_waveform,
                   style="Hi.TButton").pack(side=tk.LEFT, padx=(0, 6))

        self._draggy_btn_var = tk.StringVar(value="◇ Draggy")
        self._draggy_btn = ttk.Button(
            bar, textvariable=self._draggy_btn_var,
            command=self._toggle_draggy, style="TButton")
        self._draggy_btn.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Button(bar, text="↺ Reset Zoom",
                   command=self._reset_wave_zoom,
                   style="TButton").pack(side=tk.LEFT, padx=(0, 4))

        # snip metadata text box
        self._meta = tk.Text(
            parent, height=6,
            bg=C_PANEL, fg=C_TEXT, insertbackground=C_TEXT,
            font=FONT_SM, relief=tk.FLAT, bd=0, padx=8, pady=6,
            state=tk.DISABLED,
        )
        self._meta.pack(fill=tk.X, side=tk.BOTTOM, padx=4, pady=(0, 4))

    # nav panel
    def _build_nav(self, parent: tk.Frame) -> None:
        self._sh(parent, "SESSIONS")
        self._t_sess = self._mk_tree(
            parent,
            cols=("started", "session_id", "snips"),
            hdrs=("Started", "Session ID", "Snips"),
            wids=(140, 155, 60),
            on_sel=self._on_session, height=5,
        )

        # Small button to export YAML settings for the selected session to JSON
        sess_btn_row = tk.Frame(parent, bg=C_BG)
        sess_btn_row.pack(fill=tk.X, padx=4, pady=(1, 2))
        ttk.Button(sess_btn_row, text="Save Session YAML → JSON",
                   command=self._export_session_yaml_to_json,
                   style="TButton").pack(side=tk.LEFT)

        self._sh(parent, "DAY")
        self._t_day = self._mk_tree(
            parent,
            cols=("label", "n"), hdrs=("Date", "Snips"),
            wids=(140, 70), on_sel=self._on_day, height=4,
        )

        # Hour + Minute side-by-side
        hm = tk.Frame(parent, bg=C_BG)
        hm.pack(fill=tk.BOTH)

        for side_frame, label, tree_attr, on_sel, wids, height in (
            (tk.Frame(hm, bg=C_BG), "HOUR",   "_t_hour", self._on_hour,   (68, 54), 7),
            (tk.Frame(hm, bg=C_BG), "MINUTE", "_t_min",  self._on_minute, (56, 54), 7),
        ):
            side_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            tk.Label(side_frame, text=label, font=FONT_B,
                     fg=C_CYAN, bg=C_PANEL, padx=6, pady=3).pack(fill=tk.X)
            tree = self._mk_tree(side_frame,
                cols=("label", "n"), hdrs=(label.capitalize(), "Snips"),
                wids=wids, on_sel=on_sel, height=height,
            )
            setattr(self, tree_attr, tree)

        # Seek bar
        self._build_seek_bar(parent)

        # Snip list
        tk.Label(parent, text="SNIPS", font=FONT_B,
                 fg=C_CYAN, bg=C_PANEL, padx=8, pady=3).pack(fill=tk.X)
        self._vtree = _VirtualTree(parent, on_select=self._on_snip_select)

        # Status line (packed last → sits at bottom of nav column)
        self._status_var = tk.StringVar(value="ready")
        tk.Label(parent, textvariable=self._status_var,
                 font=FONT_SM, fg=C_DIM2, bg=C_PANEL,
                 anchor="w", padx=6, pady=3,
                 ).pack(fill=tk.X, side=tk.BOTTOM)

    def _build_seek_bar(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent, bg=C_BG)
        row.pack(fill=tk.X, padx=4, pady=3)

        tk.Label(row, text="SEEK", font=FONT_SM,
                 fg=C_DIM2, bg=C_BG).pack(side=tk.LEFT)

        self._seek_var = tk.StringVar()
        entry = tk.Entry(row, textvariable=self._seek_var, width=11,
                         bg="#06080d", fg=C_TEXT, insertbackground=C_TEXT,
                         font=FONT, relief=tk.FLAT, bd=3)
        entry.pack(side=tk.LEFT, padx=(6, 4))
        entry.bind("<Return>", lambda _: self._seek())

        ttk.Button(row, text="Go", command=self._seek,
                   style="Hi.TButton").pack(side=tk.LEFT)

    #  widget helpers
    def _sh(self, parent: tk.Frame, title: str) -> tk.Label:
        """Render a section header bar and return its counter label."""
        bar = tk.Frame(parent, bg=C_PANEL, bd=0)
        bar.pack(fill=tk.X)
        tk.Label(bar, text=title, font=FONT_B,
                 fg=C_CYAN, bg=C_PANEL, padx=8, pady=4).pack(side=tk.LEFT)
        counter = tk.Label(bar, text="", font=FONT_SM, fg=C_DIM2, bg=C_PANEL)
        counter.pack(side=tk.LEFT)
        return counter

    def _mk_tree(self, parent: tk.Frame, cols: tuple, hdrs: tuple,
                 wids: tuple, on_sel, height: int = 5) -> ttk.Treeview:
        """Build a scrollable Treeview inside its own grid frame."""
        frame = tk.Frame(parent, bg=C_BG)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        tree = ttk.Treeview(frame, columns=cols, show="headings",
                            selectmode="browse", height=height)
        for col, hdr, width in zip(cols, hdrs, wids):
            tree.heading(col, text=hdr)
            tree.column(col, width=width, anchor="w" if col == "label" else "e")

        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0,  column=1, sticky="ns")

        tree.bind("<<TreeviewSelect>>", on_sel)
        return tree

    # misc helpers
    def _status(self, msg: str, loading: bool = False) -> None:
        prefix = ">> " if loading else ""
        self._status_var.set(prefix + msg)

    def _set_meta(self, text: str) -> None:
        self._meta.configure(state=tk.NORMAL)
        self._meta.delete("1.0", tk.END)
        self._meta.insert(tk.END, text)
        self._meta.configure(state=tk.DISABLED)

    def _clr(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _pick_dir(self) -> None:
        chosen = filedialog.askdirectory(title="Select CAPPY data directory")
        if chosen:
            self.data_dir = Path(chosen)
            self.captures = self.data_dir / "captures"
            self._dir_var.set(chosen)
            self._refresh_sessions()

    def _iter_db(self):
        """Yield (db_path_str, day_dir) for every snips_*.sqlite found under captures/."""
        if not self.captures.exists():
            return
        for day_dir in sorted(self.captures.glob("*/*/*")):
            if not day_dir.is_dir():
                continue
            try:
                datetime.strptime(day_dir.name, "%Y-%m-%d")
            except ValueError:
                continue
            idx_dir = day_dir / "index"
            if not idx_dir.exists():
                continue
            for db in sorted(idx_dir.glob("snips_*.sqlite")):
                yield str(db), day_dir

    # level 0: sessions 
    def _refresh_sessions(self) -> None:
        self._cache.clear()
        self._db_map = {}
        db_paths = [p for p, _ in self._iter_db()]
        if not db_paths:
            self._status("no archive databases found"); return

        self._status("loading sessions…", loading=True)
        tok = self._tok()
        self._worker.submit(
            0, "list_sessions", db_paths, tok,
            lambda r: self.after(0, lambda: self._on_sessions(r, tok)),
        )

    def _on_sessions(self, r: dict, tok: int) -> None:
        if r.get("token") != tok:
            return
        rows = r.get("rows", [])
        self._clr(self._t_sess)
        if not rows:
            self._status("no sessions found"); return

        total_snips = 0
        for row in rows:
            sid    = str(row["session_id"])
            t0     = datetime.fromtimestamp(int(row["t0"] or 0) / 1e9, tz=self._tz)
            n      = int(row["n"])
            total_snips += n
            self._db_map[sid] = (row["db_path"], None)
            self._t_sess.insert("", tk.END, iid=f"sess_{sid}",
                values=(t0.strftime("%Y-%m-%d %H:%M:%S"), sid, f"{n:,}"))

        self._status(f"{len(rows)} sessions  ·  {total_snips:,} snips total")

    def _export_session_yaml_to_json(self) -> None:
        """Export the YAML settings associated with the selected session to a .json file."""
        sel = self._t_sess.selection()
        if not sel:
            messagebox.showinfo("Export", "Select a session first.")
            return
        vals = self._t_sess.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return
        sid = str(vals[1]).strip()
        if not sid or sid not in self._db_map:
            messagebox.showinfo("Export", "Session not found in map.")
            return

        db_path, day_dir = self._db_map[sid]

        # Try to find the YAML config used for this session.
        # Sessions typically store a run config alongside the data.
        yaml_settings = {}
        found_yaml = False

        # Search for run.yaml or config yaml in the day directory
        if day_dir:
            day_p = Path(str(day_dir))
            for yaml_pattern in ["*.run.yaml", "*.yaml", "*.yml"]:
                for yf in sorted(day_p.glob(yaml_pattern)):
                    try:
                        import yaml as _yaml
                        raw = yf.read_text(encoding="utf-8")
                        parsed = _yaml.safe_load(raw)
                        if isinstance(parsed, dict):
                            yaml_settings = parsed
                            found_yaml = True
                            break
                    except Exception:
                        continue
                if found_yaml:
                    break

        # If no YAML found on disk, try to reconstruct basic settings from the DB
        if not yaml_settings:
            try:
                import sqlite3 as _sq
                conn = _sq.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
                conn.row_factory = _sq.Row
                row = conn.execute(
                    "SELECT sample_rate_hz, n_samples, channels_mask "
                    "FROM snips WHERE session_id = ? LIMIT 1",
                    (sid,),
                ).fetchone()
                if row:
                    yaml_settings = {
                        "session_id": sid,
                        "sample_rate_hz": float(row["sample_rate_hz"] or 0),
                        "n_samples": int(row["n_samples"] or 0),
                        "channels_mask": str(row["channels_mask"] or "CHANNEL_A"),
                        "_note": "Reconstructed from DB — no YAML config file found for this session.",
                    }
                conn.close()
            except Exception:
                yaml_settings = {
                    "session_id": sid,
                    "_note": "No YAML config found and DB query failed.",
                }

        yaml_settings["_exported_session_id"] = sid
        yaml_settings["_exported_at"] = datetime.now().isoformat()

        # Ask user where to save
        import json as _json
        save_path = filedialog.asksaveasfilename(
            title="Save Session YAML as JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile=f"session_{sid}_settings.json",
        )
        if not save_path:
            return
        try:
            with open(save_path, "w", encoding="utf-8") as fh:
                _json.dump(yaml_settings, fh, indent=2, default=str)
            self._status(f"Exported session {sid} settings to {save_path}")
        except Exception as ex:
            messagebox.showerror("Export", f"Failed to save JSON:\n{ex}")

    def _on_session(self, _=None) -> None:
        sel = self._t_sess.selection()
        if not sel:
            return
        vals = self._t_sess.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return
        sid = str(vals[1]).strip()
        if not sid or sid not in self._db_map:
            return

        self._cur_sid = sid
        db_path, _   = self._db_map[sid]
        self._cur_db = db_path

        # resolve day_dir for this db (needed for blob reads)
        for p, ddir in self._iter_db():
            if p == db_path:
                self._cur_daydir        = str(ddir)
                self._db_map[sid]       = (db_path, ddir)
                break

        for tree in (self._t_day, self._t_hour, self._t_min):
            self._clr(tree)
        self._vtree.clear()
        self._cache.clear()

        self._status("loading days…", loading=True)
        tok = self._tok()
        self._worker.submit(
            1, "list_days",
            {"db_path": db_path, "session_id": sid}, tok,
            lambda r: self.after(0, lambda: self._on_days(r, tok)),
        )

    #  level 1: days 

    def _on_days(self, r: dict, tok: int) -> None:
        if r.get("token") != tok:
            return
        rows = r.get("rows", [])
        self._clr(self._t_day)
        if not rows:
            self._status("no days found"); return

        for row in rows:
            bkt = int(row["bucket"])
            lbl = datetime.fromtimestamp(bkt * _NS_PER_DAY / 1e9,
                                         tz=self._tz).strftime("%Y-%m-%d")
            self._t_day.insert("", tk.END, iid=f"day_{bkt}",
                values=(lbl, f"{int(row['n']):,}"))

        self._status(f"{len(rows)} days")
        self._autoselect(self._t_day)

    def _on_day(self, _=None) -> None:
        bkt = self._selected_bucket(self._t_day)
        if bkt is None:
            return
        for tree in (self._t_hour, self._t_min):
            self._clr(tree)
        self._vtree.clear()

        self._status("loading hours…", loading=True)
        tok = self._tok()
        self._worker.submit(
            1, "list_hours",
            {"db_path": self._cur_db, "session_id": self._cur_sid, "bucket": bkt}, tok,
            lambda r: self.after(0, lambda: self._on_hours(r, tok)),
        )

    # level 2: hours
    def _on_hours(self, r: dict, tok: int) -> None:
        if r.get("token") != tok:
            return
        rows = r.get("rows", [])
        self._clr(self._t_hour)
        if not rows:
            self._status("no hours found"); return

        for row in rows:
            bkt = int(row["bucket"])
            lbl = datetime.fromtimestamp(bkt * _NS_PER_HOUR / 1e9,
                                         tz=self._tz).strftime("%H:00")
            self._t_hour.insert("", tk.END, iid=f"hour_{bkt}",
                values=(lbl, f"{int(row['n']):,}"))

        self._status(f"{len(rows)} hours")
        self._autoselect(self._t_hour)

    def _on_hour(self, _=None) -> None:
        bkt = self._selected_bucket(self._t_hour)
        if bkt is None:
            return
        self._clr(self._t_min)
        self._vtree.clear()

        self._status("loading minutes…", loading=True)
        tok = self._tok()
        self._worker.submit(
            1, "list_minutes",
            {"db_path": self._cur_db, "session_id": self._cur_sid, "bucket": bkt}, tok,
            lambda r: self.after(0, lambda: self._on_minutes(r, tok)),
        )

    # level 3: minutes 

    def _on_minutes(self, r: dict, tok: int) -> None:
        if r.get("token") != tok:
            return
        rows = r.get("rows", [])
        self._clr(self._t_min)
        if not rows:
            self._status("no minutes found"); return

        for row in rows:
            bkt = int(row["bucket"])
            lbl = datetime.fromtimestamp(bkt * _NS_PER_MIN / 1e9,
                                         tz=self._tz).strftime(":%M")
            self._t_min.insert("", tk.END, iid=f"min_{bkt}",
                values=(lbl, f"{int(row['n']):,}"))

        self._status(f"{len(rows)} minutes")
        self._autoselect(self._t_min)

    def _on_minute(self, _=None) -> None:
        bkt = self._selected_bucket(self._t_min)
        if bkt is None:
            return
        self._vtree.clear()

        self._status("loading snips…", loading=True)
        tok = self._tok()
        self._worker.submit(
            1, "load_minute",
            {"db_path": self._cur_db, "session_id": self._cur_sid, "bucket": bkt}, tok,
            lambda r: self.after(0, lambda: self._on_minute_loaded(r, tok)),
        )

    def _on_minute_loaded(self, r: dict, tok: int) -> None:
        if r.get("token") != tok:
            return
        if "error" in r:
            self._status(f"error: {r['error']}"); return
        rows = r.get("rows", [])
        self._vtree.load(rows)
        self._status(f"{len(rows):,} snips")

    # level 4: waveform
    def _on_snip_select(self, snip_id: int, row_idx: int) -> None:
        self._cur_snip = snip_id

        cached = self._cache.get(snip_id)
        if cached is not None:
            wa, wb = cached
            self._display(wa, wb, self._vtree.get_by_id(snip_id))
            return

        if not (self._cur_db and self._cur_daydir):
            return

        tok = self._tok()
        self._worker.submit(
            0, "load_wave",
            {"db_path": self._cur_db, "snip_id": snip_id,
             "day_dir": self._cur_daydir}, tok,
            lambda r: self.after(0, lambda: self._on_wave(r, tok, snip_id)),
        )

    def _on_wave(self, r: dict, tok: int, snip_id: int) -> None:
        # discard if user already moved to a different snip
        if snip_id != self._cur_snip:
            return
        if "error" in r:
            self._set_meta(f"error loading waveform:\n{r['error']}"); return

        wa = r.get("wa")
        if wa is None:
            self._set_meta("no waveform data in this record"); return

        wb = r.get("wb")
        self._cache.put(snip_id, wa, wb)
        self._display(wa, wb, self._vtree.get_by_id(snip_id),
                      bA=r.get("baseline_A", 0.0),
                      bB=r.get("baseline_B", 0.0))

    def _display(self, wa: np.ndarray, wb, row,
                 bA: float = 0.0, bB: float = 0.0) -> None:
        sr   = float(row["sample_rate_hz"]) if row is not None else 1.0
        tsns = int(row["timestamp_ns"])     if row is not None else 0

        if sr <= 0 or not np.isfinite(sr):
            sr = 1.0

        # auto-baseline from first 64 samples if not provided
        if bA == 0.0 and len(wa) >= 64:
            bA = float(np.mean(wa[:64]))
        if bB == 0.0 and wb is not None and len(wb) >= 64:
            bB = float(np.mean(wb[:64]))

        self._wave_plot.plot(wa, wb, sr, tsns, bA=bA, bB=bB)

        if row is None:
            return

        ts_dt   = datetime.fromtimestamp(tsns / 1e9, tz=self._tz)
        ts_str  = ts_dt.strftime("%Y-%m-%d %H:%M:%S") + f".{tsns % 1_000_000_000:09d}"
        snip_id = int(row["id"])

        self._set_meta("\n".join([
            f"snip       #{snip_id}",
            f"timestamp   {ts_str}",
            f"session     {self._cur_sid or '?'}",
            f"buffer      {int(row['buffer_index'])}   "
                f"rec {int(row['record_in_buffer'])}   "
                f"global {int(row['record_global'])}",
            f"sample_rate {sr:.6g} Hz   pts {len(wa)}",
            f"area_A      {float(row['area_A_Vs']):.6g} V·s   "
                f"peak_A {float(row['peak_A_V']):.6g} V",
            f"baseline    A={bA:.6g} V   B={bB:.6g} V",
        ]))

    # seek 

    def _seek(self) -> None:
        raw = self._seek_var.get().strip()
        if not raw:
            return
        try:
            parts  = [int(x) for x in raw.replace("-", ":").split(":")]
            hh, mm, ss = {
                1: (0,        0,        parts[0]),
                2: (0,        parts[0], parts[1]),
            }.get(len(parts), (parts[0], parts[1], parts[2]))

            now    = datetime.now(tz=self._tz)
            target = now.replace(hour=hh % 24, minute=mm % 60,
                                 second=ss % 60, microsecond=0)
            tgt_ns = int(target.timestamp() * 1e9)

            if not self._vtree.seek_to_ns(tgt_ns):
                messagebox.showinfo("Seek",
                    "Select a minute window first, then seek within it.")
        except Exception as ex:
            messagebox.showerror("Seek", f"bad time string: {ex}")

    # draggy mode
    def _toggle_draggy(self) -> None:
        self._draggy_on = self._wave_plot.toggle_draggy()
        if self._draggy_on:
            self._draggy_btn_var.set("◈ Draggy  ON")
            self._draggy_btn.configure(style="Hi.TButton")
        else:
            self._draggy_btn_var.set("◇ Draggy")
            self._draggy_btn.configure(style="TButton")

    def _reset_wave_zoom(self) -> None:
        wp = self._wave_plot
        if wp._tv is None:
            return
        wp._drag_p1 = None
        wp._clear_drag_artists()
        wp._draggy_result_var.set("")
        x0, x1 = float(wp._tv[0]), float(wp._tv[-1])
        for ax in (wp.axA, wp.axB, wp.axI):
            ax.set_xlim(x0, x1)
        wp.canvas.draw_idle()

    #  pop-out compare

    def _pop_out_waveform(self) -> None:
        if self._wave_plot._last_wa is None:
            messagebox.showinfo("Compare",
                "Select a snip first to populate the waveform view.")
            return
        self._popout_count += 1
        title = (f"compare #{self._popout_count}"
                 f"  ·  session {self._cur_sid or '?'}"
                 f"  ·  snip {self._cur_snip or '?'}")
        self._wave_plot.pop_out(title=title)

    # private utils
  
    @staticmethod
    def _autoselect(tree: ttk.Treeview) -> None:
        """Select and fire the first row of a Treeview if it has children."""
        children = tree.get_children()
        if children:
            tree.selection_set(children[0])
            tree.event_generate("<<TreeviewSelect>>")

    @staticmethod
    def _selected_bucket(tree: ttk.Treeview) -> Optional[int]:
        """Return the integer bucket encoded in the selected row's iid, or None."""
        sel = tree.selection()
        if not sel:
            return None
        try:
            return int(sel[0].split("_")[1])
        except (IndexError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_archive_db(data_dir: Path) -> int:
    data_dir.mkdir(parents=True, exist_ok=True)

    root = tk.Tk()
    root.title("cappy.arch")
    root.configure(bg=C_BG)
    root.geometry("1440x920")
    root.minsize(900, 600)

    app = ArchiveDB(data_dir, master=root)
    app.pack(fill=tk.BOTH, expand=True)

    root.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    import sys
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataFile_ATS9462")
    raise SystemExit(run_archive_db(data_path))
