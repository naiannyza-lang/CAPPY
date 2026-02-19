#!/usr/bin/env python3
"""
==================================================================================
CAPPY DUAL-BOARD UNIFIED ACQUISITION SYSTEM - MINIMAL BUFFER FIX
==================================================================================

Simultaneously manages:
- ATS-9352 (System 2, Board 1) - 250 MS/s, 2-channel
- ATS-9462 (System 1, Board 1) - 180 MS/s, 2-channel

==================================================================================
"""
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
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
import pyarrow as pa
import pyarrow.parquet as pq

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont, scrolledtext

# Plotting
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import re
plt.style.use('dark_background')

# ==================================================================================
# GLOBAL CONFIGURATION
# ==================================================================================

# Color schemes - used ONLY for plot lines (not UI elements)
COLORS = {
    '9352': {
        'A': '#FF00EE',      # Pink for Channel A plots
        'B': '#26FF00',      # Green for Channel B plots
        'name': 'ATS-9352'
    },
    '9462': {
        'A': '#00BFFF',      # Blue for Channel A plots
        'B': '#FF8C00',      # Orange for Channel B plots
        'name': 'ATS-9462'
    }
}

# UI Theme - Clean and streamlined (NEW)
UI_COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_sidebar': '#252525',
    'bg_widget': '#2a2a2a',
    'button_bg': '#2f2f2f',
    'button_fg': 'white',
    'play_button_bg': '#3f3f3f',   # Grey background
    'play_button_fg': 'white',     # White text
    'separator': '#444444',
}


# Persistent app state (last-used YAML paths and data directories)
APP_STATE_DIR = Path.home() / ".cappy"
APP_STATE_PATH = APP_STATE_DIR / "cappy_state.json"
DEFAULT_CONFIG_9352_NAME = "config_9352.yaml"
DEFAULT_CONFIG_9462_NAME = "config_9462.yaml"

STOP_REQUESTED = False

def _signal_handler(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ATS API
ATS_AVAILABLE = False
ats = None
try:
    sys.path.append('/usr/local/AlazarTech/samples/Samples_Python/Library/')
    import atsapi as ats
    ATS_AVAILABLE = True
except Exception:
    ATS_AVAILABLE = False
    ats = None

# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def get_disk_usage(path: Union[str, Path]) -> float:
    """
    Get total size of all files in directory (amount written to disk).
    Returns: size in GB
    """
    try:
        path = Path(path)
        if not path.exists():
            return 0.0
        total_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_bytes / (1024**3)
    except Exception:
        return 0.0

def _codes_to_volts_u16(u16: np.ndarray, vpp: float) -> np.ndarray:
    """
    Convert uint16 ADC codes to volts.
    Universal formula for true 16-bit ADCs (both 9352 and 9462).
    """
    return (u16.astype(np.float32) - 32768.0) * (float(vpp) / 65536.0)

def _range_name_to_vpp(range_name: str, default_vpp: float = 4.0) -> float:
    """
    Convert Alazar range strings to Vpp.
    Examples: 'PM_1_V' → 2.0, 'PM_400_MV' → 0.8
    """
    rn = (range_name or "").strip().upper().replace("INPUT_RANGE_", "")
    
    if rn.startswith("PM_") and rn.endswith("_MV"):
        try:
            mv = float(rn[3:-3])
            return 2.0 * (mv / 1000.0)
        except Exception:
            return default_vpp
            
    if rn.startswith("PM_") and rn.endswith("_V"):
        try:
            v = float(rn[3:-2])
            return 2.0 * v
        except Exception:
            return default_vpp
    
    RANGES = {
        "PM_20_MV": 0.04, "PM_40_MV": 0.08, "PM_50_MV": 0.10,
        "PM_80_MV": 0.16, "PM_100_MV": 0.20, "PM_200_MV": 0.40,
        "PM_400_MV": 0.80, "PM_500_MV": 1.00, "PM_800_MV": 1.60,
        "PM_1_V": 2.00, "PM_2_V": 4.00, "PM_4_V": 8.00,
        "PM_5_V": 10.00, "PM_8_V": 16.00, "PM_10_V": 20.00,
        "PM_20_V": 40.00, "PM_40_V": 80.00,
    }
    return float(RANGES.get(rn, default_vpp))

def format_filesize(bytes_val: float) -> str:
    """Format bytes as human-readable"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.1f} PB"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


# ==================================================================================
# DEFAULT YAML CONFIGURATIONS
# ==================================================================================

DEFAULT_YAML_9352 = r"""# ATS-9352 Configuration
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
    range: PM_1_V
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
  timeout_ms: 0
  external_startcapture: false

timing:
  bunch_spacing_samples: 450

acquisition:
  channels_mask: CHANNEL_A|CHANNEL_B
  pre_trigger_samples: 0
  post_trigger_samples: 256
  records_per_buffer: 128
  buffers_allocated: 128
  buffers_per_acquisition: 0
  wait_timeout_ms: 5000

integration:
  baseline_window_samples: [0, 32]
  integral_window_samples: [32, 256]

waveforms:
  enable: true
  full_record: true
  mode: every_n
  every_n: 1
  threshold_integral_Vs: 0.0
  threshold_peak_V: 0.0
  max_waveforms_per_sec: 50
  store_volts: true
  dc_offset_correction: false

storage:
  data_dir: dataFile_ATS9352
  session_tag: ""
  rollover_minutes: 60
  session_rotate_hours: 24
  flush_every_records: 20000
  flush_every_seconds: 2
  sqlite_commit_every_snips: 200

runtime:
  readout_mode: TR
  noise_test: false
  autotrigger_timeout_ms: 10
"""

DEFAULT_YAML_9462 = r"""# ATS-9462 Configuration
board:
  system_id: 1
  board_id: 1

clock:
  source: INTERNAL_CLOCK
  sample_rate_msps: 180.0
  edge: CLOCK_EDGE_RISING

channels:
  A:
    coupling: DC
    range: PM_400_MV
    impedance: 50_OHM
  B:
    coupling: DC
    range: PM_400_MV
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
  ext_range: ETR_2V5
  delay_samples: 0
  timeout_ms: 0
  external_startcapture: false

timing:
  bunch_spacing_samples: 352

acquisition:
  channels_mask: CHANNEL_A|CHANNEL_B
  pre_trigger_samples: 0
  post_trigger_samples: 256
  records_per_buffer: 128
  buffers_allocated: 128
  buffers_per_acquisition: 0
  wait_timeout_ms: 5000

integration:
  baseline_window_samples: [0, 32]
  integral_window_samples: [32, 256]

waveforms:
  enable: true
  full_record: true
  mode: every_n
  every_n: 1
  threshold_integral_Vs: 0.0
  threshold_peak_V: 0.0
  max_waveforms_per_sec: 50
  store_volts: true
  dc_offset_correction: false

storage:
  data_dir: dataFile_ATS9462
  session_tag: ""
  rollover_minutes: 60
  session_rotate_hours: 24
  flush_every_records: 20000
  flush_every_seconds: 2
  sqlite_commit_every_snips: 200

runtime:
  readout_mode: NPT
  noise_test: true
  autotrigger_timeout_ms: 10
"""

# ==================================================================================
# DATA CLASSES
# ==================================================================================

@dataclass
class LiveRingWriter:
    """Circular buffer for live waveform display"""
    path: Path
    nslots: int
    npts: int
    
    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = np.zeros((self.nslots, self.npts), dtype=np.float32)
        self.index = 0
        
    def write(self, waveform: np.ndarray):
        """Write waveform to circular buffer"""
        if len(waveform) != self.npts:
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(waveform))
            x_new = np.linspace(0, 1, self.npts)
            f = interp1d(x_old, waveform, kind='linear')
            waveform = f(x_new)
        
        self.data[self.index] = waveform
        self.index = (self.index + 1) % self.nslots
        
        try:
            np.save(str(self.path), self.data)
        except Exception:
            pass

@dataclass
class AcquisitionStats:
    """Statistics for acquisition monitoring"""
    rate_hz: float = 0.0
    captures: int = 0
    started: str = ""
    last_capture: str = ""
    mean_peak_a: float = 0.0
    mean_peak_b: float = 0.0
    data_written_gb: float = 0.0  # CHANGED: Amount written, not space used

"""
==================================================================================
PART 2: BOARD ACQUISITION CLASS - FULLY CORRECTED
==================================================================================
BoardAcquisition class with ATS-9352 connection fix and data_written tracking.
"""

# ==================================================================================
# BOARD ACQUISITION CLASS
# ==================================================================================

# ============================================================================
# OPTION 1: EXPLICIT TRIGGER TIMEOUT - Helper Functions
# ============================================================================

class BeamState:
    """Beam state enumeration"""
    UNKNOWN = "unknown"
    ON = "on"
    OFF = "off"
    TRANSITIONING = "transitioning"


def init_trigger_timeout_tracking(self):
    """Initialize explicit trigger timeout buffer management."""
    self._last_trigger_time = time.time()
    self._trigger_timeout_ms = 5000.0
    self._trigger_signal_threshold_v = 0.5
    self._vpp_for_detection = 4.0
    
    self._beam_state = BeamState.UNKNOWN
    self._beam_off_detected = False
    self._last_state_transition = time.time()
    
    self._noise_test_mode = False
    self._noise_test_end_time = None
    
    self._trigger_count = 0
    self._queue_max_depth = 0
    self._flush_count = 0
    self._total_buffers_flushed = 0
    self._transition_time = None
    
    self._log("[TIMEOUT] Initialized explicit trigger timeout tracking")
    self._log(f"[TIMEOUT] Trigger timeout: {self._trigger_timeout_ms}ms")


def timeout_aware_backpressure(_write_queue, buffers_allocated, self):
    """Apply backpressure with explicit beam-off detection and synchronous flushing."""
    now = time.time()
    time_since_trigger = (now - self._last_trigger_time) * 1000
    
    is_beam_on = time_since_trigger <= self._trigger_timeout_ms
    
    if time_since_trigger > self._trigger_timeout_ms:
        # No triggers - beam is OFF
        if not self._beam_off_detected:
            self._beam_off_detected = True
            self._beam_state = BeamState.OFF
            self._transition_time = now
            
            self._log(f"[TIMEOUT] Beam OFF detected (no triggers for {time_since_trigger/1000:.2f}s)")
            self._log(f"[TIMEOUT] Queue at {len(_write_queue.queue)} items - SYNCHRONOUS FLUSH STARTING")
            
            # SYNCHRONOUS FLUSH
            flush_start = now
            max_flush_time = 5.0
            
            while len(_write_queue.queue) > 0 and (time.time() - flush_start) < max_flush_time:
                time.sleep(0.01)
                
                elapsed_flush = time.time() - flush_start
                if int(elapsed_flush * 10) % 5 == 0:
                    self._log(f"[TIMEOUT] Flushing... queue={len(_write_queue.queue):3d} "
                            f"elapsed={elapsed_flush:.2f}s")
            
            flush_end = time.time()
            flush_elapsed = flush_end - flush_start
            
            if len(_write_queue.queue) == 0:
                self._log(f"[TIMEOUT] ✓ Queue flushed successfully in {flush_elapsed:.2f}s")
            else:
                self._log(f"[TIMEOUT] ⚠ Queue not fully empty after {flush_elapsed:.2f}s, "
                        f"{len(_write_queue.queue)} items remaining")
            
            self._flush_count += 1
    else:
        # Beam ON
        if self._beam_off_detected:
            self._beam_off_detected = False
            self._beam_state = BeamState.ON
            self._transition_time = now
            self._log(f"[TIMEOUT] Beam ON resumed (triggers detected)")
    
    # Apply adaptive backpressure
    if self._noise_test_mode:
        max_queue_depth = 2
    elif self._beam_off_detected:
        max_queue_depth = 1
    else:
        max_queue_depth = buffers_allocated - 4
    
    current_depth = len(_write_queue.queue)
    if current_depth > self._queue_max_depth:
        self._queue_max_depth = current_depth
    
    while len(_write_queue.queue) > max_queue_depth:
        time.sleep(0.001)
        if self.paused or not self.running:
            break


def detect_trigger_with_state(self, raw_buffer_uint16):
    """Detect triggers and update beam state."""
    if raw_buffer_uint16 is None or len(raw_buffer_uint16) == 0:
        return False
    
    adc_offset = 32768
    max_adc_count = np.max(np.abs(raw_buffer_uint16.astype(np.float32) - adc_offset))
    max_voltage = (max_adc_count / 65536.0) * self._vpp_for_detection
    
    has_trigger = max_voltage >= self._trigger_signal_threshold_v
    
    if has_trigger:
        self._last_trigger_time = time.time()
        self._trigger_count += 1
    
    return has_trigger


def enable_noise_test_timeout(self, enable=True, duration_seconds=60):
    """Enable noise test mode - flush synchronously like v1.3."""
    self._noise_test_mode = enable
    
    if enable:
        self._log("[NOISE-TEST] ===== ENABLED =====")
        self._log("[NOISE-TEST] Flushing buffers synchronously (like v1.3)")
        self._log(f"[NOISE-TEST] Duration: {duration_seconds} seconds")
        self._log("[NOISE-TEST] Expected:")
        self._log("[NOISE-TEST]   - Queue limited to 2 items")
        self._log("[NOISE-TEST]   - Constant backpressure")
        self._log("[NOISE-TEST]   - Demonstrates v1.3 stability")
        self._log("[NOISE-TEST] ========================")
        
        if duration_seconds > 0:
            self._noise_test_end_time = time.time() + duration_seconds
        else:
            self._noise_test_end_time = None
        
        self._transition_time = time.time()
    else:
        self._log("[NOISE-TEST] DISABLED - returning to normal operation")


def check_noise_test_timeout_elapsed(self):
    """Check if noise test should auto-disable"""
    if self._noise_test_mode and self._noise_test_end_time is not None:
        if time.time() > self._noise_test_end_time:
            self._log("[NOISE-TEST] Timeout reached - disabling noise test mode")
            self._noise_test_mode = False


def log_timeout_statistics(self, buf_count, now=None):
    """Log detailed beam state and queue statistics."""
    if now is None:
        now = time.time()
    
    time_since_trigger = (now - self._last_trigger_time) * 1000
    
    state_str = self._beam_state.upper()
    if self._noise_test_mode:
        state_str = "NOISE_TEST"
    
    if time_since_trigger > self._trigger_timeout_ms:
        timeout_status = f"TIMEOUT ({time_since_trigger/1000:.1f}s)"
    else:
        timeout_status = f"OK ({time_since_trigger/1000:.1f}s)"
    
    self._log(
        f"[STATS] Buf={buf_count:7d} Triggers={self._trigger_count:6d} "
        f"State={state_str:10s} Timeout={timeout_status:15s} "
        f"MaxQ={self._queue_max_depth:2d} Flushes={self._flush_count}"
    )


def log_timeout_summary(self, elapsed_seconds):
    """Log summary statistics at end of acquisition"""
    self._log("\n" + "="*80)
    self._log("[TIMEOUT] SUMMARY STATISTICS")
    self._log("="*80)
    self._log(f"Duration:           {elapsed_seconds:.1f} seconds")
    self._log(f"Triggers detected:  {self._trigger_count}")
    self._log(f"Trigger rate:       {self._trigger_count/max(elapsed_seconds, 1):.1f} Hz")
    self._log(f"Queue flushes:      {self._flush_count}")
    self._log(f"Max queue depth:    {self._queue_max_depth}")
    self._log(f"Final beam state:   {self._beam_state}")
    self._log("="*80 + "\n")


# ============================================================================
# BoardAcquisition CLASS - WITH OPTION 1 INTEGRATION
# ============================================================================

class BoardAcquisition:
    """
    Handles data acquisition for a single board (9352 or 9462).
    Can be used for both board types with appropriate configuration.
    """
    
    def __init__(self, board_type: str, config_path: Path, gui_queue: queue.Queue):
        """
        Args:
            board_type: '9352' or '9462'
            config_path: Path to YAML configuration
            gui_queue: Queue for sending updates to GUI
        """
        self.board_type = board_type
        self.config_path = config_path
        self.gui_queue = gui_queue
        
        # State
        self.running = False
        self.paused = False
        self.thread = None
        
        # Configuration
        self.config = None
        self.board = None
        
        # Statistics
        self.stats = AcquisitionStats()

        # Persistent storage session (set on start)
        self._session_dir = None
        self._index_path = None
        self._save_every = 0
        
        # Acquisition parameters
        self.sample_rate_hz = 0.0
        self.vpp_A = 0.0
        self.vpp_B = 0.0
        self.channels_enabled = {'A': True, 'B': True}
        self.trigger_level = 128
        
    def load_config(self, config_path: Path = None):
        """Load configuration from YAML file"""
        if config_path:
            self.config_path = config_path
            
        if not self.config_path.exists():
            default_yaml = DEFAULT_YAML_9352 if self.board_type == '9352' else DEFAULT_YAML_9462
            self.config_path.write_text(default_yaml)
            
        self.config = yaml.safe_load(self.config_path.read_text())
        self._extract_parameters()
        
    def _extract_parameters(self):
        """Extract key parameters from config"""
        if not self.config:
            return
            
        clock_cfg = self.config.get('clock', {})
        self.sample_rate_hz = float(clock_cfg.get('sample_rate_msps', 250.0)) * 1e6
        
        channels_cfg = self.config.get('channels', {})
        if 'A' in channels_cfg:
            range_A = channels_cfg['A'].get('range', 'PM_1_V')
            self.vpp_A = _range_name_to_vpp(range_A)
        if 'B' in channels_cfg:
            range_B = channels_cfg['B'].get('range', 'PM_1_V')
            self.vpp_B = _range_name_to_vpp(range_B)
            
        trigger_cfg = self.config.get('trigger', {})
        self.trigger_level = int(trigger_cfg.get('levelJ', 128))
        
        acq_cfg = self.config.get('acquisition', {})
        channels_mask = str(acq_cfg.get('channels_mask', 'CHANNEL_A|CHANNEL_B'))
        self.channels_enabled['A'] = 'CHANNEL_A' in channels_mask
        self.channels_enabled['B'] = 'CHANNEL_B' in channels_mask
        
    def update_trigger_level(self, level: int):
        """Update trigger level dynamically"""
        self.trigger_level = max(0, min(255, level))
        
        if self.config:
            if 'trigger' not in self.config:
                self.config['trigger'] = {}
            self.config['trigger']['levelJ'] = self.trigger_level
            
        if self.board and self.running:
            try:
                self._reconfigure_trigger()
            except Exception as e:
                self._log(f"Error updating trigger level: {e}")
                
    def update_channel_b_enabled(self, enabled: bool):
        """Enable or disable Channel B"""
        self.channels_enabled['B'] = enabled
        
        if self.config:
            acq_cfg = self.config.get('acquisition', {})
            if enabled:
                acq_cfg['channels_mask'] = 'CHANNEL_A|CHANNEL_B'
            else:
                acq_cfg['channels_mask'] = 'CHANNEL_A'
                
        if self.running:
            self._log(f"Channel B {'enabled' if enabled else 'disabled'} - restarting acquisition...")
            self.stop()
            time.sleep(0.5)
            self.start()
            
    def start(self):
        """Start acquisition in background thread"""
        if self.running:
            self._log("Already running")
            return
            
        if not ATS_AVAILABLE:
            self._log("ERROR: ATS API not available")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()
        self._log(f"Started acquisition for {COLORS[self.board_type]['name']}")
        
    def stop(self):
        """Stop acquisition"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self._log(f"Stopped acquisition for {COLORS[self.board_type]['name']}")
        
    def pause(self):
        """Pause acquisition"""
        self.paused = True
        self._log("Paused")
        
    def resume(self):
        """Resume acquisition"""
        self.paused = False
        self._log("Resumed")
        
    def _log(self, message: str):
        """Send log message to GUI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        board_name = COLORS[self.board_type]['name']
        log_msg = f"[{timestamp}] [{board_name}] {message}"
        self.gui_queue.put(('log', log_msg))
        print(log_msg)
        
    def _send_stats_update(self):
        """Send statistics update to GUI"""
        # CHANGED: Calculate amount of data written (not disk space)
        if self.config:
            data_dir = Path(self.config.get('storage', {}).get('data_dir', '.'))
            self.stats.data_written_gb = get_disk_usage(data_dir)
            
        self.gui_queue.put(('stats', self.board_type, self.stats))
        
    def _send_waveform_update(self, wfA: np.ndarray, wfB: np.ndarray = None, 
                              integralA: float = 0.0, integralB: float = 0.0):
        """Send waveform data to GUI for live plotting"""
        wf_data = {
            'board': self.board_type,
            'wfA': wfA,
            'wfB': wfB,
            'integralA': integralA,
            'integralB': integralB,
            'time': time.time()
        }
        self.gui_queue.put(('waveform', wf_data))
        
    def _configure_board(self) -> bool:
        """Configure the board according to YAML settings"""
        try:
            board_cfg = self.config.get('board', {})
            system_id = int(board_cfg.get('system_id', 1))
            board_id = int(board_cfg.get('board_id', 1))
            
            self._log(f"Connecting to System {system_id}, Board {board_id}...")
            self.board = ats.Board(systemId=system_id, boardId=board_id)
            
            # Verify board connection (Alazar ATSAPI Python wrapper compatibility)
            # Older/official examples use getChannelInfo() which returns (maxSamplesPerChannel, bitsPerSample).
            try:
                _, bits = self.board.getChannelInfo()
                self._log(f"✓ Board connected: {bits} bits/sample")
            except Exception as e:
                # Fallbacks for wrapper variants
                bits = None
                try:
                    fn = getattr(self.board, "bitsPerSample", None)
                    if callable(fn):
                        bits = fn()
                except Exception:
                    bits = None

                if bits is None:
                    self._log(f"✗ Board connection failed: {e}")
                    return False

                self._log(f"✓ Board connected: {bits} bits/sample")
            
            # Configure clock
            clock_cfg = self.config.get('clock', {})
            sample_rate_id = self._get_sample_rate_id(
                float(clock_cfg.get('sample_rate_msps', 250.0))
            )
            
            self.board.setCaptureClock(
                ats.INTERNAL_CLOCK,
                sample_rate_id,
                ats.CLOCK_EDGE_RISING,
                0
            )
            
            # Configure channels
            channels_cfg = self.config.get('channels', {})
            for ch_name, ch_mask in [('A', ats.CHANNEL_A), ('B', ats.CHANNEL_B)]:
                if ch_name in channels_cfg:
                    ch_cfg = channels_cfg[ch_name]
                    
                    coupling_name = str(ch_cfg.get('coupling', 'DC'))
                    if not coupling_name.endswith('_COUPLING'):
                        coupling_name += '_COUPLING'
                    coupling = getattr(ats, coupling_name)
                    
                    range_name = str(ch_cfg.get('range', 'PM_1_V'))
                    input_range = getattr(ats, 'INPUT_RANGE_' + range_name)
                    
                    impedance_name = str(ch_cfg.get('impedance', '50_OHM'))
                    impedance = getattr(ats, 'IMPEDANCE_' + impedance_name)
                    
                    self.board.inputControlEx(ch_mask, coupling, input_range, impedance)
                    
                    # Set bandwidth limit (ATS-9462 specific)
                    if self.board_type == '9462':
                        try:
                            self.board.setBWLimit(ch_mask, 0)
                        except Exception as e:
                            self._log(f"Warning: setBWLimit failed: {e}")
                            
            # Configure trigger
            self._configure_trigger()
            
            # Configure AUX IO (ATS-9462 specific)
            if self.board_type == '9462':
                try:
                    self.board.configureAuxIO(ats.AUX_OUT_TRIGGER, 0)
                except Exception as e:
                    self._log(f"Warning: configureAuxIO failed: {e}")
                    
            self._log("Board configured successfully")
            return True
            
        except Exception as e:
            self._log(f"ERROR configuring board: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def _configure_trigger(self):
        """Configure trigger settings"""
        trigger_cfg = self.config.get('trigger', {})
        
        operation = getattr(ats, str(trigger_cfg.get('operation', 'TRIG_ENGINE_OP_J')))
        engine1 = getattr(ats, str(trigger_cfg.get('engine1', 'TRIG_ENGINE_J')))
        engine2 = getattr(ats, str(trigger_cfg.get('engine2', 'TRIG_ENGINE_K')))
        
        sourceJ = getattr(ats, str(trigger_cfg.get('sourceJ', 'TRIG_EXTERNAL')))
        slopeJ = getattr(ats, str(trigger_cfg.get('slopeJ', 'TRIGGER_SLOPE_POSITIVE')))
        levelJ = self.trigger_level
        
        sourceK = getattr(ats, str(trigger_cfg.get('sourceK', 'TRIG_DISABLE')))
        slopeK = getattr(ats, str(trigger_cfg.get('slopeK', 'TRIGGER_SLOPE_POSITIVE')))
        levelK = int(trigger_cfg.get('levelK', 128))
        
        self.board.setTriggerOperation(
            operation, engine1, sourceJ, slopeJ, levelJ,
            engine2, sourceK, slopeK, levelK
        )
        
        ext_coupling = getattr(ats, str(trigger_cfg.get('ext_coupling', 'DC_COUPLING')))
        ext_range = getattr(ats, str(trigger_cfg.get('ext_range', 'ETR_5V')))
        self.board.setExternalTrigger(ext_coupling, ext_range)
        
        self.board.setTriggerDelay(int(trigger_cfg.get('delay_samples', 0)))
        
        timeout_ms = int(trigger_cfg.get('timeout_ms', 0))
        runtime_cfg = self.config.get('runtime', {})
        if runtime_cfg.get('noise_test', False) and timeout_ms == 0:
            timeout_ms = int(runtime_cfg.get('autotrigger_timeout_ms', 10))
            
        self.board.setTriggerTimeOut(timeout_ms)
        
    def _reconfigure_trigger(self):
        """Reconfigure trigger without stopping acquisition"""
        if not self.board:
            return
        try:
            self._configure_trigger()
            self._log(f"Trigger level updated to {self.trigger_level}")
        except Exception as e:
            self._log(f"Error reconfiguring trigger: {e}")
            
    def _get_sample_rate_id(self, rate_msps: float) -> int:
        """Convert sample rate in MS/s to ATS constant"""
        rate_map = {
            1.0: ats.SAMPLE_RATE_1MSPS,
            2.0: ats.SAMPLE_RATE_2MSPS,
            5.0: ats.SAMPLE_RATE_5MSPS,
            10.0: ats.SAMPLE_RATE_10MSPS,
            20.0: ats.SAMPLE_RATE_20MSPS,
            50.0: ats.SAMPLE_RATE_50MSPS,
            100.0: ats.SAMPLE_RATE_100MSPS,
            125.0: ats.SAMPLE_RATE_125MSPS,
            160.0: ats.SAMPLE_RATE_160MSPS,
            180.0: ats.SAMPLE_RATE_180MSPS,
            200.0: ats.SAMPLE_RATE_200MSPS,
            250.0: ats.SAMPLE_RATE_250MSPS,
            500.0: ats.SAMPLE_RATE_500MSPS,
            1000.0: ats.SAMPLE_RATE_1000MSPS,
        }
        
        if rate_msps in rate_map:
            return rate_map[rate_msps]
        else:
            self._log(f"Warning: Unsupported sample rate {rate_msps} MS/s, using 250 MS/s")
            return ats.SAMPLE_RATE_250MSPS
            
    def _disk_writer_thread(self, write_queue: queue.Queue):
        """
        Background thread that handles all disk I/O so the acquisition loop
        is never blocked by storage latency.
        """
        while True:
            item = write_queue.get()
            if item is None:          # sentinel: shut down
                write_queue.task_done()
                break
            try:
                fpath, save_kwargs, index_path, index_line = item
                np.savez_compressed(fpath, **save_kwargs)
                if index_path is not None:
                    with open(index_path, "a", encoding="utf-8") as _fp:
                        _fp.write(index_line)
            except Exception as _se:
                self._log(f"Save warning (disk writer): {_se}")
            finally:
                write_queue.task_done()

    def _acquisition_loop(self):
        """
        Main acquisition loop - runs in background thread.

        KEY CHANGE vs original:
          • postAsyncBuffer() is called IMMEDIATELY after the raw copy so the
            board gets its buffer back before any processing or disk I/O.
          • All disk writes are handed off to a dedicated background thread via
            a bounded queue, so compression/fsync never stalls the loop.
          • buffers_allocated defaults to 64 (was 16) for extra headroom.
        """
        # Background disk-writer
        _write_queue: queue.Queue = queue.Queue(maxsize=256)
        _writer = threading.Thread(
            target=self._disk_writer_thread,
            args=(_write_queue,),
            daemon=True,
            name=f"DiskWriter-{self.board_type}",
        )
        _writer.start()

        # ============================================================================
        # OPTION 1: Trigger timeout tracking - INLINE VERSION
        # ============================================================================
        self._last_trigger_time = time.time()
        self._trigger_timeout_ms = 5000.0
        self._trigger_signal_threshold_v = 0.1  # ← Adjust if triggers not detected
        self._vpp_for_detection = 4.0
        self._trigger_count = 0
        self._queue_max_depth = 0
        self._flush_count = 0
        self._beam_off_detected = False
        self._acq_start_time = time.time()  # Track for summary logging

        try:
            if not self._configure_board():
                return
                
            acq_cfg = self.config.get('acquisition', {})
            pre_trigger = int(acq_cfg.get('pre_trigger_samples', 0))
            post_trigger = int(acq_cfg.get('post_trigger_samples', 256))
            samples_per_record = pre_trigger + post_trigger
            records_per_buffer = int(acq_cfg.get('records_per_buffer', 128))
            # Default bumped from 16 → 64; override in YAML via buffers_allocated
            buffers_allocated = int(acq_cfg.get('buffers_allocated', 64))
            
            channels_mask_str = str(acq_cfg.get('channels_mask', 'CHANNEL_A|CHANNEL_B'))
            ch_mask = 0
            ch_count = 0
            if 'CHANNEL_A' in channels_mask_str:
                ch_mask |= ats.CHANNEL_A
                ch_count += 1
            if 'CHANNEL_B' in channels_mask_str and self.channels_enabled['B']:
                ch_mask |= ats.CHANNEL_B
                ch_count += 1
                
            bytes_per_sample = 2
            bytes_per_buffer = bytes_per_sample * samples_per_record * records_per_buffer * ch_count
            
            buffers = []
            for i in range(buffers_allocated):
                buffers.append(ats.DMABuffer(self.board.handle, ctypes.c_uint16, bytes_per_buffer))
                
            runtime_cfg = self.config.get('runtime', {})
            readout_mode = str(runtime_cfg.get('readout_mode', 'TR')).strip().upper()
            use_npt_mode = readout_mode in ('NPT', 'NO_PRETRIGGER', 'NO-PRETRIGGER')
            
            if use_npt_mode:
                self._log("Using NPT mode (No Pre-Trigger)")
                adma_flags = ats.ADMA_NPT | ats.ADMA_EXTERNAL_STARTCAPTURE
                self.board.beforeAsyncRead(ch_mask, 0, samples_per_record, records_per_buffer, 0x7FFFFFFF, adma_flags)
            else:
                self._log("Using Traditional mode")
                # records_per_acquisition = 0x7FFFFFFF → run indefinitely.
                # Traditional mode arms itself via startCapture() below;
                # do NOT add ADMA_EXTERNAL_STARTCAPTURE here or the board
                # waits for a software arm signal that never comes.
                adma_flags = ats.ADMA_TRADITIONAL_MODE
                self.board.beforeAsyncRead(ch_mask, -pre_trigger, samples_per_record,
                                           records_per_buffer, 0x7FFFFFFF, adma_flags)
                
            for buf in buffers:
                self.board.postAsyncBuffer(buf.addr, buf.size_bytes)
                
            self.board.startCapture()
            self._log("Acquisition started")
            
            self.stats.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---- Persistent capture storage (auto-create) ----
            try:
                _dd = Path(self.config.get('storage', {}).get('data_dir', f"dataFile_ATS{self.board_type}")).expanduser()
                if not _dd.is_absolute():
                    _dd = (Path.cwd() / _dd).resolve()
                (_dd).mkdir(parents=True, exist_ok=True)
                _captures_root = _dd / "captures"
                _date = datetime.now().strftime("%Y-%m-%d")
                _hour = datetime.now().strftime("%H%M")
                _session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._session_dir = _captures_root / _date / _hour / _session_id
                self._session_dir.mkdir(parents=True, exist_ok=True)
                self._index_path = self._session_dir / "snips.csv"
                if not self._index_path.exists():
                    self._index_path.write_text(
                        "buf_index,timestamp_ns,sample_rate_hz,file,channels\n",
                        encoding="utf-8",
                    )
                self._save_every = int(self.config.get('storage', {}).get('save_every_buffers', 1) or 1)
                self._save_every = max(1, self._save_every)
                # Persist resolved data_dir back into config for archive viewer
                self.config.setdefault('storage', {})['data_dir'] = str(_dd)
                self._log(f"Saving captures under: {self._session_dir}")
            except Exception as _e:
                self._session_dir = None
                self._index_path = None
                self._save_every = 0
                self._log(f"Warning: capture saving disabled (could not init storage): {_e}")

            buf_count = 0
            last_stats_time = time.time()
            
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # ============================================================================
                # OPTION 1: Adaptive backpressure based on triggers - INLINE VERSION
                # ============================================================================
                now = time.time()
                time_since_trigger = (now - self._last_trigger_time) * 1000

                # Detect beam OFF (no triggers for 5 seconds)
                if time_since_trigger > self._trigger_timeout_ms and not self._beam_off_detected:
                    self._beam_off_detected = True
                    self._log(f"[OPTION1] Beam OFF detected (no triggers for {time_since_trigger/1000:.1f}s)")
                    
                    # Synchronously flush queue (like v1.3)
                    self._log(f"[OPTION1] Queue at {len(self._write_queue.queue)} items - SYNCHRONOUS FLUSH STARTING")
                    flush_start = now
                    while len(self._write_queue.queue) > 0 and (time.time() - flush_start) < 5.0:
                        time.sleep(0.01)
                        elapsed = time.time() - flush_start
                        if int(elapsed * 10) % 5 == 0:
                            self._log(f"[OPTION1] Flushing... queue={len(self._write_queue.queue)} elapsed={elapsed:.2f}s")
                    
                    flush_elapsed = time.time() - flush_start
                    if len(self._write_queue.queue) == 0:
                        self._log(f"[OPTION1] ✓ Queue flushed successfully in {flush_elapsed:.2f}s")
                    self._flush_count += 1

                # Back to beam ON
                elif time_since_trigger <= self._trigger_timeout_ms:
                    self._beam_off_detected = False

                # Apply adaptive backpressure
                if self._beam_off_detected:
                    max_queue_depth = buffers_allocated // 8  # Very strict during beam-off
                else:
                    max_queue_depth = buffers_allocated - 4  # Normal during beam-on

                # Track max depth
                current = len(self._write_queue.queue)
                if current > self._queue_max_depth:
                    self._queue_max_depth = current

                # Apply throttling
                while len(self._write_queue.queue) > max_queue_depth:
                    time.sleep(0.001)
                    if self.paused or not self.running:
                        break
                    
                try:
                    buf_index = buf_count % buffers_allocated
                    buf = buffers[buf_index]
                    
                    self.board.waitAsyncBufferComplete(buf.addr, 2000)
                    
                    # ── CRITICAL: copy raw data first, then immediately recycle
                    # the buffer back to the board BEFORE doing any processing.
                    # This is the primary fix for ApiBufferOverflow.
                    raw = buf.buffer.copy()
                    self.board.postAsyncBuffer(buf.addr, buf.size_bytes)  # ← moved up

                    # ── Process the copy (board is free to fill the buffer again)
                    if ch_count == 2:
                        A = raw[0::2].reshape(records_per_buffer, samples_per_record)
                        B = raw[1::2].reshape(records_per_buffer, samples_per_record)
                    else:
                        A = raw.reshape(records_per_buffer, samples_per_record)
                        B = None

                    # ── Stale-buffer guard ─────────────────────────────────────────
                    # An uninitialized DMA buffer contains all-zero uint16 words.
                    # After volt conversion, ADC code 0x0000 maps to exactly -Vpp/2
                    # (e.g. -0.4 V for PM_400_MV).  If every sample in record[0]
                    # equals that floor value the buffer hasn't been written by the
                    # board yet; skip it rather than saving garbage.
                    _floor_code = 0          # uint16 value for uninitialized memory
                    _stale = bool(np.all(A[0] == _floor_code))
                    if _stale:
                        buf_count += 1
                        continue
                    # ──────────────────────────────────────────────────────────────

                    wfA_volts = _codes_to_volts_u16(A[0], self.vpp_A)
                    wfB_volts = _codes_to_volts_u16(B[0], self.vpp_B) if B is not None else None

                    # Convert all records (needed for stats and optionally disk save)
                    A_volts_all = _codes_to_volts_u16(A, self.vpp_A)      # shape (R, S)
                    B_volts_all = _codes_to_volts_u16(B, self.vpp_B) if B is not None else None
                    
                    integralA = np.trapezoid(wfA_volts) / self.sample_rate_hz
                    integralB = np.trapezoid(wfB_volts) / self.sample_rate_hz if wfB_volts is not None else 0.0

                    # ── Enqueue disk save (non-blocking; handled by background thread)
                    if self._session_dir is not None and self._save_every > 0 and (buf_count % self._save_every == 0):
                        ts_ns = time.time_ns()
                        fname = f"buf_{buf_count:06d}.npz"
                        fpath = self._session_dir / fname
                        # Save ALL records in this buffer (shape: [records_per_buffer, samples])
                        if B_volts_all is None:
                            save_kwargs = dict(
                                wfA_V=A_volts_all.astype(np.float32, copy=False),
                                sample_rate_hz=float(self.sample_rate_hz),
                                timestamp_ns=int(ts_ns),
                                board=str(self.board_type),
                                channels="A",
                            )
                            chs = "A"
                        else:
                            save_kwargs = dict(
                                wfA_V=A_volts_all.astype(np.float32, copy=False),
                                wfB_V=B_volts_all.astype(np.float32, copy=False),
                                sample_rate_hz=float(self.sample_rate_hz),
                                timestamp_ns=int(ts_ns),
                                board=str(self.board_type),
                                channels="A|B",
                            )
                            chs = "A|B"
                        index_line = (
                            f"{buf_count},{ts_ns},{float(self.sample_rate_hz)},{fname},{chs}\n"
                            if self._index_path is not None else None
                        )
                        try:
                            _write_queue.put_nowait((fpath, save_kwargs, self._index_path, index_line))
                        except queue.Full:
                            self._log(f"Save warning: disk writer queue full at buf {buf_count}; "
                                      "disk too slow – increase save_every_buffers or use faster storage")

                    self.stats.captures = buf_count + 1
                    self.stats.last_capture = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Peak over ALL records in the buffer for a meaningful reading
                    self.stats.mean_peak_a = float(np.max(np.abs(A_volts_all)))
                    if B_volts_all is not None:
                        self.stats.mean_peak_b = float(np.max(np.abs(B_volts_all)))
                        
                    now = time.time()
                    if buf_count > 0 and now > last_stats_time:
                        self.stats.rate_hz = records_per_buffer / (now - last_stats_time)
                    last_stats_time = now
                    
                    self._send_waveform_update(wfA_volts, wfB_volts, integralA, integralB)
                    if buf_count % 10 == 0:
                        self._send_stats_update()
                        
                    buf_count += 1
                    
                    # ============================================================================
                    # ============================================================================
                    # OPTION 1: Trigger detection using configurable threshold - INLINE VERSION
                    # ============================================================================
                    raw_u16 = buf.buffer if hasattr(buf, 'buffer') else raw
                    
                    # Detect signal in buffer
                    max_adc_count = np.max(np.abs(raw_u16.astype(np.float32) - 32768))
                    max_voltage = (max_adc_count / 65536.0) * self._vpp_for_detection
                    has_trigger = max_voltage >= self._trigger_signal_threshold_v
                    
                    if has_trigger:
                        self._last_trigger_time = time.time()
                        self._trigger_count += 1
                        if self._beam_off_detected:
                            self._log("[OPTION1] Beam ON resumed")
                        self._beam_off_detected = False
                    
                    # Periodic statistics logging (every 100 buffers)
                    if buf_count % 100 == 0:
                        state = "ON" if (now - self._last_trigger_time)*1000 <= self._trigger_timeout_ms else "OFF"
                        self._log(f"[STATS] Buf={buf_count:7d} Triggers={self._trigger_count:6d} "
                                f"State={state:3s} MaxQ={self._queue_max_depth:2d} Flushes={self._flush_count}")
                    
                except Exception as e:
                    if "ApiWaitTimeout" in str(e):
                        continue
                    else:
                        self._log(f"Error in acquisition loop: {e}")
                        break
                        
        except Exception as e:
            self._log(f"Fatal error in acquisition: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            try:
                if self.board:
                    self.board.abortAsyncRead()
            except Exception:
                pass
            self.running = False
            self._log("Acquisition stopped")
            
            # ============================================================================
            # OPTION 1: Log summary statistics before cleanup
            # ============================================================================
            if hasattr(self, '_trigger_count'):
                acq_start_time = getattr(self, '_acq_start_time', time.time())
                elapsed = time.time() - acq_start_time
                log_timeout_summary(self, elapsed)
            
            # Gracefully drain and stop the disk writer
            try:
                _write_queue.put(None)   # sentinel
                _write_queue.join()      # wait for all pending writes to finish
            except Exception:
                pass

"""
==================================================================================
PART 3: GUI IMPLEMENTATION - FULLY CORRECTED
==================================================================================
Complete GUI with ALL improvements:
- Streamlined design
- White play button on grey
- Board names once at top
- Simplified titles
- Rolling integrals
- Trigger as percentage
- Disk as "GB written"
"""

# ==================================================================================
# GUI CLASSES
# ==================================================================================

class CollapsibleFrame(tk.Frame):
    """A frame that can be collapsed/expanded"""
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.show = tk.BooleanVar(value=True)
        
        title_frame = tk.Frame(self, bg='#333')
        title_frame.pack(fill=tk.X)
        
        self.toggle_btn = tk.Button(
            title_frame, text="▼", width=2,
            command=self.toggle,
            bg='#333', fg='white', relief=tk.FLAT, cursor='hand2'
        )
        self.toggle_btn.pack(side=tk.LEFT)
        
        tk.Label(
            title_frame, text=title,
            bg='#333', fg='white', font=('Arial', 10, 'bold')
        ).pack(side=tk.LEFT, padx=5)
        
        self.content = tk.Frame(self, bg=self['bg'])
        self.content.pack(fill=tk.BOTH, expand=True)
        
    def toggle(self):
        if self.show.get():
            self.content.pack_forget()
            self.toggle_btn.config(text="▶")
            self.show.set(False)
        else:
            self.content.pack(fill=tk.BOTH, expand=True)
            self.toggle_btn.config(text="▼")
            self.show.set(True)





class ArchiveViewer(tk.Toplevel):
    """Archive browser with A/B waveform plots + integral + metadata (scope-like).

    Expected on-disk layout (from this program):
      <data_dir>/captures/YYYY-MM-DD/HHMM/YYYYMMDD_HHMMSS/buf_000123.npz
    Each .npz should contain:
      - wfA_V (N,)
      - wfB_V (N,) optional
      - sample_rate_hz (scalar)
      - timestamp_ns (scalar) optional
      - board (str) optional
      - channels (str) optional
    """

    def __init__(self, master, title: str, data_dir: Path):
        super().__init__(master)
        self.title(title)
        self.configure(bg=UI_COLORS['bg_dark'])
        self.geometry("1180x760")
        self.minsize(980, 620)

        self.data_dir = Path(data_dir)
        self.captures_root = self.data_dir / "captures"
        self.filter_var = tk.StringVar(value="")
        self.date_var = tk.StringVar(value="")
        self.hour_var = tk.StringVar(value="")
        self.mmss_var = tk.StringVar(value="")
        self._sessions = []  # list[dict]
        self._snips = []     # list[Path]
        self._current_session = None

        # --- Top bar ---
        top = tk.Frame(self, bg=UI_COLORS['bg_medium'])
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text=str(self.data_dir), bg=UI_COLORS['bg_medium'], fg='white').pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(top, text="Filter:", bg=UI_COLORS['bg_medium'], fg='#ccc').pack(side=tk.LEFT, padx=(10, 4))
        ent = tk.Entry(top, textvariable=self.filter_var, bg='#222', fg='white', insertbackground='white', relief=tk.FLAT, width=32)
        ent.pack(side=tk.LEFT, pady=8)
        ent.bind("<KeyRelease>", lambda _e=None: self.refresh_sessions())

        tk.Button(
            top, text="Open Folder",
            command=self.open_folder,
            bg=UI_COLORS['button_bg'], fg='white',
            activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.RIGHT, padx=10, pady=6)

        tk.Button(
            top, text="Refresh",
            command=self.refresh_sessions,
            bg=UI_COLORS['button_bg'], fg='white',
            activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.RIGHT, padx=6, pady=6)

        # --- Body split ---
        body = tk.Frame(self, bg=UI_COLORS['bg_dark'])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=UI_COLORS['bg_dark'], width=360)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 6), pady=10)
        left.pack_propagate(False)

        right = tk.Frame(body, bg=UI_COLORS['bg_dark'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 10), pady=10)

        # --- Left panel: Sessions ---
        tk.Label(left, text="Sessions", bg=UI_COLORS['bg_dark'], fg='white', font=('Arial', 10, 'bold')).pack(anchor='w')
        self.session_list = tk.Listbox(left, bg='#1e1e1e', fg='white', height=10,
                                       activestyle='none', selectbackground='#444', highlightthickness=0)
        self.session_list.pack(fill=tk.X, pady=(6, 10))
        self.session_list.bind("<<ListboxSelect>>", self._on_select_session)

        # --- Jump controls ---
        jump = tk.Frame(left, bg=UI_COLORS['bg_dark'])
        jump.pack(fill=tk.X, pady=(0, 10))

        tk.Label(jump, text="Date", bg=UI_COLORS['bg_dark'], fg='#ccc').grid(row=0, column=0, sticky='w')
        tk.Entry(jump, textvariable=self.date_var, bg='#222', fg='white', insertbackground='white', relief=tk.FLAT, width=12).grid(row=1, column=0, sticky='w', pady=(2, 0))

        tk.Label(jump, text="Hour", bg=UI_COLORS['bg_dark'], fg='#ccc').grid(row=2, column=0, sticky='w', pady=(10, 0))
        tk.Entry(jump, textvariable=self.hour_var, bg='#222', fg='white', insertbackground='white', relief=tk.FLAT, width=5).grid(row=3, column=0, sticky='w', pady=(2, 0))

        tk.Label(jump, text="MM:SS", bg=UI_COLORS['bg_dark'], fg='#ccc').grid(row=3, column=1, sticky='w', padx=(8, 0))
        tk.Entry(jump, textvariable=self.mmss_var, bg='#222', fg='white', insertbackground='white', relief=tk.FLAT, width=6).grid(row=3, column=2, sticky='w', pady=(2, 0), padx=(4, 0))

        tk.Button(
            jump, text="Go",
            command=self._jump_to_time,
            bg=UI_COLORS['button_bg'], fg='white',
            activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).grid(row=3, column=3, sticky='w', padx=(10, 0), pady=(2, 0))

        for c in range(4):
            jump.grid_columnconfigure(c, weight=0)

        # --- Left panel: Snips ---
        tk.Label(left, text="Waveform snippets", bg=UI_COLORS['bg_dark'], fg='white', font=('Arial', 10, 'bold')).pack(anchor='w', pady=(6, 0))

        snip_frame = tk.Frame(left, bg=UI_COLORS['bg_dark'])
        snip_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self.snip_list = tk.Listbox(snip_frame, bg='#1e1e1e', fg='white',
                                    activestyle='none', selectbackground='#444', highlightthickness=0)
        self.snip_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        snip_sb = tk.Scrollbar(snip_frame, orient=tk.VERTICAL, command=self.snip_list.yview)
        snip_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.snip_list.configure(yscrollcommand=snip_sb.set)
        # <<ListboxSelect>> fires before curselection() is updated in some Tk versions.
        # ButtonRelease-1 fires after the click is fully committed → reliable single-click.
        self.snip_list.bind("<ButtonRelease-1>", lambda _e=None: self.preview_selected())
        self.snip_list.bind("<KeyRelease-Up>",   lambda _e=None: self.preview_selected())
        self.snip_list.bind("<KeyRelease-Down>",  lambda _e=None: self.preview_selected())

        # --- Right panel: plots + metadata ---
        self.fig = plt.Figure(figsize=(7, 6), dpi=100)

        # Layout: one combined waveform axis (A+B) on top, integral axis on bottom
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.1], hspace=0.35)
        self.axW = self.fig.add_subplot(gs[0:2, 0])   # combined waveform axis
        self.axI = self.fig.add_subplot(gs[2, 0])

        for ax in (self.axW, self.axI):
            ax.grid(True, alpha=0.25)

        self.axW.set_ylabel("Voltage (V)")
        self.axI.set_ylabel("Integral (V·s)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        meta_box = tk.Frame(right, bg=UI_COLORS['bg_dark'])
        meta_box.pack(fill=tk.X, pady=(8, 0))

        self.meta_text = scrolledtext.ScrolledText(
            meta_box, height=7, bg="#0a0a0a", fg="#ddd",
            font=("Courier", 9), wrap=tk.WORD
        )
        self.meta_text.pack(fill=tk.X, expand=False)

        btns = tk.Frame(right, bg=UI_COLORS['bg_dark'])
        btns.pack(fill=tk.X, pady=(8, 0))

        tk.Button(
            btns, text="Preview",
            command=self.preview_selected,
            bg=UI_COLORS['button_bg'], fg='white',
            activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=12
        ).pack(side=tk.LEFT)

        # Record navigator (for multi-record buffers)
        self._record_var = tk.IntVar(value=0)
        self._record_max = 0
        self._show_all_records = tk.BooleanVar(value=False)

        rec_frame = tk.Frame(btns, bg=UI_COLORS['bg_dark'])
        rec_frame.pack(side=tk.LEFT, padx=(16, 0))

        tk.Label(rec_frame, text="Record:", bg=UI_COLORS['bg_dark'], fg='#ccc').pack(side=tk.LEFT)
        self._rec_spin = tk.Spinbox(
            rec_frame, from_=0, to=0, width=5,
            textvariable=self._record_var,
            bg='#222', fg='white', insertbackground='white',
            buttonbackground='#333', relief=tk.FLAT,
            command=self.preview_selected
        )
        self._rec_spin.pack(side=tk.LEFT, padx=(4, 0))
        self._rec_label = tk.Label(rec_frame, text="/ 0", bg=UI_COLORS['bg_dark'], fg='#888')
        self._rec_label.pack(side=tk.LEFT, padx=(2, 8))

        tk.Checkbutton(
            rec_frame, text="All", variable=self._show_all_records,
            bg=UI_COLORS['bg_dark'], fg='#ccc', selectcolor='#333',
            activebackground=UI_COLORS['bg_dark'], activeforeground='white',
            command=self.preview_selected
        ).pack(side=tk.LEFT)

        tk.Button(
            btns, text="Close",
            command=self.destroy,
            bg=UI_COLORS['button_bg'], fg='white',
            activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=12
        ).pack(side=tk.RIGHT)

        self.refresh_sessions()

    def open_folder(self):
        try:
            subprocess.Popen(["xdg-open", str(self.data_dir)])
        except Exception:
            try:
                subprocess.Popen(["gio", "open", str(self.data_dir)])
            except Exception:
                pass

    def _iter_session_dirs(self):
        if not self.captures_root.exists():
            return []
        out = []
        # session dir contains buf_*.npz
        for p in self.captures_root.rglob("buf_*.npz"):
            out.append(p.parent)
        # unique, sorted by name
        uniq = sorted({d for d in out})
        return uniq

    def refresh_sessions(self):
        f = (self.filter_var.get() or "").strip().lower()
        self._sessions.clear()
        self.session_list.delete(0, tk.END)
        self.snip_list.delete(0, tk.END)
        self._snips = []
        self._current_session = None

        sess_dirs = self._iter_session_dirs()
        for d in sess_dirs:
            # d name like YYYYMMDD_HHMMSS
            sid = d.name
            # date inferred from parent structure if present
            date_str = d.parent.parent.name if d.parent.parent.name.count('-') == 2 else ""
            # snips count
            try:
                n = len(list(d.glob("buf_*.npz")))
            except Exception:
                n = 0

            label = f"{date_str} {sid[-6:-4]}:{sid[-4:-2]}:{sid[-2:]}  {sid}  snips={n}"
            if f and f not in label.lower():
                continue

            self._sessions.append({"dir": d, "sid": sid, "date": date_str, "n": n, "label": label})
            self.session_list.insert(tk.END, label)

        if self._sessions:
            self.session_list.selection_set(0)
            self._on_select_session()
        else:
            self.meta_text.delete("1.0", tk.END)
            self.meta_text.insert(tk.END, f"No sessions found under: {self.captures_root}\n")

    def _on_select_session(self, _evt=None):
        sel = self.session_list.curselection()
        if not sel or not self._sessions:
            return
        s = self._sessions[int(sel[0])]
        self._current_session = s
        d = Path(s["dir"])

        files = sorted(d.glob("buf_*.npz"))
        self._snips = files

        self.snip_list.delete(0, tk.END)
        for p in files:
            # show buf id only
            self.snip_list.insert(tk.END, p.name)

        if files:
            self.snip_list.selection_clear(0, tk.END)
            self.snip_list.selection_set(0)
            self.preview_selected()

    def _jump_to_time(self):
        # Best-effort: select first session matching date + hour.
        date = (self.date_var.get() or "").strip()
        hour = (self.hour_var.get() or "").strip()
        if not date or not hour:
            return
        # normalize hour
        try:
            hour_i = int(hour)
            hour = f"{hour_i:02d}"
        except Exception:
            hour = hour.zfill(2)[:2]

        for i, s in enumerate(self._sessions):
            if s.get("date") == date:
                # hour derived from captures path captures/YYYY-MM-DD/HHMM/...
                # HHMM folder is parent of session dir
                try:
                    hhmm = Path(s["dir"]).parent.name
                    if hhmm[:2] == hour:
                        self.session_list.selection_clear(0, tk.END)
                        self.session_list.selection_set(i)
                        self.session_list.see(i)
                        self._on_select_session()
                        return
                except Exception:
                    continue

    def _load_npz(self, p: Path):
        z = np.load(str(p), allow_pickle=True)
        def _get(name, default=None):
            return z[name] if name in z.files else default
        wfA = _get('wfA_V', None)
        wfB = _get('wfB_V', None)
        sr = float(_get('sample_rate_hz', 0.0) or 0.0)
        ts_ns = _get('timestamp_ns', None)
        board = _get('board', None)
        channels = _get('channels', None)

        # Return raw arrays (may be 1D or 2D); let preview_selected decide how to use them
        if wfA is not None:
            wfA = np.asarray(wfA, dtype=np.float32)
        if wfB is not None:
            wfB = np.asarray(wfB, dtype=np.float32)

        return wfA, wfB, sr, ts_ns, board, channels

    def preview_selected(self):
        sel = self.snip_list.curselection()
        if not sel or not self._snips:
            return
        p = self._snips[int(sel[0])]

        try:
            wfA, wfB, sr, ts_ns, board, channels = self._load_npz(p)
            if wfA is None or sr <= 0:
                raise ValueError("Missing wfA_V or sample_rate_hz")

            # ── Handle 1D (legacy single record) vs 2D (multi-record buffer) ──
            n_records = 1
            if wfA.ndim == 2:
                n_records = wfA.shape[0]
            if wfB is not None and wfB.ndim == 2:
                n_records = max(n_records, wfB.shape[0])

            # Update record spinbox range
            self._record_max = n_records - 1
            self._rec_spin.config(to=self._record_max)
            self._rec_label.config(text=f"/ {self._record_max}")
            rec_idx = max(0, min(int(self._record_var.get()), self._record_max))
            self._record_var.set(rec_idx)

            show_all = bool(self._show_all_records.get())

            # Extract the waveform(s) to plot
            wfA_2d = wfA if wfA.ndim == 2 else wfA.reshape(1, -1)
            wfB_2d = (wfB if wfB.ndim == 2 else wfB.reshape(1, -1)) if wfB is not None else None

            if show_all:
                # Overlay all records; each row is one trigger record
                wfA_rows = wfA_2d                         # shape (R, S)
                wfB_rows = wfB_2d                         # shape (R, S) or None
                wfA_plot  = wfA_2d.mean(axis=0)           # mean trace for integrals/stats
                wfB_plot  = wfB_2d.mean(axis=0) if wfB_2d is not None else None
                plot_title = f"{p.name}  [all {n_records} records]"
            else:
                wfA_rows = wfA_2d[rec_idx:rec_idx+1]
                wfB_rows = wfB_2d[rec_idx:rec_idx+1] if wfB_2d is not None else None
                wfA_plot  = wfA_2d[rec_idx]
                wfB_plot  = wfB_2d[rec_idx] if wfB_2d is not None else None
                plot_title = f"{p.name}  [record {rec_idx}/{self._record_max}]"

            wfA_plot = np.asarray(wfA_plot, dtype=np.float32)
            wfB_plot = np.asarray(wfB_plot, dtype=np.float32) if wfB_plot is not None else None

            # --- Time axis with auto units (based on samples per record) ---
            n_plot_samples = wfA_rows.shape[1]
            t_s = np.arange(n_plot_samples, dtype=np.float64) / float(sr)
            dur_s = float(t_s[-1]) if len(t_s) > 1 else 0.0
            if dur_s < 2e-6:
                t = t_s * 1e9;  t_unit = "ns"
            elif dur_s < 2e-3:
                t = t_s * 1e6;  t_unit = "µs"
            elif dur_s < 2.0:
                t = t_s * 1e3;  t_unit = "ms"
            else:
                t = t_s;        t_unit = "s"

            # --- Baseline subtraction ---
            def _baseline(wf: np.ndarray) -> float:
                n = len(wf)
                k = max(1, min(256, n // 20))
                return float(np.mean(wf[:k]))

            baseA = _baseline(wfA_plot)
            wfA0  = wfA_plot - baseA
            baseB = None;  wfB0 = None
            if wfB_plot is not None:
                baseB = _baseline(wfB_plot)
                wfB0  = wfB_plot - baseB

            # --- Cumulative integrals ---
            dt   = 1.0 / float(sr)
            intA = np.cumsum(wfA0) * dt
            intB = np.cumsum(wfB0) * dt if wfB0 is not None else None

            # --- Colors by board ---
            board_key = ("9352" if ("9352" in str(board).lower() or "9352" in self.title().lower())
                         else ("9462" if ("9462" in str(board).lower() or "9462" in self.title().lower())
                               else "9352"))
            colA = COLORS.get(board_key, COLORS["9352"])["A"]
            colB = COLORS.get(board_key, COLORS["9352"])["B"]

            # --- Plot waveforms: individual records semi-transparent + mean bold ---
            self.axW.clear()
            self.axW.grid(True, alpha=0.25)

            n_rows = wfA_rows.shape[0]
            alpha_ind = max(0.05, min(0.5, 1.2 / max(1, n_rows ** 0.6)))

            for i, row in enumerate(wfA_rows):
                self.axW.plot(t, row, color=colA, linewidth=0.6,
                              alpha=alpha_ind, label="_" if i else "Ch A (individual)")
            if wfB_rows is not None:
                for i, row in enumerate(wfB_rows):
                    self.axW.plot(t, row, color=colB, linewidth=0.6,
                                  alpha=alpha_ind, linestyle="--", label="_" if i else "Ch B (individual)")

            # Bold mean on top (only meaningful when show_all; single record it's identical)
            if n_rows > 1:
                self.axW.plot(t, wfA_plot, color=colA, linewidth=1.8, label="Ch A (mean)")
                if wfB_plot is not None:
                    self.axW.plot(t, wfB_plot, color=colB, linewidth=1.8,
                                  linestyle="--", label="Ch B (mean)")
            else:
                self.axW.plot(t, wfA_plot, color=colA, linewidth=1.4, label="Channel A")
                if wfB_plot is not None:
                    self.axW.plot(t, wfB_plot, color=colB, linewidth=1.4,
                                  linestyle="--", label="Channel B")

            self.axW.set_title(plot_title, fontsize=9)
            self.axW.set_ylabel("Voltage (V)")
            self.axW.set_xlabel(f"Time ({t_unit})")
            self.axW.legend(loc="upper right", fontsize=8, framealpha=0.85)

            # --- Plot integrals ---
            self.axI.clear()
            self.axI.grid(True, alpha=0.25)
            self.axI.plot(t, intA, color=colA, linewidth=0.8, label="∫A dt")
            if intB is not None:
                self.axI.plot(t, intB, color=colB, linewidth=0.8, linestyle="--", label="∫B dt")
            self.axI.set_ylabel("Integral (V·s)")
            self.axI.set_xlabel(f"Time ({t_unit})")
            self.axI.legend(loc='best', fontsize=8)

            # --- Metadata ---
            pkA  = float(np.max(wfA_plot) - np.min(wfA_plot))
            pkB  = float(np.max(wfB_plot) - np.min(wfB_plot)) if wfB_plot is not None else 0.0
            areaA = float(np.trapezoid(wfA0, dx=dt))
            areaB = float(np.trapezoid(wfB0, dx=dt)) if wfB0 is not None else 0.0

            self.meta_text.delete("1.0", tk.END)
            self.meta_text.insert(tk.END, f"File: {p}\n")
            if self._current_session:
                self.meta_text.insert(tk.END, f"Session: {self._current_session.get('sid','')}\n")
            if ts_ns is not None:
                self.meta_text.insert(tk.END, f"Timestamp_ns: {ts_ns}\n")
            if board is not None:
                self.meta_text.insert(tk.END, f"Board: {board}\n")
            if channels is not None:
                self.meta_text.insert(tk.END, f"Channels: {channels}\n")
            self.meta_text.insert(tk.END, f"Sample rate: {sr:.6g} Hz\n")
            self.meta_text.insert(tk.END, f"Records in buffer: {n_records}  |  Showing: {'all' if show_all else rec_idx}\n")
            self.meta_text.insert(tk.END, f"Samples shown: {len(wfA_plot)}\n")
            self.meta_text.insert(tk.END, f"Peak-to-peak A: {pkA:.6g} V\n")
            if wfB_plot is not None:
                self.meta_text.insert(tk.END, f"Peak-to-peak B: {pkB:.6g} V\n")
            self.meta_text.insert(tk.END, f"Baseline A: {baseA:.6g} V\n")
            if wfB_plot is not None and baseB is not None:
                self.meta_text.insert(tk.END, f"Baseline B: {baseB:.6g} V\n")
            self.meta_text.insert(tk.END, f"Area A (baseline-sub): {areaA:.6g} V·s\n")
            if wfB0 is not None:
                self.meta_text.insert(tk.END, f"Area B (baseline-sub): {areaB:.6g} V·s\n")

            self.canvas.draw_idle()

        except Exception as e:
            self.meta_text.delete("1.0", tk.END)
            self.meta_text.insert(tk.END, f"Failed to preview {p.name}: {e}\n")



class DualBoardGUI:
    """Main GUI - fully corrected with all improvements"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CAPPY Dual-Board Acquisition System")
        self.root.geometry("1800x1000")
        self.root.configure(bg=UI_COLORS['bg_dark'])
        
        self.running = False
        self.sidebar_visible = True
        self.sidebar_popout_win = None
        self._sidebar_main_container = None
        self._sidebar_toggle_btn = None
        self._board_running = {'9352': False, '9462': False}
        
        self.gui_queue = queue.Queue()
        self._log_buffer = []
        self.log_text = None
        
        self.acq_9352 = None
        self.acq_9462 = None
        
        self.config_path_9352 = Path("config_9352.yaml")
        self.config_path_9462 = Path("config_9462.yaml")


        # Load persisted UI state (last YAML/data dirs) and ensure defaults exist
        self._load_app_state_and_defaults()
        
        self.waveform_data = {
            '9352': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': [], 'streamA': np.empty((0,), dtype=np.float32), 'streamB': np.empty((0,), dtype=np.float32)},
            '9462': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': [], 'streamA': np.empty((0,), dtype=np.float32), 'streamB': np.empty((0,), dtype=np.float32)}
        }
        # Side-scrolling window (in points) for live waveform stream
        self.stream_window_pts = 20000
        self.max_plot_points = 1000
        
        # ADDED: Rolling time window for integrals
        self.integral_time_window = 20.0  # seconds
        self.integral_start_time = {}
        
        self.stats_labels = {}
        self.trigger_vars = {}
        self.channel_b_vars = {}
        
        self.setup_ui()
        self.update_from_queue()
        
    def setup_ui(self):
        """Build UI"""
        self.create_toolbar()
        
        main_container = tk.Frame(self.root, bg=UI_COLORS['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        self.create_sidebar(main_container)
        self.create_content_area(main_container)
        
    def create_toolbar(self):
        """Streamlined toolbar with white play button"""
        toolbar = tk.Frame(self.root, bg=UI_COLORS['bg_medium'], height=60)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        toolbar.pack_propagate(False)
        
        # Simple white play button on grey
        self.play_pause_btn = tk.Button(
            toolbar, text="▶ Start", 
            command=self.toggle_acquisition,
            bg=UI_COLORS['play_button_bg'],
            fg=UI_COLORS['play_button_fg'], 
            font=('Arial', 12, 'bold'),
            width=12, height=2,
            relief=tk.FLAT,
            cursor='hand2'
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=15, pady=10)

        # Controls toggle (for hidden sidebar)
        tk.Button(
            toolbar, text="Controls",
            command=self.toggle_sidebar,
            bg=UI_COLORS['button_bg'], fg=UI_COLORS['button_fg'], activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10, height=2
        ).pack(side=tk.LEFT, padx=(0, 10), pady=10)
        
        tk.Frame(toolbar, width=2, bg=UI_COLORS['separator']).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        # ATS-9352 - white label
        frame_9352 = tk.Frame(toolbar, bg=UI_COLORS['bg_medium'])
        frame_9352.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(
            frame_9352, text="ATS-9352", 
            bg=UI_COLORS['bg_medium'], fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.TOP)
        
        btn_frame_9352 = tk.Frame(frame_9352, bg=UI_COLORS['bg_medium'])
        btn_frame_9352.pack(side=tk.TOP)
        
        tk.Button(
            btn_frame_9352, text="Load YAML",
            command=lambda: self.load_yaml('9352'),
            bg=UI_COLORS['button_bg'], fg=UI_COLORS['button_fg'], activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame_9352, text="Data Dir",
            command=lambda: self.select_data_dir('9352'),
            bg=UI_COLORS['button_bg'], fg=UI_COLORS['button_fg'], activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Frame(toolbar, width=2, bg=UI_COLORS['separator']).pack(side=tk.LEFT, fill=tk.Y, padx=15)
        
        # ATS-9462 - white label
        frame_9462 = tk.Frame(toolbar, bg=UI_COLORS['bg_medium'])
        frame_9462.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(
            frame_9462, text="ATS-9462",
            bg=UI_COLORS['bg_medium'], fg='white',
            font=('Arial', 10, 'bold')
        ).pack(side=tk.TOP)
        
        btn_frame_9462 = tk.Frame(frame_9462, bg=UI_COLORS['bg_medium'])
        btn_frame_9462.pack(side=tk.TOP)
        
        tk.Button(
            btn_frame_9462, text="Load YAML",
            command=lambda: self.load_yaml('9462'),
            bg=UI_COLORS['button_bg'], fg=UI_COLORS['button_fg'], activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            btn_frame_9462, text="Data Dir",
            command=lambda: self.select_data_dir('9462'),
            bg=UI_COLORS['button_bg'], fg=UI_COLORS['button_fg'], activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.LEFT, padx=2)
        
    def create_sidebar(self, parent):
        """Create the main (docked) sidebar."""
        self.sidebar_frame = tk.Frame(parent, bg=UI_COLORS['bg_sidebar'], width=330)
        self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar_frame.pack_propagate(False)

        # Host where we can rebuild the sidebar contents (also reused for popout)
        self._sidebar_main_container = tk.Frame(self.sidebar_frame, bg=UI_COLORS['bg_sidebar'])
        self._sidebar_main_container.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar_contents(self._sidebar_main_container, mode="docked")

    def _build_sidebar_contents(self, parent, mode: str = "docked"):
        """Build sidebar UI into a given parent (docked or popout)."""
        for child in parent.winfo_children():
            child.destroy()

        header = tk.Frame(parent, bg=UI_COLORS['bg_sidebar'])
        header.pack(fill=tk.X)

        if mode == "docked":
            self._sidebar_toggle_btn = tk.Button(
                header, text="◀ Hide",
                command=self.toggle_sidebar,
                bg='#333', fg='white', font=('Arial', 9),
                relief=tk.FLAT, cursor='hand2'
            )
            self._sidebar_toggle_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

            pop_btn = tk.Button(
                header, text="Pop out",
                command=self.popout_sidebar,
                bg='#333', fg='white', font=('Arial', 9),
                relief=tk.FLAT, cursor='hand2'
            )
            pop_btn.pack(side=tk.RIGHT)
        else:
            dock_btn = tk.Button(
                header, text="Dock ◀",
                command=self.dock_sidebar,
                bg='#333', fg='white', font=('Arial', 9),
                relief=tk.FLAT, cursor='hand2'
            )
            dock_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Scroll area (with mouse wheel + width sync so popout isn't cropped)
        canvas = tk.Canvas(parent, bg=UI_COLORS['bg_sidebar'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=UI_COLORS['bg_sidebar'])

        win_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

        def _sync_scrollregion(_evt=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(_evt=None):
            try:
                canvas.itemconfigure(win_id, width=canvas.winfo_width())
            except Exception:
                pass

        scrollable_frame.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)

        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                delta = int(-1 * (event.delta / 120)) if hasattr(event, "delta") else 0
                if delta:
                    canvas.yview_scroll(delta, "units")

        for w in (canvas, scrollable_frame):
            w.bind("<MouseWheel>", _on_mousewheel)
            w.bind("<Button-4>", _on_mousewheel)
            w.bind("<Button-5>", _on_mousewheel)

        def _bind_all(_e=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_all(_e=None):
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            except Exception:
                pass

        canvas.bind("<Enter>", _bind_all)
        canvas.bind("<Leave>", _unbind_all)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Archives
        tk.Label(
            scrollable_frame, text="ARCHIVES",
            bg=UI_COLORS['bg_sidebar'], fg='white',
            font=('Arial', 11, 'bold')
        ).pack(pady=(10, 6))

        tk.Button(
            scrollable_frame, text="ATS-9352 Archive",
            command=lambda: self.open_archive('9352'),
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            font=('Arial', 10),
            relief=tk.FLAT, cursor='hand2',
            height=2
        ).pack(fill=tk.X, padx=10, pady=3)

        tk.Button(
            scrollable_frame, text="ATS-9462 Archive",
            command=lambda: self.open_archive('9462'),
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            font=('Arial', 10),
            relief=tk.FLAT, cursor='hand2',
            height=2
        ).pack(fill=tk.X, padx=10, pady=3)

        tk.Frame(scrollable_frame, height=2, bg='#555').pack(fill=tk.X, pady=10)

        # Per-board panels (Acquire / Vertical / Trigger / Stats)
        self.create_board_panel(scrollable_frame, '9352')
        tk.Frame(scrollable_frame, height=2, bg='#555').pack(fill=tk.X, pady=10)
        self.create_board_panel(scrollable_frame, '9462')

    def popout_sidebar(self):
        """Pop the side panel into a separate window."""
        try:
            if self.sidebar_popout_win is not None and self.sidebar_popout_win.winfo_exists():
                self.sidebar_popout_win.lift()
                self.sidebar_popout_win.focus_force()
                return
        except Exception:
            self.sidebar_popout_win = None

        self.sidebar_popout_win = tk.Toplevel(self.root)
        self.sidebar_popout_win.title("CAPPY Controls")
        self.sidebar_popout_win.geometry("420x950")
        self.sidebar_popout_win.minsize(380, 600)
        self.sidebar_popout_win.resizable(True, True)
        self.sidebar_popout_win.configure(bg=UI_COLORS['bg_sidebar'])

        pop_host = tk.Frame(self.sidebar_popout_win, bg=UI_COLORS['bg_sidebar'])
        pop_host.pack(fill=tk.BOTH, expand=True)
        self._build_sidebar_contents(pop_host, mode="popout")

        def _on_close():
            try:
                self.sidebar_popout_win.destroy()
            except Exception:
                pass
            self.sidebar_popout_win = None

        self.sidebar_popout_win.protocol("WM_DELETE_WINDOW", _on_close)

    def dock_sidebar(self):
        """Close the popout and ensure the docked sidebar is visible."""
        try:
            if self.sidebar_popout_win is not None and self.sidebar_popout_win.winfo_exists():
                self.sidebar_popout_win.destroy()
        except Exception:
            pass
        self.sidebar_popout_win = None

        if not self.sidebar_visible:
            self.toggle_sidebar()
        
    def create_board_panel(self, parent, board_type):
        """Side-panel controls (Acquire / Vertical / Trigger / Stats) for one board."""
        board_name = COLORS[board_type]['name']
        color_a = COLORS[board_type]['A']
        color_b = COLORS[board_type]['B']
        
        frame = CollapsibleFrame(parent, title=board_name, bg=UI_COLORS['bg_widget'])
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        content = frame.content

        # -------------------------
        # Acquire
        # -------------------------
        acquire = tk.LabelFrame(
            content, text="Acquire",
            bg=UI_COLORS['bg_widget'], fg='white',
            bd=1, relief=tk.GROOVE,
            font=('Arial', 10, 'bold'),
            labelanchor='nw'
        )
        acquire.pack(fill=tk.X, padx=10, pady=(8, 6))

        acq_row = tk.Frame(acquire, bg=UI_COLORS['bg_widget'])
        acq_row.pack(fill=tk.X, padx=8, pady=6)

        state_var = tk.StringVar(value="Stopped")
        setattr(self, f"_statevar_{board_type}", state_var)

        tk.Button(
            acq_row, text="Start",
            command=lambda bt=board_type: self.start_board(bt),
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', width=8
        ).pack(side=tk.LEFT)

        tk.Button(
            acq_row, text="Stop",
            command=lambda bt=board_type: self.stop_board(bt),
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', width=8
        ).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(
            acq_row, text="State",
            bg=UI_COLORS['bg_widget'], fg='#aaa',
            font=('Arial', 9)
        ).pack(side=tk.LEFT, padx=(12, 6))

        tk.Label(
            acq_row, textvariable=state_var,
            bg=UI_COLORS['bg_widget'], fg='white',
            font=('Arial', 9, 'bold')
        ).pack(side=tk.LEFT)

        # -------------------------
        # Vertical (Channel controls)
        # -------------------------
        vertical = tk.LabelFrame(
            content, text="Vertical",
            bg=UI_COLORS['bg_widget'], fg='white',
            bd=1, relief=tk.GROOVE,
            font=('Arial', 10, 'bold'),
            labelanchor='nw'
        )
        vertical.pack(fill=tk.X, padx=10, pady=6)

        # Range options (Alazar input ranges)
        range_opts = [
            ("20 mV", "PM_20_MV"), ("40 mV", "PM_40_MV"), ("50 mV", "PM_50_MV"),
            ("80 mV", "PM_80_MV"), ("100 mV", "PM_100_MV"), ("200 mV", "PM_200_MV"),
            ("400 mV", "PM_400_MV"), ("800 mV", "PM_800_MV"),
            ("1 V", "PM_1_V"), ("2 V", "PM_2_V"), ("4 V", "PM_4_V"),
        ]
        range_labels = [x[0] for x in range_opts]
        range_map = {lbl: val for lbl, val in range_opts}

        coupling_labels = ["DC", "AC"]
        impedance_labels = ["50 Ω", "1 MΩ"]
        impedance_map = {"50 Ω": "50_OHM", "1 MΩ": "1M_OHM"}

        def _channel_block(ch: str, title: str, title_fg: str):
            blk = tk.LabelFrame(
                vertical, text=title,
                bg=UI_COLORS['bg_widget'], fg=title_fg,
                bd=1, relief=tk.GROOVE,
                font=('Arial', 10, 'bold'),
                labelanchor='nw'
            )
            blk.pack(fill=tk.X, padx=8, pady=6)

            row1 = tk.Frame(blk, bg=UI_COLORS['bg_widget'])
            row1.pack(fill=tk.X, padx=6, pady=(4, 2))

            # Disabled
            disabled_var = tk.BooleanVar(value=False)
            setattr(self, f"_ch_disabled_{board_type}_{ch}", disabled_var)

            tk.Checkbutton(
                row1, text="Disabled",
                variable=disabled_var,
                command=lambda: self._apply_channel_disabled(board_type, ch, disabled_var.get()),
                bg=UI_COLORS['bg_widget'], fg='#ddd',
                selectcolor='#1e1e1e',
                activebackground=UI_COLORS['bg_widget'],
                activeforeground='#ddd',
                cursor='hand2'
            ).pack(side=tk.RIGHT)

            # Range
            tk.Label(row1, text="Range", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
            range_var = tk.StringVar(value=range_labels[-3])
            setattr(self, f"_ch_range_{board_type}_{ch}", range_var)

            om = ttk.OptionMenu(
                row1, range_var, range_var.get(), *range_labels,
                command=lambda _=None: self._apply_channel_setting(
                    board_type, ch, 'range', range_map.get(range_var.get(), 'PM_1_V')
                )
            )
            om.pack(side=tk.LEFT, padx=(6, 0))

            row2 = tk.Frame(blk, bg=UI_COLORS['bg_widget'])
            row2.pack(fill=tk.X, padx=6, pady=(2, 6))

            # Coupling
            tk.Label(row2, text="Coupling", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
            coup_var = tk.StringVar(value="DC")
            setattr(self, f"_ch_coup_{board_type}_{ch}", coup_var)
            ttk.OptionMenu(
                row2, coup_var, coup_var.get(), *coupling_labels,
                command=lambda _=None: self._apply_channel_setting(board_type, ch, 'coupling', coup_var.get())
            ).pack(side=tk.LEFT, padx=(6, 14))

            # Impedance
            tk.Label(row2, text="Impedance", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
            imp_var = tk.StringVar(value="50 Ω")
            setattr(self, f"_ch_imp_{board_type}_{ch}", imp_var)
            ttk.OptionMenu(
                row2, imp_var, imp_var.get(), *impedance_labels,
                command=lambda _=None: self._apply_channel_setting(
                    board_type, ch, 'impedance', impedance_map.get(imp_var.get(), '50_OHM')
                )
            ).pack(side=tk.LEFT, padx=(6, 0))

        _channel_block('A', 'Channel A', color_a)
        _channel_block('B', 'Channel B', color_b)

        # -------------------------
        # Trigger
        # -------------------------
        trigger_box = tk.LabelFrame(
            content, text="Trigger",
            bg=UI_COLORS['bg_widget'], fg='white',
            bd=1, relief=tk.GROOVE,
            font=('Arial', 10, 'bold'),
            labelanchor='nw'
        )
        trigger_box.pack(fill=tk.X, padx=10, pady=6)

        trig_row = tk.Frame(trigger_box, bg=UI_COLORS['bg_widget'])
        trig_row.pack(fill=tk.X, padx=8, pady=(6, 2))

        tk.Button(
            trig_row, text="Force",
            command=lambda: self._force_trigger(board_type),
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', width=8
        ).pack(side=tk.RIGHT)

        # Trigger source
        tk.Label(trig_row, text="Source", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
        src_labels = ["External", "Channel A", "Channel B"]
        src_map = {"External": "TRIG_EXTERNAL", "Channel A": "TRIG_CHAN_A", "Channel B": "TRIG_CHAN_B"}
        src_var = tk.StringVar(value="External")
        setattr(self, f"_trig_src_{board_type}", src_var)
        ttk.OptionMenu(
            trig_row, src_var, src_var.get(), *src_labels,
            command=lambda _=None: self._apply_trigger_setting(board_type, 'sourceJ', src_map.get(src_var.get(), 'TRIG_EXTERNAL'))
        ).pack(side=tk.LEFT, padx=(6, 12))

        # Trigger slope
        tk.Label(trig_row, text="Slope", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
        slope_labels = ["Positive", "Negative"]
        slope_map = {"Positive": "TRIGGER_SLOPE_POSITIVE", "Negative": "TRIGGER_SLOPE_NEGATIVE"}
        slope_var = tk.StringVar(value="Positive")
        setattr(self, f"_trig_slope_{board_type}", slope_var)
        ttk.OptionMenu(
            trig_row, slope_var, slope_var.get(), *slope_labels,
            command=lambda _=None: self._apply_trigger_setting(board_type, 'slopeJ', slope_map.get(slope_var.get(), 'TRIGGER_SLOPE_POSITIVE'))
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Readout mode (TR vs NPT)
        mode_row = tk.Frame(trigger_box, bg=UI_COLORS['bg_widget'])
        mode_row.pack(fill=tk.X, padx=8, pady=(0, 2))

        tk.Label(mode_row, text="Readout", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)
        readout_var = tk.StringVar(value="TR")
        setattr(self, f"_readout_{board_type}", readout_var)
        ttk.OptionMenu(
            mode_row, readout_var, readout_var.get(), "TR", "NPT",
            command=lambda _=None: self.update_readout_mode(board_type, readout_var.get())
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Level as percent
        lvl_row = tk.Frame(trigger_box, bg=UI_COLORS['bg_widget'])
        lvl_row.pack(fill=tk.X, padx=8, pady=(2, 6))

        tk.Label(lvl_row, text="Level (%)", bg=UI_COLORS['bg_widget'], fg='#aaa').pack(side=tk.LEFT)

        trigger_var = tk.DoubleVar(value=50.0)  # 50% = 128/255

        def update_trigger_from_percent(event=None):
            percent = float(trigger_var.get())
            level_255 = int(max(0.0, min(100.0, percent)) * 255.0 / 100.0)
            self.update_trigger_level(board_type, level_255)

        spinbox = tk.Spinbox(
            lvl_row,
            from_=0, to=100,
            textvariable=trigger_var,
            width=6, font=('Arial', 10),
            command=update_trigger_from_percent,
            increment=1.0
        )
        spinbox.pack(side=tk.LEFT, padx=(6, 3))
        spinbox.bind('<Return>', update_trigger_from_percent)

        tk.Label(lvl_row, text="%", bg=UI_COLORS['bg_widget'], fg='#aaa', font=('Arial', 10)).pack(side=tk.LEFT)

        self.trigger_vars[board_type] = trigger_var

        # -------------------------
        # Stats (kept, but visually grouped)
        # -------------------------
        stats_box = tk.LabelFrame(
            content, text="Stats",
            bg=UI_COLORS['bg_widget'], fg='white',
            bd=1, relief=tk.GROOVE,
            font=('Arial', 10, 'bold'),
            labelanchor='nw'
        )
        stats_box.pack(fill=tk.X, padx=10, pady=(6, 10))

        meta_frame = tk.Frame(stats_box, bg=UI_COLORS['bg_widget'])
        meta_frame.pack(fill=tk.X, padx=10, pady=6)
        
        labels = {}
        
        row = 0
        for label_text, key, default, fg_color in [
            ("Rate:", 'rate', "0 Hz", 'white'),
            ("Captures:", 'captures', "0", 'white'),
            ("Last:", 'last', "--", 'white'),
            ("Peak A:", 'peak_a', "0.000 V", color_a),
            ("Peak B:", 'peak_b', "0.000 V", color_b),
            ("Disk:", 'disk', "0.0 GB written", '#ffaa00'),  # CHANGED: "GB written"
        ]:
            tk.Label(meta_frame, text=label_text, bg=UI_COLORS['bg_widget'], fg='#aaa', anchor='w').grid(row=row, column=0, sticky='w', pady=2)
            labels[key] = tk.Label(meta_frame, text=default, bg=UI_COLORS['bg_widget'], fg=fg_color, anchor='w', font=('Arial', 9, 'bold'))
            labels[key].grid(row=row, column=1, sticky='w', padx=5, pady=2)
            row += 1
        
        self.stats_labels[board_type] = labels
        # Keep a boolean var around for programmatic enable/disable.
        self.channel_b_vars[board_type] = tk.BooleanVar(value=True)
        
    def create_content_area(self, parent):
        """Main content with tabs"""
        content_frame = tk.Frame(parent, bg=UI_COLORS['bg_dark'])
        content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=UI_COLORS['bg_dark'], borderwidth=0)
        style.configure('TNotebook.Tab', background=UI_COLORS['bg_medium'], foreground='white', padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', '#3d3d3d')])
        
        self.notebook = ttk.Notebook(content_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.create_overview_tab()
        self.create_log_tab()
        
    def create_overview_tab(self):
        """Overview - board name once, simplified titles, rolling integrals"""
        overview_frame = tk.Frame(self.notebook, bg=UI_COLORS['bg_dark'])
        self.notebook.add(overview_frame, text='Overview')
        
        self.fig = Figure(figsize=(14, 10), facecolor=UI_COLORS['bg_dark'])
        
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        self.ax_9352_A = self.fig.add_subplot(gs[0, 0])
        self.ax_9352_B = self.fig.add_subplot(gs[1, 0])
        self.ax_9352_int = self.fig.add_subplot(gs[2, 0])
        
        self.ax_9462_A = self.fig.add_subplot(gs[0, 1])
        self.ax_9462_B = self.fig.add_subplot(gs[1, 1])
        self.ax_9462_int = self.fig.add_subplot(gs[2, 1])
        
        # Board name ONCE at top only
        self.style_axis(self.ax_9352_A, "A (V)", COLORS['9352']['A'], board_name="ATS-9352")
        self.style_axis(self.ax_9352_B, "B (V)", COLORS['9352']['B'])
        self.style_axis(self.ax_9352_int, "Integral (V·s)", COLORS['9352']['A'])
        
        self.style_axis(self.ax_9462_A, "A (V)", COLORS['9462']['A'], board_name="ATS-9462")
        self.style_axis(self.ax_9462_B, "B (V)", COLORS['9462']['B'])
        self.style_axis(self.ax_9462_int, "Integral (V·s)", COLORS['9462']['A'])
        
        self.lines = {}
        self.lines['9352_A'], = self.ax_9352_A.plot([], [], color=COLORS['9352']['A'], linewidth=1.5)
        self.lines['9352_B'], = self.ax_9352_B.plot([], [], color=COLORS['9352']['B'], linewidth=1.5)
        self.lines['9352_intA'], = self.ax_9352_int.plot([], [], color=COLORS['9352']['A'], linewidth=2, label='Ch A')
        self.lines['9352_intB'], = self.ax_9352_int.plot([], [], color=COLORS['9352']['B'], linewidth=2, label='Ch B')
        self.ax_9352_int.legend(loc='upper left', fontsize=8)
        
        self.lines['9462_A'], = self.ax_9462_A.plot([], [], color=COLORS['9462']['A'], linewidth=1.5)
        self.lines['9462_B'], = self.ax_9462_B.plot([], [], color=COLORS['9462']['B'], linewidth=1.5)
        self.lines['9462_intA'], = self.ax_9462_int.plot([], [], color=COLORS['9462']['A'], linewidth=2, label='Ch A')
        self.lines['9462_intB'], = self.ax_9462_int.plot([], [], color=COLORS['9462']['B'], linewidth=2, label='Ch B')
        self.ax_9462_int.legend(loc='upper left', fontsize=8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=overview_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def style_axis(self, ax, ylabel, color, board_name=None):
        """Style axis - board name shown once at top"""
        ax.set_facecolor('#0a0a0a')
        ax.set_ylabel(ylabel, color=color, fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (µs)', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.15, color='white', linestyle='--', linewidth=0.5)
        
        if board_name:
            ax.set_title(board_name, color='white', fontsize=12, fontweight='bold', pad=10)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#555')
            spine.set_linewidth(0.5)
        
    def create_log_tab(self):
        """Log tab"""
        log_frame = tk.Frame(self.notebook, bg=UI_COLORS['bg_dark'])
        self.notebook.add(log_frame, text='Log')
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg='#0a0a0a', fg='#00ff00',
            font=('Courier', 9),
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_message("CAPPY Dual-Board System Initialized")
        self.log_message("-" * 80)

    # ==================================================================================
    # PERSISTED PATHS (YAML + DATA DIR) AND DEFAULT CREATION
    # ==================================================================================

    def _load_app_state_and_defaults(self) -> None:
        """Load last-used YAML/data dirs and create defaults if missing."""
        _ensure_dir(APP_STATE_DIR)
        state = _read_json(APP_STATE_PATH)

        p9352 = state.get("9352", {}).get("yaml_path")
        p9462 = state.get("9462", {}).get("yaml_path")

        self.config_path_9352 = Path(p9352) if p9352 else (APP_STATE_DIR / DEFAULT_CONFIG_9352_NAME)
        self.config_path_9462 = Path(p9462) if p9462 else (APP_STATE_DIR / DEFAULT_CONFIG_9462_NAME)

        if not self.config_path_9352.exists():
            self.config_path_9352.write_text(DEFAULT_YAML_9352, encoding="utf-8")
        if not self.config_path_9462.exists():
            self.config_path_9462.write_text(DEFAULT_YAML_9462, encoding="utf-8")

        # Preload configs so archive buttons work immediately
        try:
            if self.acq_9352 is None:
                self.acq_9352 = BoardAcquisition('9352', self.config_path_9352, self.gui_queue)
                self.acq_9352.load_config()
            if self.acq_9462 is None:
                self.acq_9462 = BoardAcquisition('9462', self.config_path_9462, self.gui_queue)
                self.acq_9462.load_config()
        except Exception:
            pass

        # Ensure data dirs exist
        for bt in ("9352", "9462"):
            acq = self._get_acq(bt)
            if not acq or not isinstance(acq.config, dict):
                continue
            data_dir = Path(acq.config.get("storage", {}).get("data_dir", str(APP_STATE_DIR / f"data_{bt}")))
            if not data_dir.is_absolute():
                data_dir = (Path.cwd() / data_dir).resolve()
            _ensure_dir(data_dir)
            acq.config.setdefault("storage", {})["data_dir"] = str(data_dir)

        self._save_app_state()

    def _save_app_state(self) -> None:
        state = _read_json(APP_STATE_PATH)
        for bt in ("9352", "9462"):
            acq = self._get_acq(bt)
            if not acq:
                continue
            state.setdefault(bt, {})
            state[bt]["yaml_path"] = str(self.config_path_9352 if bt == "9352" else self.config_path_9462)
            try:
                dd = Path(acq.config.get("storage", {}).get("data_dir", ""))
                state[bt]["data_dir"] = str(dd)
            except Exception:
                pass
        _write_json(APP_STATE_PATH, state)
    # ==================================================================================
    # EVENT HANDLERS
    # ==================================================================================

    def _get_acq(self, board_type: str) -> Optional[BoardAcquisition]:
        return self.acq_9352 if board_type == '9352' else self.acq_9462

    def _ensure_acq(self, board_type: str) -> Optional[BoardAcquisition]:
        # We allow config loading / archive browsing even if the ATS driver is unavailable.
        # Hardware access is only required when starting acquisition.

        if board_type == '9352':
            if self.acq_9352 is None:
                self.acq_9352 = BoardAcquisition('9352', self.config_path_9352, self.gui_queue)
                self.acq_9352.load_config()
            return self.acq_9352
        else:
            if self.acq_9462 is None:
                self.acq_9462 = BoardAcquisition('9462', self.config_path_9462, self.gui_queue)
                self.acq_9462.load_config()
            return self.acq_9462

    def start_board(self, board_type: str):
        if not ATS_AVAILABLE:
            messagebox.showerror("Error", "ATS API not available!")
            return
        acq = self._ensure_acq(board_type)
        if acq is None:
            messagebox.showerror("Error", "Failed to initialize board acquisition")
            return
        acq.start()
        self._board_running[board_type] = True
        sv = getattr(self, f"_statevar_{board_type}", None)
        if isinstance(sv, tk.StringVar):
            sv.set("Running")
        # Keep global state consistent for toolbar button
        self.running = any(self._board_running.values())
        if self.running:
            self.play_pause_btn.config(text="⏸ Pause", bg='#FF5722')

    def stop_board(self, board_type: str):
        acq = self._get_acq(board_type)
        if acq:
            acq.stop()
        self._board_running[board_type] = False
        sv = getattr(self, f"_statevar_{board_type}", None)
        if isinstance(sv, tk.StringVar):
            sv.set("Stopped")

        self.running = any(self._board_running.values())
        if not self.running:
            self.play_pause_btn.config(text="▶ Start", bg=UI_COLORS['play_button_bg'])

    def _write_config(self, board_type: str):
        acq = self._get_acq(board_type)
        if not acq or not acq.config:
            return
        try:
            cfg_path = acq.config_path
            cfg_path.write_text(yaml.safe_dump(acq.config, sort_keys=False))
        except Exception as e:
            self.log_message(f"Warning: failed to write YAML for {COLORS[board_type]['name']}: {e}")

    def _sync_sidebar_from_config(self, board_type: str):
        """Best-effort sync of the sidebar widgets from the loaded YAML."""
        acq = self._get_acq(board_type)
        if not acq or not acq.config:
            return
        cfg = acq.config
        ch_cfg = cfg.get('channels', {})
        for ch in ('A', 'B'):
            if ch not in ch_cfg:
                continue
            r = str(ch_cfg[ch].get('range', 'PM_1_V'))
            c = str(ch_cfg[ch].get('coupling', 'DC'))
            z = str(ch_cfg[ch].get('impedance', '50_OHM'))

            rv = getattr(self, f"_ch_range_{board_type}_{ch}", None)
            cv = getattr(self, f"_ch_coup_{board_type}_{ch}", None)
            zv = getattr(self, f"_ch_imp_{board_type}_{ch}", None)

            # map range code -> label if possible
            range_label = None
            for lbl, code in [
                ("20 mV", "PM_20_MV"), ("40 mV", "PM_40_MV"), ("50 mV", "PM_50_MV"),
                ("80 mV", "PM_80_MV"), ("100 mV", "PM_100_MV"), ("200 mV", "PM_200_MV"),
                ("400 mV", "PM_400_MV"), ("800 mV", "PM_800_MV"), ("1 V", "PM_1_V"),
                ("2 V", "PM_2_V"), ("4 V", "PM_4_V"),
            ]:
                if code == r:
                    range_label = lbl
                    break
            if isinstance(rv, tk.StringVar) and range_label is not None:
                rv.set(range_label)
            if isinstance(cv, tk.StringVar):
                cv.set("AC" if "AC" in c.upper() else "DC")
            if isinstance(zv, tk.StringVar):
                zv.set("1 MΩ" if "1M" in z.upper() else "50 Ω")

        # Trigger percent
        trig = cfg.get('trigger', {})
        level = int(trig.get('levelJ', 128))
        percent = max(0.0, min(100.0, level * 100.0 / 255.0))
        tv = self.trigger_vars.get(board_type)
        if isinstance(tv, tk.DoubleVar):
            tv.set(percent)

        # Channel mask -> B enabled
        acq_cfg = cfg.get('acquisition', {})
        mask = str(acq_cfg.get('channels_mask', 'CHANNEL_A|CHANNEL_B'))
        b_enabled = 'CHANNEL_B' in mask
        dv = getattr(self, f"_ch_disabled_{board_type}_B", None)
        if isinstance(dv, tk.BooleanVar):
            dv.set(not b_enabled)
    
    def toggle_sidebar(self):
        if self.sidebar_visible:
            self.sidebar_frame.pack_forget()
            self.sidebar_visible = False
        else:
            self.sidebar_frame.pack(side=tk.LEFT, fill=tk.Y, before=self.notebook.master)
            self.sidebar_visible = True

    def _apply_channel_setting(self, board_type: str, ch: str, key: str, value: str):
        acq = self._ensure_acq(board_type)
        if acq is None:
            return
        if acq.config is None:
            acq.load_config()
        acq.config.setdefault('channels', {}).setdefault(ch, {})[key] = value
        self._write_config(board_type)

        # If acquisition is running, restart to apply channel inputControlEx.
        if acq.running:
            self.log_message(f"Applying {COLORS[board_type]['name']} {ch} {key}={value} (restart)")
            self.stop_board(board_type)
            time.sleep(0.2)
            self.start_board(board_type)

    def _apply_channel_disabled(self, board_type: str, ch: str, disabled: bool):
        if ch == 'A' and disabled:
            messagebox.showwarning("Not supported", "Channel A cannot be disabled (used for timing/plotting).")
            dv = getattr(self, f"_ch_disabled_{board_type}_A", None)
            if isinstance(dv, tk.BooleanVar):
                dv.set(False)
            return

        if ch == 'B':
            # Channel B enable/disable is implemented via channels_mask.
            self.update_channel_b(board_type, not disabled)

    def _apply_trigger_setting(self, board_type: str, key: str, value: str):
        acq = self._ensure_acq(board_type)
        if acq is None:
            return
        if acq.config is None:
            acq.load_config()
        acq.config.setdefault('trigger', {})[key] = value
        self._write_config(board_type)

        if acq.running:
            try:
                acq._reconfigure_trigger()
            except Exception as e:
                self.log_message(f"Trigger update failed: {e}")

    def _force_trigger(self, board_type: str):
        acq = self._get_acq(board_type)
        if not acq or not acq.board:
            self.log_message(f"[{COLORS[board_type]['name']}] Force: board not connected")
            return
        # Some ATS models support forceTrigger; if not available, just log.
        try:
            if hasattr(acq.board, 'forceTrigger'):
                acq.board.forceTrigger()
                self.log_message(f"[{COLORS[board_type]['name']}] Forced trigger")
            else:
                self.log_message(f"[{COLORS[board_type]['name']}] Force trigger not supported by API")
        except Exception as e:
            self.log_message(f"[{COLORS[board_type]['name']}] Force trigger failed: {e}")
            
    def toggle_acquisition(self):
        if not self.running:
            self.start_acquisition()
        else:
            self.stop_acquisition()
            
    def start_acquisition(self):
        if not ATS_AVAILABLE:
            messagebox.showerror("Error", "ATS API not available!")
            return
            
        if self.acq_9352 is None:
            self.acq_9352 = BoardAcquisition('9352', self.config_path_9352, self.gui_queue)
            self.acq_9352.load_config()
            
        if self.acq_9462 is None:
            self.acq_9462 = BoardAcquisition('9462', self.config_path_9462, self.gui_queue)
            self.acq_9462.load_config()
            
        self.acq_9352.start()
        self.acq_9462.start()
        self._board_running['9352'] = True
        self._board_running['9462'] = True
        sv1 = getattr(self, "_statevar_9352", None)
        sv2 = getattr(self, "_statevar_9462", None)
        if isinstance(sv1, tk.StringVar):
            sv1.set("Running")
        if isinstance(sv2, tk.StringVar):
            sv2.set("Running")
        
        self.running = True
        self.play_pause_btn.config(
            text="⏸ Pause",
            bg='#FF5722'
        )
        self.log_message("=== ACQUISITION STARTED ===")
        
    def stop_acquisition(self):
        if self.acq_9352:
            self.acq_9352.stop()
        if self.acq_9462:
            self.acq_9462.stop()

        self._board_running['9352'] = False
        self._board_running['9462'] = False
        sv1 = getattr(self, "_statevar_9352", None)
        sv2 = getattr(self, "_statevar_9462", None)
        if isinstance(sv1, tk.StringVar):
            sv1.set("Stopped")
        if isinstance(sv2, tk.StringVar):
            sv2.set("Stopped")
            
        self.running = False
        self.play_pause_btn.config(
            text="▶ Start",
            bg=UI_COLORS['play_button_bg']
        )
        self.log_message("=== ACQUISITION STOPPED ===")
        
    def load_yaml(self, board_type):
        filename = filedialog.askopenfilename(
            title=f"Select {COLORS[board_type]['name']} Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        
        if filename:
            if board_type == '9352':
                self.config_path_9352 = Path(filename)
                if self.acq_9352:
                    self.acq_9352.load_config(self.config_path_9352)
            else:
                self.config_path_9462 = Path(filename)
                if self.acq_9462:
                    self.acq_9462.load_config(self.config_path_9462)
                    
            self.log_message(f"Loaded config for {COLORS[board_type]['name']}: {filename}")
            # Ensure storage dir exists and persist
            acq = self._get_acq(board_type)
            if acq and isinstance(acq.config, dict):
                dd = Path(acq.config.get('storage', {}).get('data_dir', '.'))
                if not dd.is_absolute():
                    dd = (Path.cwd() / dd).resolve()
                _ensure_dir(dd)
                acq.config.setdefault('storage', {})['data_dir'] = str(dd)
            self._save_app_state()
            self._sync_sidebar_from_config(board_type)
            
    def select_data_dir(self, board_type):
        directory = filedialog.askdirectory(
            title=f"Select {COLORS[board_type]['name']} Data Directory"
        )
        
        if directory:
            if board_type == '9352' and self.acq_9352:
                self.acq_9352.config['storage']['data_dir'] = directory
            elif board_type == '9462' and self.acq_9462:
                self.acq_9462.config['storage']['data_dir'] = directory
                
            dd = Path(directory)
            _ensure_dir(dd)
            self.log_message(f"Data directory for {COLORS[board_type]['name']}: {dd}")
            self._save_app_state()
            
    def update_trigger_level(self, board_type, level):
        acq = self.acq_9352 if board_type == '9352' else self.acq_9462
        if acq:
            acq.update_trigger_level(level)
            
    def update_channel_b(self, board_type, enabled):
        acq = self.acq_9352 if board_type == '9352' else self.acq_9462
        if acq:
            acq.update_channel_b_enabled(enabled)
        dv = getattr(self, f"_ch_disabled_{board_type}_B", None)
        if isinstance(dv, tk.BooleanVar):
            dv.set(not enabled)
            
    
    def update_readout_mode(self, board_type: str, mode: str):
        """Set TR vs NPT mode (persists to YAML; restarts board if running)."""
        mode_u = str(mode).strip().upper()
        if mode_u not in ("TR", "NPT"):
            mode_u = "TR"

        acq = self._ensure_acq(board_type)
        if acq is None:
            return
        if acq.config is None:
            acq.load_config()

        acq.config.setdefault('runtime', {})['readout_mode'] = mode_u
        self._write_config(board_type)

        if acq.running:
            self.log_message(f"Applying {COLORS[board_type]['name']} readout_mode={mode_u} (restart)")
            self.stop_board(board_type)
            time.sleep(0.2)
            self.start_board(board_type)

    def open_archive(self, board_type: str):
        """Open the archive viewer for a given board ('9352' or '9462')."""
        board_type = str(board_type).strip()
        if board_type not in ("9352", "9462"):
            messagebox.showwarning("Warning", f"Unknown board type: {board_type}")
            return

        acq = self._ensure_acq(board_type)
        if acq is None:
            messagebox.showwarning("Warning", "Acquisition object not available for this board")
            return
        if acq.config is None:
            acq.load_config()

        data_dir = Path(acq.config.get('storage', {}).get('data_dir', '.')).expanduser()
        # If relative, anchor to ~/.cappy so it is stable across launches
        if not data_dir.is_absolute():
            data_dir = (Path.home() / ".cappy" / data_dir).resolve()

        # Ensure directory exists
        try:
            data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not create/open data directory:\n{data_dir}\n{e}")
            return

        # Persist normalized path back into config/state so it stays attached
        acq.config.setdefault('storage', {})['data_dir'] = str(data_dir)
        self._write_config(board_type)

        if not hasattr(self, "_archive_windows"):
            self._archive_windows = {}

        # Reuse an existing viewer per board
        win = self._archive_windows.get(board_type)
        try:
            if win is not None and win.winfo_exists():
                win.set_data_dir(data_dir) if hasattr(win, "set_data_dir") else None
                win.lift()
                win.focus_force()
                return
        except Exception:
            pass

        title = f"{COLORS[board_type]['name']} Archive"
        win = ArchiveViewer(self.root, title=title, data_dir=data_dir)
        self._archive_windows[board_type] = win
    def log_message(self, message: str):
        """Thread-safe-ish logger: prints to terminal and writes to the Log tab if available."""
        try:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        except Exception:
            ts = ''
        line = f"[{ts}] {message}" if ts else str(message)
        # Always print so you have logs even if GUI isn't ready yet
        try:
            print(line)
        except Exception:
            pass

        # If log widget not created yet, buffer messages
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = []
        if not hasattr(self, 'log_text') or self.log_text is None:
            self._log_buffer.append(line)
            # prevent unbounded growth
            if len(self._log_buffer) > 2000:
                self._log_buffer = self._log_buffer[-1000:]
            return

        # Flush buffered lines first
        if self._log_buffer:
            try:
                for b in self._log_buffer:
                    self.log_text.insert(tk.END, b + "\n")
            except Exception:
                pass
            self._log_buffer.clear()

        try:
            self.log_text.insert(tk.END, line + "\n")
            self.log_text.see(tk.END)
        except Exception:
            # widget may be destroyed during shutdown
            pass
    def update_from_queue(self):
        try:
            while True:
                msg = self.gui_queue.get_nowait()
                msg_type = msg[0]
            
                if msg_type == 'log':
                    self.log_message(msg[1])
                
                elif msg_type == 'stats':
                    board_type = msg[1]
                    stats = msg[2]
                    self.update_stats_display(board_type, stats)
                
                elif msg_type == 'waveform':
                    self.update_waveform_display(msg[1])
                
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.update_from_queue)
        
    def update_stats_display(self, board_type, stats: AcquisitionStats):
        if board_type not in self.stats_labels:
            return
        
        labels = self.stats_labels[board_type]
    
        labels['rate'].config(text=f"{stats.rate_hz:.1f} Hz")
        labels['captures'].config(text=f"{stats.captures}")
        labels['last'].config(text=stats.last_capture.split()[1] if stats.last_capture else "--")
        labels['peak_a'].config(text=f"{stats.mean_peak_a:.6f} V")
        labels['peak_b'].config(text=f"{stats.mean_peak_b:.6f} V")
        labels['disk'].config(text=f"{stats.data_written_gb:.1f} GB written")  # CHANGED
    
    def update_waveform_display(self, wf_data):
        board_type = wf_data['board']
        wfA = wf_data['wfA']
        wfB = wf_data.get('wfB', None)
        integralA = float(wf_data.get('integralA', 0.0))
        integralB = float(wf_data.get('integralB', 0.0))
        timestamp = float(wf_data.get('time', time.time()))

        data = self.waveform_data[board_type]

        # Keep a short history of most recent *buffers*
        data['A'].append(wfA)
        data['B'].append(wfB)  # keep None if B not present
        data['intA'].append(integralA)
        data['intB'].append(integralB)
        data['time'].append(timestamp)

        if len(data['A']) > self.max_plot_points:
            data['A'].pop(0)
            data['B'].pop(0)
            data['intA'].pop(0)
            data['intB'].pop(0)
            data['time'].pop(0)

        # ---- Side-scrolling live stream (concatenate each buffer's representative waveform) ----
        try:
            data['streamA'] = np.concatenate((data['streamA'], np.asarray(wfA, dtype=np.float32, order='C')))
            if wfB is not None:
                data['streamB'] = np.concatenate((data['streamB'], np.asarray(wfB, dtype=np.float32, order='C')))
            else:
                data['streamB'] = np.concatenate((data['streamB'], np.full((len(wfA),), np.nan, dtype=np.float32)))

            if data['streamA'].size > self.stream_window_pts:
                data['streamA'] = data['streamA'][-self.stream_window_pts:]
            if data['streamB'].size > self.stream_window_pts:
                data['streamB'] = data['streamB'][-self.stream_window_pts:]
        except Exception:
            pass

        # Refresh plots more responsively (throttled)
        # FIX #4: Reduced from 0.05 (50ms) to 0.1 (100ms) to reduce matplotlib overhead
        # This halves plotting CPU usage while maintaining responsive visual feedback
        now = time.time()
        if not hasattr(self, '_last_plot_time'):
            self._last_plot_time = 0.0
        if (now - self._last_plot_time) >= 0.1:
            self._last_plot_time = now
            self.refresh_plots()

    def refresh_plots(self):
        """Redraw plots with ROLLING TIME for integrals"""
        for board_type in ['9352', '9462']:
            data = self.waveform_data[board_type]
        
            if not data['A']:
                continue
            
            # Use concatenated stream for v1_3-style side scrolling when available
            sample_rate = 250e6 if board_type == '9352' else 180e6
            streamA = data.get('streamA', None)
            streamB = data.get('streamB', None)

            if isinstance(streamA, np.ndarray) and streamA.size > 0:
                wfA_plot = streamA
                wfB_plot = streamB if isinstance(streamB, np.ndarray) and streamB.size == streamA.size else None
            else:
                wfA_plot = data['A'][-1]
                wfB_plot = data['B'][-1]

            # Time axis for the displayed window
            time_us = np.arange(len(wfA_plot)) / sample_rate * 1e6
            # Keep axis readable: show time relative to left edge of the window
            if time_us.size > 0:
                time_us = time_us - time_us[0]

            prefix = f"{board_type}_"
            self.lines[prefix + 'A'].set_data(time_us, wfA_plot)

            if wfB_plot is None:
                # Hide B when not present to avoid a misleading flat line
                self.lines[prefix + 'B'].set_data([], [])
            else:
                self.lines[prefix + 'B'].set_data(time_us, wfB_plot)

        
            # ROLLING TIME for integrals
            if len(data['time']) > 0:
                if board_type not in self.integral_start_time:
                    self.integral_start_time[board_type] = data['time'][0]
            
                int_times = np.array(data['time']) - self.integral_start_time[board_type]
            
                current_time = int_times[-1]
                window_start = max(0, current_time - self.integral_time_window)
            
                mask = int_times >= window_start
                int_times_windowed = int_times[mask]
                intA_windowed = np.array(data['intA'])[mask]
                intB_windowed = np.array(data['intB'])[mask]
            
                self.lines[prefix + 'intA'].set_data(int_times_windowed, intA_windowed)
                self.lines[prefix + 'intB'].set_data(int_times_windowed, intB_windowed)
            
                ax_int = self.ax_9352_int if board_type == '9352' else self.ax_9462_int
                # Guard against identical x-limits (can happen at startup)
                if current_time <= window_start:
                    current_time = window_start + 1e-6
                ax_int.set_xlim(window_start, current_time)
                ax_int.set_xlabel('Time (s)', color='white', fontsize=9)
        
            ax_map = {
                '9352': (self.ax_9352_A, self.ax_9352_B),
                '9462': (self.ax_9462_A, self.ax_9462_B)
            }
        
            for ax in ax_map[board_type]:
                ax.relim()
                ax.autoscale_view()
            
        self.canvas.draw_idle()


def main() -> int:
    try:
        root = tk.Tk()
        _app = DualBoardGUI(root)
        root.mainloop()
        return 0
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

# ==================================================================================
# ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    sys.exit(main())
