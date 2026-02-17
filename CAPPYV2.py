#!/usr/bin/env python3
"""
==================================================================================
CAPPY DUAL-BOARD UNIFIED ACQUISITION SYSTEM
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
  buffers_allocated: 16
  buffers_per_acquisition: 0
  wait_timeout_ms: 1000

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
  buffers_allocated: 16
  buffers_per_acquisition: 0
  wait_timeout_ms: 1000

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
            
    def _acquisition_loop(self):
        """Main acquisition loop - runs in background thread"""
        try:
            if not self._configure_board():
                return
                
            acq_cfg = self.config.get('acquisition', {})
            pre_trigger = int(acq_cfg.get('pre_trigger_samples', 0))
            post_trigger = int(acq_cfg.get('post_trigger_samples', 256))
            samples_per_record = pre_trigger + post_trigger
            records_per_buffer = int(acq_cfg.get('records_per_buffer', 128))
            buffers_allocated = int(acq_cfg.get('buffers_allocated', 16))
            
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
                adma_flags = ats.ADMA_TRADITIONAL_MODE | ats.ADMA_EXTERNAL_STARTCAPTURE
                self.board.beforeAsyncRead(ch_mask, -pre_trigger, samples_per_record, records_per_buffer, 0, adma_flags)
                
            for buf in buffers:
                self.board.postAsyncBuffer(buf.addr, buf.size_bytes)
                
            self.board.startCapture()
            self._log("Acquisition started")
            
            self.stats.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            buf_count = 0
            last_stats_time = time.time()
            
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                try:
                    buf_index = buf_count % buffers_allocated
                    buf = buffers[buf_index]
                    
                    self.board.waitAsyncBufferComplete(buf.addr, 1000)
                    
                    raw = buf.buffer.copy()
                    
                    if ch_count == 2:
                        A = raw[0::2].reshape(records_per_buffer, samples_per_record)
                        B = raw[1::2].reshape(records_per_buffer, samples_per_record)
                    else:
                        A = raw.reshape(records_per_buffer, samples_per_record)
                        B = None
                        
                    wfA_volts = _codes_to_volts_u16(A[0], self.vpp_A)
                    wfB_volts = _codes_to_volts_u16(B[0], self.vpp_B) if B is not None else None
                    
                    integralA = np.trapz(wfA_volts) / self.sample_rate_hz
                    integralB = np.trapz(wfB_volts) / self.sample_rate_hz if wfB_volts is not None else 0.0
                    
                    self.stats.captures = buf_count + 1
                    self.stats.last_capture = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.stats.mean_peak_a = float(np.max(np.abs(wfA_volts)))
                    if wfB_volts is not None:
                        self.stats.mean_peak_b = float(np.max(np.abs(wfB_volts)))
                        
                    now = time.time()
                    if buf_count > 0 and now > last_stats_time:
                        self.stats.rate_hz = 1.0 / (now - last_stats_time)
                    last_stats_time = now
                    
                    self._send_waveform_update(wfA_volts, wfB_volts, integralA, integralB)
                    if buf_count % 10 == 0:
                        self._send_stats_update()
                        
                    self.board.postAsyncBuffer(buf.addr, buf.size_bytes)
                    buf_count += 1
                    
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
    """Lightweight archive browser that auto-attaches to a data_dir (no file picking)."""

    def __init__(self, parent, data_dir: Path, title: str = "Archive"):
        super().__init__(parent)
        self.title(title)
        self.configure(bg=UI_COLORS['bg_dark'])
        self.geometry("900x650")
        self.minsize(700, 450)

        self.data_dir = Path(data_dir)
        self.captures_root = self.data_dir / "captures"

        top = tk.Frame(self, bg=UI_COLORS['bg_medium'])
        top.pack(fill=tk.X)

        tk.Label(top, text=title, bg=UI_COLORS['bg_medium'], fg="white",
                 font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=10, pady=8)

        tk.Button(top, text="Open Folder",
                  command=self._open_folder,
                  bg=UI_COLORS['button_bg'], fg="white",
                  activebackground="#4a4a4a", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2").pack(side=tk.RIGHT, padx=10, pady=8)

        body = tk.Frame(self, bg=UI_COLORS['bg_dark'])
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg=UI_COLORS['bg_dark'])
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        tk.Label(left, text="Sessions", bg=UI_COLORS['bg_dark'], fg="white",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.session_list = tk.Listbox(left, width=42, bg="#0f0f0f", fg="white",
                                       selectbackground="#333", activestyle="none")
        self.session_list.pack(fill=tk.Y, expand=True, pady=(6, 0))
        self.session_list.bind("<<ListboxSelect>>", self._on_select)

        right = tk.Frame(body, bg=UI_COLORS['bg_dark'])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)

        tk.Label(right, text="Details", bg=UI_COLORS['bg_dark'], fg="white",
                 font=("Arial", 10, "bold")).pack(anchor="w")

        self.details = scrolledtext.ScrolledText(
            right, bg="#0a0a0a", fg="#ddd", font=("Courier", 9),
            wrap=tk.WORD
        )
        self.details.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        self._sessions = self._scan_sessions()
        for s in self._sessions:
            self.session_list.insert(tk.END, s["label"])

        if not self._sessions:
            self.details.insert(tk.END, f"No sessions found under: {self.captures_root}\n")

    def _open_folder(self):
        target = self.captures_root if self.captures_root.exists() else self.data_dir
        try:
            subprocess.Popen(["xdg-open", str(target)])
        except Exception:
            try:
                subprocess.Popen(["gio", "open", str(target)])
            except Exception:
                messagebox.showinfo("Folder", f"Path: {target}")

    def _scan_sessions(self) -> List[dict]:
        sessions: List[dict] = []
        root = self.captures_root
        if not root.exists():
            return sessions

        for idx in sorted(root.rglob("session_index.parquet")):
            try:
                df = pq.read_table(idx).to_pandas()
                day_dir = idx.parent
                for _, row in df.iterrows():
                    sid = str(row.get("session_id", "")).strip()
                    if not sid:
                        continue
                    first_ns = int(row.get("first_timestamp_ns", 0) or 0)
                    last_ns = int(row.get("last_timestamp_ns", 0) or 0)
                    date_s = str(row.get("date", "")) if row.get("date", "") else day_dir.name
                    label = f"{date_s}  {sid}"
                    sessions.append({
                        "label": label,
                        "session_id": sid,
                        "day_dir": day_dir,
                        "meta": row.to_dict(),
                        "first_ns": first_ns,
                        "last_ns": last_ns,
                    })
            except Exception:
                continue

        if sessions:
            sessions.sort(key=lambda x: x.get("last_ns", 0), reverse=True)
            return sessions

        for p in sorted(root.rglob("session_*.txt"), reverse=True):
            try:
                sid = p.stem.replace("session_", "")
                day_dir = p.parent.parent if p.parent.name == "index" else p.parent
                label = f"{day_dir.name}  {sid}"
                sessions.append({"label": label, "session_id": sid, "day_dir": day_dir, "meta": {"marker": str(p)}})
            except Exception:
                pass
        return sessions

    def _on_select(self, _evt=None):
        sel = self.session_list.curselection()
        if not sel:
            return
        s = self._sessions[int(sel[0])]
        self.details.delete("1.0", tk.END)

        day_dir = Path(s["day_dir"])
        sid = s["session_id"]
        self.details.insert(tk.END, f"Data dir: {self.data_dir}\n")
        self.details.insert(tk.END, f"Day dir:  {day_dir}\n")
        self.details.insert(tk.END, f"Session:  {sid}\n\n")

        meta = s.get("meta", {}) or {}
        for k in sorted(meta.keys()):
            self.details.insert(tk.END, f"{k}: {meta[k]}\n")

        idx_dir = day_dir / "index"
        self.details.insert(tk.END, "\nLikely files:\n")
        self.details.insert(tk.END, f"  - {idx_dir / ('snips_' + sid + '.sqlite')}\n")
        self.details.insert(tk.END, f"  - reduced/ and waveforms/ under hour folders in: {day_dir}\n")



class ArchiveViewer(tk.Toplevel):
    """Simple archive browser bound to a board's data_dir.

    - Lists files under data_dir (recursive)
    - Double-click to preview .npy/.npz waveforms
    """

    def __init__(self, master, title: str, data_dir: Path):
        super().__init__(master)
        self.title(title)
        self.configure(bg=UI_COLORS['bg_dark'])
        self.geometry("980x720")
        self.minsize(820, 560)

        self.data_dir = Path(data_dir)
        self.filter_var = tk.StringVar(value="")
        self._files = []

        top = tk.Frame(self, bg=UI_COLORS['bg_medium'])
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, text=str(self.data_dir), bg=UI_COLORS['bg_medium'], fg='white').pack(side=tk.LEFT, padx=10, pady=8)

        tk.Label(top, text="Filter:", bg=UI_COLORS['bg_medium'], fg='#ccc').pack(side=tk.LEFT, padx=(10, 4))
        ent = tk.Entry(top, textvariable=self.filter_var, bg='#222', fg='white', insertbackground='white', relief=tk.FLAT, width=28)
        ent.pack(side=tk.LEFT, pady=8)
        ent.bind("<KeyRelease>", lambda _e=None: self.refresh_list())

        tk.Button(
            top, text="Refresh",
            command=self.refresh_list,
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.RIGHT, padx=10, pady=6)

        tk.Button(
            top, text="Open Folder",
            command=self.open_folder,
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=10
        ).pack(side=tk.RIGHT, padx=6, pady=6)

        body = tk.Frame(self, bg=UI_COLORS['bg_dark'])
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: file list
        left = tk.Frame(body, bg=UI_COLORS['bg_dark'])
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.listbox = tk.Listbox(left, bg='#1e1e1e', fg='white', width=55, activestyle='none',
                                  selectbackground='#444', highlightthickness=0)
        self.listbox.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=(10, 0), pady=10)

        sb = tk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        sb.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        self.listbox.configure(yscrollcommand=sb.set)

        self.listbox.bind("<Double-Button-1>", lambda _e=None: self.preview_selected())

        # Right: preview plot
        right = tk.Frame(body, bg=UI_COLORS['bg_dark'])
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Preview")
        self.ax.grid(True, alpha=0.2)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        btns = tk.Frame(right, bg=UI_COLORS['bg_dark'])
        btns.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Button(
            btns, text="Preview",
            command=self.preview_selected,
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=12
        ).pack(side=tk.LEFT)

        tk.Button(
            btns, text="Close",
            command=self.destroy,
            bg=UI_COLORS['button_bg'], fg='white', activebackground='#4a4a4a', activeforeground='white',
            relief=tk.FLAT, cursor='hand2', padx=12
        ).pack(side=tk.RIGHT)

        self.refresh_list()

    def open_folder(self):
        try:
            import subprocess
            subprocess.Popen(["xdg-open", str(self.data_dir)])
        except Exception:
            pass

    def _iter_files(self):
        if not self.data_dir.exists():
            return []
        out = []
        for p in self.data_dir.rglob("*"):
            if p.is_file():
                # keep common capture formats
                if p.suffix.lower() in (".npy", ".npz", ".bin", ".dat", ".csv", ".txt"):
                    out.append(p)
        return sorted(out)

    def refresh_list(self):
        flt = self.filter_var.get().strip().lower()
        self._files = self._iter_files()
        self.listbox.delete(0, tk.END)

        for p in self._files:
            s = str(p.relative_to(self.data_dir))
            if flt and flt not in s.lower():
                continue
            self.listbox.insert(tk.END, s)

    def _get_selected_path(self):
        sel = self.listbox.curselection()
        if not sel:
            return None
        rel = self.listbox.get(sel[0])
        return self.data_dir / rel

    def preview_selected(self):
        p = self._get_selected_path()
        if p is None:
            return

        self.ax.clear()
        self.ax.grid(True, alpha=0.2)

        try:
            if p.suffix.lower() == ".npy":
                arr = np.load(p, allow_pickle=False)
                y = np.ravel(arr)
                self.ax.plot(y)
                self.ax.set_title(str(p.name))

            elif p.suffix.lower() == ".npz":
                z = np.load(p, allow_pickle=False)
                # Try common keys
                for key in ("A", "B", "chA", "chB", "wfA", "wfB", "waveform", "data"):
                    if key in z.files:
                        y = np.ravel(z[key])
                        self.ax.plot(y)
                        self.ax.set_title(f"{p.name} : {key}")
                        break
                else:
                    # plot first array-like
                    y = np.ravel(z[z.files[0]])
                    self.ax.plot(y)
                    self.ax.set_title(f"{p.name} : {z.files[0]}")
            else:
                self.ax.text(0.02, 0.98, f"Preview not supported for {p.suffix}", transform=self.ax.transAxes, va='top')
                self.ax.set_title(str(p.name))

        except Exception as e:
            self.ax.text(0.02, 0.98, f"Failed to load: {e}", transform=self.ax.transAxes, va='top')
            self.ax.set_title(str(p.name))

        self.canvas.draw_idle()

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
        
        self.acq_9352 = None
        self.acq_9462 = None
        
        self.config_path_9352 = Path("config_9352.yaml")
        self.config_path_9462 = Path("config_9462.yaml")


        # Load persisted UI state (last YAML/data dirs) and ensure defaults exist
        self._load_app_state_and_defaults()
        
        self.waveform_data = {
            '9352': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': []},
            '9462': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': []}
        }
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

        def open_archive(self, board_type):
            acq = self.acq_9352 if board_type == '9352' else self.acq_9462
            if not acq or not acq.config:
                messagebox.showwarning("Warning", "No configuration loaded for this board")
                return

        data_dir = Path(acq.config.get('storage', {}).get('data_dir', '.'))
        if not data_dir.exists():
            # Create if missing (matches your request)
            try:
                data_dir.mkdir(parents=True, exist_ok=True)
                self.log_message(f"Created missing data directory: {data_dir}")
            except Exception as e:
                messagebox.showwarning("Warning", f"Data directory does not exist and could not be created:\n{data_dir}\n{e}")
                return

            if not hasattr(self, "_archive_windows"):
                self._archive_windows = {}

            # Reuse an existing viewer per board
            win = self._archive_windows.get(board_type)
            try:
                if win is not None and win.winfo_exists():
                    win.lift()
                    win.focus_force()
                    return
            except Exception:
                pass

            title = f"{COLORS[board_type]['name']} Archive"
            win = ArchiveViewer(self.root, title=title, data_dir=data_dir)
            self._archive_windows[board_type] = win

        def log_message(self, message):
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
            self.log_text.see(tk.END)
        
        # ==================================================================================
        # GUI UPDATE LOOP WITH ROLLING INTEGRALS
        # ==================================================================================
    
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
            wfB = wf_data['wfB']
            integralA = wf_data['integralA']
            integralB = wf_data['integralB']
            timestamp = wf_data['time']
        
            data = self.waveform_data[board_type]
            data['A'].append(wfA)
            data['B'].append(wfB if wfB is not None else np.zeros_like(wfA))
            data['intA'].append(integralA)
            data['intB'].append(integralB)
            data['time'].append(timestamp)
        
            if len(data['A']) > self.max_plot_points:
                data['A'].pop(0)
                data['B'].pop(0)
                data['intA'].pop(0)
                data['intB'].pop(0)
                data['time'].pop(0)
            
            # Refresh plots more responsively (throttled)
            now = time.time()
            if not hasattr(self, '_last_plot_time'):
                self._last_plot_time = 0.0
            if (now - self._last_plot_time) >= 0.08:
                self._last_plot_time = now
                self.refresh_plots()
            
        def refresh_plots(self):
            """Redraw plots with ROLLING TIME for integrals"""
            for board_type in ['9352', '9462']:
                data = self.waveform_data[board_type]
            
                if not data['A']:
                    continue
                
                wfA = data['A'][-1]
                wfB = data['B'][-1]
            
                sample_rate = 250e6 if board_type == '9352' else 180e6
                time_us = np.arange(len(wfA)) / sample_rate * 1e6
            
                prefix = f"{board_type}_"
                self.lines[prefix + 'A'].set_data(time_us, wfA)
                self.lines[prefix + 'B'].set_data(time_us, wfB)
            
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
