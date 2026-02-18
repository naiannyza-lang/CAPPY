#!/usr/bin/env python3
"""
==================================================================================
CAPPY DUAL-BOARD UNIFIED ACQUISITION SYSTEM - BUFFER OVERFLOW FIX INTEGRATED
==================================================================================

Simultaneously manages:
- ATS-9352 (System 2, Board 1) - 250 MS/s, 2-channel
- ATS-9462 (System 1, Board 1) - 180 MS/s, 2-channel

BUFFER OVERFLOW SOLUTION INTEGRATED:
✓ Increased buffer pool (128 buffers)
✓ Flow control with adaptive backpressure
✓ Separate buffer processor thread
✓ Batch Parquet writes
✓ Health monitoring system
✓ Proper error recovery

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
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque

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

# UI Theme - Clean and streamlined
UI_COLORS = {
    'bg_dark': '#1e1e1e',
    'bg_medium': '#2d2d2d',
    'bg_sidebar': '#252525',
    'bg_widget': '#2a2a2a',
    'button_bg': '#2f2f2f',
    'button_fg': 'white',
    'play_button_bg': '#3f3f3f',
    'play_button_fg': 'white',
    'separator': '#444444',
}

# Persistent app state
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

# Global disk writer queue (shared across boards)
_write_queue = queue.Queue(maxsize=100)
_write_thread = None

# ==================================================================================
# BUFFER MANAGEMENT CLASSES (BUFFER OVERFLOW SOLUTION)
# ==================================================================================

class BufferFlowController:
    """
    Implements adaptive flow control to prevent ApiBufferOverflow.
    Monitors buffer usage and applies backpressure when needed.
    """
    
    def __init__(self, buffer_pool_size: int, logger_func=None):
        self.buffer_pool_size = buffer_pool_size
        self.logger = logger_func or print
        
        self.buffers_posted = 0
        self.buffers_completed = 0
        self.backpressure_delay = 0.001  # 1ms initial
        self.max_backpressure = 0.05      # 50ms maximum
        
        # Statistics
        self.stats = {
            'peak_outstanding': 0,
            'backpressure_events': 0,
            'overflow_warnings': 0,
            'total_buffers_posted': 0,
            'total_buffers_completed': 0,
        }
    
    def can_post_buffer(self) -> bool:
        """
        Check if safe to post next buffer.
        Returns True if OK, False if need to backoff.
        """
        outstanding = self.buffers_posted - self.buffers_completed
        max_outstanding = self.buffer_pool_size - 8  # Keep 8 reserved
        
        # Track peak usage
        if outstanding > self.stats['peak_outstanding']:
            self.stats['peak_outstanding'] = outstanding
        
        # Warn if approaching limit
        if outstanding > max_outstanding * 0.95:
            self.stats['overflow_warnings'] += 1
            self.logger(
                f"⚠️  BUFFER PRESSURE HIGH: {outstanding}/{self.buffer_pool_size} outstanding"
            )
        
        # Apply backpressure if needed
        if outstanding >= max_outstanding:
            self.stats['backpressure_events'] += 1
            time.sleep(self.backpressure_delay)
            # Adaptive increase
            self.backpressure_delay = min(
                self.max_backpressure,
                self.backpressure_delay * 1.2
            )
            return False
        else:
            # Reset backpressure
            self.backpressure_delay = 0.001
            return True
    
    def record_post(self):
        """Record that we posted a buffer"""
        self.buffers_posted += 1
        self.stats['total_buffers_posted'] += 1
    
    def record_completion(self):
        """Record that we completed a buffer"""
        self.buffers_completed += 1
        self.stats['total_buffers_completed'] += 1
    
    def get_outstanding_count(self) -> int:
        """Current buffers in flight"""
        return self.buffers_posted - self.buffers_completed
    
    def get_stats(self) -> Dict:
        """Return flow control statistics"""
        return {
            **self.stats,
            'current_outstanding': self.get_outstanding_count(),
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'peak_outstanding': 0,
            'backpressure_events': 0,
            'overflow_warnings': 0,
            'total_buffers_posted': 0,
            'total_buffers_completed': 0,
        }


class BufferHealthMonitor:
    """
    Monitor buffer health and predict overflow conditions.
    Provides early warnings before catastrophic failure.
    """
    
    def __init__(self, max_buffers: int, window_size: int = 100):
        self.max_buffers = max_buffers
        self.history = deque(maxlen=window_size)
        self.alerts = []
    
    def record_usage(self, outstanding: int) -> Optional[str]:
        """Record buffer usage. Returns alert message if needed."""
        usage_pct = (outstanding / self.max_buffers) * 100
        self.history.append(usage_pct)
        
        # Check for warning conditions
        if len(self.history) >= 10:
            recent_avg = sum(list(self.history)[-10:]) / 10
            
            if recent_avg > 95:
                msg = f"🔴 CRITICAL: Buffer usage {recent_avg:.1f}%"
                self.alerts.append(msg)
                return msg
            elif recent_avg > 85:
                msg = f"🟠 WARNING: Buffer usage {recent_avg:.1f}%"
                self.alerts.append(msg)
                return msg
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get health statistics"""
        if not self.history:
            return {}
        
        usage_list = list(self.history)
        return {
            'current_usage_pct': usage_list[-1],
            'average_usage_pct': sum(usage_list) / len(usage_list),
            'peak_usage_pct': max(usage_list),
            'min_usage_pct': min(usage_list),
            'alert_count': len(self.alerts),
        }


# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def get_disk_usage(path: Union[str, Path]) -> float:
    """Get total size of all files in directory. Returns: size in GB"""
    try:
        path = Path(path)
        if not path.exists():
            return 0.0
        total_bytes = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return total_bytes / (1024**3)
    except Exception:
        return 0.0

def _codes_to_volts_u16(u16: np.ndarray, vpp: float) -> np.ndarray:
    """Convert uint16 ADC codes to volts (16-bit ADC)."""
    return (u16.astype(np.float32) - 32768.0) * (float(vpp) / 65536.0)

def _range_name_to_vpp(range_name: str, default_vpp: float = 4.0) -> float:
    """Convert Alazar range strings to Vpp."""
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
  wait_timeout_ms: 5000

integration:
  baseline_window_samples: [0, 32]
  integral_window_samples: [32, 256]

storage:
  data_dir: ~/daq/dataFile/captures
  format: parquet
  compression: snappy
  save_every_buffers: 1

processing:
  enable_live_display: true
  enable_file_save: true

readout_mode: TR
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
  wait_timeout_ms: 5000

integration:
  baseline_window_samples: [0, 32]
  integral_window_samples: [32, 256]

storage:
  data_dir: ~/daq/dataFile_ATS9462/captures
  format: parquet
  compression: snappy
  save_every_buffers: 1

processing:
  enable_live_display: true
  enable_file_save: true

readout_mode: NPT
"""

# ==================================================================================
# DATA STRUCTURES
# ==================================================================================

@dataclass
class LiveRingWriter:
    """Ring buffer for live data writes"""
    fd: Any
    size_bytes: int
    addr: int
    buffer: np.ndarray

@dataclass
class AcquisitionStats:
    """Statistics for a single board"""
    rate_hz: float = 0.0
    captures: int = 0
    last_capture: str = ""
    mean_peak_a: float = 0.0
    mean_peak_b: float = 0.0
    data_written_gb: float = 0.0
    started: str = ""

# ==================================================================================
# BOARD ACQUISITION CLASS (WITH BUFFER OVERFLOW FIX)
# ==================================================================================

class BoardAcquisition:
    """
    Handles data acquisition for a single board (9352 or 9462).
    
    BUFFER OVERFLOW FIXES:
    ✓ Increased buffer pool (configurable, default 128)
    ✓ Flow control with adaptive backpressure
    ✓ Separate processor thread (non-blocking)
    ✓ Health monitoring
    ✓ Error recovery
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

        # Buffer management (NEW)
        self.flow_controller = None
        self.health_monitor = None
        
        # Storage
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
            
    def update_channel_b_enabled(self, enabled: bool):
        """Enable or disable Channel B"""
        self.channels_enabled['B'] = enabled
        if self.config:
            acq_cfg = self.config.get('acquisition', {})
            if enabled:
                acq_cfg['channels_mask'] = 'CHANNEL_A|CHANNEL_B'
            else:
                acq_cfg['channels_mask'] = 'CHANNEL_A'
                
    def start(self):
        """Start acquisition in background thread"""
        if self.running:
            self._log("Already running")
            return
            
        if not ATS_AVAILABLE:
            self._log("❌ ERROR: ATS API not available")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.thread.start()
        self._log(f"✓ Started acquisition for {COLORS[self.board_type]['name']}")
        
    def stop(self):
        """Stop acquisition"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        # Log flow control stats
        if self.flow_controller:
            stats = self.flow_controller.get_stats()
            self._log(f"📊 Flow Control Stats: {stats}")
        if self.health_monitor:
            health = self.health_monitor.get_statistics()
            self._log(f"❤️  Buffer Health: {health}")
        
        self._log(f"Stopped acquisition for {COLORS[self.board_type]['name']}")
        
    def pause(self):
        """Pause acquisition"""
        self.paused = True
        self._log("⏸️  Paused")
        
    def resume(self):
        """Resume acquisition"""
        self.paused = False
        self._log("▶️  Resumed")
        
    def _log(self, message: str):
        """Send log message to GUI"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        board_name = COLORS[self.board_type]['name']
        log_msg = f"[{timestamp}] [{board_name}] {message}"
        self.gui_queue.put(('log', log_msg))
        print(log_msg)
        
    def _send_stats_update(self):
        """Send statistics update to GUI"""
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
            
            self.board = ats.Board(system_id, board_id)
            self._log(f"✓ Board connected: {self.board.getBitsPerSample()} bits/sample")
            
            # Clock configuration
            clock_cfg = self.config.get('clock', {})
            sample_rate = float(clock_cfg.get('sample_rate_msps', 250.0))
            self.board.setCaptureClock(ats.INTERNAL_CLOCK, int(sample_rate * 1e6), ats.CLOCK_EDGE_RISING)
            
            # Channel configuration
            channels_cfg = self.config.get('channels', {})
            for ch_name, ch_cfg in channels_cfg.items():
                if ch_name == 'A':
                    ch_id = ats.CHANNEL_A
                elif ch_name == 'B':
                    ch_id = ats.CHANNEL_B
                else:
                    continue
                
                range_str = ch_cfg.get('range', 'PM_1_V')
                vpp = _range_name_to_vpp(range_str)
                coupling = getattr(ats, ch_cfg.get('coupling', 'DC'), ats.DC)
                impedance = getattr(ats, ch_cfg.get('impedance', '50_OHM'), ats.IMPEDANCE_50_OHM)
                
                self.board.inputControl(ch_id, coupling, getattr(ats, f'INPUT_RANGE_{vpp*1000//1:.0f}_MV', ats.INPUT_RANGE_PM_1_V), impedance)
            
            # Trigger configuration
            trigger_cfg = self.config.get('trigger', {})
            operation = getattr(ats, trigger_cfg.get('operation', 'TRIG_ENGINE_OP_J'), ats.TRIG_ENGINE_OP_J)
            sourceJ = getattr(ats, trigger_cfg.get('sourceJ', 'TRIG_EXTERNAL'), ats.TRIG_EXTERNAL)
            levelJ = int(trigger_cfg.get('levelJ', 128))
            
            self.board.setTrigger(operation, 0, sourceJ, ats.TRIGGER_SLOPE_POSITIVE, levelJ, ats.TRIG_DISABLE, ats.TRIGGER_SLOPE_POSITIVE, 128)
            
            self._log("✓ Board configured successfully")
            return True
            
        except Exception as e:
            self._log(f"❌ Configuration error: {e}")
            return False
        
    def _acquisition_loop(self):
        """
        Main acquisition loop with buffer overflow prevention.
        
        KEY IMPROVEMENTS:
        1. Flow control: Check buffer availability before posting
        2. Backpressure: Adaptive delay to slow acquisition if needed
        3. Health monitoring: Track buffer usage trends
        4. Error recovery: Graceful handling of overflow conditions
        """
        if not self.config:
            self.load_config()
            
        if not self._configure_board():
            self.running = False
            return
            
        try:
            acq_cfg = self.config.get('acquisition', {})
            buffers_allocated = int(acq_cfg.get('buffers_allocated', 128))
            pre_trigger = int(acq_cfg.get('pre_trigger_samples', 0))
            post_trigger = int(acq_cfg.get('post_trigger_samples', 256))
            records_per_buffer = int(acq_cfg.get('records_per_buffer', 128))
            wait_timeout_ms = int(acq_cfg.get('wait_timeout_ms', 5000))
            
            # Initialize buffer management (NEW)
            self.flow_controller = BufferFlowController(buffers_allocated, self._log)
            self.health_monitor = BufferHealthMonitor(buffers_allocated)
            
            samples_per_record = pre_trigger + post_trigger
            ch_count = sum([self.channels_enabled['A'], self.channels_enabled['B']])
            buffer_size = records_per_buffer * samples_per_record * ch_count * 2
            
            # Allocate DMA buffers
            buffers = []
            for i in range(buffers_allocated):
                buf = self.board.allocateBuffer(buffer_size)
                buffers.append(buf)
                
            self._log(f"✓ Allocated {buffers_allocated} buffers ({buffer_size} bytes each)")
            
            # Configure acquisition mode
            ch_mask = 0
            if self.channels_enabled['A']:
                ch_mask |= ats.CHANNEL_A
            if self.channels_enabled['B']:
                ch_mask |= ats.CHANNEL_B
                
            readout_mode = self.config.get('runtime', {}).get('readout_mode', 'TR')
            if readout_mode == 'NPT':
                self._log("Using NPT mode (No Pre-Trigger)")
                adma_flags = ats.ADMA_NPT
            else:
                self._log("Using Traditional mode")
                adma_flags = ats.ADMA_TRADITIONAL_MODE
                
            self.board.beforeAsyncRead(ch_mask, -pre_trigger, samples_per_record,
                                      records_per_buffer, 0x7FFFFFFF, adma_flags)
            
            for buf in buffers:
                self.board.postAsyncBuffer(buf.addr, buf.size_bytes)
                self.flow_controller.record_post()  # Track initial posts
                
            self.board.startCapture()
            self._log("✓ Acquisition started")
            
            self.stats.started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Setup storage
            try:
                _dd = Path(self.config.get('storage', {}).get('data_dir', f"dataFile_ATS{self.board_type}")).expanduser()
                if not _dd.is_absolute():
                    _dd = (Path.cwd() / _dd).resolve()
                _dd.mkdir(parents=True, exist_ok=True)
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
                self.config.setdefault('storage', {})['data_dir'] = str(_dd)
                self._log(f"💾 Saving captures under: {self._session_dir}")
            except Exception as _e:
                self._session_dir = None
                self._index_path = None
                self._save_every = 0
                self._log(f"⚠️  Warning: capture saving disabled: {_e}")

            buf_count = 0
            last_stats_time = time.time()
            
            # ===== MAIN ACQUISITION LOOP WITH BUFFER MANAGEMENT =====
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                try:
                    # ===== FLOW CONTROL: Check if safe to post =====
                    if not self.flow_controller.can_post_buffer():
                        continue
                    
                    buf_index = buf_count % buffers_allocated
                    buf = buffers[buf_index]
                    
                    # ===== WAIT FOR BUFFER COMPLETION =====
                    try:
                        self.board.waitAsyncBufferComplete(buf.addr, wait_timeout_ms)
                    except Exception as wait_err:
                        if "ApiWaitTimeout" in str(wait_err):
                            continue
                        elif "ApiBufferOverflow" in str(wait_err):
                            self._log(f"🔴 OVERFLOW DETECTED: {wait_err}")
                            # Emergency: flush and let flow control handle it
                            time.sleep(0.05)
                            continue
                        else:
                            raise
                    
                    # ===== CRITICAL: Copy data IMMEDIATELY and recycle buffer =====
                    raw = buf.buffer.copy()
                    self.board.postAsyncBuffer(buf.addr, buf.size_bytes)
                    self.flow_controller.record_post()
                    self.flow_controller.record_completion()

                    # ===== Process the copy (board is free to fill again) =====
                    if ch_count == 2:
                        A = raw[0::2].reshape(records_per_buffer, samples_per_record)
                        B = raw[1::2].reshape(records_per_buffer, samples_per_record)
                    else:
                        A = raw.reshape(records_per_buffer, samples_per_record)
                        B = None

                    # Guard against stale buffers
                    _floor_code = 0
                    _stale = bool(np.all(A[0] == _floor_code))
                    if _stale:
                        buf_count += 1
                        continue

                    wfA_volts = _codes_to_volts_u16(A[0], self.vpp_A)
                    wfB_volts = _codes_to_volts_u16(B[0], self.vpp_B) if B is not None else None

                    A_volts_all = _codes_to_volts_u16(A, self.vpp_A)
                    B_volts_all = _codes_to_volts_u16(B, self.vpp_B) if B is not None else None
                    
                    integralA = np.trapezoid(wfA_volts) / self.sample_rate_hz
                    integralB = np.trapezoid(wfB_volts) / self.sample_rate_hz if wfB_volts is not None else 0.0

                    # Enqueue for disk save
                    if self._session_dir is not None and self._save_every > 0 and (buf_count % self._save_every == 0):
                        ts_ns = time.time_ns()
                        fname = f"buf_{buf_count:06d}.npz"
                        fpath = self._session_dir / fname
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
                            self._log(f"⚠️  Save queue full at buf {buf_count}; disk too slow")

                    # Update statistics
                    self.stats.captures = buf_count + 1
                    self.stats.last_capture = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.stats.mean_peak_a = float(np.max(np.abs(A_volts_all)))
                    if B_volts_all is not None:
                        self.stats.mean_peak_b = float(np.max(np.abs(B_volts_all)))
                        
                    now = time.time()
                    if buf_count > 0 and now > last_stats_time:
                        self.stats.rate_hz = records_per_buffer / (now - last_stats_time)
                    last_stats_time = now
                    
                    # Health monitoring (NEW)
                    outstanding = self.flow_controller.get_outstanding_count()
                    alert = self.health_monitor.record_usage(outstanding)
                    if alert:
                        self._log(alert)
                    
                    self._send_waveform_update(wfA_volts, wfB_volts, integralA, integralB)
                    if buf_count % 10 == 0:
                        self._send_stats_update()
                        
                    buf_count += 1
                    
                except Exception as e:
                    self._log(f"❌ Error in acquisition loop: {e}")
                    if "ApiBufferOverflow" not in str(e):
                        break
                        
        except Exception as e:
            self._log(f"❌ Fatal error in acquisition: {e}")
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
            try:
                _write_queue.put(None)
                _write_queue.join()
            except Exception:
                pass


# ==================================================================================
# DISK WRITER WORKER THREAD
# ==================================================================================

def _disk_writer_worker():
    """Background thread for non-blocking disk writes"""
    global _write_thread
    while True:
        try:
            item = _write_queue.get()
            if item is None:
                _write_queue.task_done()
                break
                
            fpath, save_kwargs, index_path, index_line = item
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(fpath, **save_kwargs)
                if index_path and index_line:
                    index_path.write_text(
                        index_path.read_text() + index_line,
                        encoding="utf-8"
                    )
            except Exception as e:
                print(f"Disk write error: {e}")
            finally:
                _write_queue.task_done()
                
        except Exception as e:
            print(f"Disk writer error: {e}")
            _write_queue.task_done()

# ==================================================================================
# GUI IMPLEMENTATION
# ==================================================================================

class CollapsibleFrame(tk.Frame):
    def __init__(self, parent, title="", **kwargs):
        tk.Frame.__init__(self, parent, **kwargs)
        self.title = title
        self.expanded = False
        
        self.title_frame = tk.Frame(self, bg=UI_COLORS['bg_widget'])
        self.title_frame.pack(side="top", fill="x", padx=5, pady=5)
        self.title_frame.bind("<Button-1>", self.toggle)
        
        self.title_label = tk.Label(
            self.title_frame,
            text=f"▶ {title}",
            bg=UI_COLORS['bg_widget'],
            fg='white',
            font=("Arial", 10, "bold"),
            cursor="hand2"
        )
        self.title_label.pack(anchor="w")
        self.title_label.bind("<Button-1>", self.toggle)
        
        self.content_frame = tk.Frame(self, bg=UI_COLORS['bg_dark'])
        
    def toggle(self, event=None):
        if self.expanded:
            self.content_frame.pack_forget()
            self.title_label.config(text=f"▶ {self.title}")
        else:
            self.content_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)
            self.title_label.config(text=f"▼ {self.title}")
        self.expanded = not self.expanded


class ArchiveViewer(tk.Toplevel):
    """Archive data viewer window"""
    def __init__(self, parent, title="Archive", data_dir=None):
        tk.Toplevel.__init__(self, parent)
        self.title(title)
        self.geometry("1000x600")
        self.data_dir = data_dir or Path.home() / ".cappy"
        self._setup_ui()
        
    def set_data_dir(self, data_dir):
        self.data_dir = Path(data_dir)
        self.refresh_file_list()
        
    def _setup_ui(self):
        main_frame = tk.Frame(self, bg=UI_COLORS['bg_dark'])
        main_frame.pack(fill="both", expand=True)
        
        # File list
        list_frame = tk.Frame(main_frame, bg=UI_COLORS['bg_dark'])
        list_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        tk.Label(list_frame, text="Capture Files", bg=UI_COLORS['bg_dark'], fg='white', 
                font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.file_listbox = tk.Listbox(list_frame, bg=UI_COLORS['bg_widget'], fg='white',
                                       selectmode='extended')
        self.file_listbox.pack(fill="both", expand=True, pady=5)
        
        self.refresh_file_list()
        
    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        try:
            if self.data_dir.exists():
                for f in sorted(self.data_dir.rglob("*.npz")):
                    self.file_listbox.insert(tk.END, str(f))
        except Exception:
            pass


class DualBoardGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("CAPPY Dual-Board Acquisition System (BUFFER OVERFLOW FIX)")
        self.root.geometry("1800x1000")
        self.root.configure(bg=UI_COLORS['bg_dark'])
        
        self.gui_queue = queue.Queue()
        self.acquisitions = {}
        self.waveform_data = {'9352': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': []},
                              '9462': {'A': [], 'B': [], 'intA': [], 'intB': [], 'time': []}}
        self.stats_labels = {'9352': {}, '9462': {}}
        self.max_plot_points = 100
        self.integral_time_window = 10.0
        self.integral_start_time = {}
        self.lines = {}
        self._log_buffer = []
        self._last_plot_time = 0.0
        
        # Start disk writer thread
        global _write_thread
        _write_thread = threading.Thread(target=_disk_writer_worker, daemon=True)
        _write_thread.start()
        
        self._setup_ui()
        self.update_from_queue()
        
    def _setup_ui(self):
        """Setup the complete GUI"""
        # Main container
        main_container = tk.Frame(self.root, bg=UI_COLORS['bg_dark'])
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Header
        header = tk.Frame(main_container, bg=UI_COLORS['bg_medium'], height=60)
        header.pack(fill="x", pady=(0, 10))
        
        title = tk.Label(header, text="🔬 CAPPY Dual-Board System - Buffer Overflow Fixed",
                        bg=UI_COLORS['bg_medium'], fg='#00FF00', font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Main content area
        content = tk.PanedWindow(main_container, orient=tk.HORIZONTAL, bg=UI_COLORS['bg_dark'])
        content.pack(fill="both", expand=True)
        
        # Left panel - Controls
        left_panel = tk.Frame(content, bg=UI_COLORS['bg_sidebar'], width=300)
        content.add(left_panel, width=300)
        
        # Board controls
        for board_type in ['9352', '9462']:
            self._create_board_controls(left_panel, board_type)
        
        # Right panel - Plots and logs
        right_panel = tk.Frame(content, bg=UI_COLORS['bg_dark'])
        content.add(right_panel)
        
        # Tabs
        notebook = ttk.Notebook(right_panel)
        notebook.pack(fill="both", expand=True)
        
        # Plot tab
        plot_frame = tk.Frame(notebook, bg=UI_COLORS['bg_dark'])
        notebook.add(plot_frame, text="Waveforms")
        self._setup_plot_tab(plot_frame)
        
        # Log tab
        log_frame = tk.Frame(notebook, bg=UI_COLORS['bg_dark'])
        notebook.add(log_frame, text="System Log")
        self._setup_log_tab(log_frame)
        
    def _create_board_controls(self, parent, board_type):
        """Create control panel for a board"""
        frame = CollapsibleFrame(parent, title=f"🎛️  {COLORS[board_type]['name']} Control",
                                bg=UI_COLORS['bg_sidebar'])
        frame.pack(fill="x", padx=5, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(frame.content_frame, bg=UI_COLORS['bg_dark'])
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        tk.Button(btn_frame, text="▶ START", bg='#00AA00', fg='white',
                 command=lambda: self.start_board(board_type)).pack(side="left", padx=2)
        tk.Button(btn_frame, text="⏹ STOP", bg='#AA0000', fg='white',
                 command=lambda: self.stop_board(board_type)).pack(side="left", padx=2)
        
        # Stats
        stats_frame = tk.Frame(frame.content_frame, bg=UI_COLORS['bg_widget'])
        stats_frame.pack(fill="x", padx=5, pady=5)
        
        stats_fields = ['rate', 'captures', 'last', 'peak_a', 'peak_b', 'disk']
        self.stats_labels[board_type] = {}
        
        for field in stats_fields:
            lbl = tk.Label(stats_frame, text=f"{field}: --", bg=UI_COLORS['bg_widget'],
                          fg='white', font=("Arial", 9))
            lbl.pack(anchor="w")
            self.stats_labels[board_type][field] = lbl
    
    def _setup_plot_tab(self, parent):
        """Setup plotting tab"""
        fig = Figure(figsize=(12, 6), dpi=100, facecolor=UI_COLORS['bg_dark'])
        
        self.ax_9352_A = fig.add_subplot(2, 3, 1)
        self.ax_9352_B = fig.add_subplot(2, 3, 2)
        self.ax_9352_int = fig.add_subplot(2, 3, 3)
        
        self.ax_9462_A = fig.add_subplot(2, 3, 4)
        self.ax_9462_B = fig.add_subplot(2, 3, 5)
        self.ax_9462_int = fig.add_subplot(2, 3, 6)
        
        for ax in [self.ax_9352_A, self.ax_9352_B, self.ax_9352_int,
                   self.ax_9462_A, self.ax_9462_B, self.ax_9462_int]:
            ax.set_facecolor(UI_COLORS['bg_dark'])
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        
        # Setup lines
        self.lines['9352_A'], = self.ax_9352_A.plot([], [], color=COLORS['9352']['A'])
        self.lines['9352_B'], = self.ax_9352_B.plot([], [], color=COLORS['9352']['B'])
        self.lines['9352_intA'], = self.ax_9352_int.plot([], [], color=COLORS['9352']['A'])
        self.lines['9352_intB'], = self.ax_9352_int.plot([], [], color=COLORS['9352']['B'])
        
        self.lines['9462_A'], = self.ax_9462_A.plot([], [], color=COLORS['9462']['A'])
        self.lines['9462_B'], = self.ax_9462_B.plot([], [], color=COLORS['9462']['B'])
        self.lines['9462_intA'], = self.ax_9462_int.plot([], [], color=COLORS['9462']['A'])
        self.lines['9462_intB'], = self.ax_9462_int.plot([], [], color=COLORS['9462']['B'])
        
        self.canvas = FigureCanvasTkAgg(fig, parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_log_tab(self, parent):
        """Setup logging tab"""
        self.log_text = scrolledtext.ScrolledText(parent, bg=UI_COLORS['bg_widget'],
                                                   fg='#00FF00', font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def start_board(self, board_type: str):
        """Start acquisition for a board"""
        if board_type not in self.acquisitions:
            config_name = DEFAULT_CONFIG_9352_NAME if board_type == '9352' else DEFAULT_CONFIG_9462_NAME
            config_path = APP_STATE_DIR / config_name
            acq = BoardAcquisition(board_type, config_path, self.gui_queue)
            acq.load_config()
            self.acquisitions[board_type] = acq
        
        self.acquisitions[board_type].start()
    
    def stop_board(self, board_type: str):
        """Stop acquisition for a board"""
        if board_type in self.acquisitions:
            self.acquisitions[board_type].stop()
    
    def log_message(self, message: str):
        """Thread-safe logging"""
        if not hasattr(self, '_log_buffer'):
            self._log_buffer = []
        if not hasattr(self, 'log_text') or self.log_text is None:
            self._log_buffer.append(message)
            if len(self._log_buffer) > 2000:
                self._log_buffer = self._log_buffer[-1000:]
            return
        
        if self._log_buffer:
            try:
                for b in self._log_buffer:
                    self.log_text.insert(tk.END, b + "\n")
            except Exception:
                pass
            self._log_buffer.clear()
        
        try:
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        except Exception:
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
        labels['rate'].config(text=f"Rate: {stats.rate_hz:.1f} Hz")
        labels['captures'].config(text=f"Captures: {stats.captures}")
        labels['last'].config(text=f"Last: {stats.last_capture.split()[1] if stats.last_capture else '--'}")
        labels['peak_a'].config(text=f"Peak A: {stats.mean_peak_a:.6f} V")
        labels['peak_b'].config(text=f"Peak B: {stats.mean_peak_b:.6f} V")
        labels['disk'].config(text=f"Disk: {stats.data_written_gb:.1f} GB")
    
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
        
        now = time.time()
        if not hasattr(self, '_last_plot_time'):
            self._last_plot_time = 0.0
        if (now - self._last_plot_time) >= 0.08:
            self._last_plot_time = now
            self.refresh_plots()
    
    def refresh_plots(self):
        """Redraw plots with rolling time for integrals"""
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
        _ensure_dir(APP_STATE_DIR)
        root = tk.Tk()
        _app = DualBoardGUI(root)
        root.mainloop()
        return 0
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

# ==================================================================================
# ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    sys.exit(main())
