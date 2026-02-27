# =========================
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
  slopeJ: TRIGGER_SLOPE_NEGATIVE
  levelJ: 26                  # ~10% (FrontPanel Level = 10%)

  sourceK: TRIG_DISABLE
  slopeK: TRIGGER_SLOPE_POSITIVE
  levelK: 26                  # ~10% (unused while sourceK=TRIG_DISABLE)

  ext_coupling: DC_COUPLING
  ext_range: ETR_5V
  delay_samples: 0
  timeout_ms: 0                   # 0 = wait forever (unless runtime.noise_test=true)
  timeout_pause_s: 0.0            # if >0 and no triggers arrive for this long, pause then rearm

  external_startcapture: false

timing:
  bunch_spacing_samples: 424      # adjust to your bunch spacing

acquisition:
  channels_mask: CHANNEL_A
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
  show_channel_b: false
  preview_mode: archive_match
