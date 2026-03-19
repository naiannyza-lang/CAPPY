# CAPPY User Guide

**CAPPY** stands for **Capture and Acquisition Program in Python**. It is a waveform acquisition and archive system for AlazarTech digitizers using the `atsapi` interface. It is designed for triggered capture, live monitoring, and structured archival of reduced data and selected waveform snippets.

This document serves as a concise operating reference for acquisition sessions and archive review.

In this guide:

- **CAPPY** refers to the acquisition program.
- **CAPPY.ARCH** refers to the archive browser.

Repository: [github.com/naiannyza-lang/CAPPY](https://github.com/naiannyza-lang/CAPPY)

---

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Program Functionality](#program-functionality)
4. [Installation and User Setup](#installation-and-user-setup)
5. [Operating Procedure](#operating-procedure)
6. [Graphical User Interface](#graphical-user-interface)
7. [Configuration Reference](#configuration-reference)
8. [Archive Browser](#archive-browser)
9. [Waveform Inspection Tools](#waveform-inspection-tools)
10. [Interpretation Guidelines](#interpretation-guidelines)
11. [Support](#support)

---

## Introduction

CAPPY is a waveform acquisition and archive system for AlazarTech digitizers using the `atsapi` interface. It is intended for:

- triggered pulse capture,
- live waveform monitoring,
- structured archival of reduced data and waveform snippets.

---

## System Overview

CAPPY is designed to support:

1. triggered pulse acquisition,
2. reduced-data and waveform storage,
3. live monitoring and archive inspection.

The program combines hardware control, DMA-based acquisition, numerical reduction, selective waveform storage, and archive browsing in a single workflow.

---

## Program Functionality

CAPPY v1.3 and CAPPY v1.0 are Python-based acquisition and archiving systems for AlazarTech digitizers through the `atsapi` interface. The main difference between versions is board identity and calibration. The acquisition workflow is otherwise the same.

At startup, the program loads a YAML configuration containing:

- acquisition settings,
- channel settings,
- trigger settings,
- waveform-save rules,
- runtime settings,
- archive settings,
- preview settings.

The program then configures the digitizer clock, active channels, voltage range, coupling, impedance, trigger source, trigger slope, and record length.

Acquisition is performed as triggered records with pre-trigger and post-trigger regions. Multiple records are grouped into DMA buffers for transfer to host memory.

After each completed buffer:

- raw payload data are interpreted,
- waveform arrays are formed for Channel A and, if enabled, Channel B,
- ADC codes are converted to calibrated voltages,
- each record is reduced into compact observables such as baseline, integrated area, and peak amplitude.

Each record is also assigned metadata including:

- DMA buffer number,
- record number within buffer,
- global record number,
- timestamp.

Waveforms are saved according to the configured selection rules. The program can save:

- every `N`th waveform,
- waveforms above a peak threshold,
- waveforms above an area threshold,
- combinations of these rules.

Reduced data are written to rolling Parquet files. Selected waveform snippets are written to binary waveform files and indexed in SQLite. The archive is organized by year, month, day, and hour.

---

## Installation and User Setup

### Adding a New User

If a new Linux user account must be created for instrument operation and should inherit the same graphical environment and theme configuration as an existing account, the following procedure may be used.

```bash
# Copy configuration and local data
sudo cp -ra /home/user1/.config /home/user2/
sudo cp -ra /home/user1/.local /home/user2/

# Copy theme and icon folders if they exist
sudo cp -ra /home/user1/.themes /home/user2/ 2>/dev/null
sudo cp -ra /home/user1/.icons /home/user2/ 2>/dev/null

# Ensure the new user owns their home directory
sudo chown -R user2:user2 /home/user2/
