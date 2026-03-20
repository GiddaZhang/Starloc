# Data Folder Guide

The `data/` directory is organized by data capture timestamps (for example, `12Mar2024_23_17`).

Each timestamped folder contains three files:

- `tracked_tones.mat`  
  Due to the large size of raw radio data, in this repo we only provide its processed version. This file contains three variables
  - `tones_dict`: A dictionary containing beacon-signal measurements for multiple satellites. Each entry is an $N_t \times 5$ array, where $N_t$ is the number of time samples. Column 1 is time (seconds from the start of capture), column 2 is the Fourier-transform frequency-bin index, and columns 3-5 are the phases of RX0, RX1, and RX2.
  - `main_tone_dict`: Similar to `tones_dict`, but keeps only the single longest tracked beacon tone among the 9 tones. Another difference is the second column, which is frequency in Hz instead of frequency bin.
  - `id_dict`: Satellite ID used for ground truth simulation.
- `calibration.mat`  
  Calibration parameters needed to correct measurements and align them with the processing pipeline.

- `Starlink.tle`  
  Two-line element (TLE) orbital data used to compute satellite position/orbit context for the same capture window.

## IMU Files

The `imus/` subfolder stores IMU text logs (one file per session), named like `imu_utctime_YYYY_MM_DD_HH_MM_SS.txt`.
