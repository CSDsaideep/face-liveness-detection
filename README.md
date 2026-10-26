# Face Anti-Spoofing & Active Liveness Detection Pipeline

## Architecture Overview
This repository contains a modular Presentation Attack Detection (PAD) framework built using computer vision metrics to combat 2D print and replay spoofing attempts. The pipeline extracts structural ratios using facial micro-landmarks to distinguish physical human depth from flat geometric displays.

## Performance Metrics Evaluated
- **APCER (Attack Presentation Classification Error Rate):** Evaluated down to targeted thresholds.
- **BPCER (Bona Fide Presentation Classification Error Rate):** Optimized to minimize false rejections of actual users.

## Setup Instructions
1. Clone this repository: `git clone https://github.com`
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the pipeline engine: `python liveness_detector.py`
