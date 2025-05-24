# Hive Guard Bee Density Analyzer
A simple and user-friendly Streamlit web app that integrates a custom AI model for bee tracking, detection, and classification. Beekeepers can upload beehive videos, specify the entrance area, and receive a downloadable Excel log of bee activity, including timestamps for bees entering, exiting, and carrying pollen.

---

## Features

- Upload and analyze a video of a beehive entrance.
- press start analysis and a new pop up window will show
- Draw two areas:
  - An **unfilled square** to define the **entrance zone** then press enter.
  - A **filled square** to define the **guard bee density zone** then press enter.
- wait for automatically processes the video and tracks bees in the defined areas.
- Download an Excel report of the analysis results.

---

## Requirements

- Python 3.8+
- Libraries:
  - `opencv-python`
  - `numpy`
  - `pandas`
  - `openpyxl`
  
Install required packages using:

```bash
pip install opencv-python numpy pandas openpyxl

