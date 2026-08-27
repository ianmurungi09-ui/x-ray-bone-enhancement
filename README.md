
🦴 Bone Edge Enhancement Tool

An interactive computer vision web application designed to isolate and enhance subtle bone edges, hairline fractures, and high-frequency structural details in X-ray scans. The processing engine utilizes **2D Fourier High-Pass Filtering** combined with spatial-domain contrast normalization.

---

## 📌 Features

* **Interactive Web UI**: Adjust filter parameters dynamically with real-time visual feedback using Streamlit.
* **2D Fourier High-Pass Filtering**: Isolates sharp edges and fine structural details by suppressing low-frequency background signals.
* **Built-in Synthetic Phantom**: Automatically generates a procedural bone phantom if no input file is uploaded.
* **6-Panel Pipeline Visualizer**: Displays a full decomposition of the Fourier pipeline (Original → FFT Spectrum → High-Pass Mask → Masked Spectrum → Edges → Enhanced Composite).
* **Export Capability**: High-resolution download options for both the edge-only map and the final enhanced composite scan.

---

## 🛠️ Project Structure

```text
x-ray-bone-enhancement/
├── app.py              # Main Streamlit web application & Fourier processing engine
├── requirements.txt    # Python dependencies required for deployment
└── README.md           # Project documentation