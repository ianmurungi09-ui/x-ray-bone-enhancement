import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── App Header ──────────────────────────────────────────────
st.set_page_config(page_title="Bone Edge Enhancement", layout="wide")
st.title("🦴 Bone Edge Enhancement Tool")
st.write("Upload an X-ray image or use the built-in demo phantom to enhance bone edges using 2D Fourier High-Pass Filtering.")

# ── Sidebar Controls ─────────────────────────────────────────
st.sidebar.header("Filter Settings")
cutoff_radius = st.sidebar.slider("Cutoff Radius (Pixels)", min_value=5, max_value=150, value=30, step=5)
blend_alpha = st.sidebar.slider("Blend Alpha (Edge Overlay)", min_value=0.0, max_value=1.0, value=0.65, step=0.05)

# ── Core Functions ───────────────────────────────────────────
def build_synthetic_phantom(h=512, w=512):
    rng = np.random.default_rng(42)
    canvas = np.zeros((h, w), dtype=np.float64)
    y, x = np.mgrid[0:h, 0:w]
    canvas += 25 * np.sin(np.pi * y / h) * np.sin(np.pi * x / w) + rng.normal(0, 4, (h, w))
    bones = [
        (256, 256, 170, 95, 0), (150, 180, 55, 28, 30), (370, 320, 50, 22, -20),
        (200, 355, 40, 65, 15), (315, 150, 32, 58, -10), (420, 200, 28, 20, 0)
    ]
    for (cy, cx, ry, rx, angle) in bones:
        cos_a, sin_a = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
        dy, dx = y - cy, x - cx
        yr, xr = cos_a * dy + sin_a * dx, -sin_a * dy + cos_a * dx
        canvas[(yr / ry)**2 + (xr / rx)**2 <= 1] += 185
    return np.clip(canvas, 0, 255)

def process_fourier(image, cutoff, alpha):
    fft_shifted = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fft_shifted))

    rows, cols = image.shape
    cy, cx = rows // 2, cols // 2
    y, x = np.mgrid[0:rows, 0:cols]
    mask = np.ones((rows, cols), dtype=np.float64)
    mask[np.sqrt((y - cy)**2 + (x - cx)**2) <= cutoff] = 0

    filtered_spectrum = fft_shifted * mask
    edges = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)))
    masked_mag = np.log1p(np.abs(filtered_spectrum))

    edges_norm = edges / (edges.max() + 1e-9) * 255.0
    composite = np.clip(image + alpha * edges_norm, 0, 255)

    return magnitude, mask, masked_mag, edges, composite

# ── Image Input ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload an X-ray Image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE).astype(np.float64)
    st.success("Loaded uploaded X-ray image!")
else:
    st.info("No file uploaded. Displaying default synthetic X-ray phantom.")
    image = build_synthetic_phantom()

# ── Processing & Visualisation ──────────────────────────────
mag, mask, masked_mag, edges, composite = process_fourier(image, cutoff_radius, blend_alpha)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Original X-Ray")
    st.image(image / 255.0, width="stretch")

with col2:
    st.subheader("Bone-Pop Composite")
    st.image(composite / 255.0, width="stretch")

with st.expander("View 6-Panel Fourier Pipeline Decomposition"):
    fig = plt.figure(figsize=(15, 8), facecolor="#0a0a0f")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.2)
    panels = [
        (image, "1. Original", "gray"), (mag, "2. FFT Spectrum", "inferno"),
        (mask, "3. High-Pass Mask", "RdYlGn"), (masked_mag, "4. Masked Spectrum", "inferno"),
        (edges, "5. Edges Only", "gray"), (composite, "6. Enhanced Composite", "gray")
    ]
    for idx, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_title(title, color="#e8e8f0", fontsize=10)
        ax.axis("off")
    st.pyplot(fig)

# ── Export & Download Features ──────────────────────────────
st.markdown("---")
st.header("📥 Export Processed Results")
st.write("Save the enhanced outputs locally to use in your report or presentation slides.")

col_d1, col_d2 = st.columns(2)

# Encode processed array buffers to PNG format
composite_bytes = cv2.imencode('.png', cv2.cvtColor(composite.astype(np.uint8), cv2.COLOR_GRAY2BGR))[1].tobytes()
edges_norm = (edges / (edges.max() + 1e-9) * 255.0).astype(np.uint8)
edges_bytes = cv2.imencode('.png', cv2.cvtColor(edges_norm, cv2.COLOR_GRAY2BGR))[1].tobytes()

with col_d1:
    st.download_button(
        label="💾 Download Bone-Pop Composite",
        data=composite_bytes,
        file_name="bone_pop_composite.png",
        mime="image/png",
        width="stretch"
    )

with col_d2:
    st.download_button(
        label="💾 Download High-Pass Edge Map",
        data=edges_bytes,
        file_name="highpass_edges.png",
        mime="image/png",
        width="stretch"
    )