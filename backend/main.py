from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import cv2
import numpy as np

app = FastAPI(title="X-Ray Processing API")

def create_high_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask = x**2 + y**2 >= cutoff**2
    return mask.astype(np.float32)

@app.post("/enhance")
async def enhance_xray(
    file: UploadFile = File(...),
    cutoff_radius: int = Form(30),
    blend_alpha: float = Form(0.65)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # 2D Fourier High-Pass Filtering
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        hp_filter = create_high_pass_filter(img.shape, cutoff_radius)
        fshift_filtered = fshift * hp_filter
        
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        edge_map = np.abs(img_back)
        edge_map_norm = cv2.normalize(edge_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # CLAHE Contrast Adjustment
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(img)

        # Blend original with edges
        composite = cv2.addWeighted(clahe_img, 1.0, edge_map_norm, blend_alpha, 0)

        _, encoded_img = cv2.imencode('.png', composite)
        return {
            "status": "success",
            "filename": file.filename,
            "image_bytes": encoded_img.tobytes().hex()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))