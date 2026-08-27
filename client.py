import os
import requests
import tkinter as tk
from tkinter import filedialog

# Launch graphical file picker to easily select your X-ray image
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)  # Bring window to the front

print("Please select your X-ray image file...")
INPUT_FILE_PATH = filedialog.askopenfilename(
    title="Select X-ray Image",
    filetypes=[("Image Files", "*.jpeg *.jpg *.png *.bmp")]
)

if not INPUT_FILE_PATH:
    print("No file was selected. Canceling process.")
    exit()

URL = "http://127.0.0.1:8000/enhance"
OUTPUT_FILE_PATH = "enhanced_xray.png"

PAYLOAD = {
    "cutoff_radius": 30,
    "blend_alpha": 0.65
}

try:
    with open(INPUT_FILE_PATH, "rb") as image_file:
        files = {"file": (os.path.basename(INPUT_FILE_PATH), image_file, "image/jpeg")}
        print(f"Uploading '{INPUT_FILE_PATH}' to server...")
        response = requests.post(URL, data=PAYLOAD, files=files)

    if response.status_code == 200:
        data = response.json()
        hex_bytes = data.get("image_bytes")
        if hex_bytes:
            with open(OUTPUT_FILE_PATH, "wb") as out_file:
                out_file.write(bytes.fromhex(hex_bytes))
            print(f"\nSuccess! Enhanced image saved to your folder as: {OUTPUT_FILE_PATH}")
        else:
            print("Error: 'image_bytes' missing from response payload.")
    else:
        print(f"Server Error [{response.status_code}]: {response.text}")

except Exception as e:
    print(f"An error occurred: {e}")