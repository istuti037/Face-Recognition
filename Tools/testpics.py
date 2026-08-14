import cv2
import os
from PIL import Image

KNOWN_FACES_DIR = "known_faces"

print("=== IMAGE TEST STARTED ===\n")

for root, dirs, files in os.walk(KNOWN_FACES_DIR):
    for file in files:
        path = os.path.join(root, file)

        print(f"Testing: {path}")

        # ---------- Test 1: OpenCV check ----------
        img = cv2.imread(path)
        if img is None:
            print("❌ OpenCV FAILED → Image is corrupted or unreadable.\n")
            continue
        else:
            print("✔ OpenCV OK")

        # ---------- Test 2: PIL corruption check ----------
        try:
            pil_img = Image.open(path)
            pil_img.verify()  # verifies integrity
            print("✔ PIL OK (image is not corrupted)")
        except Exception as e:
            print("❌ PIL FAILED →", e)
            print()
            continue

        # ---------- Test 3: DeepFace compatibility check ----------
        try:
            import numpy as np
            if img.shape[0] < 30 or img.shape[1] < 30:
                print("⚠️ Too small for DeepFace")
            else:
                print("✔ DeepFace-compatible size")
        except:
            print("⚠️ DeepFace check skipped")

        print()

print("=== TEST COMPLETE ===")