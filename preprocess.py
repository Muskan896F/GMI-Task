import cv2
import numpy as np
from scipy.ndimage import rotate
import os

# ---------- Skew correction ----------
def correct_skew(image, delta=1, limit=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    scores = []
    angles = np.arange(-limit, limit+delta, delta)
    for angle in angles:
        data = rotate(thresh, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1)
        score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
        scores.append(score)

    best_angle = angles[scores.index(max(scores))]
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return corrected

# ---------- Preprocess pipeline ----------
def preprocess_image(input_path, output_dir):
    # Read input image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read input image: {input_path}")

    # Step 1: Skew correction
    corrected = correct_skew(image)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

    # Step 3: Denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    # Step 4: Adaptive thresholding for clarity
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    # Step 5: Sharpen image
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save final output image
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, f"corrected_{filename}")
    cv2.imwrite(output_path, sharpened)

    print(f"[INFO] Corrected image saved at: {output_path}")
    return output_path

# ---------- Main run ----------
if __name__ == "__main__":
    # 🔹 Give your input image path manually
    input_path = r"gmindia-challlenge-012024-datas\quonto\octobre,_2021_page0.jpg"

    # 🔹 Output directory
    output_dir = r"C:\Users\HP\GMI-TASK\output\corrected_images"

    preprocess_image(input_path, output_dir)