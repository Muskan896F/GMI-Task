import cv2
import numpy as np
from scipy.ndimage import rotate

def correct_skew_hough(image, delta=0.5, limit=15):
    """Enhanced skew correction with noise reduction and adaptive thresholding."""
    def determine_score(arr, angle):
        data = rotate(arr, angle, reshape=False, order=0)
        histogram = np.sum(data, axis=1, dtype=float)
        return np.sum((histogram[1:] - histogram[:-1]) ** 2)

    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Denoising and contrast enhancement
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # Find optimal angle
    angles = np.arange(-limit, limit + delta, delta)
    scores = [determine_score(thresh, angle) for angle in angles]
    best_angle = angles[np.argmax(scores)]

    # Rotate image
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), best_angle, 1.0)
    corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return best_angle, corrected

def preprocess_document(image):
    """Full preprocessing pipeline: denoise → deskew → normalize."""
    # Normalize input shape
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Step 1: Denoise
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    # Step 2: Skew correction
    angle, deskewed = correct_skew_hough(denoised)

    # Step 3: Normalize contrast
    lab = cv2.cvtColor(deskewed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    return angle, final

# 🔧 Test block
if __name__ == '__main__':
    image_path = r'C:\Users\HP\GMI-TASK\skew_check\doc_skew.png'
    output_path = r'C:\Users\HP\GMI-TASK\skew_check\corrected_final_best.png'

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"❌ Image not found: {image_path}")
    else:
        angle, corrected = preprocess_document(image)
        print(f"✅ Final skew angle: {angle:.2f}°")
        cv2.imwrite(output_path, corrected)
        print(f"📷 Corrected image saved at: {output_path}")
