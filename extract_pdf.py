import os
import fitz  # PyMuPDF
import cv2
from pdf2image import convert_from_path
from extract_ocr import extract_text_ocr
from preprocess import preprocess_document

def extract_text_pdf(pdf_path, multiple_pages=True, max_page_count=3, max_tokens=16000, lang='eng'):
    """
    Direct PDF text extraction without OCR using PyMuPDF.
    Good for searchable PDFs.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"❌ Could not open PDF: {pdf_path} — {e}")
        return ""

    page_count = len(doc)
    pages_to_process = min(page_count, max_page_count) if multiple_pages else 1

    for page_num in range(pages_to_process):
        page = doc.load_page(page_num)
        page_text = page.get_text("text")
        text += page_text + "\n"

    doc.close()

    if not text.strip():
        print("⚠️ No text found in PDF — it might be scanned. Try OCR method.")
    return text.strip()

def extract_text_pdf_with_preprocessing(pdf_path, output_dir, max_page_count=None, max_tokens=16000, lang='eng'):
    """
    Convert PDF to images, preprocess, then run OCR on each page.
    Good for scanned PDFs.
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"❌ Error converting PDF to images: {e}")
        return ""

    page_count = len(images)
    if max_page_count:
        page_count = min(page_count, max_page_count)

    print(f"📄 PDF has {len(images)} pages. Processing {page_count} pages...")

    all_text = ""
    pdf_images_dir = os.path.join(output_dir, "pdf_images", pdf_name)
    os.makedirs(pdf_images_dir, exist_ok=True)

    corrected_dir = os.path.join(output_dir, "corrected_pdf_images", pdf_name)
    os.makedirs(corrected_dir, exist_ok=True)

    for i, image in enumerate(images[:page_count], start=1):
        print(f"🔄 Processing page {i}/{page_count}...")

        # Save raw image
        page_image_path = os.path.join(pdf_images_dir, f"page_{i}.jpg")
        image.save(page_image_path, "JPEG")
        print(f"💾 Page {i} saved as image: {page_image_path}")

        # Preprocess image (deskew, enhance)
        try:
            cv_img = cv2.imread(page_image_path)
            angle, corrected_image = preprocess_document(cv_img)
            print(f"✅ Skew corrected. Angle: {angle:.2f}°")

            corrected_image_path = os.path.join(corrected_dir, f"corrected_page_{i}.jpg")
            cv2.imwrite(corrected_image_path, corrected_image)
            print(f"📷 Corrected page {i} saved: {corrected_image_path}")

            # Run OCR
            ocr_text = extract_text_ocr(corrected_image_path, add_spaces=True, max_tokens=max_tokens, lang=lang)
            all_text += ocr_text + "\n"

        except Exception as e:
            print(f"❌ Error processing page {i}: {e}")
            continue

    print("✅ PDF processing complete. Total text length:", len(all_text))
    return all_text.strip()

# -------------------- Test --------------------
if __name__ == "__main__":
    test_pdf = r"C:\Users\HP\Downloads\GMI-TASK\GMI-TASK\gmindia-challlenge-012024-datas\quonto\test2.pdf"
    output_dir = r"C:\Users\HP\Downloads\GMI-TASK\GMI-TASK\ocr_output.pdf"

    print("\n---- TESTING SIMPLE TEXT EXTRACTION ----")
    text1 = extract_text_pdf(test_pdf, multiple_pages=True, max_page_count=2)
    print(text1[:500])

    print("\n---- TESTING OCR PDF EXTRACTION ----")
    text2 = extract_text_pdf_with_preprocessing(test_pdf, output_dir, max_page_count=2, lang='eng')
    print(text2[:500])
