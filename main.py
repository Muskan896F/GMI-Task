# ✅ FINAL main.py — Fully Updated for image/pdf input → OCR → LLM → JSON + Excel

import os
import json
import cv2
import unicodedata
import re
from dotenv import load_dotenv
from preprocess import preprocess_document
from extract_ocr import extract_text_ocr
from extract_pdf import extract_text_pdf, extract_text_pdf_with_preprocessing
from parse_with_LLM import (
    parse_structured_data,
    postprocess_task3,
    export_table_to_excel_openpyxl
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

def ensure_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY is not set.")
    os.environ["OPENAI_API_KEY"] = api_key

def slugify_filename(filename):
    nfkd = unicodedata.normalize('NFKD', filename)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r'[^\w\-_. ]', '_', ascii_str)

def process_file(input_path, output_dir, prompt="", add_spaces=True, lang='en', use_enhanced_pdf=True):
    ensure_api_key()
    file_ext = os.path.splitext(input_path)[1].lower()
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clean_name = slugify_filename(base_name)

    # STEP 1: Extract text
    if file_ext in [".jpg", ".jpeg", ".png"]:
        print("🖼️ Image detected. Running preprocessing + OCR...")
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Failed to read image: {input_path}")
            return

        angle, corrected_image = preprocess_document(image)
        print(f"✅ Skew corrected. Angle: {angle:.2f}°")

        corrected_dir = os.path.join(output_dir, "corrected_images")
        os.makedirs(corrected_dir, exist_ok=True)
        corrected_image_path = os.path.join(corrected_dir, f"corrected_{clean_name}.jpg")
        cv2.imwrite(corrected_image_path, corrected_image)
        print(f"📷 Corrected image saved at: {corrected_image_path}")

        extracted_text = extract_text_ocr(corrected_image_path, add_spaces=add_spaces, max_tokens=16000, lang=lang)

    elif file_ext == ".pdf":
        if use_enhanced_pdf:
            print("📄 PDF detected. Converting all pages to images and processing with OCR...")
            extracted_text = extract_text_pdf_with_preprocessing(
                input_path, 
                output_dir, 
                max_page_count=None,  
                max_tokens=16000, 
                lang=lang
            )
        else:
            print("📄 PDF detected. Using standard PDF text extraction...")
            extracted_text = extract_text_pdf(input_path, multiple_pages=True, max_page_count=3, max_tokens=16000, lang=lang)

    else:
        print(f"❌ Unsupported file type: {file_ext}")
        return

    if not extracted_text.strip():
        print("⚠️ No text extracted. Skipping file.")
        return

    print("\n🔍 Extracted Text Preview (first 500 chars):\n")
    print(extracted_text[:500])

    debug_text_path = os.path.join(output_dir, f"{clean_name}_ocr_debug.txt")
    with open(debug_text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    print(f"📝 Full OCR text saved for debugging: {debug_text_path}")

    # STEP 2: GPT Parsing — Updated Prompt to match parse_with_LLM.py
    enhanced_prompt = f"""
You are an expert at extracting structured data from bank statements.

Cleaned Bank Statement Text:
{extracted_text}

Extract the following information and return it as a JSON object:
- account_number: The account number if found
- bank_name: The bank name if found
- account_holder: The account holder name if found
- statement_period: The statement period (from date to date)
- opening_balance: The opening/starting balance (as a number)
- closing_balance: The closing/ending balance (as a number)
- transactions: A list of transactions, each with:
  - date: Transaction date (in YYYY-MM-DD format if possible)
  - description: Transaction description
  - amount: Transaction amount (positive for credits, negative for debits)
  - balance: Running balance after transaction (if available)
  - transaction_type: "debit" or "credit" based on the amount

If any field is not found or unclear, use null. Don't make assumptions.
Return only the JSON object:
"""

    try:
        parsed_json = parse_structured_data(enhanced_prompt)
        parsed_json = postprocess_task3(parsed_json)
    except Exception as e:
        print(f"❌ GPT parsing failed: {e}")
        return

    # STEP 3: Save JSON
    json_output_dir = os.path.join(output_dir, "json")
    os.makedirs(json_output_dir, exist_ok=True)
    json_path = os.path.join(json_output_dir, f"{clean_name}_parsed.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=4, ensure_ascii=False)
    print(f"📝 JSON saved to: {json_path}")

    # STEP 4: Save Excel
    excel_output_dir = os.path.join(output_dir, "excel")
    os.makedirs(excel_output_dir, exist_ok=True)
    excel_path = os.path.join(excel_output_dir, f"{clean_name}_parsed.xlsx")
    try:
        table_data = {
            "columns": ["Date", "Description", "Amount", "Balance"],
            "rows": [
                [txn.get("date", ""), txn.get("description", ""), txn.get("amount", ""), txn.get("balance", "")]
                for txn in parsed_json.get("transactions", [])
                if txn.get("date") and txn.get("amount")
            ]
        }
        export_table_to_excel_openpyxl(table_data, excel_path)
        print(f"📊 Excel saved to: {excel_path}")
    except Exception as e:
        print(f"❌ Excel export failed: {e}")

    print("✅ All tasks completed successfully!")

def batch_process_files(input_directory, output_directory, file_pattern="*", lang='en', use_enhanced_pdf=True):
    """Batch process multiple files in a directory."""
    import glob
    
    pattern_path = os.path.join(input_directory, file_pattern)
    files = glob.glob(pattern_path)
    
    if not files:
        print(f"❌ No files found matching pattern: {pattern_path}")
        return
    
    print(f"📂 Found {len(files)} files to process...")
    
    for i, file_path in enumerate(files, 1):
        print(f"\n🔄 Processing file {i}/{len(files)}: {os.path.basename(file_path)}")
        try:
            process_file(file_path, output_directory, "", lang=lang, use_enhanced_pdf=use_enhanced_pdf)
            print(f"✅ File {i}/{len(files)} completed successfully!")
        except Exception as e:
            print(f"❌ Failed to process {file_path}: {e}")
            continue
    
    print(f"\n🎉 Batch processing completed! Processed {len(files)} files.")

# -------------------- Run Script --------------------
if __name__ == "__main__":
    # Test with PDF file (enhanced processing)
    input_path = r"C:\Users\HP\GMI-Task\gmindia-challlenge-012024-datas\quonto\test.pdf"
    output_directory = r"C:\Users\HP\GMI-TASK\output\output/test_pdf_output"
    process_file(input_path, output_directory, lang='en', use_enhanced_pdf=True)
    


