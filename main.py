# ✅ FINAL main.py — Process whole dataset into ONE JSON, ONE Excel, ONE TXT

import os
import json
import cv2
import unicodedata
import re
import tempfile
from dotenv import load_dotenv
from preprocess import preprocess_document
from extract_ocr import extract_text_ocr
from extract_pdf import extract_text_pdf, extract_text_pdf_with_preprocessing
from parse_with_LLM import (
    parse_structured_data,
    postprocess_task3,
    export_table_to_excel_openpyxl
)

# Load .env properly
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

def ensure_api_key():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ OPENAI_API_KEY is not set.")
    os.environ["OPENAI_API_KEY"] = api_key

def slugify_filename(filename):
    nfkd = unicodedata.normalize('NFKD', filename)
    ascii_str = nfkd.encode('ASCII', 'ignore').decode('utf-8')
    return re.sub(r'[^\w\-. ]', '', ascii_str)

def process_file(input_path, add_spaces=True, lang='en', use_enhanced_pdf=True):
    """Extract text and parse JSON for a single file, return (text, parsed_json)."""
    file_ext = os.path.splitext(input_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    clean_name = slugify_filename(base_name)

    extracted_text = ""

    # STEP 1: Extract text
    if file_ext in [".jpg", ".jpeg", ".png"]:
        print("🖼 Image detected. Running preprocessing + OCR...")
        image = cv2.imread(input_path)
        if image is None:
            print(f"❌ Failed to read image: {input_path}")
            return "", None

        angle, corrected_image = preprocess_document(image)
        print(f"✅ Skew corrected. Angle: {angle:.2f}°")

        # Save corrected image temporarily for OCR
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
            cv2.imwrite(tmpfile.name, corrected_image)
            extracted_text = extract_text_ocr(tmpfile.name, add_spaces=add_spaces, max_tokens=16000, lang=lang)

    elif file_ext == ".pdf":
        if use_enhanced_pdf:
            print("📄 PDF detected. Converting all pages to images and processing with OCR...")
            extracted_text = extract_text_pdf_with_preprocessing(
                input_path, 
                None, 
                max_page_count=None,  
                max_tokens=16000, 
                lang=lang
            )
        else:
            print("📄 PDF detected. Using standard PDF text extraction...")
            extracted_text = extract_text_pdf(input_path, multiple_pages=True, max_page_count=3, max_tokens=16000, lang=lang)
    else:
        print(f"❌ Unsupported file type: {file_ext}")
        return "", None

    if not extracted_text.strip():
        print("⚠ No text extracted. Skipping file.")
        return "", None

    # STEP 2: GPT Parsing
    enhanced_prompt = f"""
You are an expert at extracting structured data from bank statements.

Cleaned Bank Statement Text:
{extracted_text}

Extract the following information and return it as a JSON object:
- account_number
- bank_name
- account_holder
- statement_period
- opening_balance
- closing_balance
- transactions: list of transactions with date, description, amount, balance, transaction_type
"""
    try:
        parsed_json = parse_structured_data(enhanced_prompt)
        parsed_json = postprocess_task3(parsed_json)
    except Exception as e:
        print(f"❌ GPT parsing failed: {e}")
        return extracted_text, None

    return extracted_text, parsed_json


# -------------------- Run Script --------------------
if __name__ == "__main__":
    ensure_api_key()

    dataset_dir = r"C:\Users\HP\GMI-Task\gmindia-challlenge-012024-datas"
    output_dir = r"C:\Users\HP\GMI-TASK\output\dataset_output"
    os.makedirs(output_dir, exist_ok=True)

    all_files = []
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".pdf", ".jpg", ".jpeg", ".png")):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("❌ No PDF or image files found in dataset folder.")
    else:
        print(f"📂 Found {len(all_files)} files across all subfolders.")

        combined_text = ""
        combined_json = {"documents": []}
        combined_transactions = {
            "columns": ["File", "Date", "Description", "Amount", "Balance"],
            "rows": []
        }

        for i, file_path in enumerate(all_files, 1):
            print(f"\n🔄 Processing file {i}/{len(all_files)}: {file_path}")
            try:
                extracted_text, parsed_json = process_file(file_path)

                if extracted_text:
                    combined_text += f"\n\n===== {os.path.basename(file_path)} =====\n\n"
                    combined_text += extracted_text

                if parsed_json:
                    combined_json["documents"].append({
                        "file": os.path.basename(file_path),
                        "data": parsed_json
                    })

                    for txn in parsed_json.get("transactions", []):
                        combined_transactions["rows"].append([
                            os.path.basename(file_path),
                            txn.get("date", ""),
                            txn.get("description", ""),
                            txn.get("amount", ""),
                            txn.get("balance", "")
                        ])

                print(f"✅ File {i}/{len(all_files)} added to combined output.")
            except Exception as e:
                print(f"❌ Failed to process {file_path}: {e}")
                continue

        # Save ONE TXT
        txt_path = os.path.join(output_dir, "combined_output.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print(f"📝 Combined TXT saved to: {txt_path}")

        # Save ONE JSON
        json_path = os.path.join(output_dir, "combined_output.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_json, f, indent=4, ensure_ascii=False)
        print(f"📝 Combined JSON saved to: {json_path}")

        # Save ONE Excel
        excel_path = os.path.join(output_dir, "combined_output.xlsx")
        try:
            export_table_to_excel_openpyxl(combined_transactions, excel_path)
            print(f"📊 Combined Excel saved to: {excel_path}")
        except Exception as e:
            print(f"❌ Excel export failed: {e}")

        print(f"\n🎉 Finished processing {len(all_files)} files into ONE JSON, ONE Excel, ONE TXT.")
