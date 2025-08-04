import os
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

load_dotenv()

# -----------------------
# Utility Functions
# -----------------------

def handle_json(response_text):
    """Extract valid JSON portion from response."""
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return response_text[start:end]
    except Exception:
        return response_text

def fix_ocr_text(text):
    """Fix common OCR mistakes."""
    corrections = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1',
        ',': '.', '|': ''
    }
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text.strip()

def is_valid_date(date_str):
    """Check if date matches expected formats."""
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            datetime.strptime(date_str, fmt)
            return True
        except:
            continue
    return False

def is_valid_amount(value):
    """Check if amount can be converted to float."""
    try:
        float(value.replace(",", "."))
        return True
    except:
        return False

# -----------------------
# Main Parsing Function
# -----------------------

def parse_structured_data(input_text: str) -> dict:
    """Send cleaned OCR text to LLM for structured parsing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an expert at extracting structured data from bank statements.

Cleaned Bank Statement Text:
{input_text}

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
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial statement parser."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        raw_response = result.choices[0].message.content
        return json.loads(handle_json(raw_response))
    except Exception as e:
        print("❌ Error parsing structured data:", e)
        return {}

# -----------------------
# Post-processing
# -----------------------

def postprocess_task3(data):
    """Clean extracted fields and validate dates/amounts."""
    for txn in data.get("transactions", []):
        txn["date"] = fix_ocr_text(str(txn.get("date", "")))
        txn["description"] = fix_ocr_text(str(txn.get("description", "")))
        txn["amount"] = fix_ocr_text(str(txn.get("amount", "")))
        txn["balance"] = fix_ocr_text(str(txn.get("balance", "")))

        if not is_valid_date(txn["date"]):
            txn["date_valid"] = False
        if not is_valid_amount(txn["amount"]):
            txn["amount_valid"] = False
    return data

# -----------------------
# Excel Export
# -----------------------

def export_table_to_excel_openpyxl(table_data, output_path):
    """Export extracted transactions to Excel."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Transactions"

    headers = table_data.get("columns", ["Date", "Description", "Amount", "Balance"])
    ws.append(headers)

    for col_num, col in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num)
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")

    for row in table_data.get("rows", []):
        ws.append(row)

    for column_cells in ws.columns:
        length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
        ws.column_dimensions[column_cells[0].column_letter].width = length + 2

    wb.save(output_path)
    print(f"📁 Excel file saved to: {output_path}")
