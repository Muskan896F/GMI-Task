import os
import pytesseract
from PIL import Image
import numpy as np
import tiktoken
import itertools
from operator import itemgetter

# ----- Tesseract path (update if needed for Windows) -----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Automatically set TESSDATA_PREFIX
tessdata_dir = r"C:\Program Files\Tesseract-OCR\tessdata"
if os.path.exists(tessdata_dir):
    os.environ["TESSDATA_PREFIX"] = tessdata_dir   # ✅ fixed bug

def num_tokens(text, model="gpt-3.5-turbo-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("⚠️ Model not found, falling back to cl100k_base.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def limit_tokens(text, max_tokens=16000):
    return text[:max_tokens] if num_tokens(text) > max_tokens else text

def cluster_list(xs, tolerance=0):
    if tolerance == 0 or len(xs) < 2:
        return [[x] for x in sorted(xs)]
    groups, current_group = [], [xs[0]]
    for x in xs[1:]:
        if x <= current_group[-1] + tolerance:
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]
    groups.append(current_group)
    return groups

def make_cluster_dict(values, tolerance):
    clusters = cluster_list(list(set(values)), tolerance)
    nested_tuples = [[(val, i) for val in cluster] for i, cluster in enumerate(clusters)]
    return dict(itertools.chain(*nested_tuples))

def cluster_objects(xs, tolerance):
    key_fn = lambda x: (x['coordinates'][1] + x['coordinates'][3]) / 2
    values = map(key_fn, xs)
    cluster_dict = make_cluster_dict(values, tolerance)
    get_0, get_1 = itemgetter(0), itemgetter(1)
    cluster_tuples = sorted(((x, cluster_dict.get(key_fn(x))) for x in xs), key=get_1)
    grouped = itertools.groupby(cluster_tuples, key=get_1)
    return [list(map(get_0, v)) for k, v in grouped]

def get_avg_char_width(data):
    height = 1000
    sum_widths = 0.0
    cnt = 0
    for datum in data:
        height = min(height, abs(datum['coordinates'][3] - datum['coordinates'][1]))
        sum_widths += datum['coordinates'][2] - datum['coordinates'][0]
        cnt += len(datum['value'])
    return height / 2, sum_widths // cnt if cnt > 0 else 10

def collate_line(line_chars, tolerance, add_spaces) -> str:
    coll = ""
    last_x1 = 0
    for char in sorted(line_chars, key=lambda x: x['coordinates'][0]):
        coll += ' '
        last_x1 += tolerance
        while last_x1 + tolerance < char['coordinates'][0] and add_spaces:
            coll += " "
            last_x1 += tolerance
        coll += char['value']
        last_x1 = char['coordinates'][2]
    return coll[1:] if add_spaces else coll.strip()

def extract_text(data, add_spaces=True, max_tokens=16000):
    if not data:
        return ""
    min_height, x_tolerance = get_avg_char_width(data)
    doctop_clusters = cluster_objects(data, tolerance=min_height)
    lines = (collate_line(line_chars, x_tolerance, add_spaces) for line_chars in doctop_clusters)
    text = "\n".join(lines)
    return limit_tokens(text, max_tokens)

def extract_text_ocr(image_path, add_spaces=True, max_tokens=16000, lang="eng"):
    """OCR extraction with safe default language 'eng'"""
    if lang.lower() == "en":
        lang = "eng"

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"❌ Could not open image {image_path}: {e}")
        return ""

    ocr_data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)

    data = []
    for i in range(len(ocr_data['text'])):
        text = ocr_data['text'][i].strip()
        try:
            conf = int(ocr_data['conf'][i])
        except ValueError:
            conf = 0
        if text and conf > 50:  # only keep confident OCR results
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            datum = {
                'value': text,
                'coordinates': [x, y, x + w, y + h]
            }
            data.append(datum)

    final_text = extract_text(data, add_spaces, max_tokens)
    if not final_text.strip():
        print(f"⚠️ OCR completed but no text found in: {image_path}")
    return final_text

# ----------------- 🧪 Test -----------------
if __name__ == "__main__":
    input_path = r'C:\Users\HP\Downloads\GMI-TASK\GMI-TASK\gmindia-challlenge-012024-datas\banquepopulaire\avril6BP.jpg'
    output_txt = r'C:\Users\HP\Downloads\GMI-TASK\GMI-TASK\output\ocr_output.txt'

    text = extract_text_ocr(input_path, add_spaces=True, lang="eng")
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)

    print("✅ OCR extraction complete. Text saved to:")
    print(output_txt)
