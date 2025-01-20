import os
import re
import logging
import ollama
import pytesseract
import pdf2image
from collections import defaultdict
from PyPDF2 import PdfReader, PdfWriter
from dotenv import load_dotenv
from datetime import datetime

# Load Environment Variables
load_dotenv()

# Configure Logging
logging.basicConfig(
    filename="document_classification.log",
    level=logging.INFO,
    encoding="utf-8",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

detailed_logger = logging.getLogger("detailed")
detailed_handler = logging.FileHandler("detailed_classification.log", encoding="utf-8")
detailed_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
detailed_logger.addHandler(detailed_handler)
detailed_logger.setLevel(logging.INFO)

def log_message(message):
    print(message)  # Display in terminal/UI
    logging.info(message)
    detailed_logger.info(message)

# Set Paths
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"
POPPLER_PATH = r"D:\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + POPPLER_PATH

main_classified_folder = f"D:\\Jupyter_projects\\unstructured_Multi_pdf_files\\llama_classified_{datetime.now().strftime('%Y%m%d')}"
os.makedirs(main_classified_folder, exist_ok=True)

# Define Categories
allowed_categories = ["Bank Statement", "Utility Bill", "W-2 Form", "Phone Bill", "Invoice", "Address Proof", "Other"]
category_paths = {cat: os.path.join(main_classified_folder, cat) for cat in allowed_categories}
for path in category_paths.values():
    os.makedirs(path, exist_ok=True)

# Keywords for Classification
keywords_w2 = ["W-2", "Wage and Tax Statement", "Social Security Wages", "Employer", "Employee", "Tax Withheld"]
keywords_bank = ["Bank", "Account Statement", "Transaction", "Deposit", "Withdrawal", "Balance", "Statement Period"]
keywords_utility = [
    "Utility Bill", "Billing Statement", "Service Charge",
    "Water", "Sewer", "Electricity", "Power", "Gas",
    "Meter Reading", "Energy Usage", "Total Amount Due", 
    "Payment Due Date", "Billing Summary", "Current Charges",
    "Amount Due", "Account Information", "Billing Date", "Due Date"
]

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of the PDF using OCR."""
    try:
        images = pdf2image.convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        page_texts = [pytesseract.image_to_string(img, config="--psm 6").strip() for img in images]
        return page_texts
    except Exception as e:
        log_message(f"⚠️ Error extracting text from `{pdf_path}`: {e}")
        return None

def extract_category(text):
    """Extracts category from the Llama response text."""
    match = re.search(r"Category: (?P<category>.+?) \(Confidence", text)
    return match.group("category").strip() if match and match.group("category") in allowed_categories else "Other"

def classify_document(text):
    """Classifies a document using keywords and Llama 3."""
    keyword_classification = "Other"

    if any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_w2):
        keyword_classification = "W-2 Form"
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_bank):
        keyword_classification = "Bank Statement"
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_utility):
        keyword_classification = "Utility Bill"

    log_message(f"🔎 Keyword-Based Classification: {keyword_classification}")

    # **Ensure Llama 3 is also used for verification**
    llama_prediction = classify_with_llama3(text)
    log_message(f"🦙 Llama-Based Classification: {llama_prediction}")

    # If keyword classification is uncertain OR if Llama predicts something different, use Llama's result
    if keyword_classification == "Other" or llama_prediction != "Other":
        return llama_prediction

    return keyword_classification


def classify_with_llama3(text):
    """Classifies document content using Llama 3 model."""
    try:
        log_message(f"📡 Sending text to Llama for classification: {text[:100]}...")  # Logging first 100 chars

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": """
                You are an AI document classifier. Categorize the document into:
                - Phone Bill
                - Utility Bill
                - Address Proof
                - Invoice
                - Bank Statement
                - W-2 Form
                - Other

                **Classification Rules:**
                - "W-2 Form" if it mentions terms like "Wage and Tax Statement", "Social Security Tax", "Employee ID".
                - "Utility Bill" if it contains "Billing Statement", "Service Charge", "Meter Reading", "Power Usage", "Total Amount Due".
                - "Bank Statement" if it includes "Account Number", "Deposit", "Withdrawal", "Transaction History".
                - "Phone Bill" if it has "Mobile Services", "Call Summary", "Minutes Used", "Text Messages".
                - "Invoice" if it mentions "Invoice Number", "Billing Period", "Payment Due Date".
                - "Address Proof" if it includes "Residential Address", "Proof of Address", "Utility Service Name".
                - If the document does not fit any category, classify it as "Other".

                **Response Format:**
                **Category: <category> (Confidence: xx%)**
                """},
                {"role": "user", "content": text}
            ]
        )

        log_message(f"📨 Raw Llama Response: {response}")  # ✅ Log entire response

        if "message" in response and "content" in response["message"]:
            result = response["message"]["content"].strip()
            log_message(f"🔍 Parsed Llama Response: {result}")
            return extract_category(result)
        else:
            log_message("⚠️ No content received from Llama API.")
            return "Other"

    except Exception as e:
        log_message(f"⚠️ Llama 3 API Error: {e}")
        return "Other"


def split_pdf_pages(pdf_path):
    """Splits a multi-page PDF into individual pages."""
    pdf_reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(pdf_reader.pages):
        page_filename = f"{pdf_path[:-4]}_page_{i+1}.pdf"
        pdf_writer = PdfWriter()
        pdf_writer.add_page(page)

        with open(page_filename, "wb") as f:
            pdf_writer.write(f)

        pages.append(page_filename)
    return pages

def merge_pdf_pages(category_pdfs, category):
    """Merges classified pages into a single PDF per category."""
    if not category_pdfs:
        return

    output_pdf = os.path.join(category_paths[category], f"{category}_merged.pdf")
    pdf_writer = PdfWriter()
    for pdf in sorted(category_pdfs):
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    with open(output_pdf, "wb") as f:
        pdf_writer.write(f)

    log_message(f"📂 Merged {len(category_pdfs)} pages into `{output_pdf}`")

def process_documents(folder_path, log_message):
    """Processes documents, classifies each page, and merges into respective categories."""
    if not os.path.exists(folder_path):
        log_message(f"❌ Error: Folder `{folder_path}` does not exist.")
        return

    category_page_map = defaultdict(list)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not filename.lower().endswith('.pdf'):
            continue

        log_message(f"📂 Processing file: {filename}")

        split_pages = split_pdf_pages(file_path)
        for page_file in split_pages:
            page_text = extract_text_from_pdf(page_file)
            if not page_text:
                log_message(f"⚠️ No text extracted from `{page_file}`")
                continue

            final_category = classify_document(page_text[0])
            log_message(f"📜 Page `{page_file}` classified as `{final_category}`")

            category_page_map[final_category].append(page_file)

    for category, pdf_list in category_page_map.items():
        merge_pdf_pages(pdf_list, category)


def get_latest_combined_folder():
    """Returns the most recently created classified folder."""
    base_folder = os.path.dirname(main_classified_folder)
    subfolders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    return os.path.join(base_folder, max(subfolders, key=lambda x: os.path.getctime(os.path.join(base_folder, x)))) if subfolders else None

if __name__ == "__main__":
    folder_path = r"D:\Jupyter_projects\unstructured_Multi_pdf_files\dataset\Combined"
    process_documents(folder_path)
