import os
import re
import logging
import shutil
import openai
import pytesseract
import cv2
import pdf2image
import time
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load OpenAI API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("❌ OpenAI API key is missing. Set it in the .env file or environment variables.")

client = OpenAI(api_key=openai_api_key)

# Configure logging
logging.basicConfig(
    filename="document_classification.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"

# Set Poppler path for PDF text extraction
POPPLER_PATH = r"D:\poppler-24.08.0\Library\bin"
os.environ["PATH"] += os.pathsep + POPPLER_PATH

# **Updated Keyword Categories**
keywords_w2 = ["W-2", "Wage and Tax Statement", "Social Security Wages", "Employer", "Employee", "Tax Withheld"]
keywords_bank = ["Bank", "Account Statement", "Transaction", "Deposit", "Withdrawal", "Balance", "Statement Period"]
keywords_utility = [
    "Electricity", "Utility Bill", "Billing Statement", "Service Charge",
    "Water", "Sewer Service", "Gas Bill", "Power Usage", "Consumption", "Meter Reading"
]

# **Extract Text from Images**
def extract_text_from_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            logging.error(f"Error: Could not load image {image_path}.")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {image_path}: {e}")
        return None

# **Extract Text from PDFs**
def extract_text_from_pdf(pdf_path):
    try:
        images = pdf2image.convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        full_text = "\n".join(pytesseract.image_to_string(img, config="--psm 6") for img in images)
        return full_text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return None

# **Classify Document using Keywords & OpenAI**
def classify_document(text):
    """First tries keyword-based classification, then falls back to OpenAI GPT"""

    keyword_classification = "Unknown"
    classification_reason = "No keywords matched"

    if any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_w2):
        keyword_classification = "W-2 Form"
        classification_reason = "Matched W-2 related keywords"
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_bank):
        keyword_classification = "Bank Statement"
        classification_reason = "Matched Bank-related keywords"
    elif any(re.search(keyword, text, re.IGNORECASE) for keyword in keywords_utility):
        keyword_classification = "Utility Bill"
        classification_reason = "Matched Utility-related keywords"

    logging.info(f"Keyword-Based Classification: {keyword_classification} (Reason: {classification_reason})")

    openai_classification, confidence_score, token_cost = classify_with_openai(text)

    logging.info(f"Final Classification: Keyword-Based={keyword_classification}, OpenAI={openai_classification}, Confidence={confidence_score}%, Cost=${token_cost}")

    return keyword_classification, openai_classification, confidence_score, token_cost

# **Use OpenAI for Classification**
def classify_with_openai(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": """
                You are an AI document classifier. Categorize the document into:
                - W-2 Form
                - Bank Statement
                - Utility Bill
                - Unknown (if not sure)

                Return the category name exactly as written above, followed by a confidence score.
                Example response: "Bank Statement (Confidence: 98%)"
                """},
                {"role": "user", "content": text}
            ],
            max_tokens=20,
            temperature=0.5
        )

        result = response.choices[0].message.content.strip()
        match = re.search(r"(W-2 Form|Bank Statement|Utility Bill|Unknown)\s*\(Confidence:\s*(\d+)%\)", result)
        category = match.group(1) if match else "Unknown"
        confidence_score = int(match.group(2)) if match else 60

        token_usage = response.usage.total_tokens  
        token_cost = round((token_usage / 1000) * 0.03, 4)  

        return category, confidence_score, token_cost

    except openai.OpenAIError as e:
        logging.error(f"⚠️ OpenAI API error: {e}")
        return "Unknown", 60, 0.00 

# **Copy Classified Files to the Correct Folder**
def copy_file_to_classified_folder(file_path, category, base_folder):
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        classified_folder = os.path.join(base_folder, f"Combined_{date_str}")

        if not os.path.exists(classified_folder):
            os.makedirs(classified_folder)
            logging.info(f"📁 Created base classified folder: {classified_folder}")

        category_folder = os.path.join(classified_folder, category)

        if not os.path.exists(category_folder):
            os.makedirs(category_folder)
            logging.info(f"📁 Created category folder: {category_folder}")

        filename = os.path.basename(file_path)
        target_path = os.path.join(category_folder, filename)

        if not os.path.exists(target_path):
            shutil.copy2(file_path, target_path)
            logging.info(f"✅ Copied {file_path} → {target_path}")
            print(f"✅ Copied {filename} to {category}")
        else:
            logging.warning(f"⚠️ File {filename} already exists in {category_folder}, skipping copy.")

    except Exception as e:
        logging.error(f"❌ Error copying {file_path} to {category_folder}: {e}")

# **Process Documents**
def process_documents(folder_path, log_callback):
    if not os.path.exists(folder_path):
        log_callback(f"❌ Error: Folder path {folder_path} does not exist.")
        return

    files = os.listdir(folder_path)
    log_callback(f"📂 Found {len(files)} files in the folder.")

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        extracted_text = None

        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            extracted_text = extract_text_from_image(file_path)
        elif filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(file_path)
        else:
            log_callback(f"⚠️ Skipping unsupported file: {filename}")
            continue

        if extracted_text:
            keyword_category, openai_category, confidence, cost = classify_document(extracted_text)
            log_callback(f"📄 Processed: {filename}")
            log_callback(f"🔍 Keyword-Based Category: {keyword_category}")
            log_callback(f"🤖 OpenAI-Based Category: {openai_category} | Confidence: {confidence}% | Cost: ${cost}\n")

            copy_file_to_classified_folder(file_path, openai_category, folder_path)
        else:
            log_callback(f"⚠️ No text extracted from {filename}.")

def get_latest_combined_folder():
    base_folder = "D:\\Jupyter_projects\\unstructured_Multi_pdf_files\\dataset\\CCM_Test Data\\CCM_Test Data\\Combined"
    subfolders = [f for f in os.listdir(base_folder) if f.startswith("Combined_")]
    if not subfolders:
        return "No classified folder found."

    latest_folder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(base_folder, x)))
    return os.path.join(base_folder, latest_folder)

# **Run the Script**
if __name__ == "__main__":
    folder_path = r"D:\Jupyter_projects\unstructured_Multi_pdf_files\dataset\CCM_Test Data\CCM_Test Data\Combined"
    process_documents(folder_path)
