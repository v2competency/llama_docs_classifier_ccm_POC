from flask import Flask, render_template, request, jsonify
import os
import threading
from datetime import datetime
import logging
from document_classification import process_documents, get_latest_combined_folder

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # 🔥 Forces Flask to reload templates

# Store logs in-memory for real-time display
log_messages = []

def log_message(message):
    """Stores logs for UI display."""
    log_messages.append(message)
    logging.info(message)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        folder_path = request.form.get("folder_path")
        if not os.path.exists(folder_path):
            log_message(f"❌ Error: Folder {folder_path} does not exist.")
            return render_template("index.html", error="Folder does not exist.", logs=log_messages)

        threading.Thread(target=process_documents, args=(folder_path, log_message)).start()
        log_message(f"🚀 Classification started for folder: {folder_path}")

    return render_template("index.html", logs=log_messages)

@app.route("/logs")
def get_logs():
    return jsonify(logs=log_messages)

@app.route("/final-folder")
def final_folder():
    latest_folder = get_latest_combined_folder()
    return jsonify({"final_folder": latest_folder})

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    app.run(debug=True)
