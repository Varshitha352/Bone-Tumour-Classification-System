
import io
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import (
    Flask, render_template, request, jsonify,
    send_file, send_from_directory, url_for
)
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "database" / "predictions.db"
MODEL_PATH = BASE_DIR / "models" / "bone_tumour_cnn.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                benign_score REAL,
                malignant_score REAL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


init_db()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def get_prediction(record_id=None):
    """Return one saved result, or the latest result."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        if record_id is not None:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?",
                (record_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT 1"
            ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["image_url"] = url_for(
        "uploaded_image", filename=result["filename"]
    )
    result["report_url"] = url_for(
        "report", record_id=result["id"]
    )

    return result


def predict_image(image):
    """Use the trained model if available."""
    if not MODEL_PATH.exists():
        return None, None, None, "demo_mode"

    try:
        import torch
        from model import BoneTumourCNN, transform_image

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = BoneTumourCNN()
        state = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        tensor = transform_image(image).unsqueeze(0).to(device)

        with torch.no_grad():
            probabilities = torch.softmax(
                model(tensor), dim=1
            )[0].cpu().numpy()

        predicted_index = int(np.argmax(probabilities))

        # This assumes class index 0 = benign and index 1 = malignant.
        return (
            predicted_index,
            float(probabilities[0]),
            float(probabilities[1]),
            "trained_model"
        )

    except Exception:
        app.logger.exception("Could not load or run the trained model")
        return None, None, None, "model_error"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/results")
def results_page():
    """Show a requested record, or the latest saved record."""
    record_id = request.args.get("id", type=int)

    prediction = get_prediction(record_id)

    if record_id is not None and prediction is None:
        return render_template(
            "results.html",
            prediction=None,
            error="The requested result was not found."
        ), 404

    return render_template(
        "results.html",
        prediction=prediction,
        error=None
    )


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "message": (
            "Model file found."
            if MODEL_PATH.exists()
            else "No trained model found. Train the model before predicting."
        )
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please choose an X-ray image."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({
            "error": "Unsupported format. Use JPG, JPEG, PNG, or WEBP."
        }), 400

    try:
        image = Image.open(uploaded.stream).convert("RGB")
        image.thumbnail((1024, 1024))
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "The uploaded file is not a valid image."
        }), 400

    # Always use .jpg because the image is saved in JPEG format.
    safe_name = f"{uuid.uuid4().hex}.jpg"
    image.save(UPLOAD_DIR / safe_name, format="JPEG", quality=95)

    original_filename = Path(uploaded.filename).name
    idx, benign, malignant, mode = predict_image(image)

    if mode == "trained_model":
        label = (
            "Benign (Osteochondroma)"
            if idx == 0
            else "Malignant (Osteosarcoma)"
        )
        benign_pct = round(benign * 100, 1)
        malignant_pct = round(malignant * 100, 1)
        message = (
            "Model prediction generated. This is a research prototype, "
            "not a medical diagnosis."
        )
    elif mode == "model_error":
        label = "Prediction unavailable"
        benign_pct = None
        malignant_pct = None
        message = (
            "The model file exists, but the model could not produce a "
            "prediction. Check the application terminal for the error."
        )
    else:
        label = "Model not trained"
        benign_pct = None
        malignant_pct = None
        message = (
            "The upload was saved, but no trained model is available. "
            "No tumour classification was generated."
        )

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO predictions (
                filename, predicted_class, benign_score,
                malignant_score, mode, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                safe_name,
                label,
                benign_pct,
                malignant_pct,
                mode,
                datetime.now().isoformat(timespec="seconds")
            )
        )
        record_id = cursor.lastrowid
        conn.commit()

    result = get_prediction(record_id)

    return jsonify({
        "id": record_id,
        "filename": original_filename,
        "prediction": label,
        "benign": benign_pct,
        "malignant": malignant_pct,
        "mode": mode,
        "image_url": result["image_url"],
        "report_url": result["report_url"],
        "results_url": url_for("results_page", id=record_id),
        "message": message
    })


@app.route("/uploads/<path:filename>")
def uploaded_image(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/prediction/<int:record_id>")
def prediction_details(record_id):
    result = get_prediction(record_id)

    if result is None:
        return jsonify({"error": "Record not found."}), 404

    return jsonify(result)


@app.route("/api/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()

    results = []

    for row in rows:
        item = dict(row)
        item["image_url"] = url_for(
            "uploaded_image", filename=item["filename"]
        )
        item["report_url"] = url_for(
            "report", record_id=item["id"]
        )
        results.append(item)

    return jsonify(results)


@app.route("/api/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "DELETE FROM predictions WHERE id = ?", (record_id,)
        )
        conn.commit()

    if cursor.rowcount == 0:
        return jsonify({"error": "Record not found."}), 404

    return jsonify({"deleted": True})


@app.route("/api/report/<int:record_id>")
def report(record_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (record_id,)
        ).fetchone()

    if row is None:
        return jsonify({"error": "Record not found."}), 404

    report_text = (
        "BONE TUMOUR AI - EDUCATIONAL RESEARCH REPORT\n"
        f"Record ID: {row['id']}\n"
        f"Filename: {row['filename']}\n"
        f"Result: {row['predicted_class']}\n"
        f"Benign score: {row['benign_score']}\n"
        f"Malignant score: {row['malignant_score']}\n"
        f"Mode: {row['mode']}\n"
        f"Created: {row['created_at']}\n\n"
        "IMPORTANT: This software is an educational research prototype, "
        "not a medical device. Do not use this report to diagnose, rule "
        "out, or treat any condition. Consult a qualified clinician.\n"
    )

    return send_file(
        io.BytesIO(report_text.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=f"bone_tumour_report_{record_id}.txt"
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
