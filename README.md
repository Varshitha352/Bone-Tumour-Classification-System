# Bone Tumour AI — X-ray Classification Research Prototype

A Flask web application with a responsive frontend, upload validation, SQLite history, downloadable text reports, CNN architecture and training script.

## Important
This is an educational/research software scaffold, not a medical device. It does not ship with clinically valid weights. Until you train and validate it on an appropriate, licensed, de-identified, expert-labelled radiograph dataset, the app explicitly reports **Model not trained** instead of inventing a prediction. Never use it to diagnose or treat a person.

## Requirements
- Windows 10/11
- Python 3.10–3.12 (Python 3.11 recommended)
- Internet for initial dependency installation
- Disk space for PyTorch/torchvision packages

## Quick start (Windows)
1. Extract the ZIP to a normal folder (not inside the ZIP preview).
2. Install Python 3.11 and tick "Add Python to PATH".
3. Double-click `run_windows.bat`, or open PowerShell in this folder and run:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python app.py
   ```
4. Open http://127.0.0.1:5000
5. API health check: http://127.0.0.1:5000/api/health

The first installation may take a while because PyTorch is large. Keep the terminal open while using the website.

## Train the CNN
Put real, properly licensed and de-identified images in this layout:
```
dataset/
  train/
    benign/
    malignant/
  val/
    benign/
    malignant/
```
Folder names must be exactly `benign` and `malignant`. Avoid splitting images from the same patient across train and validation sets. Run from the project root:
```powershell
.\.venv\Scripts\Activate.ps1
python training\train_model.py --data dataset --epochs 10 --batch-size 16
```
This writes `models/bone_tumour_cnn.pth`. Restart the app after training. This starter trainer is for experimentation only; it does not ensure good performance or clinical validity. Evaluate with patient-level test data and metrics including sensitivity, specificity, AUROC and calibration, and seek appropriate ethics/privacy review.

## Features
- Home/About/Results pages
- JPG/JPEG/PNG/WEBP image upload
- CNN inference when trained weights exist
- SQLite prediction history (`database/predictions.db`)
- Upload storage (`uploads/`)
- Downloadable text report
- JSON endpoints: `/api/health`, `/api/predict`, `/api/history`

## Troubleshooting
- `py` not recognized: reinstall Python and enable PATH.
- PowerShell activation blocked: use `run_windows.bat` or Command Prompt.
- `No module named torch`: install requirements in the active `.venv`.
- Port 5000 busy: stop the other app or change `port=5000` in `app.py`.
- Upload shows "Model not trained": expected until you train using the dataset layout above.
