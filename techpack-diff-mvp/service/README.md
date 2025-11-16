# Techpack Diff MVP

Production-ready FastAPI backend that compares two garment tech pack PDFs, annotates the latest version, and correlates findings with review spreadsheets. Outputs are grouped under `out/` so downstream workflow engines (n8n, Airflow, etc.) can ingest them directly.

## Features
- PyMuPDF-driven text extraction with automatic header/footer suppression
- Camelot-based BOM and measurement table diffing (row add/remove + cell deltas)
- Measurement tolerance handling (HIGH/MED tiers)
- Lightweight SSIM visual diffing for sketches/callouts
- Evidence thumbnails plus annotated PDF created via PyMuPDF drawing primitives
- Review Excel row mapping with fuzzy matching and priority-based statuses
- Dockerized FastAPI API with `/compare` endpoint returning artifact paths

## Installation
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r service/requirements.txt
```
Install Ghostscript, poppler, and Tesseract locally if you plan to run Camelot + PDF rendering outside Docker.

## Running Locally
```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Usage (cURL)
```bash
curl -X POST "http://localhost:8000/compare" \
  -F "old=@/path/TechPack1.pdf" \
  -F "new=@/path/TechPack2.pdf" \
  -F "review=@/path/ReviewReport.xlsx"
```
Response:
```json
{
  "annotated_pdf": "techpack-diff-mvp/out/job_1700000000/annotated_TechPack2.pdf",
  "change_log": "techpack-diff-mvp/out/job_1700000000/change_log.csv",
  "review_mapping": "techpack-diff-mvp/out/job_1700000000/review_mapping.json",
  "evidence_dir": "techpack-diff-mvp/out/job_1700000000/evidence"
}
```

## n8n Integration Notes
- Use the HTTP Request node to call `POST /compare`
- Chain a Move Binary Data node to download artifacts via the returned paths
- Optional: place the Dockerized service behind an API Gateway and trigger n8n via webhook

## Threshold Tuning
Edit `service/config.yaml` to tweak:
- `text_similarity_threshold`, `significant_text_ratio` for textual noise suppression
- `measurement_*_tolerance_mm` for measurement priorities
- `image_ssim_threshold` to make sketch diffs more/less sensitive
- `domain_keywords` to bias priority scoring for apparel-specific terms

Reload the service after changes; the config is read on each comparison run, so re-deployments are not required when invoking directly via the engine.

## Output Folder Layout
```
out/
└── job_<timestamp>/
    ├── annotated_TechPack2.pdf   # highlighted HIGH+MED changes
    ├── change_log.csv            # structured list of all change records
    ├── review_mapping.json       # Excel row to change-id linkage
    └── evidence/                 # cropped before/after PNGs
```

## Docker
```bash
cd service
docker build -t techpack-diff .
docker run -p 8000:8000 -v "$(pwd)/../out:/app/out" techpack-diff
```

The container already includes Ghostscript, poppler, Tesseract, and libGL dependencies for Camelot/OpenCV.
