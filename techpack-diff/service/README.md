# TechPack Diff Service

A production-grade backend that semantically compares two garment tech pack PDFs, detects structural/textual/visual differences, and produces artifacts suitable for downstream automation (n8n, UI front-ends, QA workflows).

## Highlights
- **Multi-modal page fingerprinting** (text + tables + perceptual thumbnail hash) eliminates fragile page-index assumptions.
- **Key-based BOM diffing** with part-ID heuristics and numeric delta detection for measurements/quantities.
- **Sketch-aware visual diffing** that only evaluates overlapping callout regions and ignores minor pixel noise.
- **Global block fallback** ensures orphaned pages still find legacy references via semantic block searches.
- **Diff PDF output** annotates TP2 pages (left panel) and injects a sidebar containing clickable change references.
- **FastAPI webhook** tailored for n8n plus standalone `/compare` endpoint for manual uploads.

## Setup
```powershell
cd service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Install system dependencies when running outside Docker: Ghostscript, Poppler, Tesseract, and libGL (Windows users can leverage the Docker image instead).

## Running Locally
```powershell
uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
```

Health probe: `GET /health`

### Manual Comparison Endpoint
```powershell
curl -X POST http://localhost:8000/compare ^
     -F "techpack1=@sample_data/TechPack1.pdf" ^
     -F "techpack2=@sample_data/TechPack2.pdf"
```
Response:
```json
{
  "job_id": "job_1731800000000",
  "changes_json": ".../out/job_1731800000000/changes.json",
  "diff_pdf": ".../out/job_1731800000000/diff.pdf",
  "page_mapping": {"0": [[0, 0.94]]}
}
```

### n8n Webhook Flow
1. Create an **HTTP Request** node pointed at `POST /webhook/compare`.
2. Send `techpack1` and `techpack2` as binary file fields (multipart/form-data).
3. Consume the JSON response and follow-up with `HTTP Request` nodes (or `Move Binary Data`) to download `changes_json_url` and `diff_pdf_url` from the local file paths or your mounted volume.

#### Example n8n Node Configuration
- Method: `POST`
- URL: `http://service-host:8000/webhook/compare`
- Authentication: none (behind VPN/VNet recommended)
- Body Content Type: `multipart-form-data`
- Binary Property: `data` (map `techpack1` / `techpack2` from upstream nodes)

## Threshold Tuning
Modify `service/config.yaml` to adjust sensitivity:
- `page_combined_score_threshold`: tighten/relax multi-modal page matching.
- `text_similarity_threshold` & `block_fallback_threshold`: govern text block matches and fallback semantics.
- `ssim_threshold` & `ssim_min_area`: tune visual diff strictness and ignore tiny sketches.
- `annotation_colors`: customize RGB overlays for added/removed/modified highlights.
- `llm_verification_enabled`: toggle the optional `llm_verifier` hook for external QA models.

Changes are hot-loaded on each comparison run—no server restart required.

## Docker
```powershell
cd service
docker build -t techpack-diff .
docker run -p 8000:8000 -v "${PWD}/../out:/app/out" techpack-diff
```
The container ships with Ghostscript, Poppler, Tesseract, and libGL, so Camelot and PyMuPDF work out of the box.

## Outputs
Each job lives under `out/job_<timestamp>/` and contains:
- `changes.json` – canonical list of change objects for UI binding.
- `diff.pdf` – annotated TP2 pages with sidebar change references.
- `page_mapping.json` – serialized semantic mapping for each TP2 page.

## Sample Data
`sample_data/` ships with placeholder PDFs—replace them with real TechPacks for realistic tests.
