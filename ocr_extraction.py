#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv, json, io
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import cv2
import numpy as np
import re
import shutil

# --------- Répertoires ----------
ROOT = Path(__file__).resolve().parents[1]     # racine du repo
IN_DIR = ROOT / "data" / "frames_raw"
SORTED_DIR = ROOT / "data" / "frames_sorted"
OUT_CSV = ROOT / "data" / "frames_metadata.csv"
OUT_JSON = ROOT / "data" / "frames_metadata.json"
DBG_DIR = ROOT / "data" / "ocr_debug"

# --------- Regex pour date & frame ----------
# 2022-03-05, 2022/3/5, 2022.03.05 (mois 1..12, jour 1..31 — évite 00)
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[./-]?(0?[1-9]|1[0-2])[./-]?(0?[1-9]|[12]\d|3[01])(?!\d)")
# “F388”, “Frame: 388”, “#388”, “388” précédé de F ou mot frame
FRAME_RE = re.compile(r"(?:F(?:rame)?\s*[:#]?\s*|#\s*)(\d{1,6})")

# --------- Params via env ----------
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
OCR_SLEEP  = float(os.getenv("OCR_SLEEP", "0.35"))
VERBOSE    = bool(int(os.getenv("VERBOSE", "1")))
OCR_DEBUG  = bool(int(os.getenv("OCR_DEBUG", "0")))

# ROI mode par défaut (sera essayé en premier)
OCR_ROI = os.getenv("OCR_ROI", "bottom25").lower()

# Azure endpoint/clé (accepte COMPUTER_VISION_*, AZURE_VISION_* et AZURE_CV_*)
AZ_ENDPOINT = (
    os.getenv("COMPUTER_VISION_ENDPOINT")
    or os.getenv("AZURE_VISION_ENDPOINT")
    or os.getenv("code go here")
    or ""
)
AZ_KEY = (
    os.getenv("COMPUTER_VISION_KEY")
    or os.getenv("AZURE_VISION_KEY")
    or os.getenv("")
    or ""
)
OCR_URL = AZ_ENDPOINT.rstrip("/") + "/vision/v3.1/ocr"

@dataclass
class OCRResult:
    filename: str
    date: Optional[str]
    frame_number: Optional[int]
    raw: str

# --------- Utils ----------
def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def parse_date_and_frame(text: str) -> Tuple[Optional[str], Optional[int]]:
    date = None
    frame = None
    m_date = DATE_RE.search(text)
    if m_date:
        y, m, d = m_date.group(1), m_date.group(2), m_date.group(3)
        date = f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    m_fr = FRAME_RE.search(text)
    if m_fr:
        try:
            frame = int(m_fr.group(1))
        except ValueError:
            frame = None
    return date, frame

def crop_roi(img: np.ndarray, mode: str) -> np.ndarray:
    """Bandeau ROI simple (le texte est toujours au même endroit)."""
    h, w = img.shape[:2]
    if mode == "bottom25":
        y0 = int(h * 0.75)
        return img[y0:h, 0:w]
    if mode == "top25":
        y1 = int(h * 0.25)
        return img[0:y1, 0:w]
    return img  # full

def enhance(gray: np.ndarray) -> np.ndarray:
    """Boost de lisibilité avant OCR (upscale + sharpen + Otsu)."""
    h, w = gray.shape[:2]
    big = cv2.resize(gray, (int(w*1.8), int(h*1.8)), interpolation=cv2.INTER_CUBIC)
    den = cv2.medianBlur(big, 3)
    sharp = cv2.addWeighted(den, 1.5, cv2.GaussianBlur(den, (0, 0), 1.0), -0.5, 0)
    _, bin_ = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_ = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_

def rest_ocr_image_bytes(image_bytes: bytes) -> List[str]:
    """Appel REST /vision/v3.1/ocr conformément au quickstart."""
    headers = {
        "Ocp-Apim-Subscription-Key": AZ_KEY,
        "Content-Type": "application/octet-stream",
    }
    params = {"language": "unk", "detectOrientation": "true"}
    resp = requests.post(OCR_URL, headers=headers, params=params, data=image_bytes, timeout=30)
    # Gestion douce des erreurs pour éviter de brûler le quota
    if resp.status_code in (429, 500, 503):
        if VERBOSE:
            print(f"[REST] {resp.status_code} -> backoff 1.2s")
        time.sleep(1.2)
        return []
    if resp.status_code == 403:
        print("⛔ 403 Quota Exceeded (arrêt pour éviter des coûts).")
        raise SystemExit(1)
    resp.raise_for_status()
    analysis = resp.json()
    lines_out: List[str] = []
    for region in analysis.get("regions", []):
        for line in region.get("lines", []):
            words = [w.get("text", "") for w in line.get("words", [])]
            s = " ".join(w for w in words if w)
            if s.strip():
                lines_out.append(s)
    return lines_out

# ---- ICI: candidats ROI essayés automatiquement avant le full ----
ROI_CANDIDATES = {
    "bottom25": lambda h, w: (int(h * 0.75), h, 0, w),
    "bottom35": lambda h, w: (int(h * 0.65), h, 0, w),
    "top25":    lambda h, w: (0, int(h * 0.25), 0, w),
}

def ocr_one(path: Path) -> OCRResult:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return OCRResult(path.name, None, None, "")

    h, w = img_bgr.shape[:2]
    # ordre d’essai : ROI demandé en premier, puis variantes
    tried_modes = [OCR_ROI] + [m for m in ("bottom35", "bottom25", "top25") if m != OCR_ROI]

    text, date, frame = "", None, None

    for mode in tried_modes:
        if mode in ROI_CANDIDATES:
            y0, y1, x0, x1 = ROI_CANDIDATES[mode](h, w)
            roi = img_bgr[y0:y1, x0:x1]
        else:
            roi = img_bgr  # fallback improbable

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bin_ = enhance(gray)

        ok, buf = cv2.imencode(".png", bin_)
        if not ok:
            continue

        if OCR_DEBUG:
            DBG_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DBG_DIR / f"dbg_{path.stem}_{mode}.png"), bin_)

        lines = rest_ocr_image_bytes(buf.tobytes())
        text = "\n".join(lines)
        date, frame = parse_date_and_frame(text)

        # si les deux trouvés → on s'arrête
        if date and frame is not None:
            break

    # ultime fallback : image complète
    if (date is None or frame is None):
        gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        ok2, buf2 = cv2.imencode(".png", enhance(gray_full))
        if ok2:
            lines2 = rest_ocr_image_bytes(buf2.tobytes())
            text2 = "\n".join(lines2)
            d2, f2 = parse_date_and_frame(text2)
            if d2 and (f2 is not None):
                date, frame, text = d2, f2, text2

    if VERBOSE:
        print(f"[{path.name}] date={date} frame={frame}")

    return OCRResult(path.name, date, frame, text)

def classify_into_folders(results: List[OCRResult]):
    """Optionnel: classer les images dans data/frames_sorted/<date>/"""
    for r in results:
        if not r.date:
            continue
        src = IN_DIR / r.filename
        if not src.exists():
            continue
        dst_dir = SORTED_DIR / r.date
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / r.filename
        if not dst.exists():
            shutil.copy2(src, dst)

def main():
    if not AZ_ENDPOINT or not AZ_KEY:
        print("❌ Défini d'abord COMPUTER_VISION_ENDPOINT et COMPUTER_VISION_KEY (ou AZURE_VISION_*/AZURE_CV_*).")
        sys.exit(1)

    imgs = list_images(IN_DIR)
    if not imgs:
        print(f"❌ Aucune image dans {IN_DIR}")
        sys.exit(1)
    if MAX_IMAGES > 0:
        imgs = imgs[:MAX_IMAGES]

    print(f"📂 {len(imgs)} image(s) depuis {IN_DIR}")
    print(f"🌐 Endpoint: {AZ_ENDPOINT}")
    print(f"🔑 Key head/tail: {AZ_KEY[:6]}...{AZ_KEY[-6:]}")
    print(f"🧭 ROI mode: {OCR_ROI}")

    rows, results = [], []
    for i, p in enumerate(imgs, 1):
        try:
            rec = ocr_one(p)
        except SystemExit:
            # quota exceeded -> stoppe pour éviter des coûts
            break
        except Exception as e:
            if VERBOSE:
                print(f"❌ OCR error {p.name}: {e}")
            rec = OCRResult(p.name, None, None, "")
        results.append(rec)
        rows.append({"filename": rec.filename, "date": rec.date, "frame_number": rec.frame_number})

        if i % 25 == 0 or i == len(imgs):
            print(f"[OCR] {i}/{len(imgs)} processed...")
        time.sleep(OCR_SLEEP)

    # Sauvegardes
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "date", "frame_number"])
        w.writeheader(); w.writerows(rows)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in rows if r["date"] and r["frame_number"] is not None)
    print(f"✅ {ok}/{len(rows)} lignes avec date+frame")
    print(f"📝 CSV : {OUT_CSV}")
    print(f"📝 JSON: {OUT_JSON}")

    # Classement dans frames_sorted (facultatif via env)
    if bool(int(os.getenv("CLASSIFY_BY_DATE", "1"))):
        classify_into_folders(results)
        print(f"🗂️  Images classées par date dans {SORTED_DIR}")

if __name__ == "__main__":
    main()
