#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import cv2
import numpy as np
import re
import shutil

# --------- Répertoires ----------
ROOT       = Path(__file__).resolve().parents[1]  # racine du repo
IN_DIR     = ROOT / "data" / "frames_raw"
SORTED_DIR = ROOT / "data" / "frames_sorted"
OUT_CSV    = ROOT / "data" / "frames_metadata.csv"
OUT_JSON   = ROOT / "data" / "frames_metadata.json"
DBG_DIR    = ROOT / "data" / "ocr_debug"

# --------- Params via env ----------
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
OCR_SLEEP  = float(os.getenv("OCR_SLEEP", "0.35"))
VERBOSE    = bool(int(os.getenv("VERBOSE", "1")))
OCR_DEBUG  = bool(int(os.getenv("OCR_DEBUG", "0")))
# ROI demandé en premier (sera suivi d'autres variantes automatiquement)
OCR_ROI    = os.getenv("OCR_ROI", "bottom35").lower()

# Azure endpoint/clé (accepte COMPUTER_VISION_* et AZURE_VISION_* / AZURE_CV_*)
AZ_ENDPOINT = (
    os.getenv("COMPUTER_VISION_ENDPOINT")
    or os.getenv("AZURE_VISION_ENDPOINT")
    or os.getenv("https://REDACTED.cognitiveservices.azure.com/")
    or ""
)
AZ_KEY = (
    os.getenv("COMPUTER_VISION_KEY")
    or os.getenv("AZURE_VISION_KEY")
    or os.getenv("REDACTED_AZURE_KEY")
    or ""
)

if AZ_ENDPOINT and not AZ_ENDPOINT.endswith("/"):
    AZ_ENDPOINT += "/"
OCR_URL = AZ_ENDPOINT + "vision/v3.1/ocr"

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

# ======== Parseur robuste (dates + frame/étage) ========

def _normalize_ocr(s: str) -> str:
    """Corrige quelques confusions OCR courantes avant regex."""
    t = s
    # O/o pris pour 0 dans/près d'une date
    t = re.sub(r'(?<=\d)[Oo](?=[\s\./-]?\d)', '0', t)
    # l/I au milieu de chiffres -> 1
    t = re.sub(r'(?<=\d)[lI](?=\d)', '1', t)
    # variantes typiques de 'Frame'
    t = re.sub(r'Fr?am[e3]', 'Frame', t, flags=re.IGNORECASE)
    # 'F 388' -> 'F388'
    t = re.sub(r'\bF\s+(\d)', r'F\1', t)
    return t

# Dates complètes : YYYY[-./]MM[-./]DD
DATE_YMD = re.compile(r'(?<!\d)(20\d{2})[\s\./-]?(0?[1-9]|1[0-2])[\s\./-]?(0?[1-9]|[12]\d|3[01])(?!\d)')
# Dates réduites : YYYY[-./]MM (sans jour)
DATE_YM  = re.compile(r'(?<!\d)(20\d{2})[\s\./-]?(0?[1-9]|1[0-2])(?!\d)')

# Étages / numéros : Floor F388 / Floor 388 / F388 / Frame: 388 / #388
FRAME_FLOOR   = re.compile(r'\bFloor\s*F?\s*([0-9]{1,4})\b', re.IGNORECASE)
FRAME_F       = re.compile(r'\bF\s*([0-9]{1,6})\b', re.IGNORECASE)
FRAME_GENERIC = re.compile(r'\b(?:Frame|FR|#)\s*[:#]?\s*([0-9]{1,6})\b', re.IGNORECASE)

def _safe_date(y: int, m: int, d: Optional[int]) -> Optional[str]:
    """Valide la date (pas de mois/jour 00, bornes simples). Si jour manquant -> 01."""
    if not (2000 <= y <= 2099): return None
    if not (1 <= m <= 12): return None
    if d is not None and not (1 <= d <= 31): return None
    if d is None: d = 1
    return f"{y:04d}-{m:02d}-{d:02d}"

def parse_date_and_frame(text: str) -> Tuple[Optional[str], Optional[int]]:
    t = _normalize_ocr(text)

    # 1) Date complète
    date: Optional[str] = None
    m = DATE_YMD.search(t)
    if m:
        y, m_, d_ = int(m.group(1)), int(m.group(2)), int(m.group(3))
        date = _safe_date(y, m_, d_)

    # 2) Sinon YYYY-MM (jour=01)
    if date is None:
        m2 = DATE_YM.search(t)
        if m2:
            y, m_ = int(m2.group(1)), int(m2.group(2))
            date = _safe_date(y, m_, None)

    # 3) Frame / étage
    frame: Optional[int] = None
    for rx in (FRAME_FLOOR, FRAME_F, FRAME_GENERIC):
        mf = rx.search(t)
        if mf:
            try:
                frame = int(mf.group(1))
            except ValueError:
                frame = None
            break

    return date, frame

# ================== fin parseur ==================

def enhance(gray: np.ndarray) -> np.ndarray:
    """Boost de lisibilité avant OCR (upscale + sharpen + Otsu)."""
    h, w = gray.shape[:2]
    big   = cv2.resize(gray, (int(w*1.8), int(h*1.8)), interpolation=cv2.INTER_CUBIC)
    den   = cv2.medianBlur(big, 3)
    sharp = cv2.addWeighted(den, 1.5, cv2.GaussianBlur(den, (0, 0), 1.0), -0.5, 0)
    _, bin_ = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_    = cv2.morphologyEx(bin_, cv2.MORPH_OPEN, kernel, iterations=1)
    return bin_

def rest_ocr_image_bytes(image_bytes: bytes) -> List[str]:
    """Appel REST /vision/v3.1/ocr conformément au quickstart."""
    headers = {
        "Ocp-Apim-Subscription-Key": AZ_KEY,
        "Content-Type": "application/octet-stream",
    }
    params = {"language": "unk", "detectOrientation": "true"}
    resp = requests.post(OCR_URL, headers=headers, params=params, data=image_bytes, timeout=30)

    # gestion douce pour préserver le quota gratuit
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

# ---- Candidats ROI essayés automatiquement avant le full ----
ROI_CANDIDATES = {
    "bottom25": lambda h, w: (int(h * 0.75), h, 0, w),
    "bottom35": lambda h, w: (int(h * 0.65), h, 0, w),
    "top25":    lambda h, w: (0, int(h * 0.25), 0, w),
    "top35":    lambda h, w: (0, int(h * 0.35), 0, w),
}

def ocr_one(path: Path) -> OCRResult:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return OCRResult(path.name, None, None, "")

    h, w = img_bgr.shape[:2]

    # ordre d’essai : ROI demandé, puis autres variantes, puis full
    tried_modes = [OCR_ROI] + [m for m in ("bottom35", "top25", "top35", "bottom25") if m != OCR_ROI] + ["full"]

    text, date, frame = "", None, None

    for mode in tried_modes:
        if mode in ROI_CANDIDATES:
            y0, y1, x0, x1 = ROI_CANDIDATES[mode](h, w)
            roi = img_bgr[y0:y1, x0:x1]
        else:
            roi = img_bgr  # "full"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bin_ = enhance(gray)

        ok, buf = cv2.imencode(".png", bin_)
        if not ok:
            continue

        if OCR_DEBUG:
            DBG_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(DBG_DIR / f"dbg_{path.stem}_{mode}.png"), bin_)

        lines = rest_ocr_image_bytes(buf.tobytes())
        text  = "\n".join(lines)
        date, frame = parse_date_and_frame(text)

        if date and frame is not None:
            break  # on a les deux → stop

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
        print("❌ Défini d'abord COMPUTER_VISION_ENDPOINT et COMPUTER_VISION_KEY "
              "(ou AZURE_VISION_*/AZURE_CV_*).")
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
            break  # quota exceeded → on arrête
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

    if bool(int(os.getenv("CLASSIFY_BY_DATE", "1"))):
        classify_into_folders(results)
        print(f"🗂️  Images classées par date dans {SORTED_DIR}")

if __name__ == "__main__":
    main()
