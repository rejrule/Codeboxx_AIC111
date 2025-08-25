#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import requests
import cv2
import numpy as np
import re
import shutil
import math
import random

# --------- Répertoires ----------
ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "frames_raw"
SORTED_DIR = ROOT / "data" / "frames_sorted"
OUT_CSV = ROOT / "data" / "frames_metadata.csv"
OUT_JSON = ROOT / "data" / "frames_metadata.json"
DBG_DIR = ROOT / "data" / "ocr_debug"

# --------- Params via env ----------
MAX_IMAGES            = int(os.getenv("MAX_IMAGES", "0"))
OCR_SLEEP             = float(os.getenv("OCR_SLEEP", "0.35"))
VERBOSE               = bool(int(os.getenv("VERBOSE", "1")))
OCR_DEBUG             = bool(int(os.getenv("OCR_DEBUG", "0")))
MAX_CALLS_PER_FIELD   = int(os.getenv("MAX_CALLS_PER_FIELD", "8"))  # essais/variantes par champ
TIMEOUT_POLL_SECONDS  = float(os.getenv("READ_TIMEOUT", "25"))
MAX_RETRIES_HTTP      = int(os.getenv("MAX_RETRIES_HTTP", "4"))     # backoff 429/5xx

# Azure endpoint/clé (COMPUTER_VISION_* préféré)
AZ_ENDPOINT = (
    os.getenv("COMPUTER_VISION_ENDPOINT")
    or os.getenv("AZURE_VISION_ENDPOINT")
    or os.getenv("AZURE_CV_ENDPOINT")
    or ""
)
AZ_KEY = (
    os.getenv("COMPUTER_VISION_KEY")
    or os.getenv("AZURE_VISION_KEY")
    or os.getenv("AZURE_CV_KEY")
    or ""
)

# Read v3.2 (analyse + polling)
READ_ANALYZE_URL = AZ_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

@dataclass
class OCRResult:
    filename: str
    date: Optional[str]
    frame_number: Optional[int]
    raw: str

# ================== Parsing robuste ==================

def _normalize_ocr(s: str) -> str:
    """Corrections courantes avant regex."""
    t = s

    # Confusions de caractères
    t = re.sub(r'(?<=\d)[Oo](?=[\s\./:\-]?\d)', '0', t)   # O→0 entre chiffres
    t = re.sub(r'(?<=\d)[lI](?=\d)', '1', t)             # l/I→1 entre chiffres
    t = re.sub(r'Fr?am[e3]', 'Frame', t, flags=re.I)     # Fram3→Frame
    t = re.sub(r'\bF\s+(\d)', r'F\1', t)                 # F 388 → F388

    # Unifier tirets unicode & enlever espaces invisibles
    t = t.replace('\u2010','-').replace('\u2011','-').replace('\u2012','-') \
         .replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')
    t = t.replace('\u200b','').replace('\u200c','').replace('\u2060','')

    # Unifier tous les séparateurs entre chiffres en '-'
    t = re.sub(r'(?<=\d)[\s\.:/\\\-\u2010-\u2015\u2212]+(?=\d)', '-', t)

    # Années "croquées" (début de ligne/bandeau)
    t = re.sub(r'(?m)^(?<!\d)0(2\d)(?=[\s\./:\-])', r'20\1', t)   # 022-.. -> 2022-..
    t = re.sub(r'(?m)^(?<!\d)(2\d)(?=[\s\./:\-])', r'20\1', t)    # 22-..  -> 2022-..

    return t

# Dates (permissives sur séparateurs)
DATE_YMD_TIGHT = re.compile(r'(?<!\d)(20\d{2})(0?\d|1[0-2])(0?\d|[12]\d|3[01])(?!\d)')
DATE_YMD       = re.compile(r'(?<!\d)(20\d{2})[\s\./:\-]*(0?\d|1[0-2])[\s\./:\-]*(0?\d|[12]\d|3[01])(?!\d)')
DATE_YM        = re.compile(r'(?<!\d)(20\d{2})[\s\./:\-]*(0?\d|1[0-2])(?!\d)')
# Fallback année sur 2 chiffres (YY-M-D) si vraiment nécessaire
DATE_YMD_2DIG  = re.compile(r'(?<!\d)(\d{2})[\s\./:\-]*(0?\d|1[0-2])[\s\./:\-]*(0?\d|[12]\d|3[01])(?!\d)')

# Frames
FRAME_FLOOR   = re.compile(r'\bFloor\s*F?\s*([0-9]{1,4})\b', re.I)
FRAME_F       = re.compile(r'\bF\s*([0-9]{1,4})\b', re.I)
FRAME_GENERIC = re.compile(r'\b(?:Fr?am[e3]|FR|F|#)\s*[:#\-]?\s*([0-9]{1,6})\b', re.I)

def _safe_date(y: int, m: int, d: Optional[int]) -> Optional[str]:
    if not (2000 <= y <= 2099): return None
    if not (1 <= m <= 12): return None
    if d is not None and not (1 <= d <= 31): return None
    if d is None: d = 1
    return f"{y:04d}-{m:02d}-{d:02d}"

def parse_date_and_frame(text: str) -> Tuple[Optional[str], Optional[int]]:
    t = _normalize_ocr(text)

    # Nettoyage léger pour nombres type 'F1-7)' -> 'F17'
    t = re.sub(r'F\s*([0-9])[^\d]{0,2}([0-9])', lambda m: f"F{m.group(1)}{m.group(2)}", t)

    # --- Date
    date: Optional[str] = None
    m0 = DATE_YMD_TIGHT.search(t)
    if m0:
        date = _safe_date(int(m0.group(1)), int(m0.group(2)), int(m0.group(3)))
    if date is None:
        m = DATE_YMD.search(t)
        if m:
            date = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if date is None:
        m2 = DATE_YM.search(t)
        if m2:
            date = _safe_date(int(m2.group(1)), int(m2.group(2)), None)
    if date is None:
        m3 = DATE_YMD_2DIG.search(t)
        if m3:
            yy = int(m3.group(1))
            year = 2000 + yy  # heuristique vidéo récente
            date = _safe_date(year, int(m3.group(2)), int(m3.group(3)))

    # --- Frame
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

# ================== Pré-traitements ==================

def _clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def _sharpen(img: np.ndarray, alpha=1.5, sigma=1.0) -> np.ndarray:
    gauss = cv2.GaussianBlur(img, (0,0), sigma)
    return cv2.addWeighted(img, alpha, gauss, -alpha + 1.0, 0)

def _to_gray_variants(bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """Retourne ('gray'|'L'|'V', image_gris)"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    L = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:,:,0]
    V = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:,:,2]
    return [("gray", gray), ("L", L), ("V", V)]

def _preprocess_variants_from_gray(gray: np.ndarray) -> List[np.ndarray]:
    """
    Renvoie une liste de variantes (du plus doux au plus agressif).
    Texte noir → on travaille principalement en *inverse*.
    """
    variants: List[np.ndarray] = []

    # upscale
    h, w = gray.shape[:2]
    big = cv2.resize(gray, (int(w*1.8), int(h*1.8)), interpolation=cv2.INTER_CUBIC)

    # v1: CLAHE + Otsu inverse
    g1 = _clahe(big)
    _, b1 = cv2.threshold(g1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(b1)

    # v2: bilateral + sharpen + Otsu inverse + petite dilatation verticale
    g2 = cv2.bilateralFilter(big, d=5, sigmaColor=35, sigmaSpace=35)
    g2 = _sharpen(g2, 1.6, 1.0)
    _, b2 = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    b2 = cv2.dilate(b2, np.ones((2,1), np.uint8), iterations=1)
    variants.append(b2)

    # v3: Adaptive inverse (pour éclairage difficile)
    b3 = cv2.adaptiveThreshold(
        _clahe(big), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 7
    )
    variants.append(b3)

    return variants

def _deskew(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation légère pour rattraper les bascules (~±3°)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# ================== Read API (analyse + polling) ==================

def _sleep_backoff(attempt: int):
    # backoff expo avec jitter
    base = 0.8 * (2 ** attempt)
    time.sleep(min(8.0, base + random.random() * 0.3))

def read_ocr(image_bytes: bytes, want_geometry: bool=False):
    """Appel Read v3.2 avec polling; retourne lignes ou blocs avec bbox."""
    headers = {
        "Ocp-Apim-Subscription-Key": AZ_KEY,
        "Content-Type": "application/octet-stream",
    }

    # POST avec retry/backoff
    post_resp = None
    for attempt in range(MAX_RETRIES_HTTP):
        try:
            post_resp = requests.post(READ_ANALYZE_URL, headers=headers, data=image_bytes, timeout=20)
        except Exception as e:
            if VERBOSE: print(f"[READ] POST error: {e}")
            _sleep_backoff(attempt)
            continue

        if post_resp.status_code == 403:
            print("⛔ 403 Quota Exceeded (arrêt pour éviter des coûts).")
            raise SystemExit(1)
        if post_resp.status_code in (429, 500, 503):
            if VERBOSE: print(f"[READ] POST {post_resp.status_code} -> backoff")
            _sleep_backoff(attempt)
            continue
        if post_resp.status_code not in (202, 200):
            if VERBOSE: print(f"[READ] POST status={post_resp.status_code}: {post_resp.text[:200]}")
            _sleep_backoff(attempt)
            continue
        break
    else:
        return []  # échec POST après retries

    resp = post_resp
    op_url = resp.headers.get("Operation-Location")
    if not op_url:
        if VERBOSE: print("[READ] Missing Operation-Location")
        return []

    # Polling
    t0 = time.time()
    attempt = 0
    while time.time() - t0 < TIMEOUT_POLL_SECONDS:
        try:
            r = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": AZ_KEY}, timeout=10)
        except Exception as e:
            if VERBOSE: print(f"[READ] GET error: {e}")
            _sleep_backoff(attempt); attempt += 1
            continue

        if r.status_code in (429, 500, 503):
            if VERBOSE: print(f"[READ] GET {r.status_code} -> backoff")
            _sleep_backoff(attempt); attempt += 1
            continue

        if r.status_code != 200:
            if VERBOSE: print(f"[READ] GET status={r.status_code}")
            time.sleep(0.6)
            continue

        data = r.json()
        status = data.get("status", "")
        if status.lower() == "succeeded":
            if not want_geometry:
                lines: List[str] = []
                for blk in data.get("analyzeResult", {}).get("readResults", []):
                    for line in blk.get("lines", []):
                        txt = line.get("text", "")
                        if txt.strip():
                            lines.append(txt)
                return lines
            else:
                blocks: List[Dict[str,Any]] = []
                for blk in data.get("analyzeResult", {}).get("readResults", []):
                    for line in blk.get("lines", []):
                        blocks.append({
                            "text": line.get("text",""),
                            "bbox": line.get("boundingBox", []),  # [x1,y1,x2,y2,x3,y3,x4,y4]
                            "words": [{"text": w.get("text",""), "bbox": w.get("boundingBox", [])}
                                      for w in line.get("words",[])]
                        })
                return blocks

        if status.lower() == "failed":
            if VERBOSE: print("[READ] status=failed")
            return []
        time.sleep(0.5)

    if VERBOSE: print("[READ] timeout polling")
    return []

def read_ocr_lines(image_bytes: bytes) -> List[str]:
    """Wrapper compat : ne retourne que le texte (lignes)."""
    out = read_ocr(image_bytes, want_geometry=False)
    return out if isinstance(out, list) else []

# ================== ROIs ==================

# Zones fixes (en proportion de l'image). Date en haut-gauche, Frame en haut-droit.
DATE_ROIS  = [
    (0.00, 0.18, 0.00, 0.40),  # TL serré
    (0.00, 0.20, 0.02, 0.45),  # TL un peu plus large et décalé
]

FRAME_ROIS = [
    (0.00, 0.18, 0.65, 1.00),  # TR serré
    (0.02, 0.22, 0.62, 1.00),  # TR un peu plus bas et large
]

# Bandeau haut plein : fallback unique pour récupérer date + frame d’un coup
TOP_STRIP_ROIS = [
    (0.00, 0.22, 0.00, 1.00),  # élargi (était 0.18)
    (0.00, 0.24, 0.00, 1.00),  # élargi (était 0.20)
]

def _crop_norm(bgr: np.ndarray, roi: Tuple[float,float,float,float]) -> np.ndarray:
    h, w = bgr.shape[:2]
    y0 = int(round(h * roi[0])); y1 = int(round(h * roi[1]))
    x0 = int(round(w * roi[2])); x1 = int(round(w * roi[3]))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    if y1 <= y0 or x1 <= x0:
        return bgr.copy()
    return bgr[y0:y1, x0:x1]

def _save_dbg(img: np.ndarray, tag: str):
    if not OCR_DEBUG: return
    DBG_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DBG_DIR / f"{tag}.png"), img)

def _try_roi(bgr: np.ndarray, roi, variant_id: int, tag: str) -> Tuple[str, List[str]]:
    """
    Essaie plusieurs canaux (gray/L/V) et variantes.
    Retourne texte concaténé + liste de lignes.
    """
    crop = _crop_norm(bgr, roi)

    collected_lines: List[str] = []
    best_text: str = "None"

    for ch_name, gray in _to_gray_variants(crop):
        variants = _preprocess_variants_from_gray(gray)

        # borne variant_id si appelé à l'ancienne
        v_idx = max(0, min(variant_id, len(variants)-1))
        # on essaie d'abord la variante demandée, puis les autres
        order = list(range(len(variants)))
        if v_idx != 0:
            order.remove(v_idx); order.insert(0, v_idx)

        for idx in order:
            var = variants[idx]

            if OCR_DEBUG:
                y0,y1,x0,x1 = roi
                dbg_tag = f"roi_{tag}_y{y0:.2f}-{y1:.2f}_x{x0:.2f}-{x1:.2f}_{ch_name}_v{idx+1}"
                _save_dbg(var, dbg_tag)

            ok, buf = cv2.imencode(".png", var)
            if not ok:
                continue
            lines = read_ocr_lines(buf.tobytes())
            if lines:
                collected_lines.extend(lines)
                best_text = "\n".join(collected_lines)
                if VERBOSE:
                    y0,y1,x0,x1 = roi
                    print(f"   [{tag}] roi=y{y0:.2f}-{y1:.2f}_x{x0:.2f}-{x1:.2f} {ch_name} v{idx+1} → '{best_text[:80]}'  ({len(lines)} lines)")
                return best_text, collected_lines

    if VERBOSE:
        y0,y1,x0,x1 = roi
        print(f"   [{tag}] roi=y{y0:.2f}-{y1:.2f}_x{x0:.2f}-{x1:.2f} → None (0 lines)")
    return best_text, collected_lines

def _deskew_try(bgr: np.ndarray, roi, angles=(+3.0, +2.0, +1.0, -1.0, -2.0, -3.0)) -> Tuple[Optional[str], Optional[int]]:
    """Ultime recours : deskew ±3° sur top strip, parse date+frame."""
    crop = _crop_norm(bgr, roi)
    for ang in angles:
        rot = _deskew(crop, ang)
        for ch_name, gray in _to_gray_variants(rot):
            for v_id, var in enumerate(_preprocess_variants_from_gray(gray)):
                if OCR_DEBUG:
                    y0,y1,x0,x1 = roi
                    _save_dbg(var, f"deskew_{ang:+.1f}_{ch_name}_v{v_id+1}")
                ok, buf = cv2.imencode(".png", var)
                if not ok: continue
                lines = read_ocr_lines(buf.tobytes())
                if not lines: continue
                d, f = parse_date_and_frame("\n".join(lines))
                if d or f is not None:
                    return d, f
    return None, None

# --------- Heuristique spatiale (avec bounding boxes) ---------

def _pick_date_frame_spatial(blocks: List[Dict[str,Any]], crop_w: int, crop_h: int) -> Tuple[Optional[str], Optional[int]]:
    """Sélectionne la DATE (haut-gauche) et FRAME (gros nombre haut-droite) via bbox normalisées."""
    best_frame = None
    best_frame_x = -1.0
    candidate_dates: List[Tuple[float,float,str]] = []  # (y_norm, x_norm, date)

    for b in blocks:
        txt = (b.get("text") or "").strip()
        bb  = b.get("bbox") or []
        if not txt or len(bb) != 8:
            continue
        xs = bb[0::2]; ys = bb[1::2]
        x_center = sum(xs)/4.0; y_center = sum(ys)/4.0
        x_norm = x_center / float(crop_w) if crop_w else 0.0
        y_norm = y_center / float(crop_h) if crop_h else 0.0

        # Frame: nombre 3-6 chiffres dans le quart haut, privilégier le plus à droite
        m = re.search(r'\b([0-9]{3,6})\b', txt)
        if m and y_norm < 0.28:
            if x_norm > best_frame_x:
                try:
                    cand = int(m.group(1))
                except:
                    cand = None
                if cand is not None:
                    best_frame_x = x_norm
                    best_frame = cand

        # Date: dans le haut-gauche, toute chaîne qui "ressemble" à une date
        d, _ = parse_date_and_frame(txt)
        if d and y_norm < 0.28 and x_norm < 0.55:
            candidate_dates.append( (y_norm, x_norm, d) )

    best_date = None
    if candidate_dates:
        candidate_dates.sort(key=lambda t: (t[0], t[1]))  # le plus haut puis le plus à gauche
        best_date = candidate_dates[0][2]

    return best_date, best_frame

def _try_roi_spatial(bgr: np.ndarray, roi, tag: str) -> Tuple[Optional[str], Optional[int]]:
    """Lis le top-strip/crop en mode geometry et applique l'heuristique spatiale."""
    crop = _crop_norm(bgr, roi)
    h, w = crop.shape[:2]

    # IMPORTANT: pour de meilleures bbox, on envoie le crop "naturel" (non binarisé)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None, None
    blocks = read_ocr(buf.tobytes(), want_geometry=True)
    if not isinstance(blocks, list) or not blocks:
        return None, None

    d, f = _pick_date_frame_spatial(blocks, w, h)
    if VERBOSE:
        y0,y1,x0,x1 = roi
        print(f"   [{tag}-spatial] roi=y{y0:.2f}-{y1:.2f}_x{x0:.2f}-{x1:.2f} → date={d} frame={f}")
    return d, f

# ================== OCR pipeline par image ==================

def ocr_one(path: Path) -> OCRResult:
    bgr = cv2.imread(str(path))
    if bgr is None:
        return OCRResult(path.name, None, None, "")

    date: Optional[str] = None
    frame: Optional[int] = None
    raw_all: List[str] = []

    # 1) Date : essais ROIs & variantes
    calls = 0
    for roi in DATE_ROIS:
        for v_id in range(3):
            txt, lines = _try_roi(bgr, roi, v_id, "date")
            raw_all.extend(lines)
            d, _ = parse_date_and_frame("\n".join(lines))
            calls += 1
            if d:
                date = d; break
            if calls >= MAX_CALLS_PER_FIELD: break
        if date or calls >= MAX_CALLS_PER_FIELD: break

    # 2) Frame : essais ROIs & variantes
    calls = 0
    for roi in FRAME_ROIS:
        for v_id in range(3):
            txt, lines = _try_roi(bgr, roi, v_id, "frame")
            raw_all.extend(lines)
            _, f = parse_date_and_frame("\n".join(lines))
            calls += 1
            if f is not None:
                frame = f; break
            if calls >= MAX_CALLS_PER_FIELD: break
        if frame is not None or calls >= MAX_CALLS_PER_FIELD: break

    # 3) Fallback top strip (on tente de lire les deux en un coup)
    if date is None or frame is None:
        for roi in TOP_STRIP_ROIS:
            # d’abord sans deskew
            for v_id in range(3):
                txt, lines = _try_roi(bgr, roi, v_id, "topstrip")
                raw_all.extend(lines)
                d, f = parse_date_and_frame("\n".join(lines))
                if date is None and d:  date = d
                if frame is None and f is not None: frame = f
                if date and frame is not None: break
            if date and frame is not None: break

        # 3b) Spatial heuristic (bbox) si encore manquant
        if date is None or frame is None:
            for roi in TOP_STRIP_ROIS:
                d, f = _try_roi_spatial(bgr, roi, "topstrip")
                if date is None and d: date = d
                if frame is None and f is not None: frame = f
                if date and frame is not None: break

        # 3c) Ultime recours : deskew ±3° sur top strip
        if date is None or frame is None:
            for roi in TOP_STRIP_ROIS[:1]:
                d, f = _deskew_try(bgr, roi, angles=(+3.0,+2.0,+1.0,-1.0,-2.0,-3.0))
                if date is None and d: date = d
                if frame is None and f is not None: frame = f
                if date and frame is not None: break

    if VERBOSE:
        print(f"[{path.name}] date={date} frame={frame}")

    # petit rythme pour éviter 429 si boucle
    time.sleep(OCR_SLEEP)

    return OCRResult(path.name, date, frame, "\n".join(raw_all))

# ================== Classement & I/O ==================

def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def classify_into_folders(results: List[OCRResult]):
    for r in results:
        if not r.date: continue
        src = IN_DIR / r.filename
        if not src.exists(): continue
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
    print(f"🧪 MAX_CALLS_PER_FIELD={MAX_CALLS_PER_FIELD}, OCR_SLEEP={OCR_SLEEP}")

    rows, results = [], []
    for i, p in enumerate(imgs, 1):
        try:
            rec = ocr_one(p)
        except SystemExit:
            break
        except Exception as e:
            if VERBOSE: print(f"❌ OCR error {p.name}: {e}")
            rec = OCRResult(p.name, None, None, "")
        results.append(rec)
        rows.append({"filename": rec.filename, "date": rec.date, "frame_number": rec.frame_number})

        if i % 10 == 0 or i == len(imgs):
            print(f"[OCR] {i}/{len(imgs)} processed...")

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
