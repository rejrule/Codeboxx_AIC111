#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import csv
import cv2
import os
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "frames_raw"
SORTED_DIR = ROOT / "data" / "frames_sorted"
META_CSV = ROOT / "data" / "frames_metadata.csv"
OUT_DIR = ROOT / "data" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def read_meta_rows(csv_path: Path):
    rows = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "filename": r.get("filename") or "",
                "date": (r.get("date") or "").strip() or None,
                "frame_number": int(r["frame_number"]) if (r.get("frame_number") not in (None,"") and r["frame_number"].isdigit()) else None
            })
    return rows

def draw_overlay(img, date=None, frame=None):
    if date or frame is not None:
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 900)
        thick = max(1, int(2 * scale))

        if date:
            cv2.putText(img, date, (10, int(40*scale)), font, scale*1.4, (0,0,0), thick+2, cv2.LINE_AA)
            cv2.putText(img, date, (10, int(40*scale)), font, scale*1.4, (255,255,255), thick, cv2.LINE_AA)
        if frame is not None:
            text = f"F{frame}"
            (tw, th), _ = cv2.getTextSize(text, font, scale*1.4, thick)
            x = w - tw - 10
            y = int(40*scale)
            cv2.putText(img, text, (x, y), font, scale*1.4, (0,0,0), thick+2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale*1.4, (255,255,255), thick, cv2.LINE_AA)
    return img

def write_video(frames_paths, out_path: Path, fps: float, overlay=False, meta=None, size=None):
    if not frames_paths:
        print(f"⚠️  Aucun frame à écrire pour {out_path.name}")
        return 0
    # déterminer la taille de sortie
    first = None
    for p in frames_paths:
        img = cv2.imread(str(p))
        if img is None: 
            continue
        first = img
        break
    if first is None:
        print(f"⚠️  Impossible de lire la première image pour {out_path.name}")
        return 0
    H, W = first.shape[:2] if size is None else size

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    written = 0
    for p in frames_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        if overlay and meta:
            info = meta.get(p.name) or {}
            img = draw_overlay(img, info.get("date"), info.get("frame_number"))
        vw.write(img)
        written += 1
    vw.release()
    print(f"✅ {written} frames → {out_path}")
    return written

def main():
    ap = argparse.ArgumentParser(description="Reconstruire une vidéo depuis des frames.")
    ap.add_argument("--fps", type=float, default=15.0, help="FPS de sortie (défaut: 15)")
    ap.add_argument("--use-csv", action="store_true", help="Utiliser frames_metadata.csv pour l'ordre (recommandé)")
    ap.add_argument("--split-by-date", action="store_true", help="Un MP4 par date (nécessite --use-csv)")
    ap.add_argument("--overlay", action="store_true", help="Surimprimer date/frame sur la vidéo")
    ap.add_argument("--source", choices=["raw","sorted"], default="sorted",
                    help="Dossier source des images: 'sorted' (par date) ou 'raw'")
    ap.add_argument("--limit", type=int, default=0, help="Limiter le nombre d'images (0 = pas de limite)")
    args = ap.parse_args()

    meta_rows = read_meta_rows(META_CSV) if args.use_csv else []
    meta_map = {}
    for r in meta_rows:
        meta_map[r["filename"]] = {"date": r["date"], "frame_number": r["frame_number"]}

    if args.use_csv and meta_rows:
        # ordre par date puis frame_number, fallback par nom
        items = []
        for r in meta_rows:
            fname = r["filename"]
            d = r["date"] or ""
            f = r["frame_number"] if r["frame_number"] is not None else 10**9
            if args.source == "sorted" and r["date"]:
                p = SORTED_DIR / r["date"] / fname
            else:
                p = RAW_DIR / fname
            if p.exists():
                items.append((d, f, p))
        # tri
        items.sort(key=lambda t: (t[0], t[1], t[2].name))
        if args.limit > 0:
            items = items[:args.limit]

        if args.split_by_date:
            buckets = defaultdict(list)
            for d, f, p in items:
                buckets[d].append(p)
            total = 0
            for d, paths in sorted(buckets.items()):
                out_path = OUT_DIR / f"recon_{d or 'unknown'}.mp4"
                total += write_video(paths, out_path, args.fps, overlay=args.overlay, meta=meta_map)
            print(f"🎬 Total écrit: {total} frames (toutes dates)")
        else:
            out_path = OUT_DIR / "recon_all.mp4"
            write_video([p for _,_,p in items], out_path, args.fps, overlay=args.overlay, meta=meta_map)
    else:
        # sans CSV : ordre alphabétique des fichiers dans raw/sorted
        base = SORTED_DIR if args.source == "sorted" else RAW_DIR
        if args.source == "sorted" and base.exists():
            # concaténer toutes les dates
            all_paths = []
            for ddir in sorted([d for d in base.iterdir() if d.is_dir()]):
                all_paths.extend(list_images(ddir))
        else:
            all_paths = list_images(base)
        if args.limit > 0:
            all_paths = all_paths[:args.limit]
        out_path = OUT_DIR / "recon_all.mp4"
        write_video(all_paths, out_path, args.fps, overlay=False, meta=None)

if __name__ == "__main__":
    main()
