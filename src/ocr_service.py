import os
import io
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

AZ_ENDPOINT = os.getenv("AZURE_CV_ENDPOINT") or os.getenv("COMPUTER_VISION_ENDPOINT") or ""
AZ_KEY = os.getenv("AZURE_CV_KEY") or os.getenv("COMPUTER_VISION_KEY") or ""
READ_URL = AZ_ENDPOINT.rstrip("/") + "/vision/v3.2/read/analyze"

app = FastAPI(title="Simple OCR Proxy (Azure Vision Read)")

def call_read(image_bytes: bytes):
    headers = {"Ocp-Apim-Subscription-Key": AZ_KEY, "Content-Type": "application/octet-stream"}
    r = requests.post(READ_URL, headers=headers, data=image_bytes, timeout=20)
    if r.status_code not in (200, 202):
        raise HTTPException(status_code=500, detail=f"Read POST failed: {r.status_code} {r.text[:200]}")
    op = r.headers.get("Operation-Location")
    if not op:
        raise HTTPException(status_code=500, detail="Missing Operation-Location")
    # poll
    for _ in range(60):
        g = requests.get(op, headers={"Ocp-Apim-Subscription-Key": AZ_KEY}, timeout=10)
        if g.status_code != 200:
            continue
        data = g.json()
        if data.get("status","").lower() == "succeeded":
            lines = []
            for blk in data.get("analyzeResult", {}).get("readResults", []):
                for line in blk.get("lines", []):
                    if line.get("text","").strip():
                        lines.append(line["text"])
            return {"lines": lines}
        if data.get("status","").lower() == "failed":
            raise HTTPException(status_code=500, detail="Read status=failed")
    raise HTTPException(status_code=504, detail="Read timeout")

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if not (AZ_ENDPOINT and AZ_KEY):
        raise HTTPException(status_code=500, detail="AZURE_CV_ENDPOINT / AZURE_CV_KEY not set")
    content = await file.read()
    res = call_read(content)
    return JSONResponse(res)
