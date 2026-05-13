"""
로스트아크 업적 실시간 체커 - FastAPI 백엔드
- WebSocket으로 프레임 수신
- 이전 프레임과 diff 비교 → 변화 있을 때만 OCR
- 사각형 윤곽선으로 업적 행 분리
- OCR 결과를 업적 DB(JSON)와 매칭

실행:
    pip install fastapi uvicorn easyocr opencv-python numpy python-multipart websockets
    uvicorn main:app --reload --port 8000
"""

import base64
import json
import re
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

# ── EasyOCR 로드 ────────────────────────────────────────────────────────────
print("EasyOCR 로드 중...")
reader = easyocr.Reader(["ko", "en"], gpu=False)
print("EasyOCR 로드 완료")

# ── 업적 DB 로드 (없으면 빈 dict) ───────────────────────────────────────────
DB_PATH = Path(__file__).parent / "achievements_db.json"
if DB_PATH.exists():
    with open(DB_PATH, encoding="utf-8") as f:
        ACHIEVEMENT_DB: dict = json.load(f)
else:
    ACHIEVEMENT_DB = {}

# ── HTML 서빙 ────────────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PATH.read_text(encoding="utf-8")


# ────────────────────────────────────────────────────────────────────────────
# 이미지 처리 유틸
# ────────────────────────────────────────────────────────────────────────────

def b64_to_bgr(b64: str) -> np.ndarray | None:
    """base64 PNG/JPEG → BGR numpy array"""
    try:
        header, data = b64.split(",", 1)
    except ValueError:
        data = b64
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def frame_changed(prev: np.ndarray | None, curr: np.ndarray, threshold: float = 0.03) -> bool:
    """
    이전 프레임과 현재 프레임의 차이가 threshold 이상이면 변화로 판단.
    - 리사이즈해서 빠르게 비교
    """
    if prev is None:
        return True
    h, w = 120, 200
    a = cv2.resize(prev, (w, h)).astype(np.float32)
    b = cv2.resize(curr, (w, h)).astype(np.float32)
    diff = np.abs(a - b).mean() / 255.0
    return diff > threshold


# ────────────────────────────────────────────────────────────────────────────
# 사각형 감지 & 행 분리
# ────────────────────────────────────────────────────────────────────────────

def find_achievement_rects(img: np.ndarray) -> list[dict]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 90)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch == 0:
            continue
        ratio = cw / ch
        if ratio >= 3 and cw >= w * 0.4 and ch >= 30:
            rects.append((x, y, cw, ch))

    if not rects:
        return []

    # y 기준 정렬 & 근접 중복 병합
    rects.sort(key=lambda r: r[1])
    merged = []
    for rect in rects:
        x, y, cw, ch = rect
        if merged and abs(y - merged[-1][1]) < 20:
            if cw > merged[-1][2]:
                merged[-1] = rect
        else:
            merged.append(rect)

    result = []
    for x, y, cw, ch in merged:
        pad = 4
        result.append({
            "x": max(0, x - pad),
            "y": max(0, y - pad),
            "w": min(w, x + cw + pad) - max(0, x - pad),
            "h": min(h, y + ch + pad) - max(0, y - pad),
        })
    return result


def crop_rows(img: np.ndarray) -> tuple[list[np.ndarray], list[dict]]:
    rects = find_achievement_rects(img)
    if not rects:
        return [img], [{"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[0]}]
    crops = []
    for r in rects:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        crops.append(img[y:y+h, x:x+w])
    return crops, rects


# ────────────────────────────────────────────────────────────────────────────
# OCR & 달성 판단
# ────────────────────────────────────────────────────────────────────────────

_NOISE = re.compile(
    r"^(\d{4}\.\d{2}\.\d{2}|\d+[\./]\d+|\d+\.?\d*\s*%?"
    r"|완료|보상|보통|적음|희귀|영웅|전설|일반|희소|\s*)$",
    re.IGNORECASE,
)


def parse_progress(texts: list[str]) -> tuple[int | None, int | None]:
    pat = re.compile(r"(\d+)\s*/\s*(\d+)")
    for t in texts:
        m = pat.search(t)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def parse_percent(texts: list[str]) -> float | None:
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    for t in texts:
        m = pat.search(t)
        if m:
            return float(m.group(1))
    return None


def color_completed(row: np.ndarray) -> bool:
    rh, rw = row.shape[:2]
    roi = row[int(rh * 0.55):rh, 0:int(rw * 0.2)]
    if roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    return (mask > 0).sum() / mask.size > 0.03


def analyze_row(row: np.ndarray) -> dict:
    texts = reader.readtext(row, detail=0, paragraph=False)

    completed = (
        any("완료" in t for t in texts)
    )
    cur, total = parse_progress(texts)
    if cur is not None and total is not None and cur >= total:
        completed = True

    pct = parse_percent(texts)
    if pct is not None and pct >= 100.0:
        completed = True

    if not completed:
        completed = color_completed(row)

    cleaned = [t.strip() for t in texts if t.strip() and not _NOISE.match(t.strip())]
    name = cleaned[0] if cleaned else ""
    desc = cleaned[1] if len(cleaned) > 1 else ""

    # DB 매칭: 이름이 DB 키에 포함되는지 fuzzy 체크
    db_match = None
    if name and ACHIEVEMENT_DB:
        for key in ACHIEVEMENT_DB:
            if name in key or key in name:
                db_match = ACHIEVEMENT_DB[key]
                break

    return {
        "name": name,
        "description": desc,
        "completed": completed,
        "progress": f"{cur}/{total}" if cur is not None else None,
        "percent": pct,
        "db_match": db_match,
        "raw_texts": texts,
    }


def draw_overlay(img: np.ndarray, rects: list[dict], results: list[dict]) -> np.ndarray:
    vis = img.copy()
    for r, res in zip(rects, results):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        color = (50, 200, 80) if res["completed"] else (60, 60, 210)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        label = f"{'✓' if res['completed'] else '✗'} {res['name'][:12]}"
        cv2.putText(vis, label, (x + 6, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return vis


# ────────────────────────────────────────────────────────────────────────────
# WebSocket 엔드포인트
# ────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    prev_frame: np.ndarray | None = None
    last_results: list[dict] = []
    last_rects:   list[dict] = []

    print("WebSocket 연결됨")
    try:
        while True:
            # 클라이언트에서 base64 프레임 수신
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") != "frame":
                continue

            frame = b64_to_bgr(msg["data"])
            if frame is None:
                continue

            # 변화 없으면 이전 결과 재전송 (OCR 스킵)
            if not frame_changed(prev_frame, frame):
                if last_results:
                    await ws.send_text(json.dumps({
                        "type": "result",
                        "cached": True,
                        "achievements": last_results,
                    }))
                continue

            prev_frame = frame.copy()

            # OCR 분석
            t0 = time.time()
            crops, rects = crop_rows(frame)
            results_list = [analyze_row(c) for c in crops]
            elapsed = round(time.time() - t0, 2)

            last_results = results_list
            last_rects   = rects

            # 오버레이 이미지 생성
            overlay = draw_overlay(frame, rects, results_list)
            overlay_b64 = bgr_to_b64(overlay)

            await ws.send_text(json.dumps({
                "type": "result",
                "cached": False,
                "elapsed": elapsed,
                "achievements": results_list,
                "overlay": overlay_b64,
                "count": {
                    "total": len(results_list),
                    "done":  sum(1 for r in results_list if r["completed"]),
                },
            }))

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")
    except Exception as e:
        print(f"WebSocket 오류: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
