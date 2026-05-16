"""
로스트아크 업적 실시간 체커 - FastAPI 백엔드
실행: uvicorn main:app --port 8000
"""

import base64
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI()

print("PaddleOCR 로드 중...")
reader = PaddleOCR(
    use_textline_orientation=False,
    lang="korean",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec"

)
print("PaddleOCR 로드 완료")

DB_PATH = Path(__file__).parent / "achievements_db.json"
ACHIEVEMENT_DB = json.loads(DB_PATH.read_text(encoding="utf-8")) if DB_PATH.exists() else {}

HTML_PATH = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PATH.read_text(encoding="utf-8")


# ── 이미지 유틸 ──────────────────────────────────────────────────────────────

def b64_to_bgr(b64: str):
    try:
        _, data = b64.split(",", 1)
    except ValueError:
        data = b64
    arr = np.frombuffer(base64.b64decode(data), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── 설정 ─────────────────────────────────────────────────────────────────────

PANEL_LEFT   = 660
PANEL_TOP    = 5
PANEL_WIDTH  = 1160
PANEL_HEIGHT = 855

ROW_OFFSET_LEFT  = 0
ROW_OFFSET_RIGHT = 615
ROW_OFFSET_TOP   = -5
ROW_OFFSET_BOT   = 105

YELLOW_Y_MIN = 775
YELLOW_Y_MAX = 800

# ── 템플릿 로드 ───────────────────────────────────────────────────────────────

def _load_gray(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None

def _load_bgr(path):
    return cv2.imread(str(path))

_title_tmpl_gray = _load_gray(Path(__file__).parent / "achievement_title.png")
_complete_tmpl   = _load_bgr(Path(__file__).parent / "complete.png")
_incomplete_tmpl = _load_bgr(Path(__file__).parent / "incomplete.png")

print(f"타이틀 템플릿: {'로드됨' if _title_tmpl_gray is not None else '없음'}")
print(f"완료 템플릿: {'로드됨' if _complete_tmpl is not None else '없음'}")
print(f"미완료 템플릿: {'로드됨' if _incomplete_tmpl is not None else '없음'}")


# ── 패널 탐지 ─────────────────────────────────────────────────────────────────

def find_main_panel(img):
    if _title_tmpl_gray is None:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, tw = _title_tmpl_gray.shape[:2]
    best_val, best_loc = 0.0, None
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        rw, rh = int(tw*scale), int(th*scale)
        if rw < 10 or rh < 5:
            continue
        resized = cv2.resize(_title_tmpl_gray, (rw, rh))
        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_val:
            best_val, best_loc = mv, ml
    if best_val < 0.8 or best_loc is None:
        return None
    tx, ty = best_loc
    px = max(0, tx - PANEL_LEFT)
    py = max(0, ty - PANEL_TOP)
    pw = min(w - px, PANEL_WIDTH)
    ph = min(h - py, PANEL_HEIGHT)
    print(f"[DEBUG] 패널 탐지: ({px},{py},{pw},{ph}) 신뢰도:{best_val:.3f}")
    return (px, py, pw, ph)


# ── 행 탐지 ──────────────────────────────────────────────────────────────────

def find_all_matches(panel_gray, tmpl, threshold=0.8):
    tmpl_g = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
    if tmpl_g.shape[0] > panel_gray.shape[0] or tmpl_g.shape[1] > panel_gray.shape[1]:
        return []
    res = cv2.matchTemplate(panel_gray, tmpl_g, cv2.TM_CCOEFF_NORMED)
    locs = np.where(res >= threshold)
    th, tw = tmpl_g.shape[:2]
    matches = [(x, y, tw, th, float(res[y,x])) for y, x in zip(*locs)]
    matches.sort(key=lambda m: -m[4])
    filtered = []
    for m in matches:
        if all(abs(m[0]-f[0]) > tw//2 or abs(m[1]-f[1]) > th//2 for f in filtered):
            filtered.append(m)
    return filtered


def find_rows_by_template(panel):
    panel_gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    ph, pw = panel.shape[:2]
    rows = []
    if _complete_tmpl is not None:
        for x, y, tw, th, score in find_all_matches(panel_gray, _complete_tmpl):
            rows.append({"x0": max(0, x+ROW_OFFSET_LEFT), "y0": max(0, y+ROW_OFFSET_TOP),
                         "x1": min(pw, x+ROW_OFFSET_RIGHT), "y1": min(ph, y+ROW_OFFSET_BOT),
                         "completed": True})
    if _incomplete_tmpl is not None:
        for x, y, tw, th, score in find_all_matches(panel_gray, _incomplete_tmpl):
            rows.append({"x0": max(0, x+ROW_OFFSET_LEFT), "y0": max(0, y+ROW_OFFSET_TOP),
                         "x1": min(pw, x+ROW_OFFSET_RIGHT), "y1": min(ph, y+ROW_OFFSET_BOT),
                         "completed": False})
    rows.sort(key=lambda r: r["y0"])
    filtered = []
    for r in rows:
        if not filtered or r["y0"] - filtered[-1]["y0"] > 30:
            filtered.append(r)
    print(f"[DEBUG] 행: 완료 {sum(r['completed'] for r in filtered)}개, 미완료 {sum(not r['completed'] for r in filtered)}개")
    return filtered


# ── OCR ──────────────────────────────────────────────────────────────────────

def ocr_with_pos(img):
    if img is None or img.size == 0:
        return []
    result = reader.predict(img)
    if not result or not result[0]:
        return []
    texts = result[0].get('rec_texts', [])
    polys = result[0].get('rec_polys', [])
    items = []
    for t, p in zip(texts, polys):
        if not t.strip():
            continue
        xs, ys = p[:,0], p[:,1]
        items.append({"text": t.strip(), "x": int(xs.min()), "y": int(ys.min()),
                      "w": int(xs.max()-xs.min()), "h": int(ys.max()-ys.min())})
    return items


def parse_progress(texts):
    pat = re.compile(r"(\d+)\s*/\s*(\d+)")
    for t in texts:
        m = pat.search(t)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def parse_row(img):
    items  = ocr_with_pos(img)
    row_w  = img.shape[1]
    row_h  = img.shape[0]
    cx     = row_w * 0.70
    ix     = row_w * 0.18
    ny0, ny1 = row_h * 0.00, row_h * 0.25
    dy0, dy1 = row_h * 0.25, row_h * 0.70
    left       = [it for it in items if ix <= it["x"] < cx]
    name_parts = [it["text"] for it in left if ny0 <= it["y"] < ny1]
    desc_parts = [it["text"] for it in left if dy0 <= it["y"] < dy1]
    cur, total = parse_progress([it["text"] for it in items])
    return (
        " ".join(name_parts),
        " ".join(desc_parts),
        f"{cur}/{total}" if cur is not None else None,
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    cached_panel = None
    last_results = []
    print("WebSocket 연결됨")
    try:
        while True:
            data  = await ws.receive_text()
            msg   = json.loads(data)
            frame = b64_to_bgr(msg["data"])
            if frame is None:
                continue

            # 전체 화면 → 패널 탐지
            if msg.get("type") == "frame_full":
                new_panel = find_main_panel(frame)
                if new_panel is not None:
                    cached_panel = new_panel
                    px, py, pw, ph = cached_panel
                    await ws.send_text(json.dumps({"type":"panel_rect","x":px,"y":py,"w":pw,"h":ph}))
                else:
                    await ws.send_text(json.dumps({"type":"panel_not_found"}))
                continue

            # 크롭 프레임 → 행 매칭 + OCR
            if msg.get("type") != "frame_crop":
                continue

            if cached_panel is None:
                await ws.send_text(json.dumps({"type":"panel_not_found"}))
                continue

            t0 = time.time()
            px, py, pw, ph = cached_panel
            panel_img = frame

            row_infos    = find_rows_by_template(panel_img)
            results_list = []

            for info in row_infos:
                abs_y0 = py + info["y0"]
                abs_y1 = py + info["y1"]
                if abs_y0 < YELLOW_Y_MAX and abs_y1 > YELLOW_Y_MIN:
                    print(f"[SKIP] 노란 텍스트 범위 겹침 y={abs_y0}~{abs_y1}")
                    continue
                row = panel_img[info["y0"]:info["y1"], info["x0"]:info["x1"]]
                if row.shape[0] < 30 or row.shape[1] < 30:
                    continue
                name, desc, progress = parse_row(row)
                print(f"[OCR] {'완료' if info['completed'] else '미완료'}: name={name} | desc={desc} | prog={progress}")
                results_list.append({
                    "name": name,
                    "description": desc,
                    "completed": bool(info["completed"]),
                    "progress": progress,
                    "db_match": None,
                })

            elapsed = round(time.time() - t0, 2)
            last_results = results_list

            await ws.send_text(json.dumps({
                "type": "result",
                "cached": False,
                "elapsed": elapsed,
                "achievements": results_list,
                "count": {
                    "total": len(results_list),
                    "done":  sum(1 for r in results_list if r["completed"]),
                },
            }))

    except WebSocketDisconnect:
        print("WebSocket 연결 종료")
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            await ws.send_text(json.dumps({"type":"error","message":str(e)}))
        except Exception:
            pass