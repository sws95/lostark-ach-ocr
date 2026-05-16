"""
로스트아크 업적 실시간 체커 - FastAPI 백엔드 (최적화 버전)

변경사항:
- 전체 화면 Canny → 큰 네모(패널) 탐지 후 캐싱
- 패널 내부에서 수평 밝기 프로젝션으로 행 분리 (Canny 제거)
- 행 전체 OCR → 이름/진행도 영역만 OCR
- 완료 판단 → "완료" 텍스트 영역 픽셀 색상으로 우선 판단 (OCR 최소화)

실행:
    pip install fastapi uvicorn paddlepaddle-gpu paddleocr opencv-python numpy python-multipart websockets
    uvicorn main:app --reload --port 8000
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

# ── PaddleOCR 로드 ───────────────────────────────────────────────────────────
print("PaddleOCR 로드 중...")
reader = PaddleOCR(use_textline_orientation=False, lang="korean")
print("PaddleOCR 로드 완료")

# ── 업적 DB 로드 ──────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent / "achievements_db.json"
if DB_PATH.exists():
    with open(DB_PATH, encoding="utf-8") as f:
        ACHIEVEMENT_DB: dict = json.load(f)
else:
    ACHIEVEMENT_DB = {}

# ── HTML 서빙 ─────────────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent / "index.html"

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# 이미지 처리 유틸
# ─────────────────────────────────────────────────────────────────────────────

def b64_to_bgr(b64: str) -> np.ndarray | None:
    try:
        _, data = b64.split(",", 1)
    except ValueError:
        data = b64
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def bgr_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def frame_changed(prev: np.ndarray | None, curr: np.ndarray, threshold: float = 0.03) -> bool:
    if prev is None:
        return True
    h, w = 120, 200
    a = cv2.resize(prev, (w, h)).astype(np.float32)
    b = cv2.resize(curr, (w, h)).astype(np.float32)
    diff = np.abs(a - b).mean() / 255.0
    return diff > threshold


# ── 패널 오프셋 설정 (find_rects.py로 맞춘 값) ──────────────────────────────
PANEL_LEFT   = 660
PANEL_TOP    = 5
PANEL_WIDTH  = 1160
PANEL_HEIGHT = 855

# ── 업적 타이틀 템플릿 로드 ──────────────────────────────────────────────────
TITLE_TEMPLATE_PATH = Path(__file__).parent / "achievement_title.png"
_title_tmpl_gray: np.ndarray | None = None
if TITLE_TEMPLATE_PATH.exists():
    _t = cv2.imread(str(TITLE_TEMPLATE_PATH))
    _title_tmpl_gray = cv2.cvtColor(_t, cv2.COLOR_BGR2GRAY)
    print(f"타이틀 템플릿 로드: {_title_tmpl_gray.shape}")
else:
    print("경고: achievement_title.png 없음")


def find_main_panel(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """
    "업적" 타이틀 템플릿 매칭 후 오프셋으로 패널 범위 계산.
    """
    if _title_tmpl_gray is None:
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, tw = _title_tmpl_gray.shape[:2]

    best_val = 0.0
    best_loc = None
    best_scale = 1.0

    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        rw = int(tw * scale)
        rh = int(th * scale)
        if rw < 10 or rh < 5:
            continue
        resized = cv2.resize(_title_tmpl_gray, (rw, rh))
        if resized.shape[0] > gray.shape[0] or resized.shape[1] > gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv > best_val:
            best_val = mv
            best_loc = ml
            best_scale = scale

    if best_val < 0.8 or best_loc is None:
        print(f"[DEBUG] 템플릿 매칭 실패 - 최대 신뢰도: {best_val:.3f}")
        return None

    tx, ty = best_loc
    panel_x = max(0, tx - PANEL_LEFT)
    panel_y = max(0, ty - PANEL_TOP)
    panel_w = min(w - panel_x, PANEL_WIDTH)
    panel_h = min(h - panel_y, PANEL_HEIGHT)

    print(f"[DEBUG] 패널 탐지 성공 - 신뢰도:{best_val:.3f} 패널:({panel_x},{panel_y},{panel_w},{panel_h})")
    return (panel_x, panel_y, panel_w, panel_h)
    return (panel_x, panel_y, panel_w, panel_h)


# ─────────────────────────────────────────────────────────────────────────────
# 2단계: 패널 내부 행 분리 (수평 밝기 프로젝션)
# ─────────────────────────────────────────────────────────────────────────────

def find_row_boundaries(panel: np.ndarray, min_row_h: int = 60) -> list[int]:
    """
    패널 이미지에서 행 경계 y좌표 목록 반환.
    각 행 사이의 구분선은 어두운 수평선이므로
    y축 평균 밝기가 낮은 줄을 경계로 사용.
    """
    gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)  # shape: (height,)

    # 어두운 줄 = 행 경계 후보 (threshold: 평균보다 30 어두운 곳)
    global_mean = row_mean.mean()
    dark_threshold = max(global_mean * 0.6, 30)
    is_dark = row_mean < dark_threshold

    # 연속된 어두운 구간의 중앙을 경계로 사용
    boundaries = [0]
    in_dark = False
    dark_start = 0
    for y, dark in enumerate(is_dark):
        if dark and not in_dark:
            in_dark = True
            dark_start = y
        elif not dark and in_dark:
            in_dark = False
            mid = (dark_start + y) // 2
            # 이전 경계와 min_row_h 이상 떨어져 있을 때만 추가
            if mid - boundaries[-1] >= min_row_h:
                boundaries.append(mid)

    boundaries.append(panel.shape[0])
    return boundaries


# ─────────────────────────────────────────────────────────────────────────────
# 3단계: 행 내부 영역별 분석
# ─────────────────────────────────────────────────────────────────────────────

# 행 내부 상대 좌표 (패널 너비 기준 비율)
# 업적창 레이아웃: [아이콘(~9%)] [이름+설명(9%~55%)] [날짜+%(55%~75%)] [보상(75%~)]
NAME_X_RATIO   = (0.09, 0.55)   # 이름+설명 영역
PROG_X_RATIO   = (0.55, 0.78)   # n/n, % 영역
DONE_X_RATIO   = (0.01, 0.12)   # "완료" 텍스트 영역 (왼쪽 하단)
NAME_Y_RATIO   = (0.05, 0.45)   # 행 높이 기준 상단 절반 → 이름
DESC_Y_RATIO   = (0.45, 0.75)   # 행 높이 기준 하단 → 설명
DONE_Y_RATIO   = (0.65, 1.00)   # "완료" 위치


def crop_ratio(img: np.ndarray, x0r: float, x1r: float, y0r: float, y1r: float) -> np.ndarray:
    h, w = img.shape[:2]
    x0, x1 = int(w * x0r), int(w * x1r)
    y0, y1 = int(h * y0r), int(h * y1r)
    return img[y0:y1, x0:x1]


def is_completed_by_color(done_region: np.ndarray) -> bool:
    """
    '완료' 텍스트는 노란색(금색)이므로 HSV로 확인.
    OCR 없이 색상만으로 완료 판단 → 빠름.
    """
    if done_region.size == 0:
        return False
    hsv = cv2.cvtColor(done_region, cv2.COLOR_BGR2HSV)
    # 노란~금색 범위
    mask = cv2.inRange(hsv, np.array([15, 80, 100]), np.array([40, 255, 255]))
    ratio = (mask > 0).sum() / mask.size
    return ratio > 0.04


_NOISE = re.compile(
    r"^(\d{4}\.\d{2}\.\d{2}|\d+[\./]\d+|\d+\.?\d*\s*%?"
    r"|완료|보상|보통|적음|희귀|영웅|전설|일반|희소|\s*)$",
    re.IGNORECASE,
)


def ocr_texts(img: np.ndarray) -> list[str]:
    """PaddleOCR 호출 래퍼 - 텍스트 리스트 반환"""
    if img is None or img.size == 0:
        return []
    result = reader.ocr(img)
    if not result or not result[0]:
        return []
    return [line[1][0] for line in result[0] if line and line[1]]


def parse_progress(texts: list[str]) -> tuple[int | None, int | None]:
    pat = re.compile(r"(\d+)\s*/\s*(\d+)")
    for t in texts:
        m = pat.search(t)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None, None


def analyze_row(row: np.ndarray) -> dict:
    """
    행 이미지에서:
    1. 완료 여부 → 색상으로 우선 판단 (OCR 스킵 가능)
    2. 이름 → 상단 영역만 OCR
    3. 진행도 → 우측 영역만 OCR
    """
    pw = row.shape[1]

    # ── 완료 판단 (색상 우선) ──────────────────────────────────────────────
    done_region = crop_ratio(row, *DONE_X_RATIO, *DONE_Y_RATIO)
    completed = is_completed_by_color(done_region)

    # ── 이름 OCR (상단 좌측 영역만) ───────────────────────────────────────
    name_region = crop_ratio(row, *NAME_X_RATIO, *NAME_Y_RATIO)
    name_texts  = ocr_texts(name_region)
    name = name_texts[0].strip() if name_texts else ""

    # ── 설명 OCR ──────────────────────────────────────────────────────────
    desc_region = crop_ratio(row, *NAME_X_RATIO, *DESC_Y_RATIO)
    desc_texts  = ocr_texts(desc_region)
    desc = desc_texts[0].strip() if desc_texts else ""

    # ── 진행도 OCR (완료 아닐 때만) ───────────────────────────────────────
    cur, total = None, None
    if not completed:
        prog_region = crop_ratio(row, *PROG_X_RATIO, 0.0, 1.0)
        prog_texts  = ocr_texts(prog_region)
        cur, total  = parse_progress(prog_texts)
        if cur is not None and total is not None and cur >= total:
            completed = True

    # ── DB 매칭 ───────────────────────────────────────────────────────────
    db_match = None
    if name and ACHIEVEMENT_DB:
        for key in ACHIEVEMENT_DB:
            if name in key or key in name:
                db_match = ACHIEVEMENT_DB[key]
                break

    return {
        "name": name,
        "description": desc,
        "completed": bool(completed),
        "progress": f"{cur}/{total}" if cur is not None else None,
        "db_match": db_match,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 오버레이 드로잉
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    img: np.ndarray,
    panel_rect: tuple[int, int, int, int] | None,
    boundaries: list[int],
    results: list[dict],
) -> np.ndarray:
    vis = img.copy()

    if panel_rect is None:
        # 패널 못 찾으면 경고 텍스트만
        cv2.putText(vis, "업적 창을 찾을 수 없습니다",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 220), 2, cv2.LINE_AA)
        return vis

    px, py, pw, ph = panel_rect

    # 패널 전체 초록 네모 (두껍게)
    cv2.rectangle(vis, (px, py), (px + pw, py + ph), (50, 220, 80), 3)
    cv2.putText(vis, "업적 창 감지됨", (px + 6, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 220, 80), 1, cv2.LINE_AA)

    # 행별 네모
    for y0, y1, res in zip(boundaries[:-1], boundaries[1:], results):
        abs_y0 = py + y0
        abs_y1 = py + y1
        color = (50, 200, 80) if res["completed"] else (60, 60, 210)
        cv2.rectangle(vis, (px, abs_y0), (px + pw, abs_y1), color, 1)
        label = f"{'v' if res['completed'] else 'x'} {res['name'][:14]}"
        cv2.putText(vis, label, (px + 8, abs_y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket 엔드포인트
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    prev_frame:   np.ndarray | None = None
    cached_panel: tuple | None      = None   # (x, y, w, h)
    last_results: list[dict]        = []

    print("WebSocket 연결됨")
    try:
        while True:
            data  = await ws.receive_text()
            msg   = json.loads(data)
            if msg.get("type") != "frame":
                continue

            frame = b64_to_bgr(msg["data"])
            if frame is None:
                continue

            # 변화 없으면 이전 결과 재전송
            if not frame_changed(prev_frame, frame):
                if last_results:
                    await ws.send_text(json.dumps({
                        "type": "result",
                        "cached": True,
                        "achievements": last_results,
                    }))
                continue

            prev_frame = frame.copy()
            t0 = time.time()

            # ── 디버그: 첫 프레임 저장 ────────────────────────────────────
            debug_path = Path(__file__).parent / "debug_frame.png"
            if not debug_path.exists():
                cv2.imwrite(str(debug_path), frame)
                print(f"[DEBUG] 첫 프레임 저장: {debug_path}")

            # ── 1. 패널 탐지 ──────────────────────────────────────────────
            panel_rect = find_main_panel(frame)
            if panel_rect is None:
                overlay = draw_overlay(frame, None, [], [])
                await ws.send_text(json.dumps({
                    "type": "result",
                    "cached": True,
                    "achievements": last_results,
                    "overlay": bgr_to_b64(overlay),
                    "warning": "패널을 찾지 못했습니다.",
                }))
                continue
            print(f"[DEBUG] 패널 탐지 성공: {panel_rect}")

            px, py, pw, ph = panel_rect
            panel_img = frame[py:py+ph, px:px+pw]

            # ── 2. 행 경계 탐지 ───────────────────────────────────────────
            boundaries = find_row_boundaries(panel_img)

            # ── 3. 행별 분석 ──────────────────────────────────────────────
            # 정상 행 높이 추정: 전체 경계 간격의 중앙값
            row_heights = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
            normal_h = int(np.median(row_heights)) if row_heights else 130

            results_list = []
            for y0, y1 in zip(boundaries[:-1], boundaries[1:]):
                row = panel_img[y0:y1]
                if row.shape[0] < 30:           # 너무 얇은 행 스킵
                    continue
                if row.shape[0] > normal_h * 1.8:  # 펼쳐진 행(세부 목록) 스킵
                    continue
                results_list.append(analyze_row(row))

            elapsed = round(time.time() - t0, 2)
            last_results = results_list

            # ── 4. 오버레이 생성 & 전송 ───────────────────────────────────
            # 전체 화면 오버레이 (초록 네모 표시)
            px, py, pw, ph = panel_rect
            vis = frame.copy()
            cv2.rectangle(vis, (px, py), (px+pw, py+ph), (50, 220, 80), 3)
            for y0, y1, res in zip(boundaries[:-1], boundaries[1:], results_list):
                abs_y0 = py + y0
                abs_y1 = py + y1
                color = (50, 200, 80) if res["completed"] else (60, 60, 210)
                cv2.rectangle(vis, (px, abs_y0), (px+pw, abs_y1), color, 1)

            # 미리보기는 패널 크롭만
            panel_crop = vis[py:py+ph, px:px+pw]
            overlay_b64 = bgr_to_b64(panel_crop)

            await ws.send_text(json.dumps({
                "type":    "result",
                "cached":  False,
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
        import traceback
        traceback.print_exc()
        print(f"WebSocket 오류: {e}")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass