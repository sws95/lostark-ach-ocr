"""
로스트아크 코덱스 업적 크롤러
실행: python crawl_achievements.py
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import os
import openpyxl

EXCEL_FILE   = "lostarc2.xlsx"
OUTPUT_FILE  = "achievements_db.json"
IMG_DIR      = "achievement_icons"
BASE_URL     = "https://lostarkcodex.com/kr/achievement/{id}/"
HEADERS      = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

os.makedirs(IMG_DIR, exist_ok=True)


# ── 엑셀에서 ID 추출 ──────────────────────────────────────────────────────────
def get_ids_from_excel(filename):
    wb = openpyxl.load_workbook(filename)
    ids = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            for cell in row:
                if cell and str(cell).strip().isdigit() and len(str(cell).strip()) >= 7:
                    ids.append(int(str(cell).strip()))
    return sorted(set(ids))


# ── 이미지 다운로드 ───────────────────────────────────────────────────────────
def download_image(img_url, fname):
    if not img_url or os.path.exists(fname):
        return fname if os.path.exists(fname) else ""
    try:
        r = requests.get(img_url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            with open(fname, "wb") as f:
                f.write(r.content)
            return fname
    except Exception as e:
        print(f"  이미지 실패: {e}")
    return ""


# ── 페이지 파싱 ───────────────────────────────────────────────────────────────
def parse_achievement(html, ach_id):
    soup = BeautifulSoup(html, "html.parser")

    # 이름
    name_el = soup.find("span", id="item_name")
    name = name_el.get_text(strip=True) if name_el else ""
    if not name:
        return None

    # 업적 아이콘
    img_el = soup.find("img", class_="item_icon")
    img_url = ""
    if img_el and img_el.get("src"):
        src = img_el["src"]
        img_url = src if src.startswith("http") else f"https://lostarkcodex.com{src}"
    img_local = download_image(img_url, f"{IMG_DIR}/{ach_id}_{img_url.split('/')[-1]}" if img_url else "")

    # titles_cell: 타입, 카테고리, 노트
    ach_type = category = note = ""
    titles_td = soup.find("td", class_="titles_cell")
    if titles_td:
        cat_el = titles_td.find("span", class_="category_text")
        ach_type = cat_el.get_text(strip=True) if cat_el else ""
        parts = [elem.strip() for elem in titles_td.children
                 if isinstance(elem, str) and elem.strip()]
        remaining = [p for p in parts if p != ach_type]
        if len(remaining) >= 1: category = remaining[0]
        if len(remaining) >= 2: note = remaining[1]

    # hr 구간별 파싱
    description = []
    steps = []
    rewards = []
    td = soup.find("td", colspan="2")
    if td:
        hrs = td.find_all("hr", class_="tooltiphr")

        def get_texts_between(start_hr, end_hr=None):
            texts = []
            for elem in start_hr.next_siblings:
                if elem == end_hr or elem.name == "hr":
                    break
                if isinstance(elem, str):
                    t = elem.strip()
                    if t:
                        texts.append(t)
            return texts

        if len(hrs) >= 1:
            description = get_texts_between(hrs[0], hrs[1] if len(hrs) > 1 else None)
        if len(hrs) >= 2:
            steps = get_texts_between(hrs[1], hrs[2] if len(hrs) > 2 else None)
        if len(hrs) >= 3:
            reward_names = []
            for elem in hrs[2].next_siblings:
                if isinstance(elem, str):
                    t = elem.strip().lstrip("- ").strip()
                    if t and t != "보상:":
                        reward_names.append(t)

            reward_icons = []
            for wrapper in hrs[2].find_all_next("div", class_="icon_wrapper_medium"):
                img = wrapper.find("img", class_="list_icon_medium")
                rname = wrapper.get("oldtitle", "")
                if img:
                    src = img["src"]
                    icon_url = src if src.startswith("http") else f"https://lostarkcodex.com{src}"
                    icon_fname = f"{IMG_DIR}/reward_{icon_url.split('/')[-1]}"
                    download_image(icon_url, icon_fname)
                    reward_icons.append({"name": rname, "icon_url": icon_url, "icon_local": icon_fname})

            for i, icon in enumerate(reward_icons):
                rewards.append({
                    "name": reward_names[i] if i < len(reward_names) else icon["name"],
                    "icon_url": icon["icon_url"],
                    "icon_local": icon["icon_local"],
                })

    return {
        "id":          ach_id,
        "name":        name,
        "type":        ach_type,
        "category":    category,
        "note":        note,
        "description": description,
        "steps":       steps,
        "rewards":     rewards,
        "image_url":   img_url,
        "image_local": img_local,
    }


# ── 메인 ──────────────────────────────────────────────────────────────────────
print(f"엑셀 파일 읽는 중: {EXCEL_FILE}")
id_list = get_ids_from_excel(EXCEL_FILE)
print(f"총 {len(id_list)}개 ID 추출")

db = {}
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        db = json.load(f)
    print(f"기존 DB 로드: {len(db)}개")

crawled = skipped = errors = 0

for ach_id in id_list:
    if str(ach_id) in db:
        skipped += 1
        continue

    url = BASE_URL.format(id=ach_id)
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 404:
            continue
        if r.status_code != 200:
            print(f"  [{ach_id}] HTTP {r.status_code}")
            errors += 1
            continue

        data = parse_achievement(r.text, ach_id)
        if not data:
            continue

        db[str(ach_id)] = data
        crawled += 1
        print(f"[{ach_id}] {data['name']} | {data['category']}")

        if crawled % 50 == 0:
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            print(f"  → {crawled}개 저장")

        time.sleep(0.3)

    except Exception as e:
        print(f"  [{ach_id}] 오류: {e}")
        errors += 1

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(db, f, ensure_ascii=False, indent=2)

print(f"\n완료! 총 {len(db)}개 ({crawled}개 새로 추가, {skipped}개 스킵, {errors}개 오류)")