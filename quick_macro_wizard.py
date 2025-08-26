import json
import os
import time
import re
import random
import ctypes  # DPI 보정(Windows)
import cv2
import numpy as np
import pyautogui as pag
import keyboard
import easyocr

# ===== DPI 스케일 보정(Windows) =====
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# ===== 경로 설정 =====
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 인터프리터/노트북 등
    BASE_DIR = os.getcwd()

CONFIG = os.path.join(BASE_DIR, "macro_boxes.json")
OUT_DIR = os.path.join(BASE_DIR, "remainder_shots")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 유틸 =====
def ask_point(msg: str):
    input(f"\n[지정] {msg}\n마우스를 원하는 위치에 올려두고 Enter를 누르세요...")
    return pag.position()

def ask_box(name: str, tl_prompt: str, br_prompt: str):
    print(f"\n영역 지정: {name}")
    x1, y1 = ask_point(tl_prompt)
    x2, y2 = ask_point(br_prompt)
    left, top = min(x1, x2), min(y1, y2)
    right, bottom = max(x1, x2), max(y1, y2)
    center = ((left + right)//2, (top + bottom)//2)
    print(f" -> {name} 영역: ({left},{top}) ~ ({right},{bottom}), center={center}")
    return {"left": left, "top": top, "right": right, "bottom": bottom, "center": center}

def load_or_make():
    data = {}
    data["cart"] = ask_box(
        "장바구니 탭",
        "장바구니 탭 왼쪽 위를 클릭해주세요",
        "장바구니 탭 오른쪽 아래를 클릭해주세요",
    )

    n = int(input("\n신청할 과목 수를 입력하세요(정수): ").strip() or "1")
    data["subjects"] = []
    for i in range(1, n+1):
        remainder_box = ask_box(
            f"{i}번 과목 여석",
            f"{i}번 과목 여석 왼쪽 위를 클릭해주세요",
            f"{i}번 과목 여석 오른쪽 아래를 클릭해주세요",
        )
        apply_point = ask_point(f"{i}번 수강신청 버튼 위치를 클릭해주세요")
        data["subjects"].append({
            "remainder": remainder_box,
            "apply": {"x": apply_point[0], "y": apply_point[1]}
        })

    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def click_at_point(point):
    pag.moveTo(point["x"], point["y"], duration=0.1)
    time.sleep(0.1)
    pag.click()

def click_in_box(box):
    x, y = box["center"]
    pag.moveTo(x, y, duration=0.1)
    time.sleep(0.1)
    pag.click()

# ===== OCR 전처리 & 파서 =====
def preprocess_for_ocr(pil_img):
    """작은 회색 글자를 인식하기 위한 확대+이진화 전처리"""
    arr = np.array(pil_img)  # RGB
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # 작으면 3~4배 확대가 유리
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def extract_number_from_ocr(result):
    """
    EasyOCR 결과에서 숫자를 robust하게 추출.
    'O'→'0', 'I/l'→'1' 치환 후 숫자 토큰(r'\d+')을 모두 수집.
    """
    candidates = []
    for r in result:
        text = r[1] if isinstance(r, (list, tuple)) and len(r) >= 2 else str(r)
        s = text.strip()
        s = s.replace('O', '0').replace('o', '0').replace('I', '1').replace('l', '1')
        digits = re.findall(r'\d+', s)
        for d in digits:
            try:
                candidates.append(int(d))
            except Exception:
                pass
    return max(candidates) if candidates else 0

# ===== 렌더링 안정화(선택) =====
def wait_region_settle(region, timeout=2.0, interval=0.1, thresh=2.0):
    """
    영역 평균밝기 변화가 작아질 때까지 잠시 대기(페이지 렌더링 안정화).
    thresh: 평균밝기 변화 허용치(작을수록 엄격)
    """
    start = time.time()
    prev_mean = None
    while time.time() - start < timeout:
        img = pag.screenshot(region=region)
        arr = np.array(img.convert("L"))
        m = float(arr.mean())
        if prev_mean is not None and abs(m - prev_mean) < thresh:
            return
        prev_mean = m
        time.sleep(interval)

# ===== 메인 루프 =====
def run():
    # 기존 설정 강제 초기화
    if os.path.exists(CONFIG):
        os.remove(CONFIG)
        print(f"{CONFIG} 삭제됨. 새로 위치를 지정합니다.")

    conf = load_or_make()

    # 숫자만 필요하므로 언어 축소(안정적)
    reader = easyocr.Reader(['en'])

    print("3초 후 시작합니다. 대상 창을 앞으로 가져오세요(ESC 또는 Ctrl+C로 종료).")
    time.sleep(3)

    try:
        while True:
            if keyboard.is_pressed('esc'):
                print("ESC 눌림. 프로그램 종료.")
                break

            # 1) 새로고침 후 렌더링 여유
            pag.press('f5')
            time.sleep(1.5)  # 1.0 -> 1.5

            if keyboard.is_pressed('esc'):
                print("ESC 눌림. 프로그램 종료.")
                break

            # 2) 장바구니 탭 클릭 후 여유
            click_in_box(conf["cart"])
            time.sleep(0.8)  # 0.5 -> 0.8

            if keyboard.is_pressed('esc'):
                print("ESC 눌림. 프로그램 종료.")
                break

            applied = False
            for i, subj in enumerate(conf["subjects"], start=1):
                if keyboard.is_pressed('esc'):
                    print("ESC 눌림. 프로그램 종료.")
                    break

                rem_box = subj["remainder"]
                region = (
                    rem_box["left"],
                    rem_box["top"],
                    rem_box["right"] - rem_box["left"],
                    rem_box["bottom"] - rem_box["top"]
                )

                # (선택) 영역 렌더링 안정화 폴링
                wait_region_settle(region, timeout=1.5, interval=0.1, thresh=1.0)

                # 스크린샷
                img = pag.screenshot(region=region)
                raw_path = os.path.join(OUT_DIR, f"remainder_{i}.png")
                img.save(raw_path)

                # 전처리 후 OCR
                roi = preprocess_for_ocr(img)
                proc_path = os.path.join(OUT_DIR, f"remainder_{i}_proc.png")
                cv2.imwrite(proc_path, roi)

                ocr_result = reader.readtext(
                    roi,
                    detail=1,
                    allowlist='0123456789',  # 숫자만
                    rotation_info=[0],
                    contrast_ths=0.05,
                    adjust_contrast=0.5,
                    text_threshold=0.5,
                    low_text=0.3,
                    link_threshold=0.5
                )

                num = extract_number_from_ocr(ocr_result)
                print(f"{i}번 과목 여석: {num}")

                if num >= 1:
                    if keyboard.is_pressed('esc'):
                        print("ESC 눌림. 프로그램 종료.")
                        break
                    click_at_point(subj["apply"])
                    time.sleep(0.1)
                    pag.press('enter')
                    time.sleep(0.1)
                    print(f"{i}번 신청 완료")
                    applied = True

            if keyboard.is_pressed('esc'):
                print("ESC 눌림. 프로그램 종료.")
                break

            if not applied:
                print("여석 없음, 바로 새로고침")
                time.sleep(0.2)
            else:
                print("하나 이상 신청 완료, 다음 사이클 시작")

    except KeyboardInterrupt:
        print("Ctrl+C 눌림. 프로그램 종료.")

if __name__ == "__main__":
    run()
