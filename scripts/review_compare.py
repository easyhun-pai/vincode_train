"""두 모델 비교 프레임을 사람이 정답 판정하는 대화형 툴 (정답 라벨이 없는 unseen 영상용).

compare_v1_v4/ 의 좌우비교 프레임(좌 v1 / 우 v4)을 한 장씩 보며 키 입력:
    1 = v1 만 정답      2 = v4 만 정답
    3 = 둘 다 정답      4 = 둘 다 오답
    b = 이전 프레임     s = 건너뛰기(미판정)     q/ESC = 저장 후 종료
판정은 <dir>/verdicts.csv 에 저장되고, 다시 실행하면 미판정부터 이어서 한다.

실행:  python review_compare.py --dir data/260701/testVideo/compare_v1_v4
"""
import argparse
import csv
from pathlib import Path

import cv2

LABELS = {ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4}
DESC = {1: "v1 만 정답", 2: "v4 만 정답", 3: "둘다 정답", 4: "둘다 오답"}
VIEW_W = 1700


def load_csv(p):
    if not p.exists():
        return {}
    return {r["frame"]: int(r["verdict"]) for r in csv.DictReader(open(p, encoding="utf-8"))}


def save_csv(p, verdicts):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "verdict"])
        for k in sorted(verdicts):
            w.writerow([k, verdicts[k]])


def banner(img, text, sub):
    h = 74
    out = cv2.copyMakeBorder(img, h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(out, text, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, sub, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 120), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="좌우비교 프레임 폴더")
    args = ap.parse_args()
    d = Path(args.dir)
    frames = sorted(p.name for p in d.glob("*.jpg"))
    assert frames, f"프레임 없음: {d}"
    csv_p = d / "verdicts.csv"
    verdicts = load_csv(csv_p)

    # 미판정 첫 위치로
    idx = next((i for i, f in enumerate(frames) if f not in verdicts), 0)
    print(f"[i] 총 {len(frames)}장, 판정됨 {len(verdicts)}. 조작: 1/2/3/4, b=이전, s=스킵, q=종료")

    while 0 <= idx < len(frames):
        name = frames[idx]
        img = cv2.imread(str(d / name))
        scale = VIEW_W / img.shape[1]
        img = cv2.resize(img, (VIEW_W, int(img.shape[0] * scale)))
        cur = verdicts.get(name)
        sub = f"[{idx+1}/{len(frames)}]  {name}  " + (f"현재판정={cur}({DESC[cur]})" if cur else "미판정")
        cv2.imshow("review  (1:v1  2:v4  3:both  4:none  b:back  s:skip  q:quit)",
                   banner(img, "1=v1  2=v4  3=both correct  4=both wrong", sub))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("b"):
            idx = max(0, idx - 1)
        elif k == ord("s"):
            idx += 1
        elif k in LABELS:
            verdicts[name] = LABELS[k]
            save_csv(csv_p, verdicts)          # 매 판정마다 저장(안전)
            idx += 1

    cv2.destroyAllWindows()
    save_csv(csv_p, verdicts)
    n = len(verdicts)
    from collections import Counter
    c = Counter(verdicts.values())
    print(f"\n[=] 판정 {n}/{len(frames)} 저장 -> {csv_p}")
    print(f"    v1만 {c[1]} / v4만 {c[2]} / 둘다정답 {c[3]} / 둘다오답 {c[4]}")
    print("    성능표는:  python report_compare.py --dir", args.dir)


if __name__ == "__main__":
    main()
