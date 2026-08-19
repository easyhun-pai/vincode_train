"""VisoLabel export 후, '라벨 있는' 프레임 중 학습에 쓸 것만 y/n 으로 골라내는 대화형 툴.

각 프레임(박스 그려진 채)을 보고 키 입력:
    y = 학습에 사용   + 다음 장
    n = 학습에서 제외 + 다음 장
    b = 이전 장       s = 판단 보류(다음 장)     q/ESC = 저장 후 종료
결정은 <out>/keep.csv 에 저장되고, 다시 실행하면 미판단부터 이어서 한다.
종료 시 y 로 고른 것만 <out>/images, <out>/labels 로 복사 → 그대로 prepare_split 입력.

라벨 없는(빈 txt) 프레임은 애초에 보여주지 않는다(= 학습에서 자동 제외).

실행:
    python scripts/review_keep.py --images <export폴더> --out data/260731/kept
    # 라벨이 이미지와 다른 폴더면: --labels <라벨폴더> 추가
"""
import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIEW_W = 1700


def load_csv(p):
    if not p.exists():
        return {}
    return {r["frame"]: r["decision"] for r in csv.DictReader(open(p, encoding="utf-8"))}


def save_csv(p, d):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "decision"])
        for k in sorted(d):
            w.writerow([k, d[k]])


def draw_boxes(img, label_path):
    H, W = img.shape[:2]
    n = 0
    for ln in label_path.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = (float(x) for x in p[1:5])
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        n += 1
    return n


def banner(img, sub):
    out = cv2.copyMakeBorder(img, 70, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
    cv2.putText(out, "y=학습 사용   n=제외   b=이전   s=보류   q=종료",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, sub, (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 120), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="이미지 폴더 (VisoLabel export)")
    ap.add_argument("--labels", default=None, help="라벨 폴더 (기본: --images 와 동일)")
    ap.add_argument("--out", required=True, help="선별 결과(images/ labels/) 저장 폴더")
    args = ap.parse_args()

    img_dir = Path(args.images)
    lab_dir = Path(args.labels) if args.labels else img_dir
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 라벨 있는(=박스 있는) 프레임만 대상
    frames = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        lp = lab_dir / (p.stem + ".txt")
        if lp.exists() and lp.name != "classes.txt" and lp.stat().st_size > 0:
            frames.append(p.name)
    assert frames, f"라벨 있는 이미지 없음: {img_dir}"

    csv_p = out / "keep.csv"
    dec = load_csv(csv_p)
    idx = next((i for i, f in enumerate(frames) if f not in dec), 0)
    print(f"[i] 라벨 있는 프레임 {len(frames)}장 | 판단됨 {len(dec)} | 이어서 시작")

    while 0 <= idx < len(frames):
        name = frames[idx]
        img = cv2.imread(str(img_dir / name))
        nb = draw_boxes(img, lab_dir / (Path(name).stem + ".txt"))
        img = cv2.resize(img, (VIEW_W, int(img.shape[0] * VIEW_W / img.shape[1])))
        cur = dec.get(name, "-")
        sub = f"[{idx+1}/{len(frames)}]  {name}  박스 {nb}개  현재={cur}"
        cv2.imshow("review_keep", banner(img, sub))
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("y"):
            dec[name] = "keep"; save_csv(csv_p, dec); idx += 1
        elif k == ord("n"):
            dec[name] = "exclude"; save_csv(csv_p, dec); idx += 1
        elif k == ord("s"):
            idx += 1
        elif k == ord("b"):
            idx = max(0, idx - 1)

    cv2.destroyAllWindows()
    save_csv(csv_p, dec)

    # y(keep) 만 복사 → out/images, out/labels
    (out / "images").mkdir(exist_ok=True)
    (out / "labels").mkdir(exist_ok=True)
    kept = [f for f in frames if dec.get(f) == "keep"]
    for name in kept:
        shutil.copy2(img_dir / name, out / "images" / name)
        shutil.copy2(lab_dir / (Path(name).stem + ".txt"), out / "labels" / (Path(name).stem + ".txt"))
    c = Counter(dec.values())
    print(f"\n[=] 판단 {len(dec)}/{len(frames)} | keep {c['keep']} / exclude {c['exclude']} / 보류 {len(frames)-len(dec)}")
    print(f"    학습용 복사: {out/'images'} ({len(kept)}장)")
    print(f"    다음: python scripts/prepare_split.py --src {out} --split-mode session ...")


if __name__ == "__main__":
    main()
