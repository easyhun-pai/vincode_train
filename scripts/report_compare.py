"""review_compare.py 로 만든 verdicts.csv -> 성능표 PNG + 판정 얹은 비교영상.

정답 라벨이 없는 unseen 영상이라 mAP 대신 '사람 판정' 기반 정답률/승패로 비교한다.
실행:  python report_compare.py --dir data/260701/testVideo/compare_v1_v4
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

DESC = {1: "v1 correct", 2: "v4 correct", 3: "both correct", 4: "both wrong"}
COLOR = {1: (200, 120, 40), 2: (60, 180, 60), 3: (150, 150, 150), 4: (60, 60, 210)}  # BGR


def make_table_png(counts, total, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for cand in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
        if any(cand in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    c1, c2, c3, c4 = (counts.get(i, 0) for i in (1, 2, 3, 4))
    v1c, v4c = c1 + c3, c2 + c3
    rows = [
        ["v1 (vincode.pt)", f"{v1c}", f"{v1c/total*100:.1f}%" if total else "-"],
        ["v4 (vincode_v4.pt)", f"{v4c}", f"{v4c/total*100:.1f}%" if total else "-"],
        ["v4만 정답 (v4>v1)", f"{c2}", f"{c2/total*100:.1f}%" if total else "-"],
        ["v1만 정답 (v1>v4)", f"{c1}", f"{c1/total*100:.1f}%" if total else "-"],
        ["둘다 정답", f"{c3}", f"{c3/total*100:.1f}%" if total else "-"],
        ["둘다 오답", f"{c4}", f"{c4/total*100:.1f}%" if total else "-"],
    ]
    fig, ax = plt.subplots(figsize=(7.5, 3.3), dpi=200)
    ax.axis("off")
    ax.set_title(f"unseen 영상 v1 vs v4 · 사람 판정 {total}프레임", fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    t = ax.table(cellText=rows, colLabels=["항목", "프레임", "비율"],
                 cellLoc="center", loc="center", colWidths=[0.5, 0.25, 0.25])
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_facecolor("#2b3a55"); cell.set_text_props(color="white", fontweight="bold")
        elif r <= 2:
            cell.set_facecolor("#eef6ff")
            if c == 0:
                cell.set_text_props(fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    print("[+] 성능표:", out_png)


def make_video(d, verdicts, out_mp4, fps=3):
    frames = [f for f in sorted(p.name for p in d.glob("*.jpg")) if f in verdicts]
    if not frames:
        print("[!] 판정된 프레임 없음 - 영상 생략"); return
    h0, w0 = cv2.imread(str(d / frames[0])).shape[:2]
    BAR = 80
    vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0 + BAR))
    for name in frames:
        img = cv2.imread(str(d / name))
        v = verdicts[name]
        out = cv2.copyMakeBorder(img, BAR, 0, 0, 0, cv2.BORDER_CONSTANT, value=COLOR[v])
        cv2.putText(out, f"[{v}] {DESC[v]}", (18, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, name, (w0 - 260, 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(out)
    vw.release()
    print(f"[+] 판정영상: {out_mp4} ({len(frames)}프레임, {fps}fps)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--fps", type=int, default=3)
    args = ap.parse_args()
    d = Path(args.dir)
    csv_p = d / "verdicts.csv"
    assert csv_p.exists(), f"verdicts.csv 없음 (먼저 review_compare.py 실행): {csv_p}"

    verdicts = {r["frame"]: int(r["verdict"]) for r in csv.DictReader(open(csv_p, encoding="utf-8"))}
    from collections import Counter
    counts = Counter(verdicts.values())
    total = len(verdicts)
    print(f"[i] 판정 {total}프레임 | v1만 {counts[1]} / v4만 {counts[2]} / 둘다정답 {counts[3]} / 둘다오답 {counts[4]}")

    make_table_png(counts, total, d / "verdict_table.png")
    make_video(d, verdicts, d / "review_result.mp4", args.fps)


if __name__ == "__main__":
    main()
