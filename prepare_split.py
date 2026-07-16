"""검수 끝난 배치(images/ + labels/)를 학습셋(data_exist)에 **세션 단위**로 배치한다.

왜 세션 단위인가 (중요):
  같은 차량 한 대가 여러 프레임·여러 카메라에 걸쳐 찍힌다. 프레임을 랜덤 분할하면
  같은 차가 train 과 val 양쪽에 들어가 성능이 가짜로 높게 나온다(누수).
  -> 세션 통째로 train / eval 로 가른다.

배경(빈 라벨) 비율:
  배경이 과하면 모델이 '아무것도 없다'를 학습해 보수적으로 변하고 recall 이 떨어진다.
  recall 이 우선인 용도라 기본 --bg-ratio 2.0 (양성의 2배)로 제한한다.

사용:
    # 세션이 파일명에 s1_/s2_/s3_ 형태로 있는 배치
    python prepare_split.py --src data260701 --train-sessions s1,s2 --eval-sessions s3

    # 세션이 20260619_124450 형태인 배치
    python prepare_split.py --src data_new --train-sessions 20260619_124248,20260619_124450 \
        --eval-sessions 20260619_130143

옵션:
    --eval-to  valid|test|both   평가 세션을 어디에 넣을지 (기본 both:
               데이터가 적을 때 val=test 로 쓰는 구성. 독립 test 를 원하면 세션을 따로 지정)
"""
import argparse
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 기본 세션 패턴: 's1' 같은 접두사 또는 '20260619_124450' 같은 타임스탬프
DEFAULT_SESSION_RE = r"(?:^|_)(s\d+|\d{8}_\d{6})(?:_|$)"


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def place(dst_root, split, bucket, pos, bg_sel, src_lab):
    imgd = dst_root / split / "images" / bucket
    labd = dst_root / split / "labels" / bucket
    imgd.mkdir(parents=True, exist_ok=True)
    labd.mkdir(parents=True, exist_ok=True)
    for p in pos:
        shutil.copy2(p, imgd / p.name)
        shutil.copy2(src_lab / (p.stem + ".txt"), labd / (p.stem + ".txt"))
    for p in bg_sel:
        shutil.copy2(p, imgd / p.name)
        (labd / (p.stem + ".txt")).write_text("", encoding="utf-8")  # 빈 라벨 = 배경
    print(f"[+] {split:6s}: 양성 {len(pos):4d} + 배경 {len(bg_sel):4d} -> {imgd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="검수 끝난 배치 폴더 (images/ labels/ 포함)")
    ap.add_argument("--data-root", default="data_exist", help="학습셋 루트 (기본: ./data_exist)")
    ap.add_argument("--bucket", default="v2_visolabel", help="배치될 버킷 폴더명")
    ap.add_argument("--train-sessions", required=True, help="쉼표 구분 (예: s1,s2)")
    ap.add_argument("--eval-sessions", required=True, help="쉼표 구분 (예: s3)")
    ap.add_argument("--eval-to", default="both", choices=["valid", "test", "both"],
                    help="평가 세션을 valid/test/양쪽 중 어디에 배치할지")
    ap.add_argument("--bg-ratio", type=float, default=2.0, help="배경 = 양성 * 이 값 (recall 보호)")
    ap.add_argument("--session-re", default=DEFAULT_SESSION_RE, help="파일명에서 세션을 뽑는 정규식")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    src = resolve(args.src)
    src_img, src_lab = src / "images", src / "labels"
    dst_root = resolve(args.data_root)
    assert src_img.exists() and src_lab.exists(), f"images/ labels/ 없음: {src}"

    train_s = {s.strip() for s in args.train_sessions.split(",") if s.strip()}
    eval_s = {s.strip() for s in args.eval_sessions.split(",") if s.strip()}
    overlap = train_s & eval_s
    assert not overlap, f"train 과 eval 세션이 겹침(누수!): {overlap}"

    pat = re.compile(args.session_re)
    labset = {p.stem for p in src_lab.glob("*.txt")
              if p.name != "classes.txt" and p.stat().st_size > 0}

    g = {"train": {"pos": [], "bg": []}, "eval": {"pos": [], "bg": []}}
    unmatched = 0
    for p in sorted(src_img.glob("*.*")):
        m = pat.search(p.stem)
        if not m:
            unmatched += 1
            continue
        s = m.group(1)
        grp = "train" if s in train_s else ("eval" if s in eval_s else None)
        if grp is None:
            continue
        (g[grp]["pos"] if p.stem in labset else g[grp]["bg"]).append(p)

    if unmatched:
        print(f"[!] 세션 파싱 실패 {unmatched}장 (--session-re 확인)")

    def sample_bg(bg, n_pos):
        return random.sample(bg, min(len(bg), int(round(n_pos * args.bg_ratio))))

    place(dst_root, "train", args.bucket, g["train"]["pos"],
          sample_bg(g["train"]["bg"], len(g["train"]["pos"])), src_lab)

    eval_bg = sample_bg(g["eval"]["bg"], len(g["eval"]["pos"]))
    targets = ["valid", "test"] if args.eval_to == "both" else [args.eval_to]
    for t in targets:
        place(dst_root, t, args.bucket, g["eval"]["pos"], eval_bg, src_lab)
    if args.eval_to == "both":
        print("[i] valid=test (같은 세션). 데이터가 늘면 세션을 나눠 독립 test 를 만들 것.")


if __name__ == "__main__":
    main()
