"""검수 끝난 배치(images/ + labels/)를 학습셋(data/dataset)에 배치한다.

누수 방지가 핵심: 같은 차량 한 대가 여러 프레임·여러 카메라(gige1/gige2)에 걸쳐 찍힌다.
프레임을 랜덤 분할하면 같은 차가 train 과 val 양쪽에 들어가 성능이 가짜로 높게 나온다.
-> 반드시 '차량' 단위로 가른다. 두 가지 모드 제공:

  --split-mode session   : 세션을 통째로 train / eval 로 가른다. (특정 도메인을 held-out 평가)
  --split-mode stratify  : 세션은 유지하고, 각 세션 '안에서' 차량 단위로 train/val 을 나눈다.
                           모든 세션이 train·val 양쪽에 고르게 들어가므로 도메인 편중이 없다.
                           (데이터가 적고 도메인이 다양할 때 권장)

배경(빈 라벨): 과하면 모델이 '아무것도 없다'를 학습해 recall 이 떨어진다.
  --bg-ratio 로 양성 대비 상한을 둔다(배경 풀이 그보다 적으면 전부 포함).

사용:
    # 세션 안에서 차량단위 8:2 (파일명이 s1_v001_gige1 형태: 그룹=세션,차량)
    python prepare_split.py --src data/260701 --bucket v4_260701 --split-mode stratify \
        --group-re "(s\\d+)_(v\\d+)" --val-frac 0.2

    # 세션 통째 분할 (파일명이 ..._20260619_124450_... 형태)
    python prepare_split.py --src data/260619 --split-mode session \
        --train-sessions 20260619_124248,20260619_124450 --eval-sessions 20260619_130143
"""
import argparse
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # scripts/ 상위 = 프로젝트 루트

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


def split_session(src_img, labset, session_re, train_s, eval_s):
    """세션 통째 분할. -> {'train':..., 'eval':...}"""
    pat = re.compile(session_re)
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
    return g


def split_stratify(src_img, labset, group_re, val_frac, rng):
    """세션 안에서 '차량' 단위 train/val 분할. group_re 는 (세션, 차량) 2그룹."""
    pat = re.compile(group_re)
    groups, sess_veh, unmatched = {}, {}, 0
    for p in sorted(src_img.glob("*.*")):
        m = pat.search(p.stem)
        if not m:
            unmatched += 1
            continue
        key = (m.group(1), m.group(2))            # (세션, 차량)
        groups.setdefault(key, []).append(p)
        sess_veh.setdefault(m.group(1), set()).add(m.group(2))
    if unmatched:
        print(f"[!] 그룹 파싱 실패 {unmatched}장 (--group-re 확인)")

    val_keys = set()
    for s, vehs in sess_veh.items():
        vl = sorted(vehs)
        rng.shuffle(vl)
        n_val = round(len(vl) * val_frac) if len(vl) > 1 else 0
        for v in vl[:n_val]:
            val_keys.add((s, v))
        print(f"    세션 {s}: 차량 {len(vl)}대 -> train {len(vl)-n_val} / val {n_val}")

    g = {"train": {"pos": [], "bg": []}, "eval": {"pos": [], "bg": []}}
    for key, imgs in groups.items():
        grp = "eval" if key in val_keys else "train"
        for p in imgs:
            (g[grp]["pos"] if p.stem in labset else g[grp]["bg"]).append(p)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="검수 끝난 배치 폴더 (images/ labels/ 포함)")
    ap.add_argument("--data-root", default="data/dataset", help="학습셋 루트 (기본: ./data/dataset)")
    ap.add_argument("--bucket", default="v2_visolabel", help="배치될 버킷 폴더명")
    ap.add_argument("--split-mode", default="session", choices=["session", "stratify"])
    # session 모드
    ap.add_argument("--train-sessions", help="[session] 쉼표 구분 (예: s1,s2)")
    ap.add_argument("--eval-sessions", help="[session] 쉼표 구분 (예: s3)")
    ap.add_argument("--eval-to", default="both", choices=["valid", "test", "both"],
                    help="[session] 평가 세션을 valid/test/양쪽 중 어디에")
    ap.add_argument("--session-re", default=DEFAULT_SESSION_RE, help="[session] 세션 정규식")
    # stratify 모드
    ap.add_argument("--group-re", default=r"(s\d+)_(v\d+)", help="[stratify] (세션,차량) 2그룹 정규식")
    ap.add_argument("--val-frac", type=float, default=0.2, help="[stratify] 세션별 val 비율")
    # 공통
    ap.add_argument("--bg-ratio", type=float, default=2.0, help="배경 = 양성 * 이 값 (recall 보호)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = resolve(args.src)
    src_img, src_lab = src / "images", src / "labels"
    dst_root = resolve(args.data_root)
    assert src_img.exists() and src_lab.exists(), f"images/ labels/ 없음: {src}"

    labset = {p.stem for p in src_lab.glob("*.txt")
              if p.name != "classes.txt" and p.stat().st_size > 0}

    if args.split_mode == "session":
        assert args.train_sessions and args.eval_sessions, "session 모드는 --train/eval-sessions 필요"
        train_s = {s.strip() for s in args.train_sessions.split(",") if s.strip()}
        eval_s = {s.strip() for s in args.eval_sessions.split(",") if s.strip()}
        assert not (train_s & eval_s), f"train/eval 세션 겹침(누수!): {train_s & eval_s}"
        g = split_session(src_img, labset, args.session_re, train_s, eval_s)
        eval_targets = ["valid", "test"] if args.eval_to == "both" else [args.eval_to]
    else:  # stratify
        g = split_stratify(src_img, labset, args.group_re, args.val_frac, rng)
        eval_targets = ["valid"]

    def sample_bg(bg, n_pos):
        return rng.sample(bg, min(len(bg), int(round(n_pos * args.bg_ratio))))

    place(dst_root, "train", args.bucket, g["train"]["pos"],
          sample_bg(g["train"]["bg"], len(g["train"]["pos"])), src_lab)
    eval_bg = sample_bg(g["eval"]["bg"], len(g["eval"]["pos"]))
    for t in eval_targets:
        place(dst_root, t, args.bucket, g["eval"]["pos"], eval_bg, src_lab)
    if args.split_mode == "session" and args.eval_to == "both":
        print("[i] valid=test (같은 세션). 데이터가 늘면 세션을 나눠 독립 test 를 만들 것.")


if __name__ == "__main__":
    main()
