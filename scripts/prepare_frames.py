"""현장 녹화 영상에서 재학습용 프레임을 자동 선별·정제한다.

문제:
  - 영상에 운영 중인 모델의 검출 오버레이(노란 박스) + 카메라 정보 텍스트가 픽셀에 구워져 있음
    -> 그대로 학습하면 모델이 "노란 사각형=바코드"라는 가짜 단서를 학습(치명적 오염).
  - 원본은 수만 프레임인데 대부분 빈 화면(차 없음).

처리:
  1) 빈 화면 제거  : MOG2 배경차분으로 '차량 통과' 프레임만 통과 (downscale 해서 빠르게).
  2) 근접 중복 제거: dHash(64bit) 해밍거리로 직전 보존 프레임과 너무 비슷하면 버림.
  3) 오버레이 정제: 내용물이 흑백이므로 '채도 높고 밝은 픽셀 = 오버레이'.
                    마스크를 inpaint -> 박스·텍스트 동시 제거. 박스는 얇은 외곽선이라
                    지워도 안쪽 VIN 내용물은 그대로 남는다.
  4) 분류         : 노란 박스 존재 여부로
                    detected/(정탐·오탐 섞임, 박스 좌표를 YOLO 사전라벨로 기록)
                    missed/  (차는 있는데 박스 없음 = 미탐 후보, 빈 라벨)
                    로 나눠 저장. (정탐 vs 오탐 구분은 VisoLabel 검수에서 사람이 함)

출력: <--output>/{detected,missed}/<영상>_f<프레임번호>.jpg + .txt, classes.txt

사용:
    python prepare_frames.py --input data/260619/_etc/originVideo --output data/260619/frames_for_label
    python prepare_frames.py --input <영상폴더> --only cam1_0001.mp4      # 1개만(테스트)
    python prepare_frames.py --input <영상폴더> --stride 2 --dedup 6 --motion 0.05
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent   # scripts/ 상위 = 프로젝트 루트

# 영상 내용물은 흑백 -> 채도가 이 값보다 높으면 컬러 오버레이(박스/텍스트)로 간주.
# 단, 어두운 픽셀은 노이즈로 채도가 가짜로 높게 나오므로 밝기(V)도 함께 요구해야 함
# (안 그러면 노출 낮은 gige1 어두운 영역이 통째로 오인되어 inpaint 로 화면이 뭉개짐).
SAT_OVERLAY = 60
VAL_OVERLAY = 90  # 이 밝기 미만은 오버레이로 보지 않음(어두운 노이즈 보호)
# 노란 박스(검출 표시) 색범위 (OpenCV HSV: H 0~180)
YELLOW_LO = np.array([18, 80, 80])
YELLOW_HI = np.array([38, 255, 255])


def dhash(gray_small):
    """64bit difference hash. gray_small 은 9x8 회색 이미지."""
    diff = gray_small[:, 1:] > gray_small[:, :-1]
    return np.packbits(diff.flatten())


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def yellow_boxes(hsv, min_side):
    """노란 오버레이에서 박스(외곽선) 바운딩박스 목록 반환. [(x,y,w,h), ...] (픽셀)."""
    ymask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    if ymask.sum() == 0:
        return []
    # 외곽선이 끊겨 있을 수 있어 닫힘연산으로 사각형 윤곽 연결
    ymask = cv2.morphologyEx(ymask, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    cnts, _ = cv2.findContours(ymask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_side and h >= min_side:
            out.append((x, y, w, h))
    return out


def clean_overlay(bgr, hsv):
    """오버레이 제거 후 깨끗한 흑백 프레임 복원.
    - 채도 있는 픽셀 = 컬러 오버레이(노란 박스 + 상단 초록 텍스트) -> inpaint
    - 좌하단 회색 카메라정보 텍스트('Exp.. Gain..')는 채도가 낮아 안 잡히므로
      고정 코너 strip 을 함께 마스킹 (그 영역은 항상 바닥/도로라 inpaint 안전)
    """
    H, W = bgr.shape[:2]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    # 채도 AND 밝기 둘 다 높은 픽셀만 오버레이 (어두운 노이즈는 V 조건에서 탈락)
    mask = ((sat > SAT_OVERLAY) & (val > VAL_OVERLAY)).astype(np.uint8) * 255
    # 좌하단 텍스트 strip (상대좌표: 아래 3.5%, 왼쪽 10%)
    mask[int(H * 0.965):H, 0:int(W * 0.10)] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    return cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)


def process_video(path, out_dir, stride, motion_thr, dedup_thr, min_box_side, warmup):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[!] 열기 실패: {path.name}")
        return 0, 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = 512 / W                                  # 모션/해시용 축소폭
    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)
    min_side = max(20, int(min_box_side * W / 4096))  # 4096 기준값을 해상도에 맞춤

    last_hash = None
    fidx = -1
    n_det = n_miss = 0
    stem = path.stem
    while True:
        # stride: 분석 안 할 프레임은 grab()으로 디코드 비용 절약하고 건너뜀
        if not cap.grab():
            break
        fidx += 1
        if fidx % stride != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        fg = bg.apply(gray_small)
        if fidx < warmup:                            # MOG2 배경 학습 구간은 건너뜀
            continue
        fg_ratio = (fg > 0).mean()
        if fg_ratio < motion_thr:                    # 차량 없음(빈 화면) -> 버림
            last_hash = None                         # 통과 끝 -> 다음 통과는 새로 시작
            continue

        # 근접 중복 제거
        h9 = cv2.resize(gray_small, (9, 8), interpolation=cv2.INTER_AREA)
        hsh = dhash(h9)
        if last_hash is not None and hamming(hsh, last_hash) <= dedup_thr:
            continue
        last_hash = hsh

        # 오버레이 분석은 풀해상도에서
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        boxes = yellow_boxes(hsv, min_side)
        clean = clean_overlay(frame, hsv)

        name = f"{stem}_f{fidx:06d}"
        sub = "detected" if boxes else "missed"
        cv2.imwrite(str(out_dir / sub / f"{name}.jpg"), clean,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 노란 박스 좌표 -> YOLO 정규화 사전라벨 (class 0). missed 는 빈 파일.
        lines = []
        for (x, y, w, h) in boxes:
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        (out_dir / sub / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")

        if boxes:
            n_det += 1
        else:
            n_miss += 1

    cap.release()
    print(f"[+] {path.name:42s} frames={total:5d} -> detected {n_det:4d} / missed {n_miss:4d}")
    return n_det, n_miss


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="원본 영상(.mp4) 폴더")
    ap.add_argument("--output", required=True, help="프레임 출력 폴더 (detected/ missed/ 가 생성됨)")
    ap.add_argument("--names-yaml", default="data/dataset/data.yaml",
                    help="classes.txt 생성용 클래스명 출처")
    ap.add_argument("--stride", type=int, default=2, help="N프레임마다 1장 분석 (12.5fps면 2 -> ~6fps)")
    ap.add_argument("--motion", type=float, default=0.05, help="전경 비율 임계 (이상이면 차량 있음)")
    ap.add_argument("--dedup", type=int, default=6, help="dHash 해밍거리 임계 (이하면 중복으로 버림)")
    ap.add_argument("--min-box", type=int, default=40, help="검출박스 최소 변 길이(px, 4096폭 기준·해상도에 맞춰 자동 환산)")
    ap.add_argument("--warmup", type=int, default=25, help="MOG2 배경학습용 초기 스킵 프레임 수")
    ap.add_argument("--only", default=None, help="특정 파일 하나만 처리(파일명)")
    ap.add_argument("--reset", action="store_true", help="출력 폴더 비우고 시작")
    args = ap.parse_args()

    in_dir, out_dir = resolve(args.input), resolve(args.output)
    assert in_dir.exists(), f"영상 폴더 없음: {in_dir}"

    if args.reset and out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("detected", "missed"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # classes.txt (VisoLabel 참조)
    names_yaml = resolve(args.names_yaml)
    names = yaml.safe_load(names_yaml.read_text(encoding="utf-8")).get("names") \
        if names_yaml.exists() else None
    if names:
        names_list = list(names.values()) if isinstance(names, dict) else list(names)
        for sub in ("detected", "missed"):
            (out_dir / sub / "classes.txt").write_text("\n".join(names_list), encoding="utf-8")

    vids = sorted(in_dir.glob("*.mp4"))
    if args.only:
        vids = [v for v in vids if v.name == args.only]
    print(f"[i] 처리 대상 {len(vids)}개")

    tot_d = tot_m = 0
    for v in vids:
        d, m = process_video(v, out_dir, args.stride, args.motion, args.dedup,
                             args.min_box, args.warmup)
        tot_d += d
        tot_m += m
    print(f"\n[=] 합계: detected {tot_d} / missed {tot_m} (총 {tot_d + tot_m}장)")
    print(f"    출력: {out_dir}")


if __name__ == "__main__":
    main()
