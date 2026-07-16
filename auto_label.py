"""기존 모델로 신규 이미지를 1차 추론해 YOLO 포맷 라벨(.txt)을 생성한다.

VisoLabel 에서 "박스가 이미 그려진 상태"로 불러와 검수만 하기 위한 반자동 라벨링 1단계.

conf 를 낮게 잡는 이유(중요):
  여기서 나오는 오탐은 학습에 들어가지 않는다 — 검수 때 사람이 지운다(클릭 한 번).
  반대로 conf 가 높아 VIN 을 놓치면 사람이 박스를 새로 그려야 하고(느림),
  검수자가 그걸 못 보고 넘기면 '진짜 VIN 이 있는데 배경'으로 학습되어 recall 을 직접 깎는다.
  즉 사전라벨 단계에서는 과검출이 미검출보다 훨씬 싸다.

사용:
    python auto_label.py --batch data260701
    python auto_label.py --batch data260701 --conf 0.1     # 후보 더 넓게
    python auto_label.py --batch data260701 --model _model_/vincode_v2.pt
"""

import argparse
import os
import shutil
from pathlib import Path

# CUDA 단편화 완화. torch import 전에 설정해야 함.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True,
                    help="배치 폴더. 안에 images/ 가 있어야 함 (예: data260701)")
    ap.add_argument("--model", default="_model_/vincode_v3.pt", help="사전라벨에 쓸 모델")
    ap.add_argument("--names-yaml", default="data_exist/data.yaml",
                    help="classes.txt 생성용 클래스명 출처")
    ap.add_argument("--conf", type=float, default=0.2,
                    help="신뢰도 임계값. 검수용이라 낮게 (docstring 참고)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU 임계값")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="추론 입력 크기. 원본보다 크게 잡아도 이득 없음(업스케일)")
    ap.add_argument("--chunk", type=int, default=16,
                    help="한 번에 넘길 이미지 수 (아래 주석 참고). OOM 시 줄일 것")
    ap.add_argument("--device", default=None, help="'0', 'cpu' 등. 미지정시 자동")
    args = ap.parse_args()

    batch = (ROOT / args.batch) if not Path(args.batch).is_absolute() else Path(args.batch)
    model_path = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    src_images = batch / "images"
    viso_dir = batch / "visolabel"    # 검수용: 이미지 + txt 가 한 폴더
    labels_dir = batch / "labels"     # 학습 재사용용: 표준 YOLO 구조

    assert model_path.exists(), f"모델 없음: {model_path}"
    assert src_images.exists(), f"이미지 폴더 없음: {src_images}"

    images = sorted(p for p in src_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    print(f"[i] 신규 이미지 {len(images)}장 발견")

    viso_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = None
    names_yaml = ROOT / args.names_yaml
    if names_yaml.exists():
        names = yaml.safe_load(names_yaml.read_text(encoding="utf-8")).get("names")

    model = YOLO(str(model_path))
    print(f"[i] 모델 클래스: {model.names}")

    # ── 추론 ──
    # ★ predict 에 리스트를 통째로 넘기면 stream=True 여도 그 길이만큼 한 배치로 묶여
    #   GPU 메모리가 터진다(358장 -> 8.7GiB 시도). 반드시 chunk 단위로 나눠 넘길 것.
    def infer():
        for i in range(0, len(images), args.chunk):
            part = images[i:i + args.chunk]
            # stream=True 는 입력 순서를 보존하므로 원본 경로와 zip.
            # (리스트를 넘기면 ultralytics 가 r.path 를 image0/image1.. 로 매겨 원본 파일명을 잃음)
            for pair in zip(part, model.predict(
                source=[str(p) for p in part],
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                half=torch.cuda.is_available(),  # FP16: 메모리 절감, 품질 영향 없음
                stream=True,
                verbose=False,
            )):
                yield pair

    total_boxes = 0
    empty = []
    for i, (src_img, r) in enumerate(infer()):
        if torch.cuda.is_available() and i % 16 == 0:
            torch.cuda.empty_cache()
        stem = src_img.stem
        lines = []
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.int().tolist()
            xywhn = r.boxes.xywhn.tolist()   # 정규화된 cx, cy, w, h (YOLO 포맷)
            for c, (cx, cy, w, h) in zip(cls, xywhn):
                lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += len(lines)
        else:
            empty.append(stem)

        txt = "\n".join(lines)
        (labels_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")
        (viso_dir / f"{stem}.txt").write_text(txt, encoding="utf-8")
        dst = viso_dir / src_img.name
        if not dst.exists():
            shutil.copy2(src_img, dst)

    if names:
        names_list = list(names.values()) if isinstance(names, dict) else list(names)
        (viso_dir / "classes.txt").write_text("\n".join(names_list), encoding="utf-8")

    print(f"[+] 완료: 박스 {total_boxes}개, 라벨 파일 {len(images)}개 생성")
    print(f"    - 검수용(이미지+txt): {viso_dir}")
    print(f"    - 학습용 라벨:        {labels_dir}")
    if empty:
        print(f"[!] 박스 0개 이미지 {len(empty)}장 — 검수 시 직접 그려야 할 후보")


if __name__ == "__main__":
    main()
