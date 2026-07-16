"""VIN 바코드 스티커 탐지 모델 fine-tune.

설계 의도 (현장 = 흑백 / 고정화각 / 소형객체):
  - 흑백        : 색상·채도 aug 무의미 -> hsv_h=hsv_s=0, 밝기(hsv_v)만 유지
  - 고정화각    : 전단/원근/모자이크 0 (실제로 안 일어나는 변형). 설치 기울기만 degrees 로 소폭 허용
  - 좌우/상하반전: 0 (고정 카메라라 VIN 방향 일정, 거울반전은 가짜 데이터)
  - 작은 객체   : imgsz 크게(1280), scale 로 거리에 따른 크기 변동 커버
  - 미탐 최소화 : val 을 배포 도메인으로 잡아 best.pt 를 실제 환경 기준 선택
                  (배포 시 conf 를 낮추면 recall 추가 확보 — README 참고)

사용:
    python train_vincode.py                                      # 기본 통합 구성
    python train_vincode.py --data data_field_only.yaml --name fieldonly
    python train_vincode.py --base _model_/vincode_v2.pt --name v4
    python train_vincode.py --batch 4 --imgsz 1024               # OOM 시

메모리(8GB GPU 기준): OOM 나면 --batch 8 -> 4 -> 2, 그래도 나면 --imgsz 1280 -> 1024.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def resolve_data(data_yaml: Path, data_root: Path) -> Path:
    """yaml 의 상대경로를 data_root 기준 절대경로로 확정한 해석본을 .cache/ 에 생성.

    ultralytics 는 path: 가 상대경로면 저장소가 아니라 자기 settings 의 datasets_dir
    기준으로 풀어버린다. 그래서 저장소를 클론한 위치와 무관하게 동작시키려면
    실행 시점에 절대경로를 박아 넣어야 한다.
    """
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    d["path"] = str(data_root.resolve())
    out = ROOT / ".cache" / f"{data_yaml.stem}.resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_combined.yaml", help="데이터 구성 yaml")
    ap.add_argument("--data-root", default="data_exist",
                    help="yaml 안 상대경로의 기준 루트 (기본: ./data_exist)")
    ap.add_argument("--base", default="_model_/vincode_v3.pt",
                    help="warm-start 할 가중치. 처음 학습이면 COCO 사전학습(예: yolo26n.pt) 지정")
    ap.add_argument("--project", default="runs_vincode")
    ap.add_argument("--name", default="finetune",
                    help="실행 이름. 같은 이름이 있으면 자동으로 뒤에 번호가 붙어 기존 결과를 덮지 않음")
    ap.add_argument("--imgsz", type=int, default=1280, help="작은 VIN 검출 위해 크게. OOM 시 1024")
    ap.add_argument("--batch", type=int, default=8, help="OOM 시 4 -> 2")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=50, help="개선 없으면 조기 종료")
    ap.add_argument("--device", default=0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    base = (ROOT / args.base) if not Path(args.base).is_absolute() else Path(args.base)
    data_yaml = (ROOT / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    data_root = (ROOT / args.data_root) if not Path(args.data_root).is_absolute() else Path(args.data_root)

    assert data_yaml.exists(), f"데이터 yaml 없음: {data_yaml}"
    assert data_root.exists(), f"데이터 루트 없음: {data_root} (--data-root 로 지정)"
    if not base.exists():
        raise SystemExit(
            f"[!] base 가중치 없음: {base}\n"
            f"    - 이어학습: 직전 모델 경로를 --base 로 지정\n"
            f"    - 처음부터: COCO 사전학습 모델 지정 (예: --base yolo26n.pt)"
        )

    resolved = resolve_data(data_yaml, data_root)
    print(f"[i] base={base.name} | data={data_yaml.name} | data_root={data_root}")

    model = YOLO(str(base))  # scratch 아님 — 기존 가중치에서 이어학습
    model.train(
        data=str(resolved),
        project=args.project,
        name=args.name,
        exist_ok=False,     # ★ 기존 실행 결과를 덮지 않고 번호를 증가시킴 (가중치 유실 방지)

        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        patience=args.patience,
        amp=True,           # 혼합정밀 — 메모리 절감
        cache=False,        # 고해상 이미지 캐시하면 RAM 폭발
        device=args.device,
        workers=args.workers,

        # 학습률: optimizer=auto 가 데이터 규모 보고 AdamW lr 자동 선택.
        # 완만한 lr 이라 도메인 적응하면서 기존 지식을 덜 잊는다.
        optimizer="auto",

        # ── Augmentation: 배포 환경에서 '실제로 변하는 축'만 ──
        mosaic=0.0,         # 고정화각·소형객체 -> 오히려 해로움
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,        # 카메라 설치가 완전 수평이 아닐 수 있음 -> ±5°. 기울기 크면 8 정도까지
        shear=0.0,
        perspective=0.0,
        fliplr=0.0,         # 고정 카메라 -> 좌우반전은 가짜 데이터
        flipud=0.0,
        hsv_h=0.0,          # 흑백 -> 색조 무의미
        hsv_s=0.0,          # 흑백 -> 채도 무의미
        hsv_v=0.4,          # 밝기 변화(조명/날씨/역광)만 — 미탐 줄이는 데 유효
        translate=0.1,      # 차량 좌우 위치 변동
        scale=0.6,          # 거리별 크기 변동 + 근거리 차량은 VIN 이 더 큼 (0.4~1.6배)
        erasing=0.2,        # 부분 가림(먼지/반사) 약하게. 소형객체라 과하면 라벨 노이즈
    )

    save_dir = model.trainer.save_dir
    metrics = model.val(split="test")
    print("=== TEST (배포 도메인 held-out) ===")
    print(f"  precision {metrics.box.mp:.4f} | recall {metrics.box.mr:.4f}")
    print(f"  mAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f}")
    print("best.pt:", save_dir / "weights" / "best.pt")


if __name__ == "__main__":
    main()
