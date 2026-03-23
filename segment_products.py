"""
Ko‘p mahsulot/ob’yektlarni aniqlash, fonni olib tashlash va har birini alohida PNG sifatida saqlash.
FastSAM (Ultralytics) — umumiy mahsulot rasmlari uchun instance segmentatsiya.

O‘rnatish:
  pip install -r requirements.txt

Ishlatish:
  python segment_products.py sizning_rasm.jpg
  python segment_products.py papka/ -o natijalar
  python segment_products.py sizning_rasm.jpg -o natijalar --mode transparent --device cuda
  python segment_products.py rasm.jpg --layout full
  python segment_products.py rasm.jpg --layout centered --canvas-side 2048
  python segment_products.py rasm.jpg --quality high
  python segment_products.py rasm.jpg --quality max --device cuda

Standart chiqish: kirish fayli yonidagi `output` papkasi, fayllar `*_product_01_white.png` ko‘rinishida.
Standart rejim: **centered** — mahsulot qirqiladi, kvadrat markazga kattalashtiriladi.
Papka berilsa — ichidagi barcha rasmlar ketma-ket qayta ishlanadi; model bir marta yuklanadi.

Grafik interfeys (terminal shart emas): python gui.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# FastSAM: (qurilma, model fayli) bo‘yicha bitta marta yuklanadi
_fastsam_models: dict[tuple[str, str], Any] = {}

MODEL_FILES: dict[str, str] = {
    "s": "FastSAM-s.pt",
    "x": "FastSAM-x.pt",
}

# Sifat: inferens o‘lchami (yuqori = chegaralar aniqroq, sekinroq), conf/iou, model (x = aniqroq, og‘ir)
QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {
        "imgsz": 768,
        "conf": 0.28,
        "iou": 0.72,
        "model": "s",
        "refine_mask": False,
    },
    "normal": {
        "imgsz": 1024,
        "conf": 0.25,
        "iou": 0.72,
        "model": "s",
        "refine_mask": False,
    },
    "high": {
        "imgsz": 1280,
        "conf": 0.22,
        "iou": 0.68,
        "model": "s",
        "refine_mask": True,
    },
    "max": {
        "imgsz": 1536,
        "conf": 0.2,
        "iou": 0.65,
        "model": "x",
        "refine_mask": True,
    },
}


def get_inference_params(
    quality: str = "normal",
    *,
    imgsz: int | None = None,
    conf: float | None = None,
    iou: float | None = None,
    model_key: str | None = None,
    refine_mask: bool | None = None,
) -> dict[str, Any]:
    """CLI/GUI: preset + ixtiyoriy ustunliklar."""
    if quality not in QUALITY_PRESETS:
        quality = "normal"
    p = QUALITY_PRESETS[quality]
    out = {
        "imgsz": imgsz if imgsz is not None else p["imgsz"],
        "conf": conf if conf is not None else p["conf"],
        "iou": iou if iou is not None else p["iou"],
        "model_key": model_key if model_key is not None else p["model"],
        "refine_mask": refine_mask if refine_mask is not None else p["refine_mask"],
    }
    return out


def _get_fastsam(device: str, model_key: str) -> Any:
    if model_key not in MODEL_FILES:
        model_key = "s"
    path = MODEL_FILES[model_key]
    key = (device, path)
    if key not in _fastsam_models:
        from ultralytics import FastSAM

        _fastsam_models[key] = FastSAM(path)
    return _fastsam_models[key]


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_images_in_dir(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            out.append(p)
    return out


@dataclass
class SegmentResult:
    """CLI va GUI uchun bir xil natija."""

    ok: bool
    message: str
    saved_files: list[Path] = field(default_factory=list)


def _filter_subsumed_masks(
    masks: list[np.ndarray], iou_containment: float = 0.85
) -> list[np.ndarray]:
    """Kichikroq maskalar katta maska ichida bo‘lsa, ularni tashlab yuborish."""
    if not masks:
        return []
    areas = [m.sum() for m in masks]
    order = sorted(range(len(masks)), key=lambda i: areas[i], reverse=True)
    kept: list[np.ndarray] = []
    for i in order:
        mi = masks[i]
        inside = False
        for kj in kept:
            inter = np.logical_and(mi, kj).sum()
            area = mi.sum()
            if area > 0 and inter / area >= iou_containment:
                inside = True
                break
        if not inside:
            kept.append(mi)
    return kept


def segment_everything(
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    model_key: str = "s",
) -> Any:
    model = _get_fastsam(device, model_key)
    results = model(
        image_bgr,
        device=device,
        retina_masks=True,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )
    return results


def masks_from_results(results, orig_h: int, orig_w: int) -> list[np.ndarray]:
    if not results or results[0].masks is None:
        return []
    m = results[0].masks
    if m.data is None:
        return []
    data = m.data.cpu().numpy()  # (N, h, w) float 0..1
    out: list[np.ndarray] = []
    for i in range(data.shape[0]):
        mask_small = data[i]
        mask_u8 = (mask_small > 0.5).astype(np.uint8) * 255
        if mask_u8.shape[0] != orig_h or mask_u8.shape[1] != orig_w:
            mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        out.append(mask_u8 > 127)
    return out


def refine_mask_edges(mask_bool: np.ndarray) -> np.ndarray:
    """Chekka tishlarni yumshatish (morfologik yopish/ochish)."""
    m = mask_bool.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m > 127


def apply_background(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    mode: str,
) -> Image.Image:
    """mode: 'white' — RGB, fon #FFFFFF; 'transparent' — RGBA, fon shaffof."""
    if mask_bool.shape[:2] != image_bgr.shape[:2]:
        raise ValueError("Maska va rasm o‘lchamlari mos kelmaydi")
    if mode == "white":
        canvas = np.full_like(image_bgr, 255, dtype=np.uint8)
        canvas[mask_bool] = image_bgr[mask_bool]
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    if mode == "transparent":
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        alpha = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
        alpha[mask_bool] = 255
        rgba = np.dstack([rgb, alpha])
        return Image.fromarray(rgba, mode="RGBA")
    raise ValueError("mode 'white' yoki 'transparent' bo‘lishi kerak")


def _bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return 0, 0, mask_bool.shape[1], mask_bool.shape[0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def apply_background_centered(
    image_bgr: np.ndarray,
    mask_bool: np.ndarray,
    mode: str,
    canvas_side: int,
    pad_ratio: float = 0.06,
    inner_margin: float = 0.06,
) -> Image.Image:
    """
    Maska bo‘yicha qirqadi, mahsulotni masshtablab kvadrat kadrdagi markazga qo‘yadi
    (kichik ob’yektlar kattaroq ko‘rinadi).
    """
    h, w = image_bgr.shape[:2]
    x0, y0, x1, y1 = _bbox_from_mask(mask_bool)
    bw, bh = x1 - x0, y1 - y0
    pad = max(4, int(max(bw, bh) * pad_ratio))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)
    crop_img = image_bgr[y0:y1, x0:x1].copy()
    crop_mask = mask_bool[y0:y1, x0:x1].copy()
    ch, cw = crop_img.shape[:2]
    if cw < 1 or ch < 1:
        return apply_background(image_bgr, mask_bool, mode)

    canvas_side = max(256, min(8192, int(canvas_side)))
    fit = max(64, int(canvas_side * (1.0 - 2.0 * inner_margin)))
    scale = min(fit / float(cw), fit / float(ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    resized = cv2.resize(crop_img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    mask_u8 = crop_mask.astype(np.uint8) * 255
    mask_rs = cv2.resize(mask_u8, (nw, nh), interpolation=cv2.INTER_NEAREST) > 127

    ox = (canvas_side - nw) // 2
    oy = (canvas_side - nh) // 2

    if mode == "white":
        canvas = np.full((canvas_side, canvas_side, 3), 255, dtype=np.uint8)
        roi = canvas[oy : oy + nh, ox : ox + nw]
        roi[mask_rs] = resized[mask_rs]
        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    if mode == "transparent":
        rgba = np.zeros((canvas_side, canvas_side, 4), dtype=np.uint8)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        roi = rgba[oy : oy + nh, ox : ox + nw]
        for c in range(3):
            roi[:, :, c][mask_rs] = rgb[:, :, c][mask_rs]
        roi[:, :, 3][mask_rs] = 255
        return Image.fromarray(rgba, mode="RGBA")

    raise ValueError("mode 'white' yoki 'transparent' bo‘lishi kerak")


def process_image(
    input_path: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    min_area_ratio: float,
    mode: str,
    filter_subsumed: bool,
    max_objects: int,
    layout: str = "centered",
    canvas_side: int = 1024,
    pad_ratio: float = 0.06,
    model_key: str = "s",
    refine_mask: bool = False,
) -> SegmentResult:
    image_bgr = cv2.imdecode(np.fromfile(str(input_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Rasm o‘qilmadi: {input_path}")

    h, w = image_bgr.shape[:2]
    min_area = int(h * w * min_area_ratio)

    results = segment_everything(image_bgr, conf, iou, imgsz, device, model_key=model_key)
    masks = masks_from_results(results, h, w)

    filtered: list[np.ndarray] = []
    for mb in masks:
        if refine_mask:
            mb = refine_mask_edges(mb)
        if mb.sum() < min_area:
            continue
        filtered.append(mb)

    if filter_subsumed and len(filtered) > 1:
        filtered = _filter_subsumed_masks(filtered)

    filtered.sort(key=lambda m: m.sum(), reverse=True)
    if max_objects > 0:
        filtered = filtered[:max_objects]

    if not filtered:
        return SegmentResult(
            ok=False,
            message="Hech qanday ob’yekt topilmadi. Ishonchlilik (conf) ni kamaytiring yoki minimal maydonni.",
            saved_files=[],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_abs = output_dir.resolve()
    stem = input_path.stem
    saved: list[Path] = []

    for idx, mb in enumerate(filtered, start=1):
        if layout == "centered":
            pil_img = apply_background_centered(
                image_bgr, mb, mode, canvas_side, pad_ratio=pad_ratio
            )
        else:
            pil_img = apply_background(image_bgr, mb, mode)
        suffix = "rgba" if mode == "transparent" else "white"
        out_name = f"{stem}_product_{idx:02d}_{suffix}.png"
        out_path = out_abs / out_name
        pil_img.save(out_path, compress_level=3)
        saved.append(out_path)

    lay_note = (
        f" (markaz, {canvas_side}×{canvas_side} px)"
        if layout == "centered"
        else " (to‘liq kadr)"
    )
    msg = f"Saqlangan: {len(filtered)} ta fayl{lay_note}\n{out_abs}"
    return SegmentResult(ok=True, message=msg, saved_files=saved)


def process_folder(
    input_dir: Path,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    min_area_ratio: float,
    mode: str,
    filter_subsumed: bool,
    max_objects: int,
    layout: str = "centered",
    canvas_side: int = 1024,
    pad_ratio: float = 0.06,
    model_key: str = "s",
    refine_mask: bool = False,
) -> SegmentResult:
    """Papkadagi barcha rasmlarni ketma-ket qayta ishlaydi (model bir marta yuklangan)."""
    files = list_images_in_dir(input_dir)
    if not files:
        return SegmentResult(
            ok=False,
            message=f"Papka bo‘sh yoki rasmlar yo‘q: {input_dir}",
            saved_files=[],
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    all_saved: list[Path] = []
    lines: list[str] = [f"Papka: {input_dir.name} — {len(files)} ta rasm\n"]

    for fp in files:
        try:
            one = process_image(
                fp,
                output_dir,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                min_area_ratio=min_area_ratio,
                mode=mode,
                filter_subsumed=filter_subsumed,
                max_objects=max_objects,
                layout=layout,
                canvas_side=canvas_side,
                pad_ratio=pad_ratio,
                model_key=model_key,
                refine_mask=refine_mask,
            )
        except (FileNotFoundError, OSError) as e:
            lines.append(f"  ✗ {fp.name}: {e}")
            continue
        except Exception as e:
            lines.append(f"  ✗ {fp.name}: {e}")
            continue

        if one.ok:
            all_saved.extend(one.saved_files)
            lines.append(f"  ✓ {fp.name} — {len(one.saved_files)} ta ob’yekt")
        else:
            lines.append(f"  ○ {fp.name} — {one.message}")

    msg = "\n".join(lines)
    ok = bool(all_saved)
    if not ok and files:
        return SegmentResult(
            ok=False,
            message=msg + "\n\nHech qayerga saqlanmadi.",
            saved_files=[],
        )
    return SegmentResult(ok=True, message=msg, saved_files=all_saved)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Rasmdagi har bir mahsulot uchun alohida PNG (oq fon yoki shaffof fon)."
    )
    p.add_argument(
        "input",
        type=Path,
        help="Kirish rasmi yoki papka (ichidagi jpg/png/webp …)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Chiqish papkasi (standart: kirish fayli yonidagi 'output')",
    )
    p.add_argument(
        "--quality",
        choices=tuple(QUALITY_PRESETS.keys()),
        default="normal",
        help="draft=tez | normal=teng | high=yuqori aniqlik | max=FastSAM-x + 1536px (eng sekin)",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Ishonchlilik (standart: sifat profili bo‘yicha)",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=None,
        help="Maskalar IoU NMS (standart: profil)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Inferens o‘lchami px (standart: profil; katta = aniqroq chekka)",
    )
    p.add_argument(
        "--model",
        choices=("s", "x"),
        default=None,
        help="s=FastSAM-s (tez) | x=FastSAM-x (aniqroq, ~84 MB)",
    )
    p.add_argument(
        "--no-refine-mask",
        action="store_true",
        help="Yumshoq chekka (morfologiya) filtrini o‘chirish",
    )
    p.add_argument("--device", default="cpu", help="cpu yoki cuda")
    p.add_argument(
        "--min-area",
        type=float,
        default=0.002,
        help="Minimal maska maydoni (rasm ulushining ulushi, shovqinni kesish)",
    )
    p.add_argument(
        "--mode",
        choices=("white", "transparent"),
        default="white",
        help="white: fon #FFFFFF (RGB PNG); transparent: fon shaffof (RGBA PNG)",
    )
    p.add_argument(
        "--no-filter-nested",
        action="store_true",
        help="Ichma-ich maskalarni filtrlashni o‘chirish",
    )
    p.add_argument(
        "--max-objects",
        type=int,
        default=50,
        help="Eng katta maydonli maskalar soni (0 = cheksiz)",
    )
    p.add_argument(
        "--layout",
        choices=("centered", "full"),
        default="centered",
        help="centered: qirqib markazda kattalashtirilgan kvadrat; full: asl kadr o‘lchami",
    )
    p.add_argument(
        "--canvas-side",
        type=int,
        default=1024,
        help="centered rejimida chiqish kvadrat tomoni (px), masalan 1024 yoki 2048",
    )
    p.add_argument(
        "--pad-ratio",
        type=float,
        default=0.06,
        help="Qirqish atrofidagi bo‘shliq (maska nisbati)",
    )
    args = p.parse_args()

    inp = args.input
    if not inp.exists():
        print(f"Yo‘q: {inp}", file=sys.stderr)
        return 1

    out = args.output_dir
    if out is None:
        out = inp.parent / "output"

    inf = get_inference_params(
        args.quality,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        model_key=args.model,
    )
    if args.no_refine_mask:
        inf["refine_mask"] = False

    try:
        if inp.is_dir():
            result = process_folder(
                inp,
                out,
                conf=inf["conf"],
                iou=inf["iou"],
                imgsz=inf["imgsz"],
                device=args.device,
                min_area_ratio=args.min_area,
                mode=args.mode,
                filter_subsumed=not args.no_filter_nested,
                max_objects=args.max_objects,
                layout=args.layout,
                canvas_side=args.canvas_side,
                pad_ratio=args.pad_ratio,
                model_key=inf["model_key"],
                refine_mask=bool(inf["refine_mask"]),
            )
        elif inp.is_file():
            result = process_image(
                inp,
                out,
                conf=inf["conf"],
                iou=inf["iou"],
                imgsz=inf["imgsz"],
                device=args.device,
                min_area_ratio=args.min_area,
                mode=args.mode,
                filter_subsumed=not args.no_filter_nested,
                max_objects=args.max_objects,
                layout=args.layout,
                canvas_side=args.canvas_side,
                pad_ratio=args.pad_ratio,
                model_key=inf["model_key"],
                refine_mask=bool(inf["refine_mask"]),
            )
        else:
            print(f"Bu yo‘l na fayl, na papka: {inp}", file=sys.stderr)
            return 1
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Xato: {e}", file=sys.stderr)
        return 1

    if result.ok:
        print(result.message)
    else:
        print(result.message, file=sys.stderr)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
