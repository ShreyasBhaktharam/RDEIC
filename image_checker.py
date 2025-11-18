#!/usr/bin/env python3
import argparse, math, os, csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pyiqa

def load_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return a, b

def psnr_mse_mae(ref: np.ndarray, rec: np.ndarray) -> tuple[float, float, float]:
    ref, rec = ref.astype(np.float32), rec.astype(np.float32)
    mse = float(np.mean((ref - rec) ** 2))
    mae = float(np.mean(np.abs(ref - rec)))
    psnr = 20 * math.log10(255.0) - 10 * math.log10(mse + 1e-12)
    return psnr, mse, mae

def to_t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

def stem_map(dir_path: Path, exts: set[str]) -> dict[str, Path]:
    out = {}
    for p in dir_path.rglob("*"):
        if p.suffix.lower() in exts:
            out[p.stem] = p
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="Reference image or folder")
    ap.add_argument("--rec", required=True, help="Reconstructed image or folder")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--save_csv", default="", help="Optional path to save CSV summary")
    ap.add_argument("--save_diff", action="store_true", help="Save per-image abs-diff PNGs next to rec files")
    ap.add_argument("--exts", nargs="+", default=[".png",".jpg",".jpeg"], help="Extensions for folder mode")
    args = ap.parse_args()

    ref_p = Path(args.ref)
    rec_p = Path(args.rec)
    exts = set(e.lower() for e in args.exts)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    lpips_metric = pyiqa.create_metric("lpips", device=device)

    rows = []
    pairs = []

    if ref_p.is_file() and rec_p.is_file():
        pairs = [(ref_p, rec_p)]
    elif ref_p.is_dir() and rec_p.is_dir():
        ref_map = stem_map(ref_p, exts)
        rec_map = stem_map(rec_p, exts)
        shared = sorted(set(ref_map.keys()) & set(rec_map.keys()))
        if not shared:
            raise SystemExit("No matching filenames (by stem) found between the two folders.")
        pairs = [(ref_map[k], rec_map[k]) for k in shared]
    else:
        raise SystemExit("Both --ref and --rec must be files or both must be directories.")

    psnrs, mses, maes, lpips_list = [], [], [], []
    for r, c in pairs:
        ref = load_rgb(r)
        rec = load_rgb(c)
        ref, rec = center_crop_to_match(ref, rec)

        psnr, mse, mae = psnr_mse_mae(ref, rec)

        t_ref = to_t(ref).to(device)
        t_rec = to_t(rec).to(device)
        lp = float(lpips_metric(t_rec, t_ref).item())

        psnrs.append(psnr); mses.append(mse); maes.append(mae); lpips_list.append(lp)
        rows.append({"name": r.stem, "psnr": psnr, "mse": mse, "mae": mae, "lpips": lp})

        if args.save_diff and c.parent.exists():
            diff = np.clip(np.abs(ref.astype(np.int16) - rec.astype(np.int16)), 0, 255).astype(np.uint8)
            out_path = c.with_name(f"{c.stem}_diff.png")
            Image.fromarray(diff).save(out_path)

    avg = {
        "name": "AVERAGE",
        "psnr": float(np.mean(psnrs)),
        "mse": float(np.mean(mses)),
        "mae": float(np.mean(maes)),
        "lpips": float(np.mean(lpips_list)),
    }

    # Print summary
    print(f"Count: {len(pairs)}")
    print(f"PSNR:  {avg['psnr']:.3f} dB")
    print(f"MSE:   {avg['mse']:.6f}")
    print(f"MAE:   {avg['mae']:.6f}")
    print(f"LPIPS: {avg['lpips']:.6f} (lower is better)")

    # Optional CSV
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name","psnr","mse","mae","lpips"])
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            writer.writerow(avg)

if __name__ == "__main__":
    main()
