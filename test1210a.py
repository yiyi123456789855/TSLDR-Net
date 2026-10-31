import os, glob, math, csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import utils as vutils

# pip install pytorch-msssim
try:
    from pytorch_msssim import ssim as ssim_torch
except Exception as e:
    raise ImportError("请先安装: pip install pytorch-msssim") from e

# =============================
# utils (log<->lin 与训练脚本一致)
# =============================
EPS = 1e-6

def to_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp_min(EPS))

def from_log_preserve(y: torch.Tensor, ref_linear: torch.Tensor) -> torch.Tensor:
    out = torch.exp(y) - EPS
    ref_mean = ref_linear.mean(dim=[2,3], keepdim=True)
    ref_std  = ref_linear.std(dim=[2,3], keepdim=True).clamp_min(1e-6)
    out_mean = out.mean(dim=[2,3], keepdim=True)
    out_std  = out.std(dim=[2,3], keepdim=True).clamp_min(1e-6)
    out = (out - out_mean) / out_std * ref_std + ref_mean
    return out.clamp(0.0, 1.0)

def from_log_simple(y: torch.Tensor) -> torch.Tensor:
    """
    监督模型用的简化版：只从 log 还原到线性域，不再匹配噪声图的 mean/std
    """
    out = torch.exp(y) - EPS
    return out.clamp(0.0, 1.0)


@torch.no_grad()
def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target, reduction='mean').item()
    return -10.0 * math.log10(max(mse, 1e-10))

@torch.no_grad()
def calc_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ssim_torch(pred, target, data_range=1.0, size_average=True).item()

# =============================
# Dataset（严格按同名 noisy/clean 配对；不缩放，原分辨率评测）
# =============================
class PairedSet(Dataset):
    def __init__(self, noisy_dir: str, clean_dir: str, gray: bool = True):
        self.gray = gray
        n_list = self._list_images(noisy_dir)
        c_map = {os.path.basename(p): p for p in self._list_images(clean_dir)}
        self.pairs: List[Tuple[str,str]] = []
        for n in n_list:
            name = os.path.basename(n)
            if name in c_map:
                self.pairs.append((n, c_map[name]))
        if not self.pairs:
            raise ValueError("No filename-matched pairs between noisy and clean.")

    def _list_images(self, root: str) -> List[str]:
        files = []
        for e in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp"):
            files += glob.glob(os.path.join(root, e))
        return sorted(files)

    def _load_tensor(self, p: str) -> torch.Tensor:
        img = Image.open(p)
        if self.gray:
            img = img.convert("L")
            arr = np.array(img, dtype=np.uint8)[:, :, None]
        else:
            if img.mode != "RGB":
                img = img.convert("RGB")
            arr = np.array(img, dtype=np.uint8)
        t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
        return t

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        n_path, c_path = self.pairs[idx]
        return self._load_tensor(n_path), self._load_tensor(c_path), os.path.basename(n_path)

# =============================
# 模型（兼容你训练脚本的 UNet 结构）
# =============================

def conv_block(in_c, out_c, groups=32):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(groups, out_c), num_channels=out_c),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self, in_ch=1, base=96, groups=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base, groups)
        self.enc2 = conv_block(base, base*2, groups)
        self.enc3 = conv_block(base*2, base*4, groups)
        self.enc4 = conv_block(base*4, base*8, groups)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base*8, base*16, groups)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2); self.dec4 = conv_block(base*16, base*8, groups)
        self.up3 = nn.ConvTranspose2d(base*8,  base*4, 2, 2); self.dec3 = conv_block(base*8,  base*4, groups)
        self.up2 = nn.ConvTranspose2d(base*4,  base*2, 2, 2); self.dec2 = conv_block(base*4,  base*2, groups)
        self.up1 = nn.ConvTranspose2d(base*2,  base,   2, 2); self.dec1 = conv_block(base*2,  base,   groups)
        self.outc = nn.Conv2d(base, in_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.outc(d1)

# =============================
# 推理（可选 TTA x8 + 与输入融合）
# =============================
@torch.no_grad()
def tta_x8(model: nn.Module, x_log: torch.Tensor) -> torch.Tensor:
    outs = []
    for fh in [False, True]:
        for fv in [False, True]:
            for rot in [0, 1]:
                _x = x_log
                if fh: _x = torch.flip(_x, [-1])
                if fv: _x = torch.flip(_x, [-2])
                if rot: _x = _x.transpose(-1, -2)
                out = _x + model(_x)
                if rot: out = out.transpose(-1, -2)
                if fv:  out = torch.flip(out, [-2])
                if fh:  out = torch.flip(out, [-1])
                outs.append(out)
    return torch.mean(torch.stack(outs, 0), 0)

@torch.no_grad()
def forward_denoise(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    use_tta: bool = True,
    blend_alpha: float = 0.0,
    gamma: float = 1.0
) -> torch.Tensor:
    """
    监督训练的 UNet 前向推理：
    - 输入 x: [B,C,H,W]，线性域 [0,1] 的 noisy 图
    - 输出:     线性域 [0,1] 的去噪图
    """
    x = x.to(device)
    x_log = to_log(x)
    model.eval()

    # 1) 得到 log 域的预测：log(H_pred)
    if use_tta:
        out_log = tta_x8(model, x_log)
    else:
        out_log = x_log + model(x_log)

    # 2) 直接从 log 还原到线性，不再对齐噪声统计
    y = from_log_simple(out_log)

    # 3) 可选：和 noisy 做一点点融合（一般先关掉）
    if blend_alpha > 0:
        y = (1.0 - blend_alpha) * y + blend_alpha * x

    # 4) 可选：gamma 调整（一般先设 gamma=1 不启用）
    if abs(gamma - 1.0) > 1e-3:
        y = y.clamp(0.0, 1.0).pow(gamma)

    return y.clamp(0.0, 1.0)


# =============================
# 评测 & 保存
# =============================
@torch.no_grad()
@torch.no_grad()
def evaluate_and_save(model: nn.Module,
                      loader: DataLoader,
                      device: str,
                      out_dir: Path,
                      use_tta=True,
                      blend_alpha=0.0,
                      gamma=1.0,
                      save_ext='png') -> Tuple[float, float, List[Tuple[str, float, float]]]:
    """
    返回 (avg_psnr, avg_ssim, per_image_metrics)
    per_image_metrics: [(filename, psnr, ssim), ...]
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    per_metrics: List[Tuple[str, float, float]] = []

    for noisy, clean, name in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        pred = forward_denoise(model, noisy, device,
                               use_tta=use_tta, blend_alpha=blend_alpha, gamma=gamma)
        psnr = calc_psnr(pred, clean)
        ssim = calc_ssim(pred, clean)
        psnr_sum += psnr
        ssim_sum += ssim
        n += 1
        per_metrics.append((name[0], psnr, ssim))

        base, _ = os.path.splitext(name[0])
        vutils.save_image(pred.clamp(0, 1).cpu(), str(out_dir / f"{base}.{save_ext}"))

    avg_psnr = psnr_sum / n
    avg_ssim = ssim_sum / n
    return avg_psnr, avg_ssim, per_metrics


# =============================
# 权重加载（与你训练脚本的 state_dict 对齐）
# =============================

def load_weights_to_model(model: nn.Module, weights_path: str):
    sd = torch.load(weights_path, map_location='cpu')
    if isinstance(sd, dict):
        model.load_state_dict(sd, strict=False)
    else:
        raise ValueError("Unsupported weight file format.")

# =============================
# CLI
# =============================

def parse_args():
    import argparse
    p = argparse.ArgumentParser("Paired test: save denoised images + compute PSNR/SSIM")
    p.add_argument('--noisy_dir', required=True)
    p.add_argument('--clean_dir', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out_dir', default='denoise_results')
    p.add_argument('--gray', action='store_true')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--blend_alpha', type=float, default=0.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--csv', type=str, default='')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main():
    args = parse_args()

    # data
    dataset = PairedSet(args.noisy_dir, args.clean_dir, gray=args.gray)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # model
    C = 1 if args.gray else 3
    model = UNet(in_ch=C, base=96, groups=32).to(args.device)
    load_weights_to_model(model, args.weights)

    # eval & save
    out_dir = Path(args.out_dir)
    avg_psnr, avg_ssim, per_metrics = evaluate_and_save(
        model, loader, args.device, out_dir,
        use_tta=args.tta, blend_alpha=args.blend_alpha, gamma=args.gamma
    )

    print("\nPer-image metrics:")
    for name, p, s in per_metrics:
        print(f"{name}: PSNR={p:.2f} dB, SSIM={s:.4f}")

    print(f"\nAverage PSNR: {avg_psnr:.2f} dB\nAverage SSIM: {avg_ssim:.4f}")

    # optional CSV
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["filename", "psnr", "ssim"])
            for name, p, s in per_metrics:
                w.writerow([name, f"{p:.6f}", f"{s:.6f}"])
            w.writerow(["AVERAGE", f"{avg_psnr:.6f}", f"{avg_ssim:.6f}"])
        print(f"Saved per-image metrics to {args.csv}")


if __name__ == '__main__':
    main()
