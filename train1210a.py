import os, glob
from typing import Tuple
from PIL import Image
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------
# 0) Log <-> Linear + brightness alignment
# ---------------------------
EPS = 1e-6

def box_blur(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    x: [B, C, H, W]
    k: odd kernel size (e.g., 3/5/7)
    returns: blurred tensor with same shape as x
    """
    if k % 2 == 0:
        raise ValueError("box_blur kernel size k must be odd.")
    pad = k // 2
    # reflect padding to avoid dark borders
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    # depthwise conv with box kernel
    C = x.shape[1]
    weight = torch.ones(C, 1, k, k, dtype=x.dtype, device=x.device) / (k * k)
    return F.conv2d(x_pad, weight, bias=None, stride=1, padding=0, groups=C)

def to_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x.clamp_min(EPS))

def from_log_preserve(y: torch.Tensor, ref_linear: torch.Tensor) -> torch.Tensor:
    """
    y: [B,C,H,W] in log domain
    ref_linear: [B,C,H,W] in [0,1], original input (linear)
    Return linear-domain output with mean/std matched to ref (brightness preserved)
    """
    out = torch.exp(y) - EPS
    ref_mean = ref_linear.mean(dim=[2,3], keepdim=True)
    ref_std  = ref_linear.std(dim=[2,3], keepdim=True).clamp_min(1e-6)
    out_mean = out.mean(dim=[2,3], keepdim=True)
    out_std  = out.std(dim=[2,3], keepdim=True).clamp_min(1e-6)
    out = (out - out_mean) / out_std * ref_std + ref_mean
    return out.clamp(0.0, 1.0)

def alpha_blend(denoised: torch.Tensor, inp: torch.Tensor, alpha: float=0.05) -> torch.Tensor:
    return (1 - alpha) * denoised + alpha * inp

def gamma_tone(x: torch.Tensor, gamma: float=0.95) -> torch.Tensor:
    return x.clamp(0,1).pow(gamma)

# ---------------------------
# 1) Dataset (noisy-only, self-supervised)
# ---------------------------
class NoisyOnlyDataset(Dataset):
    def __init__(self, root, img_size=256, gray=True):
        self.files = []
        for ext in ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"):
            self.files += glob.glob(os.path.join(root, ext))
        if len(self.files) == 0:
            raise ValueError(f"No images found in: {root}")
        self.gray = gray
        if gray:
            self.tf = transforms.Compose([
                transforms.RandomCrop(img_size),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor()
            ])
        else:
            self.tf = transforms.Compose([
                transforms.RandomCrop(img_size),
                transforms.ToTensor()
            ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        x = self.tf(img)   # [C,H,W] in [0,1]
        return x


# ---------------------------
# 1b) 成对的 L/H 监督数据集
# ---------------------------
class PairedGammaDataset(Dataset):
    def __init__(self, root_L, root_H, img_size=256, gray=True,
                 use_poisson=False, gamma_min=500.0, gamma_max=2000.0):
        """
        root_L: 低计数/带噪图像目录 (可选)
        root_H: 干净/高计数图像目录
        use_poisson: 如果 True，则忽略磁盘上的 L，直接从 H 上 Poisson 生成 L
        gamma_min, gamma_max: 训练时随机采样的计数 scale 区间
        """
        self.root_L = root_L
        self.root_H = root_H
        self.gray = gray
        self.img_size = img_size
        self.use_poisson = use_poisson
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        self.L_files = []
        self.H_files = []
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")
        for ext in exts:
            if root_L is not None:
                self.L_files += glob.glob(os.path.join(root_L, ext))
            self.H_files += glob.glob(os.path.join(root_H, ext))

        self.L_files = sorted(self.L_files)
        self.H_files = sorted(self.H_files)

        if len(self.H_files) == 0:
            raise ValueError(f"No GT images found in: {root_H}")
        if (not self.use_poisson) and len(self.L_files) == 0:
            raise ValueError(f"No input images found in: {root_L}")
        if (not self.use_poisson) and len(self.L_files) != len(self.H_files):
            print("Warning: L/H count mismatch, use min length.")
            n = min(len(self.L_files), len(self.H_files))
            self.L_files = self.L_files[:n]
            self.H_files = self.H_files[:n]

                # -------- 在这里加：过滤掉尺寸不对的图片对 --------
        if not self.use_poisson:
            valid_L, valid_H = [], []
            for L_path, H_path in zip(self.L_files, self.H_files):
                try:
                    with Image.open(L_path) as imgL, Image.open(H_path) as imgH:
                        # 转成灰度 / RGB，方便拿 size
                        if self.gray:
                            imgL = imgL.convert('L')
                            imgH = imgH.convert('L')
                        else:
                            imgL = imgL.convert('RGB')
                            imgH = imgH.convert('RGB')
                        wL, hL = imgL.size  # PIL: (width, height)
                        wH, hH = imgH.size
                except Exception as e:
                    print(f"[WARN] skip pair {L_path} / {H_path} (load error: {e})")
                    continue

                # 条件1：L/H 尺寸必须一致，否则随机裁剪时会出问题
                if (wL != wH) or (hL != hH):
                    print(f"[WARN] skip pair {L_path} / {H_path} (size mismatch: "
                          f"L={wL}x{hL}, H={wH}x{hH})")
                    continue

                # 条件2：尺寸必须 >= img_size，否则裁不出 256x256 之类的 patch
                if (wL < self.img_size) or (hL < self.img_size):
                    print(f"[WARN] skip pair {L_path} / {H_path} (too small: "
                          f"{wL}x{hL} < {self.img_size})")
                    continue

                valid_L.append(L_path)
                valid_H.append(H_path)

            self.L_files = valid_L
            self.H_files = valid_H
            print(f"[INFO] after size filtering, {len(self.L_files)} valid pairs left")
        else:
            # use_poisson=True 时，只用 H 图，根据 H 尺寸来过滤
            valid_H = []
            for H_path in self.H_files:
                try:
                    with Image.open(H_path) as imgH:
                        if self.gray:
                            imgH = imgH.convert('L')
                        else:
                            imgH = imgH.convert('RGB')
                        wH, hH = imgH.size
                except Exception as e:
                    print(f"[WARN] skip H {H_path} (load error: {e})")
                    continue

                if (wH < self.img_size) or (hH < self.img_size):
                    print(f"[WARN] skip H {H_path} (too small: "
                          f"{wH}x{hH} < {self.img_size})")
                    continue

                valid_H.append(H_path)

            self.H_files = valid_H
            print(f"[INFO] after size filtering, {len(self.H_files)} valid GT images left")
        # -------- 尺寸过滤结束 --------

        # 只做 ToTensor，随机裁剪要保证 L/H 对齐，所以自己写
        if gray:
            self.tf = transforms.ToTensor()
        else:
            self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.H_files) if self.use_poisson else len(self.L_files)

    def _load_img(self, path):
        img = Image.open(path)
        if self.gray:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        return self.tf(img)  # [C,H,W] in [0,1]

    def _random_crop_pair(self, L, H):
        _, h, w = H.shape
        ps = self.img_size
        # 防止极端情况：空图像
        if h == 0 or w == 0:
            raise RuntimeError(f"Got empty image with shape {H.shape}")

        # 若任一边小于 patch 大小，先反射 pad 再随机裁剪
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            # pad = (left, right, top, bottom)
            pad = (pad_w // 2, pad_w - pad_w // 2,
                   pad_h // 2, pad_h - pad_h // 2)
            L = F.pad(L, pad, mode="reflect")
            H = F.pad(H, pad, mode="reflect")
            _, h, w = H.shape  # 更新尺寸

        # 到这里一定有 h >= ps, w >= ps，可以安全随机裁剪 ps×ps
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        L = L[:, top:top+ps, left:left+ps]
        H = H[:, top:top+ps, left:left+ps]
        return L.contiguous(), H.contiguous()

    def _augment_pair(self, L, H):
        # 水平翻转
        if random.random() < 0.5:
            L = torch.flip(L, dims=[2])
            H = torch.flip(H, dims=[2])
        # 垂直翻转
        if random.random() < 0.5:
            L = torch.flip(L, dims=[1])
            H = torch.flip(H, dims=[1])
        # 90° 旋转
        if random.random() < 0.5:
            L = L.transpose(1, 2)
            H = H.transpose(1, 2)
        return L.contiguous(), H.contiguous()

    def __getitem__(self, idx):
        # 1) 读 H
        H = self._load_img(self.H_files[idx])

        # 2) 生成或读取 L
        if self.use_poisson:
            clean = H.clamp(0.0, 1.0)
            scale = random.uniform(self.gamma_min, self.gamma_max)
            lam = clean * scale
            noisy_counts = torch.poisson(lam)
            L = noisy_counts / scale
        else:
            L = self._load_img(self.L_files[idx])

        # 3) 同步裁剪 & 数据增强
        L, H = self._random_crop_pair(L, H)
        L, H = self._augment_pair(L, H)

        return L, H

# ---------------------------
# 2) UNet backbone (GroupNorm to reduce BN-induced shifts)
# ---------------------------
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

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = conv_block(base*16, base*8, groups)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = conv_block(base*8, base*4, groups)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = conv_block(base*4, base*2, groups)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = conv_block(base*2, base, groups)

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

# ---------------------------
# 3) Noise2Void blind-spot masking
# ---------------------------
def make_n2v_mask(x: torch.Tensor, drop_p=0.1, radius=3) -> Tuple[torch.Tensor, torch.Tensor]:
    B,C,H,W = x.shape
    device = x.device
    mask = (torch.rand(B,1,H,W, device=device) < drop_p).float()

    dx = torch.randint(-radius, radius+1, (B,1,1,1), device=device)
    dy = torch.randint(-radius, radius+1, (B,1,1,1), device=device)

    yy, xx = torch.meshgrid(
        torch.linspace(-1,1,H,device=device),
        torch.linspace(-1,1,W,device=device),
        indexing="ij"
    )
    grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(B,1,1,1)
    grid[...,0] = grid[...,0] + dx.float().squeeze(1) / (W//2 + 1e-6)
    grid[...,1] = grid[...,1] + dy.float().squeeze(1) / (H//2 + 1e-6)

    neigh = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
    masked_x = x * (1 - mask) + neigh * mask
    return masked_x, mask

# ---------------------------
# 4) Losses (detail-friendly)
# ---------------------------
def charbonnier(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps).mean()

def sobel_grad(x):
    """ channel-wise Sobel magnitude """
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    C = x.shape[1]
    gx = F.conv2d(x, kx.repeat(C,1,1,1), padding=1, groups=C)
    gy = F.conv2d(x, ky.repeat(C,1,1,1), padding=1, groups=C)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

def detail_loss(pred_lin, ref_lin, weight=0.01):
    """ preserve edges softly: compare Sobel magnitude of denoised vs lightly-blurred ref """
    ref_smooth = box_blur(ref_lin, k=3)  # reduce noise influence
    g_pred = sobel_grad(pred_lin)
    g_ref  = sobel_grad(ref_smooth)
    return weight * F.l1_loss(g_pred, g_ref)

# 新增：TV 平滑约束 --------------------------------
def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Total Variation loss，鼓励图像整体更平滑
    x: [B,C,H,W] in [0,1]
    """
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    return (dx.abs().mean() + dy.abs().mean())

# ---------------------------
# 5) Train / Validate (log-domain residual + brightness/contrast + detail)
# ---------------------------
def train_one_epoch(model, loader, opt, device,
                    drop_p=0.15, w_lum=0.05, w_ct=0.02, w_edge=0.02, w_identity=0.02):
    model.train()
    total = 0.0
    for noisy in loader:
        noisy = noisy.to(device)          # linear [B,C,H,W]
        y_log = to_log(noisy)             # log-domain
        masked, m = make_n2v_mask(y_log, drop_p=drop_p)

        out_delta = model(masked)         # predict residual in log
        pred_log  = masked + out_delta

        # Reconstruction on masked pixels
        rec = charbonnier((pred_log - y_log) * m)

        # Brightness & contrast regularization in linear domain
        pred_lin = torch.exp(pred_log) - EPS
        y_lin    = torch.exp(y_log) - EPS

        lum = (pred_lin.mean(dim=[2,3]) - y_lin.mean(dim=[2,3])).abs().mean()
        std_pred = pred_lin.std(dim=[2,3]).clamp_min(1e-6)
        std_ref  = y_lin.std(dim=[2,3]).clamp_min(1e-6)
        ct = torch.relu(std_pred - std_ref).mean()

        # Identity on unmasked pixels (prevents over-smoothing)
        iden = F.l1_loss(pred_log*(1 - m), y_log*(1 - m))

        # Edge/detail preservation (small weight)
        edge = detail_loss(pred_lin, y_lin, weight=w_edge)

        loss = rec + w_lum*lum + w_ct*ct + w_identity*iden + edge

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

def train_sup_epoch(model, loader, opt, device,
                    w_l1=1.0, w_lum=0.02, w_ct=0.01, w_edge=0.01, w_tv=0.0):
    """
    监督学习版本:
      - 输入: L (noisy, linear [0,1])
      - GT:   H (clean/high-count, linear [0,1])
      - 模型: 接收 log(L), 输出 log 残差, pred_log = log(L) + delta
    """
    model.train()
    total = 0.0
    for L, H in loader:
        L = L.to(device)
        H = H.to(device)

        L_log = to_log(L)
        H_log = to_log(H)

        out_delta = model(L_log)
        pred_log  = L_log + out_delta

        # 1) 主重建损失：log 域 Charbonnier（类似 L1）
        rec = charbonnier(pred_log - H_log)

        # 2) 在线性域做一点 brightness / contrast / edge 正则（延续原 train22a3 风格）
        pred_lin = torch.exp(pred_log) - EPS
        H_lin    = H

        # 亮度
        lum = (pred_lin.mean(dim=[2,3]) - H_lin.mean(dim=[2,3])).abs().mean()

        # 对比度：只惩罚“比 GT 更大的 std”，允许去噪后变平滑
        std_pred = pred_lin.std(dim=[2,3]).clamp_min(1e-6)
        std_ref  = H_lin.std(dim=[2,3]).clamp_min(1e-6)
        ct = torch.relu(std_pred - std_ref).mean()

        # 边缘保持（利用之前的 detail_loss）
        edge = detail_loss(pred_lin, H_lin, weight=w_edge)
        # 新增：整体平滑项（不依赖 GT，只看输出本身）
        tv = tv_loss(pred_lin)

        loss = w_l1*rec + w_lum*lum + w_ct*ct + edge + w_tv * tv

        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate_sup(model, loader, device,
                 w_l1=1.0, w_lum=0.02, w_ct=0.01, w_edge=0.01, w_tv=0.0):
    model.eval()
    total = 0.0
    for L, H in loader:
        L = L.to(device)
        H = H.to(device)

        L_log = to_log(L)
        H_log = to_log(H)

        out_delta = model(L_log)
        pred_log  = L_log + out_delta

        rec = charbonnier(pred_log - H_log)

        pred_lin = torch.exp(pred_log) - EPS
        H_lin    = H

        lum = (pred_lin.mean(dim=[2,3]) - H_lin.mean(dim=[2,3])).abs().mean()
        std_pred = pred_lin.std(dim=[2,3]).clamp_min(1e-6)
        std_ref  = H_lin.std(dim=[2,3]).clamp_min(1e-6)
        ct = torch.relu(std_pred - std_ref).mean()
        edge = detail_loss(pred_lin, H_lin, weight=w_edge)
        tv   = tv_loss(pred_lin)
        loss = w_l1*rec + w_lum*lum + w_ct*ct + edge + w_tv * tv
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device,
             drop_p=0.15, w_lum=0.05, w_ct=0.02, w_edge=0.02, w_identity=0.02):
    model.eval()
    total = 0.0
    for noisy in loader:
        noisy = noisy.to(device)
        y_log = to_log(noisy)
        masked, m = make_n2v_mask(y_log, drop_p=drop_p)

        out_delta = model(masked)
        pred_log  = masked + out_delta

        rec = charbonnier((pred_log - y_log) * m)

        pred_lin = torch.exp(pred_log) - EPS
        y_lin    = torch.exp(y_log) - EPS

        lum = (pred_lin.mean(dim=[2,3]) - y_lin.mean(dim=[2,3])).abs().mean()
        std_pred = pred_lin.std(dim=[2,3]).clamp_min(1e-6)
        std_ref  = y_lin.std(dim=[2,3]).clamp_min(1e-6)
        ct = torch.relu(std_pred - std_ref).mean()

        iden = F.l1_loss(pred_log*(1 - m), y_log*(1 - m))
        edge = detail_loss(pred_lin, y_lin, weight=w_edge)

        loss = rec + w_lum*lum + w_ct*ct + w_identity*iden + edge
        total += loss.item()
    return total / len(loader)

# ---------------------------
# 6) Inference
# ---------------------------
@torch.no_grad()
def denoise_infer(model, img_tensor, device,
                  do_blend=True, blend_alpha=0,
                  do_gamma=True, gamma=0.95):
    """
    img_tensor: [B,C,H,W] in [0,1] linear
    returns: denoised image (linear, brightness preserved)
    """
    model.eval()
    y_log = to_log(img_tensor.to(device))
    out_delta = model(y_log)       # residual
    out_log   = y_log + out_delta  # log-domain output

    out = from_log_preserve(out_log, img_tensor.to(device))
    if do_blend and blend_alpha > 0:
        out = alpha_blend(out, img_tensor.to(device), alpha=blend_alpha)
    if do_gamma and abs(gamma - 1.0) > 1e-3:
        out = gamma_tone(out, gamma=gamma)
    return out.clamp(0,1)

# ---------------------------
# 7) Main
# ---------------------------
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='data/train')
    p.add_argument('--val_dir',   default='data/val')
    p.add_argument('--epochs',    type=int, default=100)
    p.add_argument('--batch',     type=int, default=8)
    p.add_argument('--img_size',  type=int, default=256)
    p.add_argument('--gray', action='store_true')
    p.add_argument('--lr',   type=float, default=1e-3)
    p.add_argument('--drop_p', type=float, default=0.1)
    p.add_argument('--w_lum', type=float, default=0.05)
    p.add_argument('--w_ct',  type=float, default=0.02)
    p.add_argument('--w_edge',type=float, default=0.01)
    p.add_argument('--w_id',  type=float, default=0.05)
    p.add_argument('--save',  default='n2v_bright_detail.pth')
    p.add_argument('--patience', type=int, default=200)
     # --- 新加：监督 / 伽马 Poisson 相关 ---
    p.add_argument('--supervised', action='store_true',
                   help='使用 L/H 监督训练，而不是 N2V 自监督')
    p.add_argument('--train_L', default=None,
                   help='监督训练时，低计数/带噪图像目录')
    p.add_argument('--train_H', default=None,
                   help='监督训练时，干净/高计数图像目录')
    p.add_argument('--val_L', default=None)
    p.add_argument('--val_H', default=None)
    p.add_argument('--gamma_poisson', action='store_true',
                   help='从 H 上 Poisson 再加噪生成 L（借鉴 train_gamma）')
    p.add_argument('--gamma_min', type=float, default=500.0)
    p.add_argument('--gamma_max', type=float, default=2000.0)
    p.add_argument('--w_tv', type=float, default=0.0,
               help='TV 平滑正则权重（监督训练用）')

    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    C = 1 if args.gray else 3

    def sup_collate_fn(batch):
        # batch: [(L1,H1), (L2,H2), ...]
        Ls, Hs = zip(*batch)
        return torch.stack(Ls, dim=0), torch.stack(Hs, dim=0)

    if args.supervised:
        # ----- 监督版：使用成对 L/H 数据集 -----
        train_set = PairedGammaDataset(
            root_L=None if args.gamma_poisson else args.train_L,
            root_H=args.train_H,
            img_size=args.img_size,
            gray=args.gray,
            use_poisson=args.gamma_poisson,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max
        )

        if args.val_H is not None:
            val_set = PairedGammaDataset(
                root_L=None if args.gamma_poisson else args.val_L,
                root_H=args.val_H,
                img_size=args.img_size,
                gray=args.gray,
                use_poisson=False   # 一般验证集用真实 L/H，不再随机加噪
            )
        else:
            val_set = None
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=sup_collate_fn,  # ★ 自定义 collate_fn
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=sup_collate_fn,
        ) if val_set else None

    else:

        train_set = NoisyOnlyDataset(args.train_dir, args.img_size, gray=args.gray)
        val_set = NoisyOnlyDataset(args.val_dir, args.img_size, gray=args.gray) if os.path.isdir(args.val_dir) and len(os.listdir(args.val_dir))>0 else None

    # train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,  num_workers=4, drop_last=True)
    # val_loader   = DataLoader(val_set,   batch_size=args.batch, shuffle=False, num_workers=2, drop_last=False) if val_set else None

    model = UNet(in_ch=C, base=96, groups=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e9
    bad  = 0
    for e in range(1, args.epochs+1):
        if args.supervised:
            # --------- 监督训练分支 ---------
            tr = train_sup_epoch(
                model, train_loader, opt, device,
                w_l1=1.0,
                w_lum=args.w_lum,
                w_ct=args.w_ct,
                w_edge=args.w_edge,
                w_tv=args.w_tv
       )

            if val_loader:
                va = validate_sup(
                    model, val_loader, device,
                    w_l1=1.0,
                    w_lum=args.w_lum,
                    w_ct=args.w_ct,
                    w_edge=args.w_edge,
                    w_tv=args.w_tv
                )
                print(f"Epoch {e:03d}  train:{tr:.6f}  val:{va:.6f}")
                if va < best:
                    best = va; bad = 0
                    torch.save(model.state_dict(), args.save)
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"Early stop at epoch {e}, best val={best:.6f}")
                        break
            else:
                print(f"Epoch {e:03d}  train:{tr:.6f}")
                torch.save(model.state_dict(), args.save)

        else:
            # --------- 保留原来的 N2V 自监督分支 ---------
            tr = train_one_epoch(
                model, train_loader, opt, device,
                drop_p=args.drop_p,
                w_lum=args.w_lum,
                w_ct=args.w_ct,
                w_edge=args.w_edge,
                w_identity=args.w_id
            )
            if val_loader:
                va = validate(
                    model, val_loader, device,
                    drop_p=args.drop_p,
                    w_lum=args.w_lum,
                    w_ct=args.w_ct,
                    w_edge=args.w_edge,
                    w_identity=args.w_id
                )
                print(f"Epoch {e:03d}  train:{tr:.6f}  val:{va:.6f}")
                if va < best:
                    best = va; bad = 0
                    torch.save(model.state_dict(), args.save)
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"Early stop at epoch {e}, best val={best:.6f}")
                        break
            else:
                print(f"Epoch {e:03d}  train:{tr:.6f}")
                torch.save(model.state_dict(), args.save)

    print(f"Saved best model to {args.save}")

if __name__ == '__main__':
    main()
