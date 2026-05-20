# deconv3d_core.py
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Utils
# ----------------------------
def _to_3tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x, x)

def robust_norm(vol, p_lo=0.1, p_hi=99.9, eps=1e-6):
    lo = np.percentile(vol, p_lo)
    hi = np.percentile(vol, p_hi)
    if not np.isfinite(lo): lo = float(np.nanmin(vol))
    if not np.isfinite(hi): hi = float(np.nanmax(vol))
    if not np.isfinite(lo): lo = 0.0
    if not np.isfinite(hi): hi = 1.0
    if hi - lo < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + eps)
    return vol.astype(np.float32)

# ----------------------------
# Model
# ----------------------------
class ConvStem3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch), nn.GELU(),
        )
    def forward(self, x): return self.net(x)

class WindowAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4), qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ws = _to_3tuple(window_size)
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def partition_windows(self, x):
        B, C, D, H, W = x.shape
        wd, wh, ww = self.ws
        assert D % wd == 0 and H % wh == 0 and W % ww == 0, "Dims must be multiples of window size."
        x = x.view(B, C, D//wd, wd, H//wh, wh, W//ww, ww)
        x = x.permute(0,2,4,6,1,3,5,7).contiguous()
        x = x.view(-1, C, wd, wh, ww)
        return x

    def reverse_windows(self, x, B, C, D, H, W):
        wd, wh, ww = self.ws
        nD, nH, nW = D//wd, H//wh, W//ww
        x = x.view(B, nD, nH, nW, C, wd, wh, ww)
        x = x.permute(0,4,1,5,2,6,3,7).contiguous()
        x = x.view(B, C, D, H, W)
        return x

    def forward(self, x):
        B, C, D, H, W = x.shape
        xw = self.partition_windows(x)
        Bn, Cw, d, h, w = xw.shape
        N = d*h*w
        xw = xw.view(Bn, Cw, N).transpose(1, 2)
        qkv = self.qkv(xw); q,k,v = qkv.chunk(3, dim=-1)
        q = q.view(Bn, N, self.num_heads, Cw//self.num_heads).transpose(1, 2)
        k = k.view(Bn, N, self.num_heads, Cw//self.num_heads).transpose(1, 2)
        v = v.view(Bn, N, self.num_heads, Cw//self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1); attn = self.attn_drop(attn)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(Bn, N, Cw)
        out = self.proj(out); out = self.proj_drop(out)
        out = out.transpose(1,2).contiguous().view(Bn, Cw, d, h, w)
        out = self.reverse_windows(out, B, C, D, H, W)
        return out

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(4,4,4), mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.ws = _to_3tuple(window_size)
        self.norm1 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim, eps=1e-5, affine=True)
        self.attn  = WindowAttention3D(dim, num_heads=num_heads, window_size=self.ws,
                                       qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.GroupNorm(num_groups=min(8, dim), num_channels=dim, eps=1e-5, affine=True)
        self.mlp   = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + h
        h = x
        x = self.norm2(x)
        B,C,D,H,W = x.shape
        x = x.permute(0,2,3,4,1).contiguous().view(B*D*H*W, C)
        x = self.mlp(x)
        x = x.view(B, D, H, W, C).permute(0,4,1,2,3).contiguous()
        x = x + h
        return x

def Down3D(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm3d(out_ch),
        nn.GELU(),
    )

class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1x1 = nn.Conv3d(in_ch, out_ch, 1)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=(2,2,2), mode="trilinear", align_corners=False)
        x = self.conv1x1(x)
        x = torch.cat([x, skip], dim=1)
        return x

class TinyUNETR3D(nn.Module):
    def __init__(self, in_ch=1, base_ch=24, heads=(2,4,4,8,8), window_size=(4,4,4), dec_conv=True):
        super().__init__()
        ws = _to_3tuple(window_size)
        C1, C2, C3, C4 = base_ch, base_ch*2, base_ch*4, base_ch*8

        self.stem = ConvStem3D(in_ch, C1)

        self.enc1_blk = TransformerBlock3D(C1, num_heads=heads[0], window_size=ws)
        self.down1 = Down3D(C1, C2)
        self.enc2_blk = TransformerBlock3D(C2, num_heads=heads[1], window_size=ws)
        self.down2 = Down3D(C2, C3)
        self.enc3_blk = TransformerBlock3D(C3, num_heads=heads[2], window_size=ws)
        self.down3 = Down3D(C3, C4)
        self.enc4_blk = TransformerBlock3D(C4, num_heads=heads[3], window_size=ws)

        self.bot_blk = TransformerBlock3D(C4, num_heads=heads[4], window_size=ws)

        self.up3 = Up3D(C4, C3)
        self.dec3_pre = nn.Sequential(nn.Conv3d(C3 + C3, C3, 3, padding=1),
                                      nn.InstanceNorm3d(C3), nn.GELU()) if dec_conv else nn.Identity()
        self.dec3_blk = TransformerBlock3D(C3, num_heads=heads[2], window_size=ws)

        self.up2 = Up3D(C3, C2)
        self.dec2_pre = nn.Sequential(nn.Conv3d(C2 + C2, C2, 3, padding=1),
                                      nn.InstanceNorm3d(C2), nn.GELU()) if dec_conv else nn.Identity()
        self.dec2_blk = TransformerBlock3D(C2, num_heads=heads[1], window_size=ws)

        self.up1 = Up3D(C2, C1)
        self.dec1_pre = nn.Sequential(nn.Conv3d(C1 + C1, C1, 3, padding=1),
                                      nn.InstanceNorm3d(C1), nn.GELU()) if dec_conv else nn.Identity()
        self.dec1_blk = TransformerBlock3D(C1, num_heads=heads[0], window_size=ws)

        self.head = nn.Conv3d(C1, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.stem(x);  x1 = self.enc1_blk(x1)
        x2 = self.down1(x1); x2 = self.enc2_blk(x2)
        x3 = self.down2(x2); x3 = self.enc3_blk(x3)
        x4 = self.down3(x3); x4 = self.enc4_blk(x4)
        xb = self.bot_blk(x4)

        y3 = self.up3(xb, x3); y3 = self.dec3_pre(y3); y3 = self.dec3_blk(y3)
        y2 = self.up2(y3, x2); y2 = self.dec2_pre(y2); y2 = self.dec2_blk(y2)
        y1 = self.up1(y2, x1); y1 = self.dec1_pre(y1); y1 = self.dec1_blk(y1)
        return self.head(y1)

# ----------------------------
# Sliding window inference
# ----------------------------
def gaussian_weight(win):
    def g1(n):
        x = np.linspace(-1, 1, n)
        return np.exp(-4*(x**2))
    wz, wy, wx = (g1(win[0]), g1(win[1]), g1(win[2]))
    W = np.outer(wz, wy).reshape(win[0], win[1], 1) * wx.reshape(1,1,win[2])
    return (W / W.max()).astype(np.float32)

def pad_to_multiple(vol, mult=(4,4,4), min_size=(96,96,96), mode="reflect"):
    D, H, W = vol.shape
    md, mh, mw = mult
    td = max(min_size[0], int(np.ceil(D/md) * md))
    th = max(min_size[1], int(np.ceil(H/mh) * mh))
    tw = max(min_size[2], int(np.ceil(W/mw) * mw))

    pd, ph, pw = max(0, td-D), max(0, th-H), max(0, tw-W)
    if pd==ph==pw==0:
        return vol, (0,0,0), (D,H,W)

    pd0 = pd // 2; pd1 = pd - pd0
    ph0 = ph // 2; ph1 = ph - ph0
    pw0 = pw // 2; pw1 = pw - pw0
    out = np.pad(vol, ((pd0,pd1),(ph0,ph1),(pw0,pw1)), mode=mode)
    return out, (pd0,ph0,pw0), (D,H,W)

def unpad(vol, pads, orig_shape):
    pd0,ph0,pw0 = pads
    D,H,W = orig_shape
    return vol[pd0:pd0+D, ph0:ph0+H, pw0:pw0+W]

@torch.no_grad()
def infer_volume(model, vol, roi=(96,96,96), overlap=0.5, device="cpu", use_amp=False):
    model.eval()
    D,H,W = vol.shape
    rd,rh,rw = roi
    sd = max(1, int(rd * (1 - overlap)))
    sh = max(1, int(rh * (1 - overlap)))
    sw = max(1, int(rw * (1 - overlap)))

    out = np.zeros((D,H,W), dtype=np.float32)
    wgt = np.zeros((D,H,W), dtype=np.float32)
    gw = gaussian_weight(roi)

    for z in range(0, max(1, D-rd+1), sd):
        for y in range(0, max(1, H-rh+1), sh):
            for x in range(0, max(1, W-rw+1), sw):
                zz, yy, xx = z, y, x
                z2, y2, x2 = zz+rd, yy+rh, xx+rw
                if z2>D: zz, z2 = D-rd, D
                if y2>H: yy, y2 = H-rh, H
                if x2>W: xx, x2 = W-rw, W

                patch = vol[zz:z2, yy:y2, xx:x2][None,None,...]
                t = torch.from_numpy(patch.astype(np.float32)).to(device)

                if device.startswith("cuda"):
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        logits = model(t)
                        pred = torch.sigmoid(logits)
                else:
                    logits = model(t)
                    pred = torch.sigmoid(logits)

                pred = torch.clamp(pred, 0.0, 1.0).float().cpu().numpy()[0,0]
                ww = gw[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
                out[zz:z2, yy:y2, xx:x2] += pred * ww
                wgt[zz:z2, yy:y2, xx:x2] += ww

    out = out / np.maximum(wgt, 1e-6)
    return np.clip(out, 0.0, 1.0)

def load_model(weights_path: str, base_ch=24, win=4, device="cpu"):
    model = TinyUNETR3D(in_ch=1, base_ch=base_ch, window_size=(win,win,win)).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
