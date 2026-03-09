"""
generate_v3.py  —  HR-VITON-inspired improvements over v2.

Critical fixes vs v2 & first v3:
  1. Auto GPU detection (was hardcoded CPU)
  2. Dense flow field GMM — replaces broken batched TPS
  3. VGG input normalisation in feature extractor ([-1,1] -> VGG ImageNet range)
  4. GMM appearance loss — warped cloth must match the person cloth region (paired GT)
  5. ALIAS normalisation in TOM decoder (HR-VITON 3.2)
  6. SSIM loss in TOM (directly optimises the evaluation metric)
  7. Mixed-precision (AMP) training for speed
  8. 5000 training pairs, batch=8, OneCycleLR
"""
import nbformat as nbf, textwrap
nb = nbf.v4.new_notebook()
cells = []

def md(s):   cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))
def code(s): cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

md("""
# Enhanced Virtual Try-On — HR-VITON Inspired (v3)
## MSDS Computer Vision 462 · Final Project

| | |
|---|---|
| **Team** | Joyati · Biraj Mishra · Murughanandam S. |
| **Course** | MSDS COMPUTERVISION 462 |
| **Date** | Spring 2026 |

## Key fixes over v2 / broken-v3

| Fix | Before | After |
|---|---|---|
| Device | CPU (hardcoded) | Auto-detect GPU |
| GMM warping | TPS batched-solve (kernel bug) | Dense flow field (UNet) |
| VGG normalisation | Missing | [-1,1] to ImageNet range |
| GMM supervision | Mask L1 only | Mask L1 + cloth appearance L1 |
| TOM width | 32/64/128 | 64/128/256/512 |
| TOM decoder | BatchNorm | ALIAS (SPADE, HR-VITON) |
| Losses | L1 + VGG + TV | L1 + VGG + SSIM + TV |
| Training pairs | 600 | 5000 |
| Speed | CPU only | AMP (mixed precision) |
""")

md("## 0. Dataset Setup")
code("""
import zipfile, subprocess
from pathlib import Path

DATASET_DIR = Path('./dataset')
ZIP_PATH    = DATASET_DIR / 'complete dataset.zip'
KAGGLE_URL  = 'https://www.kaggle.com/api/v1/datasets/download/marquis03/high-resolution-viton-zalando-dataset'

def dataset_ready():
    p = DATASET_DIR / 'train' / 'image'
    return p.exists() and len(list(p.iterdir())) > 100

if dataset_ready():
    print('Dataset already present.')
else:
    DATASET_DIR.mkdir(exist_ok=True)
    if not ZIP_PATH.exists():
        print('Downloading (~4.4 GB) ...')
        subprocess.run(['curl', '-L', '-o', str(ZIP_PATH), KAGGLE_URL], check=True)
    print('Extracting ...')
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATASET_DIR)
    print(f'Train: {len(list((DATASET_DIR/"train"/"image").iterdir())):,}')
    print(f'Test : {len(list((DATASET_DIR/"test" /"image").iterdir())):,}')
""")

md("## 1. Setup & Configuration")
code("""
import os, json, random, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import cv2
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import vgg16, VGG16_Weights
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as psnr_metric

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

BASE_DIR  = Path('./dataset')
TRAIN_DIR = BASE_DIR / 'train'
TEST_DIR  = BASE_DIR / 'test'

# Critical fix: auto GPU detection (v2 hardcoded CPU)
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = (DEVICE.type == 'cuda')

IMG_H, IMG_W = 256, 192
BATCH_SIZE   = 8
N_WORKERS    = 4
GMM_EPOCHS   = 30
TOM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7
N_TRAIN      = 5000
N_VAL        = 500
CLOTH_LABEL  = 5

LABEL_NAMES = {
    0:'Background',1:'Hat',2:'Hair',3:'Sunglasses',4:'Upper-cloth',
    5:'Dress/Upper',6:'Coat',7:'Socks',8:'Pants',9:'Right Shoe',
    10:'Left Shoe',11:'Face',12:'Left Leg',13:'Right Leg',
    14:'Left Arm',15:'Right Arm',16:'Bag',17:'Scarf',
}

print(f'Device  : {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU     : {torch.cuda.get_device_name(0)}')
    print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
print(f'AMP     : {USE_AMP}')
print(f'Img     : {IMG_H}x{IMG_W} | Batch:{BATCH_SIZE} | Workers:{N_WORKERS}')
print(f'Train   : {N_TRAIN} pairs | Val:{N_VAL} pairs')
""")

md("## 2. Exploratory Data Analysis")
code("""
def load_pairs(fp):
    pairs = []
    with open(fp) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 2: pairs.append((p[0], p[1]))
    return pairs

train_pairs = load_pairs(BASE_DIR / 'train_pairs.txt')
test_pairs  = load_pairs(BASE_DIR / 'test_pairs.txt')
random.shuffle(train_pairs)
print(f'Train pairs: {len(train_pairs):,}  |  Test pairs: {len(test_pairs):,}')
modalities = sorted([d for d in os.listdir(TRAIN_DIR) if (TRAIN_DIR/d).is_dir()])
for m in modalities:
    print(f'  {m:35s} {len(list((TRAIN_DIR/m).iterdir())):>6,}')
""")

code("""
PARSE_PALETTE = np.array([
    [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
    [255,85,0],[0,0,85],[0,119,221],[85,85,0],[0,85,85],
    [85,51,0],[52,86,128],[0,128,0],[0,0,255],[51,170,221],
    [0,255,255],[85,255,170],[170,255,85],[255,255,0],[255,170,0]
], dtype=np.uint8)

def colorize_parse(arr):
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    for lbl, col in enumerate(PARSE_PALETTE):
        rgb[arr==lbl] = col
    return rgb

N_SHOW = 3
fig, axes = plt.subplots(N_SHOW, 5, figsize=(18, 4.5*N_SHOW))
fig.suptitle('Dataset Samples', fontsize=13, fontweight='bold', y=1.01)
for row, idx in enumerate(random.sample(range(len(train_pairs)), N_SHOW)):
    pname, cname = train_pairs[idx]; base = pname.replace('.jpg','')
    imgs = [
        np.array(Image.open(TRAIN_DIR/'image'/pname)),
        np.array(Image.open(TRAIN_DIR/'cloth'/cname)),
        np.array(Image.open(TRAIN_DIR/'cloth-mask'/cname).convert('L')),
        colorize_parse(np.array(Image.open(TRAIN_DIR/'image-parse-v3'/(base+'.png')))),
        np.array(Image.open(TRAIN_DIR/'agnostic-v3.2'/pname)),
    ]
    for col, (img, cm, t) in enumerate(zip(imgs, [None,None,'gray',None,None],
            ['Person','Cloth','Mask','Parse','Agnostic'])):
        axes[row][col].imshow(img, cmap=cm); axes[row][col].axis('off')
        if row==0: axes[row][col].set_title(t, fontsize=10, fontweight='bold')
plt.tight_layout(); plt.show()
""")

md("## 3. Dataset & Data Pipeline")
code("""
class VTONDataset(Dataset):
    def __init__(self, pairs, base_dir, img_h=256, img_w=192, augment=False):
        self.pairs=pairs; self.base_dir=Path(base_dir)
        self.img_h=img_h; self.img_w=img_w; self.augment=augment
        self.img_tf = T.Compose([
            T.Resize((img_h,img_w)), T.ToTensor(),
            T.Normalize([0.5]*3,[0.5]*3)
        ])

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        pname, cname = self.pairs[idx]; base = pname.replace('.jpg','')
        person   = Image.open(self.base_dir/'image'/pname).convert('RGB')
        cloth    = Image.open(self.base_dir/'cloth'/cname).convert('RGB')
        c_mask   = Image.open(self.base_dir/'cloth-mask'/cname).convert('L')
        agnostic = Image.open(self.base_dir/'agnostic-v3.2'/pname).convert('RGB')
        parse_np = np.array(Image.open(self.base_dir/'image-parse-v3'/(base+'.png')))

        if self.augment and random.random() > 0.5:
            person=TF.hflip(person); cloth=TF.hflip(cloth)
            c_mask=TF.hflip(c_mask); agnostic=TF.hflip(agnostic)
            parse_np=np.fliplr(parse_np).copy()

        person_t  = self.img_tf(person)
        cloth_t   = self.img_tf(cloth)
        agno_t    = self.img_tf(agnostic)

        cm_arr = np.array(c_mask.resize((self.img_w,self.img_h),Image.NEAREST))
        cm_t   = torch.from_numpy((cm_arr>127).astype(np.float32)).unsqueeze(0)

        ps     = cv2.resize(parse_np,(self.img_w,self.img_h),interpolation=cv2.INTER_NEAREST)
        cr_t   = torch.from_numpy((ps==CLOTH_LABEL).astype(np.float32)).unsqueeze(0)
        parse_t= torch.from_numpy(ps).long()

        return dict(person=person_t, cloth=cloth_t, cloth_mask=cm_t,
                    agnostic=agno_t, parse=parse_t, cloth_region=cr_t)


tr_pairs  = train_pairs[:N_TRAIN]
val_pairs = train_pairs[N_TRAIN:N_TRAIN+N_VAL]

train_ds = VTONDataset(tr_pairs,  TRAIN_DIR, IMG_H, IMG_W, augment=True)
val_ds   = VTONDataset(val_pairs, TRAIN_DIR, IMG_H, IMG_W, augment=False)
test_ds  = VTONDataset(test_pairs[:200], TEST_DIR, IMG_H, IMG_W, augment=False)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=N_WORKERS, pin_memory=(DEVICE.type=='cuda'), drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=N_WORKERS, pin_memory=(DEVICE.type=='cuda'))
test_dl  = DataLoader(test_ds,  batch_size=8, shuffle=False,
                      num_workers=N_WORKERS, pin_memory=(DEVICE.type=='cuda'))

print(f'Train batches:{len(train_dl)} | Val:{len(val_dl)} | Test:{len(test_dl)}')
s = train_ds[0]
for k,v in s.items(): print(f'  {k:14s} {tuple(v.shape)}  {v.dtype}')
""")

md("""
## 4. Architecture

### GMM — Dense Flow Predictor (fix: no TPS)

```
[agnostic] -- VGGFeat(relu2_2) --+
                                  cat(256ch) -- FlowUNet -- flow_offset(2,H,W)
[cloth]    -- VGGFeat(relu2_2) --+                              |
                                                   identity_grid + 0.5*offset
                                                                |
[cloth]     ------------------------------------- F.grid_sample --> warped_cloth
[cloth_mask] ------------------------------------ F.grid_sample --> warped_mask
```

Dense flow avoids the TPS kernel dependency on theta (which varies per sample).
VGG features are properly normalised from [-1,1] to ImageNet range.

### TOM — U-Net + ALIAS (HR-VITON)

```
[agnostic | warped_cloth | warped_mask] (7ch)
  BN-Encoder (e1..e4 + bottleneck)
  ALIAS-Decoder (d4..d1, conditioned on warped_mask)
     rendered(3ch)  +  composition_mask(1ch)
output = mask * warped_cloth + (1-mask) * rendered
```
""")

md("### 4.1 VGG Feature Extractor — with ImageNet normalisation")
code("""
class VGGFeatureExtractor(nn.Module):
    # VGG16 up to relu2_2 -> (B, 128, H/4, W/4).
    # Fix: adds [-1,1] -> ImageNet renormalisation before VGG.
    # relu1_x is frozen; relu2_x is fine-tuned.
    def __init__(self):
        super().__init__()
        feats = list(vgg16(weights=VGG16_Weights.DEFAULT).features)
        self.feat = nn.Sequential(*feats[:10])   # up to relu2_2
        for p in list(self.feat.parameters())[:8]:   # freeze relu1_x
            p.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x):
        # x in [-1,1] -> convert to VGG ImageNet-normalised range
        x = (x * 0.5 + 0.5 - self.mean) / self.std
        return self.feat(x)   # (B,128,H/4,W/4)
""")

md("### 4.2 Dense Flow UNet + GMM")
code("""
def _conv_block(ic, oc, drop=0.0):
    layers = [
        nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
    ]
    if drop > 0: layers.append(nn.Dropout2d(drop))
    return nn.Sequential(*layers)


class FlowUNet(nn.Module):
    # UNet: (B,256,H/4,W/4) -> (B,2,H,W) flow offset via Tanh
    # Architecture (H=256, W=192):
    #   e0:256->128  @ H/4 (64x48)
    #   e1:128->256  @ H/8 (32x24)
    #   e2:256->512  @ H/16(16x12)
    #   bt:512->512  @ H/32(8x6)
    #   d2:(512+512)->256 @ H/16
    #   d1:(256+256)->128 @ H/8
    #   d0:(128+128)->64  @ H/4
    #   up: H/4->H/2->H, flow(2ch)
    def __init__(self, in_ch=256):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.e0   = _conv_block(in_ch, 128)
        self.e1   = _conv_block(128,   256)
        self.e2   = _conv_block(256,   512)
        self.bt   = _conv_block(512,   512)
        self.u2   = nn.ConvTranspose2d(512, 512, 2, 2)
        self.d2   = _conv_block(512+512, 256)
        self.u1   = nn.ConvTranspose2d(256, 256, 2, 2)
        self.d1   = _conv_block(256+256, 128)
        self.u0   = nn.ConvTranspose2d(128, 128, 2, 2)
        self.d0   = _conv_block(128+128, 64)
        self.up_h = nn.ConvTranspose2d(64, 32, 2, 2)   # H/4 -> H/2
        self.up_w = nn.ConvTranspose2d(32, 16, 2, 2)   # H/2 -> H
        self.out  = nn.Sequential(nn.Conv2d(16, 2, 3, padding=1), nn.Tanh())
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        # Zero-init last conv -> identity warp at epoch 0
        nn.init.zeros_(self.out[0].weight)
        nn.init.zeros_(self.out[0].bias)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(self.pool(e0))
        e2 = self.e2(self.pool(e1))
        bt = self.bt(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(bt), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        d0 = self.d0(torch.cat([self.u0(d1), e0], 1))
        return self.out(self.up_w(self.up_h(d0)))   # (B,2,H,W) Tanh


class GMM(nn.Module):
    # Dense flow GMM — no TPS, no per-sample solve.
    # Inputs : agnostic(3), cloth(3), cloth_mask(1)
    # Outputs: warped_cloth(3), warped_mask(1), flow_offset(2,H,W)
    def __init__(self, img_h=256, img_w=192):
        super().__init__()
        self.feat_p   = VGGFeatureExtractor()
        self.feat_c   = VGGFeatureExtractor()
        self.flow_net = FlowUNet(in_ch=256)
        gy, gx = torch.meshgrid(
            torch.linspace(-1,1,img_h), torch.linspace(-1,1,img_w), indexing='ij')
        self.register_buffer('identity', torch.stack([gx,gy],0))  # (2,H,W)

    def forward(self, agnostic, cloth, cloth_mask):
        fp    = self.feat_p(agnostic)
        fc    = self.feat_c(cloth)
        feats = torch.cat([fp, fc], 1)            # (B,256,H/4,W/4)
        flow  = self.flow_net(feats) * 0.5        # (B,2,H,W), scale max=0.5
        grid  = (self.identity.unsqueeze(0) + flow).permute(0,2,3,1).clamp(-1,1)
        wc = F.grid_sample(cloth,      grid, mode='bilinear', padding_mode='border', align_corners=True)
        wm = F.grid_sample(cloth_mask, grid, mode='bilinear', padding_mode='zeros',  align_corners=True)
        return wc, wm, flow


gmm = GMM(IMG_H, IMG_W).to(DEVICE)
p_gmm = sum(p.numel() for p in gmm.parameters() if p.requires_grad)
print(f'GMM trainable: {p_gmm:,}')
_b = {k: v.to(DEVICE) for k,v in next(iter(train_dl)).items()}
_wc, _wm, _fl = gmm(_b['agnostic'], _b['cloth'], _b['cloth_mask'])
print(f'  wc:{tuple(_wc.shape)} | wm:{tuple(_wm.shape)} | flow:{tuple(_fl.shape)}')
""")

md("### 4.3 TOM with ALIAS Normalisation (HR-VITON)")
code("""
class ALIASNorm(nn.Module):
    # SPADE-style adaptive norm conditioned on warped_mask (cloth region).
    def __init__(self, num_ch, cond_ch=1):
        super().__init__()
        self.norm   = nn.InstanceNorm2d(num_ch, affine=False)
        h = min(num_ch, 128)
        self.shared = nn.Sequential(nn.Conv2d(cond_ch,h,3,padding=1), nn.ReLU(True))
        self.gamma  = nn.Conv2d(h, num_ch, 3, padding=1)
        self.beta   = nn.Conv2d(h, num_ch, 3, padding=1)

    def forward(self, x, cond):
        cond_r = F.interpolate(cond, x.shape[2:], mode='nearest')
        h = self.shared(cond_r)
        return self.norm(x) * (1 + self.gamma(h)) + self.beta(h)


class ALIASBlock(nn.Module):
    def __init__(self, ic, oc, drop=0.0):
        super().__init__()
        self.c1   = nn.Conv2d(ic,oc,3,padding=1,bias=False)
        self.n1   = ALIASNorm(oc)
        self.c2   = nn.Conv2d(oc,oc,3,padding=1,bias=False)
        self.n2   = ALIASNorm(oc)
        self.act  = nn.ReLU(True)
        self.drop = nn.Dropout2d(drop) if drop>0 else nn.Identity()

    def forward(self, x, cond):
        x = self.act(self.n1(self.c1(x), cond))
        x = self.act(self.n2(self.c2(x), cond))
        return self.drop(x)


class BNBlock(nn.Module):
    def __init__(self, ic, oc, drop=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        ]
        if drop > 0: layers.append(nn.Dropout2d(drop))
        self.b = nn.Sequential(*layers)
    def forward(self, x): return self.b(x)


class TOM(nn.Module):
    # U-Net (64/128/256/512) with BN encoder + ALIAS decoder.
    # Input : 7ch [agnostic | warped_cloth | warped_mask]
    # Output: try-on image, rendered image, composition mask
    def __init__(self, in_ch=7, feats=(64,128,256,512)):
        super().__init__()
        f = list(feats); self.pool = nn.MaxPool2d(2)
        self.e1=BNBlock(in_ch,f[0],0.0); self.e2=BNBlock(f[0],f[1],0.1)
        self.e3=BNBlock(f[1],f[2],0.2);  self.e4=BNBlock(f[2],f[3],0.3)
        self.bt=BNBlock(f[3],f[3],0.3)
        self.u4=nn.ConvTranspose2d(f[3],f[3],2,2); self.d4=ALIASBlock(f[3]+f[3],f[3],0.2)
        self.u3=nn.ConvTranspose2d(f[3],f[2],2,2); self.d3=ALIASBlock(f[2]+f[2],f[2],0.2)
        self.u2=nn.ConvTranspose2d(f[2],f[1],2,2); self.d2=ALIASBlock(f[1]+f[1],f[1],0.1)
        self.u1=nn.ConvTranspose2d(f[1],f[0],2,2); self.d1=ALIASBlock(f[0]+f[0],f[0],0.0)
        self.rh=nn.Sequential(nn.Conv2d(f[0],3,1),nn.Tanh())
        self.mh=nn.Sequential(nn.Conv2d(f[0],1,1),nn.Sigmoid())
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, warped_cloth, warped_mask):
        e1=self.e1(x); e2=self.e2(self.pool(e1))
        e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
        bt=self.bt(self.pool(e4))
        d4=self.d4(torch.cat([self.u4(bt),e4],1),warped_mask)
        d3=self.d3(torch.cat([self.u3(d4),e3],1),warped_mask)
        d2=self.d2(torch.cat([self.u2(d3),e2],1),warped_mask)
        d1=self.d1(torch.cat([self.u1(d2),e1],1),warped_mask)
        rendered=self.rh(d1); mask=self.mh(d1)
        return mask*warped_cloth+(1-mask)*rendered, rendered, mask


tom = TOM(in_ch=7).to(DEVICE)
p_tom = sum(p.numel() for p in tom.parameters() if p.requires_grad)
print(f'TOM trainable: {p_tom:,}')
_xi = torch.cat([_b['agnostic'],_wc.detach(),_wm.detach()],1)
_out,_ren,_msk = tom(_xi, _wc.detach(), _wm.detach())
print(f'  output:{tuple(_out.shape)} | mask:{tuple(_msk.shape)}')
print(f'\\n{"Module":20s} {"Params":>12s}'); print('-'*34)
print(f'{"GMM":20s} {p_gmm:>12,}')
print(f'{"TOM":20s} {p_tom:>12,}')
print(f'{"Total":20s} {p_gmm+p_tom:>12,}')
""")

md("### 4.4 Losses")
code("""
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        f = list(vgg16(weights=VGG16_Weights.DEFAULT).features)
        self.s1=nn.Sequential(*f[:5]); self.s2=nn.Sequential(*f[5:10])
        self.s3=nn.Sequential(*f[10:17])
        for p in self.parameters(): p.requires_grad=False
        self.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def _prep(self,x): return ((x*0.5+0.5)-self.mean)/self.std

    def forward(self,pred,target):
        p=self._prep(pred); t=self._prep(target); loss=0.0
        for sl,w in zip([self.s1,self.s2,self.s3],[1.0,0.75,0.5]):
            p=sl(p); t=sl(t); loss=loss+w*F.l1_loss(p,t)
        return loss


def ssim_loss_fn(pred, target, win=11):
    # Differentiable 1-SSIM. pred/target in [-1,1].
    p=(pred*0.5+0.5).clamp(0,1); t=(target*0.5+0.5).clamp(0,1)
    coords=torch.arange(win,dtype=torch.float32,device=pred.device)-win//2
    g=torch.exp(-(coords**2)/(2*1.5**2)); g=g/g.sum()
    kernel=(g[:,None]*g[None,:]).unsqueeze(0).unsqueeze(0).repeat(3,1,1,1)
    pad=win//2
    mu_p=F.conv2d(p,kernel,padding=pad,groups=3)
    mu_t=F.conv2d(t,kernel,padding=pad,groups=3)
    sg_p =F.conv2d(p*p,kernel,padding=pad,groups=3)-mu_p**2
    sg_t =F.conv2d(t*t,kernel,padding=pad,groups=3)-mu_t**2
    sg_pt=F.conv2d(p*t,kernel,padding=pad,groups=3)-mu_p*mu_t
    C1,C2=0.01**2,0.03**2
    ssim_map=((2*mu_p*mu_t+C1)*(2*sg_pt+C2))/((mu_p**2+mu_t**2+C1)*(sg_p+sg_t+C2))
    return 1.0-ssim_map.mean()


vgg_loss = VGGPerceptualLoss().to(DEVICE)
_pl = vgg_loss(_out.detach(), _b['person'])
_sl = ssim_loss_fn(_out.detach(), _b['person'])
print(f'VGG loss test : {_pl.item():.4f}')
print(f'SSIM loss test: {_sl.item():.4f}')
""")

md("""
## 5. Stage 1 — GMM Training

**Loss** = L1(warped_mask, cloth_region)                    [mask alignment]
         + 0.5 x L1(warped_cloth x cloth_region, person_cloth) [appearance — paired GT!]
         + 0.001 x TV(flow)                                 [flow smoothness]

The appearance term is the key new supervision signal:
training pairs are **paired** (person wears the listed cloth),
so `person * cloth_region` is the ground truth that the warped cloth must match.
""")
code("""
def move(b): return {k: v.to(DEVICE) for k,v in b.items()}

scaler_gmm = torch.cuda.amp.GradScaler(enabled=USE_AMP)
gmm_opt = torch.optim.Adam(gmm.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
gmm_sch = torch.optim.lr_scheduler.OneCycleLR(
    gmm_opt, max_lr=LR*5, steps_per_epoch=len(train_dl),
    epochs=GMM_EPOCHS, pct_start=0.1, anneal_strategy='cos')

APPEAR_W  = 0.5
FLOW_TV_W = 0.001

def gmm_loss_fn(wc, wm, flow, cloth_region, person):
    mask_l   = F.l1_loss(wm, cloth_region)
    # Appearance: warped cloth in the cloth region should match person's cloth (paired GT)
    person_cloth  = person * cloth_region
    warped_in_reg = wc     * cloth_region
    appear_l = F.l1_loss(warped_in_reg, person_cloth.detach())
    tv_l = (torch.abs(flow[:,:,1:,:]-flow[:,:,:-1,:]).mean() +
            torch.abs(flow[:,:,:,1:]-flow[:,:,:,:-1]).mean())
    return mask_l+APPEAR_W*appear_l+FLOW_TV_W*tv_l, mask_l, appear_l, tv_l


def gmm_epoch(model, dl, opt, sch, train):
    model.train(train); tot=ml=al=tl=0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            b=move(b)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                wc,wm,fl=model(b['agnostic'],b['cloth'],b['cloth_mask'])
                loss,ml_i,al_i,tl_i=gmm_loss_fn(wc,wm,fl,b['cloth_region'],b['person'])
            if train:
                opt.zero_grad()
                scaler_gmm.scale(loss).backward()
                scaler_gmm.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler_gmm.step(opt); scaler_gmm.update(); sch.step()
            tot+=loss.item(); ml+=ml_i.item(); al+=al_i.item(); tl+=tl_i.item()
    n=len(dl); return tot/n,ml/n,al/n,tl/n


gmm_h={k:[] for k in ['tr','val','tr_m','val_m','tr_a','val_a','lr']}
best_v=float('inf'); best_gmm_w=None; pat=0

print('Stage 1 - GMM Training'); print('='*80)
for ep in range(1,GMM_EPOCHS+1):
    t0=time.time()
    tr=gmm_epoch(gmm,train_dl,gmm_opt,gmm_sch,True)
    va=gmm_epoch(gmm,val_dl,  gmm_opt,gmm_sch,False)
    lr=gmm_opt.param_groups[0]['lr']
    for k,v in zip(['tr','val','tr_m','val_m','tr_a','val_a','lr'],
                   [tr[0],va[0],tr[1],va[1],tr[2],va[2],lr]):
        gmm_h[k].append(v)
    print(f'Ep{ep:2d}/{GMM_EPOCHS} '
          f'| tr={tr[0]:.4f}(mask={tr[1]:.4f} app={tr[2]:.4f} tv={tr[3]:.5f}) '
          f'| val={va[0]:.4f}(mask={va[1]:.4f} app={va[2]:.4f}) '
          f'| lr={lr:.2e} | {time.time()-t0:.1f}s')
    if va[0]<best_v:
        best_v=va[0]; pat=0
        best_gmm_w={k:v.cpu().clone() for k,v in gmm.state_dict().items()}
        print('   v best saved')
    else:
        pat+=1
        if pat>=PATIENCE: print(f'  Early stop ep{ep}'); break
gmm.load_state_dict(best_gmm_w)
print(f'\\nGMM best val: {best_v:.4f}')
""")

code("""
ep=list(range(1,len(gmm_h['tr'])+1))
fig,axes=plt.subplots(1,3,figsize=(16,4))
fig.suptitle('GMM Training - Dense Flow + Appearance Loss',fontsize=13,fontweight='bold')
axes[0].plot(ep,gmm_h['tr'],label='Train',color='steelblue',lw=2)
axes[0].plot(ep,gmm_h['val'],label='Val',color='darkorange',lw=2)
axes[0].set_title('Total Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(ep,gmm_h['tr_m'],label='Train Mask',color='steelblue',lw=2)
axes[1].plot(ep,gmm_h['val_m'],label='Val Mask',color='darkorange',lw=2)
axes[1].plot(ep,gmm_h['tr_a'],'--',label='Train Appear',color='steelblue',lw=2)
axes[1].plot(ep,gmm_h['val_a'],'--',label='Val Appear',color='darkorange',lw=2)
axes[1].set_title('Mask & Appearance L1'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
axes[2].plot(ep,gmm_h['lr'],color='purple',lw=2)
axes[2].set_title('LR (OneCycle)'); axes[2].set_yscale('log'); axes[2].grid(alpha=0.3)
for ax in axes: ax.set_xlabel('Epoch')
plt.tight_layout(); plt.savefig('gmm_curves_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

code("""
def denorm(t):
    return ((t*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8)

gmm.eval()
N_SHOW=4
samples=[test_ds[i] for i in range(N_SHOW)]
batch={k:torch.stack([s[k] for s in samples]).to(DEVICE) for k in samples[0]}
with torch.no_grad():
    wc,wm,fl=gmm(batch['agnostic'],batch['cloth'],batch['cloth_mask'])

fig,axes=plt.subplots(N_SHOW,5,figsize=(19,4.8*N_SHOW))
fig.suptitle('GMM - Dense Flow Warping',fontsize=13,fontweight='bold',y=1.01)
for row in range(N_SHOW):
    cr=batch['cloth_region'][row,0].cpu().numpy()
    wm_np=wm[row,0].detach().cpu().numpy()
    overlay=np.zeros((IMG_H,IMG_W,3),np.float32)
    overlay[...,0]=wm_np; overlay[...,1]=cr
    imgs=[denorm(batch['person'][row]),denorm(batch['cloth'][row]),
          batch['cloth_mask'][row,0].cpu().numpy(),denorm(wc[row]),overlay]
    for col,(img,cm,t) in enumerate(zip(imgs,[None,None,'gray',None,None],
            ['Person','Cloth','Mask','Warped','Overlay'])):
        axes[row][col].imshow(img,cmap=cm,vmin=0 if cm else None,vmax=1 if cm else None)
        axes[row][col].axis('off')
        if row==0: axes[row][col].set_title(t,fontsize=10,fontweight='bold')
fig.legend(handles=[mpatches.Patch(color='red',label='Warped mask'),
                    mpatches.Patch(color='green',label='GT cloth region')],
           loc='lower center',ncol=2)
plt.tight_layout(); plt.savefig('gmm_results_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

md("""
## 6. Stage 2 — TOM Training (GMM frozen)

**Loss** = 1.0 x L1  +  0.25 x VGG perceptual  +  0.5 x (1-SSIM)  +  0.01 x mask_TV
""")
code("""
for p in gmm.parameters(): p.requires_grad=False

scaler_tom = torch.cuda.amp.GradScaler(enabled=USE_AMP)
tom_opt = torch.optim.Adam(tom.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
tom_sch = torch.optim.lr_scheduler.OneCycleLR(
    tom_opt, max_lr=LR*5, steps_per_epoch=len(train_dl),
    epochs=TOM_EPOCHS, pct_start=0.1, anneal_strategy='cos')

VGG_W=0.25; SSIM_W=0.5; TV_W=0.01

def tom_loss_fn(out, ren, mask, wc, target):
    l1  =F.l1_loss(out,target)
    perc=vgg_loss(out,target)
    sl  =ssim_loss_fn(out,target)
    tv  =(torch.abs(mask[:,:,1:,:]-mask[:,:,:-1,:]).mean()+
          torch.abs(mask[:,:,:,1:]-mask[:,:,:,:-1]).mean())
    return l1+VGG_W*perc+SSIM_W*sl+TV_W*tv, l1, perc, sl


def tom_epoch(gmm_m, tom_m, dl, opt, sch, train):
    tom_m.train(train); tot=tl1=tpc=tsl=0.0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for b in dl:
            b=move(b)
            with torch.no_grad():
                wc,wm,_=gmm_m(b['agnostic'],b['cloth'],b['cloth_mask'])
            xi=torch.cat([b['agnostic'],wc,wm],1)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out,ren,msk=tom_m(xi,wc,wm)
                loss,l1i,pci,sli=tom_loss_fn(out,ren,msk,wc,b['person'])
            if train:
                opt.zero_grad()
                scaler_tom.scale(loss).backward()
                scaler_tom.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(tom_m.parameters(),1.0)
                scaler_tom.step(opt); scaler_tom.update(); sch.step()
            tot+=loss.item(); tl1+=l1i.item(); tpc+=pci.item(); tsl+=sli.item()
    n=len(dl); return tot/n,tl1/n,tpc/n,tsl/n


tom_h={k:[] for k in ['tr','val','tr_l1','val_l1','tr_pc','val_pc','tr_sl','val_sl','lr']}
best_vt=float('inf'); best_tom_w=None; pat_t=0

print('Stage 2 - TOM Training  (GMM frozen)'); print('='*90)
for ep in range(1,TOM_EPOCHS+1):
    t0=time.time()
    tr=tom_epoch(gmm,tom,train_dl,tom_opt,tom_sch,True)
    va=tom_epoch(gmm,tom,val_dl,  tom_opt,tom_sch,False)
    lr=tom_opt.param_groups[0]['lr']
    for k,v in zip(['tr','val','tr_l1','val_l1','tr_pc','val_pc','tr_sl','val_sl','lr'],
                   [tr[0],va[0],tr[1],va[1],tr[2],va[2],tr[3],va[3],lr]):
        tom_h[k].append(v)
    print(f'Ep{ep:2d}/{TOM_EPOCHS} '
          f'| tr={tr[0]:.4f}(L1={tr[1]:.4f} VGG={tr[2]:.4f} SSIM={tr[3]:.4f}) '
          f'| val={va[0]:.4f}(L1={va[1]:.4f} SSIM={va[3]:.4f}) '
          f'| lr={lr:.2e} | {time.time()-t0:.1f}s')
    if va[0]<best_vt:
        best_vt=va[0]; pat_t=0
        best_tom_w={k:v.cpu().clone() for k,v in tom.state_dict().items()}
        print('   v best saved')
    else:
        pat_t+=1
        if pat_t>=PATIENCE: print(f'  Early stop ep{ep}'); break
tom.load_state_dict(best_tom_w)
print(f'\\nTOM best val: {best_vt:.4f}')
""")

code("""
ep=list(range(1,len(tom_h['tr'])+1))
fig,axes=plt.subplots(1,4,figsize=(20,4))
fig.suptitle('TOM Training - ALIAS + SSIM loss',fontsize=13,fontweight='bold')
for ax,tr_k,va_k,t in zip(axes,
        ['tr','tr_l1','tr_pc','tr_sl'],['val','val_l1','val_pc','val_sl'],
        ['Total','L1','VGG Perceptual','1-SSIM']):
    ax.plot(ep,tom_h[tr_k],label='Train',color='steelblue',lw=2)
    ax.plot(ep,tom_h[va_k],label='Val',  color='darkorange',lw=2)
    ax.set_title(t); ax.legend(); ax.grid(alpha=0.3); ax.set_xlabel('Epoch')
plt.tight_layout(); plt.savefig('tom_curves_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

md("## 7. Results\n### 7.1 Qualitative")
code("""
gmm.eval(); tom.eval()
N_SHOW=6
samples=[test_ds[i] for i in range(N_SHOW)]
batch={k:torch.stack([s[k] for s in samples]).to(DEVICE) for k in samples[0]}
with torch.no_grad():
    wc,wm,_    =gmm(batch['agnostic'],batch['cloth'],batch['cloth_mask'])
    out,ren,msk=tom(torch.cat([batch['agnostic'],wc,wm],1),wc,wm)

fig,axes=plt.subplots(N_SHOW,6,figsize=(22,4.3*N_SHOW))
fig.suptitle('HR-VITON-Inspired Pipeline - Results',fontsize=14,fontweight='bold',y=1.005)
for row in range(N_SHOW):
    imgs=[denorm(batch['person'][row]),denorm(batch['cloth'][row]),
          denorm(wc[row]),msk[row,0].detach().cpu().numpy(),
          denorm(out[row]),denorm(batch['person'][row])]
    for col,(img,cm,t) in enumerate(zip(imgs,
            [None,None,None,'RdYlGn',None,None],
            ['Person','Cloth','Warped','Mask','Output','GT'])):
        axes[row][col].imshow(img,cmap=cm,vmin=0 if cm else None,vmax=1 if cm else None)
        axes[row][col].axis('off')
        if row==0: axes[row][col].set_title(t,fontsize=9,fontweight='bold')
    axes[row][0].set_ylabel(f'Test {row+1}',fontsize=10)
plt.tight_layout(); plt.savefig('results_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

md("### 7.2 Quantitative Metrics")
code("""
def metrics_batch(pred_t, gt_t):
    sl,pl,ml=[],[],[]
    for p,g in zip(pred_t,gt_t):
        pn=(p*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().detach().numpy()
        gn=(g*0.5+0.5).clamp(0,1).permute(1,2,0).cpu().detach().numpy()
        sl.append(ssim_metric(pn,gn,data_range=1.0,channel_axis=2))
        pl.append(psnr_metric(gn,pn,data_range=1.0))
        ml.append(float(np.mean((pn-gn)**2)))
    return sl,pl,ml

s_nc,p_nc,m_nc=[],[],[]
s_cp,p_cp,m_cp=[],[],[]

with torch.no_grad():
    for b in test_dl:
        b=move(b)
        wc,wm,_=gmm(b['agnostic'],b['cloth'],b['cloth_mask'])
        out,_,_=tom(torch.cat([b['agnostic'],wc,wm],1),wc,wm)
        sl,pl,ml=metrics_batch(b['agnostic'],b['person'])
        s_nc+=sl; p_nc+=pl; m_nc+=ml
        sl,pl,ml=metrics_batch(out,b['person'])
        s_cp+=sl; p_cp+=pl; m_cp+=ml

res=pd.DataFrame({
    'Method':['Baseline (agnostic)','HR-VITON-Inspired (Ours)'],
    'SSIM':  [np.mean(s_nc),np.mean(s_cp)],
    'PSNR':  [np.mean(p_nc),np.mean(p_cp)],
    'MSE':   [np.mean(m_nc),np.mean(m_cp)],
})
print(res.to_string(index=False,float_format='{:.4f}'.format))

fig,axes=plt.subplots(1,3,figsize=(14,4))
fig.suptitle('Quantitative Evaluation (200 test pairs)',fontsize=13,fontweight='bold')
for ax,(m,v) in zip(axes,[('SSIM (higher=better)',[np.mean(s_nc),np.mean(s_cp)]),
                           ('PSNR (higher=better)',[np.mean(p_nc),np.mean(p_cp)]),
                           ('MSE (lower=better)', [np.mean(m_nc),np.mean(m_cp)])]):
    bars=ax.bar(['Baseline','Ours'],v,color=['#e74c3c','#2ecc71'],edgecolor='white',width=0.5)
    ax.set_title(m); ax.grid(axis='y',alpha=0.3)
    for bar,val in zip(bars,v):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.01,f'{val:.4f}',
                ha='center',va='bottom',fontsize=10,fontweight='bold')
plt.tight_layout(); plt.savefig('metrics_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

md("## 8. Overfitting Analysis")
code("""
ge=len(gmm_h['tr']); te=len(tom_h['tr'])
print('='*60); print('OVERFITTING ANALYSIS'); print('='*60)
print(f'{"":22s} {"GMM":>10s} {"TOM":>10s}'); print('-'*44)
print(f'{"Epochs trained":22s} {ge:>10d} {te:>10d}')
print(f'{"Final train loss":22s} {gmm_h["tr"][-1]:>10.4f} {tom_h["tr"][-1]:>10.4f}')
print(f'{"Final val loss":22s} {gmm_h["val"][-1]:>10.4f} {tom_h["val"][-1]:>10.4f}')
print(f'{"Train-Val gap":22s} {abs(gmm_h["val"][-1]-gmm_h["tr"][-1]):>10.4f} '
      f'{abs(tom_h["val"][-1]-tom_h["tr"][-1]):>10.4f}')

fixes=pd.DataFrame({'Fix':[
    'Auto GPU detection','Dense flow GMM (no TPS)','VGG input normalisation',
    'GMM appearance loss','TOM features 64/128/256/512','ALIAS normalisation',
    'SSIM loss','AMP mixed precision','OneCycleLR warmup','5000 training pairs'],
'Why it matters':[
    'Was hardcoded to CPU: 50x slower training',
    'TPS batched solve had kernel bug (L dep on theta)',
    'VGG expects ImageNet range not [-1,1]',
    'Paired GT: person cloth region supervises warp',
    '4x more capacity vs 32/64/128',
    'HR-VITON: handles cloth-body misalignment',
    'Directly optimises evaluation metric',
    '2-3x speed-up on modern GPUs',
    'Warmup prevents early divergence',
    'Reduces underfitting vs 600 pairs']})
print(); print(fixes.to_string(index=False))

fig,axes=plt.subplots(2,2,figsize=(14,9))
fig.suptitle('Overfitting Analysis',fontsize=14,fontweight='bold')
for (stage,hist),row in zip([('GMM',gmm_h),('TOM',tom_h)],[0,1]):
    ep=list(range(1,len(hist['tr'])+1))
    axes[row][0].plot(ep,hist['tr'],label='Train',color='steelblue',lw=2)
    axes[row][0].plot(ep,hist['val'],label='Val',color='darkorange',lw=2)
    axes[row][0].fill_between(ep,hist['tr'],hist['val'],alpha=0.12,color='red')
    axes[row][0].set_title(f'{stage} Loss'); axes[row][0].legend(); axes[row][0].grid(alpha=0.3)
    gaps=[abs(v-t) for v,t in zip(hist['val'],hist['tr'])]
    axes[row][1].bar(ep,gaps,color='salmon',edgecolor='darkred',lw=0.4)
    axes[row][1].axhline(np.mean(gaps),color='darkred',ls='--',lw=2,label=f'Mean={np.mean(gaps):.4f}')
    axes[row][1].set_title(f'{stage} Val-Train Gap'); axes[row][1].legend(); axes[row][1].grid(alpha=0.3,axis='y')
for ax in axes.flat: ax.set_xlabel('Epoch')
plt.tight_layout(); plt.savefig('overfitting_v3.png',dpi=100,bbox_inches='tight'); plt.show()
""")

md("""
## 9. Conclusion & References

### Bug Summary (v2 -> v3)

| Bug | Root cause | Fix |
|---|---|---|
| SSIM = 0.12 | CPU training + 600 pairs = undertrained | Auto GPU + 5000 pairs |
| Bad warping | Batched TPS: kernel computed from ctrl_src, must use ctrl_tgt per sample | Dense flow UNet |
| Garbled VGG features | No [-1,1] to ImageNet renorm before VGG | normalisation buffer |
| Weak GMM signal | Only mask L1, no texture/appearance supervision | Cloth-region appearance L1 |
| Slow convergence | ReduceLROnPlateau, no warmup | OneCycleLR (10% warmup) |

### References

1. **CP-VTON** Wang et al., ECCV 2018.
2. **HR-VITON** Lee et al., ECCV 2022.
3. **VITON-HD** Choi et al., CVPR 2021.
4. **SPADE/ALIAS** Park et al., CVPR 2019.
5. **PFAFN (Flow)** Ge et al., CVPR 2021.
6. **U-Net** Ronneberger et al., MICCAI 2015.
7. **Perceptual Loss** Johnson et al., ECCV 2016.
8. **SSIM** Wang et al., IEEE TIP 2004.
""")

# Build
nb.cells = cells
nb.metadata['kernelspec'] = {'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata['language_info'] = {'name':'python','version':'3.10.0'}
import nbformat
nbformat.write(nb, 'virtual_tryon_v3.ipynb')
print('Written: virtual_tryon_v3.ipynb')
