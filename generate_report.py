from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
    KeepInFrame
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

OUTPUT = "VirtualTryOn_Report.pdf"

doc = SimpleDocTemplate(
    OUTPUT, pagesize=letter,
    leftMargin=0.55*inch, rightMargin=0.55*inch,
    topMargin=0.45*inch,  bottomMargin=0.45*inch,
)
W = letter[0] - 1.1*inch

ACCENT = colors.HexColor("#1a5276")
LIGHT  = colors.HexColor("#d6eaf8")
LGREY  = colors.HexColor("#f5f5f5")

# ── Styles ────────────────────────────────────────────────────────────────────
TITLE = ParagraphStyle("T", fontName="Helvetica-Bold",   fontSize=13,
                        textColor=ACCENT, spaceAfter=1, alignment=TA_CENTER)
SUB   = ParagraphStyle("S", fontName="Helvetica",        fontSize=8.5,
                        textColor=colors.HexColor("#555"), spaceAfter=4,
                        alignment=TA_CENTER)
H1    = ParagraphStyle("H1",fontName="Helvetica-Bold",   fontSize=8.5,
                        textColor=colors.white, leftIndent=5)
BODY  = ParagraphStyle("B", fontName="Helvetica",        fontSize=7.5,
                        leading=10.2, spaceAfter=2, alignment=TA_JUSTIFY)
BUL   = ParagraphStyle("BU",fontName="Helvetica",        fontSize=7.5,
                        leading=10, leftIndent=11, firstLineIndent=-7,
                        spaceAfter=1.2)
BOLD  = ParagraphStyle("BO",fontName="Helvetica-Bold",   fontSize=7.5,
                        leading=10, spaceAfter=1.5)
FN    = ParagraphStyle("FN",fontName="Helvetica-Oblique",fontSize=6.5,
                        textColor=colors.grey, spaceAfter=1)
FOOT  = ParagraphStyle("FT",fontName="Helvetica-Oblique",fontSize=6.8,
                        textColor=colors.grey, alignment=TA_CENTER)

def b(t):  return f"<b>{t}</b>"
def it(t): return f"<i>{t}</i>"

def p(txt, st=BODY):  return Paragraph(txt, st)
def bp(txt):          return Paragraph(f"• {txt}", BUL)
def sp(h=3):          return Spacer(1, h)

def sec(txt):
    t = Table([[Paragraph(txt, H1)]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), ACCENT),
        ("TOPPADDING",    (0,0),(-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2.5),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ]))
    return t

def hr(): return HRFlowable(width="100%",thickness=0.4,
                             color=colors.HexColor("#aaa"),spaceAfter=3)

def two(L, R, lw=0.485, rw=0.485):
    g = W*0.03
    lf = KeepInFrame(W*lw, 9999, L, mode="shrink")
    rf = KeepInFrame(W*rw, 9999, R, mode="shrink")
    t  = Table([[lf, Spacer(g,1), rf]], colWidths=[W*lw, g, W*rw])
    t.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),
                            ("PADDING",(0,0),(-1,-1),0)]))
    return t

def tbl(data, col_w, row_colors=None):
    t = Table(data, colWidths=col_w)
    style = [
        ("BACKGROUND",    (0,0),(-1,0),  ACCENT),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 7.2),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#ccc")),
        ("TOPPADDING",    (0,0),(-1,-1), 2.5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2.5),
        ("LEFTPADDING",   (0,0),(-1,-1), 4),
    ]
    if row_colors:
        for r,c in row_colors:
            style.append(("BACKGROUND",(0,r),(-1,r),c))
    t.setStyle(TableStyle(style))
    return t

# ════════════════════════════════════════════════════════════════════════════
story = []

# Title
story += [
    p("Enhanced Virtual Try-On", TITLE),
    p("Joyati R· Biraj M · Murughanandam S.  |  MSDS Computer Vision 462 · Spring 2026", SUB),
    hr(), sp(1),
]

# ─── §1 Problem Statement ────────────────────────────────────────────────────
story += [
    sec("1.  Problem Statement and Objectives"), sp(3),
    two(
        [
            p("Image-based virtual try-on synthesises a photo-realistic image of a "
              "person wearing a target garment without physical fitting. The task "
              f"requires solving three simultaneous sub-problems: {b('geometric warping')} "
              f"of the garment to body pose, {b('semantic body parsing')} to preserve "
              f"identity, and {b('photo-realistic synthesis')} that blends cloth texture "
              "naturally. Applications span e-commerce, gaming, and metaverse avatars."),
        ],
        [
            p(f"{b('Objectives:')}"),
            bp("Implement a two-stage try-on pipeline (warping + synthesis)."),
            bp("Iteratively improve quality across v2 → v3 → v4."),
            bp("Benchmark using SSIM, PSNR, and MSE."),
            bp("Show the practical advantage of pretrained models."),
        ],
    ), sp(3),
]

# ─── §2 Dataset ──────────────────────────────────────────────────────────────
story += [
    sec("2.  Dataset Description and Preprocessing"), sp(3),
    two(
        [
            p(f"We use {b('VITON-HD')} (Zalando) — a high-resolution benchmark "
              "with {b('11,647 training')} and {b('2,032 test')} paired samples "
              "at native 768×1024 resolution. Each sample contains 9 modalities: "
              "person image, garment, cloth mask, agnostic image (clothing blanked), "
              "semantic parse map (18 labels), parse-agnostic map, DensePose UV, "
              "OpenPose keypoints (BODY_25, 25 pts), and OpenPose image."),
        ],
        [
            p(f"{b('Preprocessing:')}"),
            bp("Resize: bilinear for RGB; nearest-neighbour for masks/parse maps."),
            bp("Normalise to [−1, 1] (mean=0.5, std=0.5). Cloth mask binarised at 0.5."),
            bp("Clothing region = parse label 5 (Upper-clothes/Dress)."),
            bp("Augmentation: horizontal flip + colour jitter (v2/v3 only)."),
            bp("Paired setup: GT garment matches the garment worn in each photo."),
        ],
    ), sp(3),
]

# ─── §3 Architecture ─────────────────────────────────────────────────────────
story += [
    sec("3.  Mathematical Foundations and Model Architectures"), sp(3),
    p(f"All versions follow the {b('CP-VTON two-stage paradigm:')} a "
      f"{b('Geometric Matching Module (GMM)')} warps the garment and a "
      f"{b('Try-On Module (TOM)')} synthesises the final image. The architecture "
      "evolved significantly across versions as bugs were identified and fixed."),
    sp(3),
    two(
        [
            p(f"{b('v2 — TPS Warping (CP-VTON):')}", BOLD),
            p("CNN features from the agnostic image and garment are cross-correlated. "
              f"A regressor predicts offsets {it('θ ∈ ℝ^(B×2×25)')} for a 5×5 "
              f"control-point grid. The TPS system {it('Lw = v')} is solved per sample "
              f"using basis {it('U(r) = r² log r²')} to produce a dense warp grid. "
              "The TOM (U-Net 32/64/128ch) outputs a rendered image and composition "
              "mask: output = mask ⊙ warped_cloth + (1−mask) ⊙ rendered."),
            sp(3),
            p(f"{b('v3 — Dense Flow UNet + ALIAS:')}", BOLD),
            p("TPS replaced by a FlowUNet predicting a 2-ch offset field "
              f"{it('f ∈ ℝ^(B×2×H×W)')} via Tanh; grid = G_identity + 0.5·f. "
              "VGG relu2_2 features (256-ch) from agnostic and cloth drive the UNet. "
              "TOM widened to 64/128/256/512ch with ALIAS decoder blocks — "
              "SPADE-style adaptive normalisation conditioned on the warped cloth mask."),
        ],
        [
            p(f"{b('v4 — Official HR-VITON (Pretrained):')}", BOLD),
            p(f"Stage 1: {b('ConditionGenerator (tocg)')} jointly predicts a 5-scale "
              "coarse-to-fine flow field and 13-ch segmentation map from cloth + "
              "cloth-mask (4ch) and parse-agnostic + DensePose (16ch). Stage 2: "
              f"{b('SPADEGenerator')} synthesises 768×1024 output from agnostic + "
              "DensePose + warped cloth (9ch) conditioned on a 7-ch parse map via "
              "ALIAS blocks. An occlusion step (remove_overlap) prevents garment "
              "bleeding over face and arms."),
            sp(3),
            tbl(
                [["", "v2","v3","v4"],
                 ["Warping",    "TPS (buggy)", "Dense Flow",  "5-scale flow"],
                 ["TOM",        "BN 32/64/128","ALIAS 64→512","SPADEGen"],
                 ["Resolution", "256×192",     "256×192",     "768×1024"],
                 ["Train pairs","600",          "5,000",       "Full VITON-HD"],
                 ["Device",     "CPU",          "GPU+AMP",     "GPU"]],
                [W*0.10, W*0.13, W*0.13, W*0.14],
                row_colors=[(1,LGREY),(3,LGREY),(5,LGREY)],
            ),
        ],
    ), sp(3),
]

# ─── §4 Training ─────────────────────────────────────────────────────────────
story += [
    sec("4.  Training Methodology and Hyperparameter Choices"), sp(3),
    two(
        [
            p(f"{b('Sequential two-stage training (v2/v3):')}"),
            bp("GMM trains first; TOM trains with GMM frozen."),
            bp("Adam optimiser, lr=1×10⁻⁴, weight_decay=1×10⁻⁴."),
            bp("v2: ReduceLROnPlateau (×0.5, patience 3); batch=4, epochs=20/20."),
            bp("v3: OneCycleLR (max_lr=5×10⁻⁴, 10% warmup, cosine); batch=8, epochs=30/30."),
            bp("Early stopping patience=5 (v2), 3 (v3). Grad clip max_norm=1.0."),
        ],
        [
            p(f"{b('Loss functions:')}"),
            bp(f"{b('GMM v2:')} L1(warp_mask, cloth_region) + 0.01·‖θ‖₂"),
            bp(f"{b('GMM v3:')} L1(mask) + 0.5·L1(appearance) + 0.001·TV(flow)"),
            bp(f"{b('TOM:')} L1 + 0.25·VGG_perceptual + 0.5·(1−SSIM) + 0.01·TV(mask)"),
            sp(2),
            p(f"{b('v4:')} No training. Official weights loaded; inference only via "
              "two forward passes: tocg → SPADEGenerator."),
        ],
    ), sp(3),
]

# ─── §5 Metrics & Results ────────────────────────────────────────────────────
story += [
    sec("5.  Evaluation Metrics, Results, and Benchmarks"), sp(3),
    p(f"{b('SSIM')} (0–1 ↑) measures luminance, contrast, and structural similarity "
      "— closely aligned with human perception. "
      f"{b('PSNR')} (dB ↑) measures pixel-level fidelity. "
      f"{b('MSE')} (↓) is raw squared pixel error. "
      "Virtual try-on is a synthesis task; classification metrics (accuracy, F1, mAP, "
      "IoU) do not apply — SSIM/PSNR are the field-standard paired metrics; FID "
      "is standard for unpaired quality. All metrics computed in [0,1] range."),
    sp(3),
    tbl(
        [["Method","SSIM ↑","PSNR ↑","MSE ↓","Setting","Notes"],
         ["v2 — CP-VTON (CPU, buggy TPS)","0.12","~14 dB","High","Paired","Buggy warp + underfitted"],
         ["v3 — Dense Flow UNet (Colab)","0.12","~14 dB","High","Paired","Fixed bugs; too little compute"],
         ["Agnostic image (no model)","0.45","~18 dB","Med","Paired","Lower bound baseline"],
         ["v4 HR-VITON pretrained","0.85","~27 dB","Low","Paired","7× gain over v2/v3"],
         ["v4 HR-VITON pretrained","~0.70","~22 dB","Med","Unpaired","True try-on; GT shows diff. cloth"],
         ["Published HR-VITON (full dataset)","0.844","27.1 dB","—","Paired","Lee et al., ECCV 2022"]],
        [W*0.30, W*0.08, W*0.09, W*0.07, W*0.10, W*0.31],
        row_colors=[(2,LGREY),(4,LIGHT),(6,LGREY)],
    ),
    sp(3),
    p(f"{b('Analysis:')} v2 and v3 both plateau at SSIM ≈ 0.12 — demonstrating that "
      "training a competitive try-on model from scratch demands the full dataset, "
      "hundreds of GPU-hours, and a discriminator — well beyond course-project scope. "
      "v4 immediately achieves SSIM = 0.85 with pretrained weights, closely matching "
      "published results and confirming correct inference. "
      f"{b('Paired vs. unpaired:')} Paired inference uses the same garment shown in "
      "the GT photo, enabling exact SSIM/PSNR measurement. Unpaired inference swaps "
      "in a different garment — the real try-on scenario — so lower SSIM is expected "
      "and correct, not a model failure."),
    sp(3),
]

# ─── §6 Comparison ───────────────────────────────────────────────────────────
story += [
    sec("6.  Comparison with Baseline and Alternative Approaches"), sp(3),
    two(
        [
            bp(f"{b('No-change baseline (agnostic):')}"
               " Person with clothing region blanked. SSIM ≈ 0.45. Lower bound."),
            bp(f"{b('Classical warp (v1):')}"
               " Parse-guided bounding-box resize + Gaussian feathering. "
               "Deterministic; no learning. SSIM ≈ 0.50–0.55."),
            bp(f"{b('v2 — CP-VTON:')}"
               " TPS warping + U-Net synthesis. SSIM ≈ 0.12. "
               "Worse than classical baseline due to buggy TPS + CPU underfitting."),
        ],
        [
            bp(f"{b('v3 — Dense Flow:')}"
               " All v2 bugs fixed; 5,000 pairs; GPU+AMP. "
               "Still SSIM ≈ 0.12 — compute bottleneck persists on Colab free tier."),
            bp(f"{b('v4 — HR-VITON pretrained:')}"
               " SSIM = 0.85 paired. 7× over v2/v3. "
               "Demonstrates the decisive value of large-scale pretraining."),
            bp(f"{b('Published state-of-the-art (ECCV 2022):')}"
               " SSIM 0.844 / PSNR 27.1 dB on full test set — our v4 matches."),
        ],
    ), sp(3),
]

# ─── §7 Challenges ───────────────────────────────────────────────────────────
story += [
    sec("7.  Challenges Encountered and Solutions Implemented"), sp(3),
    two(
        [
            p(f"{b('TPS kernel bug (v2) →')}"
              " Per-sample solve used wrong control-point matrix, causing degenerate warps. "
              f"{b('Fix:')} Replaced TPS with dense flow UNet — no linear solve needed."),
            sp(2),
            p(f"{b('CPU-only training (v2) →')}"
              " Hardcoded CPU meant 3–8 min/epoch; only 600 pairs feasible. "
              f"{b('Fix:')} Auto GPU detection + AMP (FP16) in v3; 10× faster."),
            sp(2),
            p(f"{b('Missing VGG normalisation (v2) →')}"
              " VGG received raw [−1,1] tensors instead of ImageNet range; "
              "perceptual loss was meaningless. "
              f"{b('Fix:')} Registered normalisation buffer before every VGG pass."),
        ],
        [
            p(f"{b('SSIM ≈ 0.12 despite fixes (v3) →')}"
              " Even with all bugs resolved, from-scratch training on Colab "
              "cannot match models trained on 11k+ pairs for many GPU-days. "
              f"{b('Fix:')} Pivot to official pretrained HR-VITON (v4); SSIM → 0.85."),
            sp(2),
            p(f"{b('Checkpoint key mismatch (v4) →')}"
              " SPADEGenerator weights used legacy key names ('ace', '.Spade'). "
              f"{b('Fix:')} String replacement on keys during load_state_dict."),
            sp(2),
            p(f"{b('Occlusion / garment bleeding (v4) →')}"
              " Warped cloth overlapped face and arms. "
              f"{b('Fix:')} remove_overlap() subtracts warped mask from "
              "body-part segmentation regions at inference time."),
        ],
    ),
    sp(5), hr(),
    p("MSDS Computer Vision 462 · Spring 2026 · Enhanced Virtual Try-On", FOOT),
]

doc.build(story)
print(f"Saved → {OUTPUT}")
