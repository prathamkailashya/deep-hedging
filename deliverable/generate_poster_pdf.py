#!/usr/bin/env python3
"""Generate A0 landscape poster PDF matching the Pencil design."""

from reportlab.lib.pagesizes import A0
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, white, Color
from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

# A0 landscape
PAGE_W, PAGE_H = A0[1], A0[0]  # 1189mm x 841mm

# Colors matching Pencil design
PURPLE_DARK = HexColor("#6A1B9A")
PURPLE_MID = HexColor("#9C27B0")
PURPLE_LIGHT = HexColor("#AB47BC")
PURPLE_HEADER = HexColor("#7B1FA2")
PURPLE_BG = HexColor("#F3E5F5")
ORANGE = HexColor("#E65100")
ORANGE_BG = HexColor("#FFF3E0")
BLUE = HexColor("#1565C0")
BLUE_BG = HexColor("#E3F2FD")
RED = HexColor("#C62828")
RED_BG = HexColor("#FFEBEE")
GREEN = HexColor("#2E7D32")
GREEN_BG = HexColor("#E8F5E9")
GRAY_BG = HexColor("#F0EEF0")
WHITE = HexColor("#FFFFFF")
FAFAFA = HexColor("#FAFAFA")
TEXT_DARK = HexColor("#333333")
TEXT_MED = HexColor("#666666")
TEXT_LIGHT = HexColor("#999999")
BORDER = HexColor("#E0E0E0")
BAR_GRAY = HexColor("#9E9E9E")
BAR_LGRAY = HexColor("#BDBDBD")
ACCENT_LINE = HexColor("#E1BEE7")

# Layout constants
MARGIN = 36 * mm
HEADER_H = 85 * mm
COL_GAP = 18 * mm
ROW_GAP = 18 * mm
CONTENT_TOP = PAGE_H - HEADER_H - 20 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
COL_W = (CONTENT_W - 2 * COL_GAP) / 3
BLOCK_CORNER = 4 * mm
BLOCK_PAD = 12 * mm
HEADER_BLOCK_H = 18 * mm


def draw_rounded_rect(c, x, y, w, h, r, fill=None, stroke=None, stroke_w=0.5):
    """Draw a rounded rectangle."""
    c.saveState()
    p = c.beginPath()
    p.roundRect(x, y, w, h, r)
    if fill:
        c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_w)
    if fill and stroke:
        c.drawPath(p, fill=1, stroke=1)
    elif fill:
        c.drawPath(p, fill=1, stroke=0)
    elif stroke:
        c.drawPath(p, fill=0, stroke=1)
    c.restoreState()


def draw_block_header(c, x, y, w, title):
    """Draw purple block header with title."""
    draw_rounded_rect(c, x, y - HEADER_BLOCK_H, w, HEADER_BLOCK_H,
                       BLOCK_CORNER, fill=PURPLE_HEADER)
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x + 12 * mm, y - HEADER_BLOCK_H + 5.5 * mm, title)
    c.restoreState()
    return y - HEADER_BLOCK_H


def draw_block_body_bg(c, x, y_top, w, h):
    """Draw white body background for a block."""
    draw_rounded_rect(c, x, y_top - h, w, h, BLOCK_CORNER,
                       fill=WHITE, stroke=BORDER, stroke_w=0.5)


def draw_text(c, x, y, text, font="Helvetica", size=10, color=TEXT_DARK, max_w=None):
    """Draw text, return new y position."""
    c.saveState()
    c.setFillColor(color)
    c.setFont(font, size)
    if max_w:
        # Simple word wrap
        words = text.split()
        lines = []
        current = ""
        for w in words:
            test = current + " " + w if current else w
            if c.stringWidth(test, font, size) <= max_w:
                current = test
            else:
                if current:
                    lines.append(current)
                current = w
        if current:
            lines.append(current)
        for line in lines:
            c.drawString(x, y, line)
            y -= size * 1.4
    else:
        c.drawString(x, y, text)
        y -= size * 1.4
    c.restoreState()
    return y


def draw_bullet(c, x, y, text, font="Helvetica", size=10, color=TEXT_DARK,
                bullet_color=None, max_w=None, bullet="▸"):
    """Draw a bullet point."""
    c.saveState()
    c.setFillColor(bullet_color or color)
    c.setFont(font, size)
    c.drawString(x, y, bullet)
    bw = c.stringWidth(bullet + " ", font, size)
    c.restoreState()
    return draw_text(c, x + bw, y, text, font, size, color, max_w=max_w - bw if max_w else None)


def draw_colored_pill(c, x, y, w, h, text, bg_color, text_color, font_size=9):
    """Draw a small colored pill/badge."""
    draw_rounded_rect(c, x, y, w, h, 2 * mm, fill=bg_color)
    c.saveState()
    c.setFillColor(text_color)
    c.setFont("Helvetica-Bold", font_size)
    tw = c.stringWidth(text, "Helvetica-Bold", font_size)
    c.drawString(x + (w - tw) / 2, y + (h - font_size) / 2 + 1, text)
    c.restoreState()


def draw_bar(c, x, y, w, h, color, label=None, value=None, label_w=45*mm):
    """Draw a horizontal bar with optional label and value."""
    if label:
        c.saveState()
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(color)
        c.drawString(x, y + 1.5, label)
        c.restoreState()
    bar_x = x + label_w
    bar_w = w - label_w
    if value:
        bar_w -= 30 * mm
    draw_rounded_rect(c, bar_x, y, bar_w, h, 1.5 * mm, fill=color)
    if value:
        c.saveState()
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(color)
        c.drawString(bar_x + bar_w + 3 * mm, y + 1.5, value)
        c.restoreState()


def draw_gap_row(c, x, y, w, gap_text, solution_text, gap_bg, gap_color, sol_bg, sol_color):
    """Draw a gap → solution row."""
    pill_w = (w - 20 * mm) / 2
    pill_h = 8 * mm
    draw_rounded_rect(c, x, y, pill_w, pill_h, 2 * mm, fill=gap_bg)
    c.saveState()
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(gap_color)
    c.drawString(x + 4 * mm, y + 2, gap_text)
    c.restoreState()
    # Arrow
    c.saveState()
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(PURPLE_HEADER)
    c.drawString(x + pill_w + 4 * mm, y + 1, "→")
    c.restoreState()
    # Solution pill
    draw_rounded_rect(c, x + pill_w + 20 * mm, y, pill_w, pill_h, 2 * mm, fill=sol_bg)
    c.saveState()
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(sol_color)
    tw = c.stringWidth(solution_text, "Helvetica-Bold", 10)
    c.drawString(x + pill_w + 20 * mm + (pill_w - tw) / 2, y + 2, solution_text)
    c.restoreState()


def draw_check_item(c, x, y, text, max_w):
    """Draw a checkmark item."""
    c.saveState()
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(GREEN)
    c.drawString(x, y, "✓")
    c.restoreState()
    return draw_text(c, x + 8 * mm, y, text, "Helvetica", 10, TEXT_DARK, max_w=max_w - 8 * mm)


def draw_number_circle(c, x, y, num, r=4*mm):
    """Draw a numbered circle."""
    c.saveState()
    c.setFillColor(PURPLE_HEADER)
    c.circle(x + r, y + r, r, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 9)
    tw = c.stringWidth(str(num), "Helvetica-Bold", 9)
    c.drawString(x + r - tw / 2, y + r - 3, str(num))
    c.restoreState()


def generate_poster():
    output_path = "/Users/prathamkailasiya/Desktop/Sem9 Desktop/Thesis/deep_hedging/deliverable/A0_poster.pdf"
    c = canvas.Canvas(output_path, pagesize=(PAGE_W, PAGE_H))

    # =========================================================================
    # BACKGROUND
    # =========================================================================
    c.setFillColor(GRAY_BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    # =========================================================================
    # HEADER BAR (purple gradient approximation)
    # =========================================================================
    header_y = PAGE_H - HEADER_H
    # Gradient: draw multiple thin strips
    strips = 60
    for i in range(strips):
        frac = i / strips
        r = PURPLE_DARK.red * (1 - frac) + PURPLE_LIGHT.red * frac
        g = PURPLE_DARK.green * (1 - frac) + PURPLE_LIGHT.green * frac
        b = PURPLE_DARK.blue * (1 - frac) + PURPLE_LIGHT.blue * frac
        strip_color = Color(r, g, b)
        sy = header_y + (HEADER_H / strips) * (strips - 1 - i)
        c.setFillColor(strip_color)
        c.rect(0, sy, PAGE_W, HEADER_H / strips + 1, fill=1, stroke=0)

    # Accent line at bottom
    c.setFillColor(ACCENT_LINE)
    c.rect(0, header_y, PAGE_W, 2, fill=1, stroke=0)

    # Title
    c.saveState()
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 40)
    c.drawString(MARGIN, header_y + 50 * mm,
                 "Deep Hedging with Distributionally Robust, Curriculum,")
    c.drawString(MARGIN, header_y + 36 * mm,
                 "and Regime-Adaptive Neural Networks")
    # Author
    c.setFont("Helvetica", 16)
    c.setFillColor(HexColor("#F3E5F5"))
    c.drawString(MARGIN, header_y + 18 * mm,
                 "Pratham Kailasiya  ·  Department of Management Studies, "
                 "Indian Institute of Technology Roorkee  ·  pratham@ms.iitr.ac.in")
    c.restoreState()

    # Logo area (top right)
    logo_x = PAGE_W - MARGIN - 55 * mm
    logo_y = header_y + 25 * mm
    c.saveState()
    c.setFillColor(Color(1, 1, 1, 0.15))
    p = c.beginPath()
    p.roundRect(logo_x, logo_y, 50 * mm, 40 * mm, 4 * mm)
    c.drawPath(p, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(logo_x + 25 * mm, logo_y + 24 * mm, "IIT")
    c.drawCentredString(logo_x + 25 * mm, logo_y + 10 * mm, "ROORKEE")
    c.setFont("Helvetica", 12)
    c.setFillColor(ACCENT_LINE)
    c.drawCentredString(logo_x + 25 * mm, logo_y + 2 * mm, "DoMS")
    c.restoreState()

    # =========================================================================
    # CONTENT GRID - 3 columns × 3 rows
    # =========================================================================
    col_x = [MARGIN, MARGIN + COL_W + COL_GAP, MARGIN + 2 * (COL_W + COL_GAP)]
    body_w = COL_W - 2 * BLOCK_PAD  # usable width inside block body

    # ===================== COLUMN 1 =====================
    cx = col_x[0]
    cy = CONTENT_TOP

    # --- BLOCK 1: Introduction & Motivation ---
    bh1_y = draw_block_header(c, cx, cy, COL_W, "INTRODUCTION & MOTIVATION")
    b1_h = 130 * mm
    b1_body_top = bh1_y
    draw_block_body_bg(c, cx, b1_body_top, COL_W, b1_h)
    ty = b1_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty,
                   "Classical delta hedging assumes known dynamics and continuous "
                   "rebalancing — both violated in practice. Deep hedging learns "
                   "strategies directly from simulated paths, but three critical gaps remain:",
                   "Helvetica", 10, TEXT_DARK, max_w=body_w)
    ty -= 3 * mm
    ty = draw_bullet(c, tx, ty, "Distributional shift: trained on Heston → fails under regime change",
                     size=9.5, bullet_color=HexColor("#E91E63"), max_w=body_w)
    ty = draw_bullet(c, tx, ty, "Training instability: CVaR → entropic jump destabilises optimisation",
                     size=9.5, bullet_color=HexColor("#FF9800"), max_w=body_w)
    ty = draw_bullet(c, tx, ty, "Regime dependence: single network cannot adapt across market states",
                     size=9.5, bullet_color=BLUE, max_w=body_w)
    ty -= 2 * mm

    # Solutions
    c.saveState()
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(PURPLE_HEADER)
    c.drawString(tx, ty, "→  W-DRO-T: Wasserstein DRO with gradient-penalty dual")
    ty -= 5 * mm
    c.setFillColor(ORANGE)
    c.drawString(tx, ty, "→  3SCH: Three-stage curriculum homotopy")
    ty -= 5 * mm
    c.setFillColor(BLUE)
    c.drawString(tx, ty, "→  RSE: Regime-switching ensemble with gating")
    c.restoreState()
    ty -= 6 * mm

    # Protocol
    c.saveState()
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(TEXT_MED)
    ty = draw_text(c, tx, ty,
                   "Protocol: 80K Heston paths · N=30 steps · 10 seeds (42...942) · "
                   "10K bootstrap · Holm–Bonferroni (α=0.05) · Cohen's d · SPY + NIFTY validation",
                   "Helvetica-Oblique", 9, TEXT_MED, max_w=body_w)
    c.restoreState()
    ty -= 4 * mm

    # Gap → Solution diagram
    draw_rounded_rect(c, tx, ty - 30 * mm, body_w, 30 * mm, 3 * mm, fill=FAFAFA)
    gy = ty - 4 * mm
    draw_gap_row(c, tx + 4 * mm, gy, body_w - 8 * mm,
                 "Gap 1: Distributional Shift", "W-DRO-T", RED_BG, RED, PURPLE_BG, PURPLE_DARK)
    gy -= 10 * mm
    draw_gap_row(c, tx + 4 * mm, gy, body_w - 8 * mm,
                 "Gap 2: Training Instability", "3SCH", ORANGE_BG, ORANGE, ORANGE_BG, ORANGE)
    gy -= 10 * mm
    draw_gap_row(c, tx + 4 * mm, gy, body_w - 8 * mm,
                 "Gap 3: Regime Dependence", "RSE", BLUE_BG, BLUE, BLUE_BG, BLUE)

    # --- BLOCK 4: Key Results ---
    cy = b1_body_top - b1_h - ROW_GAP
    bh4_y = draw_block_header(c, cx, cy, COL_W, "KEY RESULTS & STATISTICS")
    b4_h = 155 * mm
    b4_body_top = bh4_y
    draw_block_body_bg(c, cx, b4_body_top, COL_W, b4_h)
    ty = b4_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty, "Simulated CVaR₉₅ Comparison (10 Seeds)",
                   "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 2 * mm

    # Bar chart
    chart_h = 38 * mm
    draw_rounded_rect(c, tx, ty - chart_h, body_w, chart_h, 3 * mm, fill=FAFAFA)
    by = ty - 5 * mm
    bar_h = 4.5 * mm
    bars = [
        ("RSE", BLUE, "3.109", 0.92),
        ("LSTM", BAR_GRAY, "3.215", 0.95),
        ("3SCH", HexColor("#FF9800"), "3.219", 0.955),
        ("W-DRO-T", PURPLE_MID, "3.227", 0.958),
        ("Trans.", BAR_LGRAY, "3.234", 0.96),
    ]
    for label, color, val, frac in bars:
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        c.drawString(tx + 3 * mm, by + 0.5 * mm, label)
        c.restoreState()
        bw = (body_w - 55 * mm) * frac
        draw_rounded_rect(c, tx + 35 * mm, by, bw, bar_h, 1.5 * mm, fill=color)
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        c.drawString(tx + 35 * mm + bw + 2 * mm, by + 0.5 * mm, val)
        c.restoreState()
        by -= 6.5 * mm

    ty = ty - chart_h - 4 * mm

    # Results table
    ty = draw_text(c, tx, ty, "Model      CVaR₉₅           95% CI              p vs LSTM   Cohen's d",
                   "Helvetica-Bold", 8, PURPLE_DARK)
    draw_rounded_rect(c, tx, ty - 2, body_w, 0.5, 0, fill=PURPLE_BG)
    ty -= 2 * mm

    table_data = [
        ("RSE", "3.109 ± 0.010", "[3.104, 3.115]", "< 0.0001", "−7.41", BLUE),
        ("LSTM", "3.215 ± 0.015", "[3.207, 3.225]", "—", "—", TEXT_MED),
        ("3SCH", "3.219 ± 0.016", "[3.210, 3.229]", "0.58", "+0.19", ORANGE),
        ("W-DRO-T", "3.227 ± 0.011", "[3.220, 3.234]", "0.10", "+0.60", PURPLE_DARK),
    ]
    for i, (model, cvar, ci, pval, d, color) in enumerate(table_data):
        bg = FAFAFA if i % 2 else WHITE
        draw_rounded_rect(c, tx, ty - 5 * mm, body_w, 5 * mm, 0, fill=bg)
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        c.drawString(tx + 2 * mm, ty - 4 * mm, model)
        c.setFont("Helvetica", 8)
        c.setFillColor(TEXT_DARK)
        c.drawString(tx + 32 * mm, ty - 4 * mm, cvar)
        c.drawString(tx + 72 * mm, ty - 4 * mm, ci)
        if pval == "< 0.0001":
            c.setFillColor(GREEN)
            c.setFont("Helvetica-Bold", 8)
        c.drawString(tx + 115 * mm, ty - 4 * mm, pval)
        c.setFont("Helvetica-Bold" if d.startswith("−") else "Helvetica", 8)
        c.setFillColor(BLUE if d.startswith("−") else TEXT_DARK)
        c.drawString(tx + 147 * mm, ty - 4 * mm, d)
        c.restoreState()
        ty -= 5 * mm

    ty -= 3 * mm
    ty = draw_text(c, tx, ty,
                   "RSE achieves lowest simulated CVaR₉₅ (3.109 ± 0.010) with "
                   "non-overlapping CI vs all baselines (p < 0.0001, |d| = 7.41). "
                   "10 seeds × 10K bootstrap.",
                   "Helvetica-Oblique", 8.5, TEXT_MED, max_w=body_w)

    # --- BLOCK 7: Experimental Pipeline ---
    cy = b4_body_top - b4_h - ROW_GAP
    bh7_y = draw_block_header(c, cx, cy, COL_W, "EXPERIMENTAL PIPELINE")
    b7_h = 105 * mm
    b7_body_top = bh7_y
    draw_block_body_bg(c, cx, b7_body_top, COL_W, b7_h)
    ty = b7_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    # Pipeline flow
    steps = [
        ("Heston Sim", "80K paths", PURPLE_BG, PURPLE_DARK),
        ("Features", "5-dim Iₖ", ORANGE_BG, ORANGE),
        ("Training", "CVaR→Ent", BLUE_BG, BLUE),
        ("7 Models", "×10 Seeds", PURPLE_BG, PURPLE_DARK),
        ("Backtest", "SPY+NIFTY", GREEN_BG, GREEN),
    ]
    step_w = 28 * mm
    step_h = 14 * mm
    gap_arrow = 6 * mm
    sx = tx + 2 * mm
    for i, (name, desc, bg, clr) in enumerate(steps):
        draw_rounded_rect(c, sx, ty - step_h, step_w, step_h, 3 * mm, fill=bg)
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(clr)
        c.drawCentredString(sx + step_w / 2, ty - 5 * mm, name)
        c.setFont("Helvetica", 7)
        c.setFillColor(TEXT_MED)
        c.drawCentredString(sx + step_w / 2, ty - 10 * mm, desc)
        c.restoreState()
        sx += step_w
        if i < len(steps) - 1:
            c.saveState()
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(PURPLE_HEADER)
            c.drawString(sx + 1 * mm, ty - 9 * mm, "→")
            c.restoreState()
            sx += gap_arrow

    ty -= step_h + 6 * mm

    # Protocol params
    params = [
        ("Paths: 80K Heston", "Seeds: 42 ... 942"),
        ("Steps: N = 30", "Epochs: 50 + 30"),
        ("Split: 50/10/20K", "LR: 10⁻³ → 10⁻⁴"),
        ("Bootstrap: 10K", "Holm–Bonferroni α = 0.05"),
    ]
    draw_rounded_rect(c, tx, ty - 26 * mm, body_w, 26 * mm, 3 * mm, fill=FAFAFA)
    py = ty - 3 * mm
    for left, right in params:
        c.saveState()
        c.setFont("Helvetica", 9)
        c.setFillColor(TEXT_DARK)
        c.drawString(tx + 4 * mm, py, left)
        c.drawString(tx + body_w / 2 + 4 * mm, py, right)
        c.restoreState()
        py -= 5.5 * mm

    ty = ty - 26 * mm - 6 * mm

    # Model cards
    ty = draw_text(c, tx, ty, "7 Models Compared", "Helvetica-Bold", 11, PURPLE_HEADER)
    ty -= 1 * mm
    model_cards = [
        [("W-DRO-T", "Transformer", PURPLE_BG, PURPLE_DARK),
         ("3SCH", "LSTM", ORANGE_BG, ORANGE),
         ("RSE", "3×LSTM", BLUE_BG, BLUE)],
        [("LSTM", "Baseline", FAFAFA, TEXT_MED),
         ("Transformer", "Baseline", FAFAFA, TEXT_MED),
         ("+2 More", "Ablations", FAFAFA, TEXT_LIGHT)],
    ]
    card_w = (body_w - 6 * mm) / 3
    card_h = 10 * mm
    for row in model_cards:
        mcx = tx
        for name, desc, bg, clr in row:
            draw_rounded_rect(c, mcx, ty - card_h, card_w, card_h, 3 * mm, fill=bg)
            c.saveState()
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(clr)
            c.drawCentredString(mcx + card_w / 2, ty - 4 * mm, name)
            c.setFont("Helvetica", 7)
            c.setFillColor(TEXT_MED)
            c.drawCentredString(mcx + card_w / 2, ty - 8.5 * mm, desc)
            c.restoreState()
            mcx += card_w + 3 * mm
        ty -= card_h + 3 * mm

    # ===================== COLUMN 2 =====================
    cx = col_x[1]
    cy = CONTENT_TOP

    # --- BLOCK 2: Mathematical Formulation ---
    bh2_y = draw_block_header(c, cx, cy, COL_W, "MATHEMATICAL FORMULATION")
    b2_h = 185 * mm
    b2_body_top = bh2_y
    draw_block_body_bg(c, cx, b2_body_top, COL_W, b2_h)
    ty = b2_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty, "Heston Stochastic-Volatility Model",
                   "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 1 * mm
    eqs = [
        "dSₜ = rSₜ dt + √vₜ Sₜ dWₜˢ",
        "dvₜ = κ(θᵥ − vₜ) dt + ξ√vₜ dWₜᵛ",
        "dWˢ · dWᵛ = ρ dt",
    ]
    for eq in eqs:
        ty = draw_text(c, tx + 4 * mm, ty, eq, "Helvetica", 10, TEXT_DARK)
    ty -= 2 * mm

    ty = draw_text(c, tx, ty, "Deep Hedging Objective", "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 1 * mm
    ty = draw_text(c, tx + 4 * mm, ty, "Π(δ) = −Z + Σₖ δₖΔSₖ − c_tc Σₖ|δₖ − δₖ₋₁|Sₖ",
                   "Helvetica", 10, TEXT_DARK)
    ty = draw_text(c, tx + 4 * mm, ty, "θ* = argmin_θ  ρ(Π(δᶿ))", "Helvetica", 10, TEXT_DARK)
    ty -= 2 * mm

    ty = draw_text(c, tx, ty, "Three Novel Risk Objectives", "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 2 * mm

    # Objective boxes
    objectives = [
        ("W-DRO-T — Wasserstein DRO dual:", PURPLE_DARK, PURPLE_BG,
         "L_DRO = E_P[ℓ(θ;X)] + ε · E_P[‖∇_X ℓ‖₂],  ε: 0 → ε_max"),
        ("3SCH — Three-stage curriculum homotopy:", ORANGE, ORANGE_BG,
         "L_mix(α) = α · CVaR₀.₉₅ + (1−α) · ρ_λ,  α: 0.8 → 0.2"),
        ("RSE — Regime-switching ensemble:", BLUE, BLUE_BG,
         "δₖᴿˢᴱ = Σₘ wₘ(rₖ) δₖ⁽ᵐ⁾,  wₘ = Σⱼ pⱼ[softmax(A/τ)]ⱼₘ"),
    ]
    for title, title_clr, bg_clr, formula in objectives:
        obj_h = 14 * mm
        draw_rounded_rect(c, tx, ty - obj_h, body_w, obj_h, 2 * mm, fill=bg_clr)
        c.saveState()
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(title_clr)
        c.drawString(tx + 4 * mm, ty - 5 * mm, title)
        c.setFont("Helvetica", 9)
        c.setFillColor(TEXT_DARK)
        c.drawString(tx + 4 * mm, ty - 11 * mm, formula)
        c.restoreState()
        ty -= obj_h + 3 * mm

    # --- BLOCK 5: Real Market Validation ---
    cy = b2_body_top - b2_h - ROW_GAP
    bh5_y = draw_block_header(c, cx, cy, COL_W, "REAL MARKET VALIDATION")
    b5_h = 200 * mm
    b5_body_top = bh5_y
    draw_block_body_bg(c, cx, b5_body_top, COL_W, b5_h)
    ty = b5_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty, "SPY Crisis Stress Testing — CVaR₉₅",
                   "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 2 * mm

    # Crisis bars
    crisis = [
        ("Normal Market", TEXT_DARK, [
            ("LSTM 20.3", BAR_GRAY, 0.20),
            ("W-DRO-T 12.0", PURPLE_MID, 0.12),
        ]),
        ("COVID-19", RED, [
            ("LSTM 86.8", BAR_GRAY, 0.87),
            ("W-DRO-T 46.9", PURPLE_MID, 0.47),
        ]),
        ("GFC-2008", ORANGE, [
            ("LSTM 100.9", BAR_GRAY, 1.0),
            ("W-DRO-T 62.3", PURPLE_MID, 0.62),
        ]),
    ]
    draw_rounded_rect(c, tx, ty - 62 * mm, body_w, 62 * mm, 3 * mm, fill=FAFAFA)
    cby = ty - 3 * mm
    for scenario, sc_color, bars_data in crisis:
        c.saveState()
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(sc_color)
        c.drawString(tx + 3 * mm, cby, scenario)
        c.restoreState()
        cby -= 5 * mm
        for lbl, clr, frac in bars_data:
            c.saveState()
            c.setFont("Helvetica", 7.5)
            c.setFillColor(clr)
            c.drawString(tx + 3 * mm, cby + 0.5 * mm, lbl)
            c.restoreState()
            bw = (body_w - 55 * mm) * frac
            draw_rounded_rect(c, tx + 42 * mm, cby, bw, 4 * mm, 1 * mm, fill=clr)
            cby -= 5.5 * mm
        cby -= 1 * mm

    ty -= 66 * mm

    # SPY findings
    ty = draw_text(c, tx, ty, "SPY (US, 5 bps cost)", "Helvetica-Bold", 11, PURPLE_DARK)
    spy_items = [
        "W-DRO-T dominates all scenarios",
        "Normal: −41% vs LSTM",
        "COVID-19: −46% vs LSTM",
        "GFC-2008: −38% vs LSTM",
        "Std P&L: 1.71 (vs 5.05, −66%)",
    ]
    for item in spy_items:
        ty = draw_bullet(c, tx, ty, item, size=9.5, max_w=body_w)
    ty -= 3 * mm

    # NIFTY findings
    ty = draw_text(c, tx, ty, "NIFTY 50 (India, 18 bps cost)", "Helvetica-Bold", 11, ORANGE)
    nifty_items = [
        "3SCH wins normal: CVaR = 622",
        "Transformer wins COVID: 1644",
        "W-DRO-T wins GFC: 2350",
        "Higher vol (0.90) amplifies cost impact",
    ]
    for item in nifty_items:
        ty = draw_bullet(c, tx, ty, item, size=9.5, max_w=body_w)

    # --- BLOCK 8: Key Contributions ---
    cy = b5_body_top - b5_h - ROW_GAP
    bh8_y = draw_block_header(c, cx, cy, COL_W, "KEY CONTRIBUTIONS")
    b8_h = 68 * mm
    b8_body_top = bh8_y
    draw_block_body_bg(c, cx, b8_body_top, COL_W, b8_h)
    ty = b8_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    contributions = [
        "W-DRO-T: Wasserstein gradient-norm penalty with 2nd-order SDPA gradients",
        "3SCH: Mixed-loss homotopy via Berge's maximum theorem",
        "RSE: Regime-gated ensemble with interpretable affinity A ∈ R⁴ˣ³",
        "Statistical rigour: 10 seeds, 10K bootstrap, Holm–Bonferroni corrections",
        "Market-specific deployment: US vs India evidence with crisis testing",
    ]
    for i, contrib in enumerate(contributions):
        draw_number_circle(c, tx, ty - 3.5 * mm, i + 1)
        ty = draw_text(c, tx + 12 * mm, ty, contrib, "Helvetica", 9.5, TEXT_DARK, max_w=body_w - 12 * mm)
        ty -= 1 * mm

    # ===================== COLUMN 3 =====================
    cx = col_x[2]
    cy = CONTENT_TOP

    # --- BLOCK 3: Method Overview & Novelty ---
    bh3_y = draw_block_header(c, cx, cy, COL_W, "METHOD OVERVIEW & NOVELTY")
    b3_h = 185 * mm
    b3_body_top = bh3_y
    draw_block_body_bg(c, cx, b3_body_top, COL_W, b3_h)
    ty = b3_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    novelty_sections = [
        ("W-DRO-T Novelty", PURPLE_DARK, [
            "Gradient-penalty dual avoids inner sup over Q",
            "ε-annealing stabilises early training",
            "Transformer with causal mask + 2nd-order SDPA",
        ]),
        ("3SCH Novelty", ORANGE, [
            "Mixed-loss convexity via Berge's maximum theorem",
            "Three-stage α-annealing: warm-up, transition, fine-tune",
            "Zero-cost rollback to LSTM baseline",
        ]),
        ("RSE Novelty", BLUE, [
            "Soft regime gating via learned affinity A ∈ R⁴ˣ³",
            "Regime features: realised vol, BiPV, returns, spread",
            "Variance reduction: Var[δᴿˢᴱ] < min_m Var[δ⁽ᵐ⁾]",
        ]),
        ("Statistical Rigour", TEXT_DARK, [
            "10,000 bootstrap resamples for 95% CIs",
            "Paired t-tests with Holm–Bonferroni (α = 0.05)",
            "Cohen's d effect size with magnitude classification",
        ]),
    ]
    for section_title, title_color, items in novelty_sections:
        ty = draw_text(c, tx, ty, section_title, "Helvetica-Bold", 11, title_color)
        for item in items:
            ty = draw_bullet(c, tx, ty, item, size=9.5, max_w=body_w)
        ty -= 3 * mm

    # --- BLOCK 6: Deployment & Regulatory ---
    cy = b3_body_top - b3_h - ROW_GAP
    bh6_y = draw_block_header(c, cx, cy, COL_W, "DEPLOYMENT & REGULATORY")
    b6_h = 185 * mm
    b6_body_top = bh6_y
    draw_block_body_bg(c, cx, b6_body_top, COL_W, b6_h)
    ty = b6_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty, "Basel III/IV Capital Savings", "Helvetica-Bold", 12, PURPLE_HEADER)
    ty -= 2 * mm

    # Capital bars
    cap_bars = [
        ("LSTM", BAR_GRAY, "$8.1M", 1.0),
        ("RSE", BLUE, "$6.8M", 0.84),
        ("3SCH", HexColor("#FF9800"), "$6.3M", 0.78),
        ("Trans.", BAR_LGRAY, "$5.8M", 0.72),
        ("W-DRO-T", PURPLE_MID, "$4.8M", 0.59),
    ]
    draw_rounded_rect(c, tx, ty - 35 * mm, body_w, 35 * mm, 3 * mm, fill=FAFAFA)
    cby = ty - 3 * mm
    for label, color, val, frac in cap_bars:
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        c.drawString(tx + 3 * mm, cby + 0.5 * mm, label)
        c.restoreState()
        bw = (body_w - 55 * mm) * frac
        draw_rounded_rect(c, tx + 35 * mm, cby, bw, 4 * mm, 1.5 * mm, fill=color)
        c.saveState()
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(color)
        c.drawString(tx + 35 * mm + bw + 2 * mm, cby + 0.5 * mm, val)
        c.restoreState()
        cby -= 6.5 * mm

    ty -= 39 * mm

    # Deployment guidance
    ty = draw_text(c, tx, ty, "Deployment Guidance", "Helvetica-Bold", 11, PURPLE_DARK)
    deploy_items = [
        "US capital efficiency: W-DRO-T (−41% capital)",
        "India normal market: 3SCH (best NIFTY CVaR)",
        "India crisis: Transformer (best COVID)",
        "Regime uncertainty: RSE (CV = 0.30%)",
        "$10B desk: $332M/yr freed capital",
    ]
    for item in deploy_items:
        ty = draw_bullet(c, tx, ty, item, size=9.5, max_w=body_w)
    ty -= 3 * mm

    # Regulatory checklist
    ty = draw_text(c, tx, ty, "Regulatory & Risk Compliance", "Helvetica-Bold", 11, PURPLE_DARK)
    ty -= 1 * mm
    checks = [
        "IFRS 9 / Ind-AS 109: 80–125% offset & R² ≥ 0.80",
        "FRTB IMA: W-DRO-T lower P&L volatility",
        "SR 11-7: RSE affinity matrix interpretable",
        "DRO degrades slower (3.92× vs 4.28×)",
    ]
    for check in checks:
        ty = draw_check_item(c, tx, ty, check, body_w)
        ty -= 1 * mm

    # --- BLOCK 9: Future Work & References ---
    cy = b6_body_top - b6_h - ROW_GAP
    bh9_y = draw_block_header(c, cx, cy, COL_W, "FUTURE WORK & REFERENCES")
    b9_h = 120 * mm
    b9_body_top = bh9_y
    draw_block_body_bg(c, cx, b9_body_top, COL_W, b9_h)
    ty = b9_body_top - BLOCK_PAD
    tx = cx + BLOCK_PAD

    ty = draw_text(c, tx, ty, "Research Roadmap", "Helvetica-Bold", 11, PURPLE_DARK)
    roadmap = [
        "Online adaptation for non-stationary markets",
        "Multi-asset portfolio extension",
        "Market impact models integration",
        "Cross-market transfer learning",
        "Hybrid 3SCH-DRO architecture",
    ]
    for item in roadmap:
        ty = draw_bullet(c, tx, ty, item, size=9.5, max_w=body_w)
    ty -= 3 * mm

    ty = draw_text(c, tx, ty, "Key References", "Helvetica-Bold", 11, PURPLE_DARK)
    refs = [
        "[1] Buehler et al. (2019) Deep Hedging, QF",
        "[2] Sinha et al. (2018) Certifying DRO, ICLR",
        "[3] Bengio et al. (2009) Curriculum Learning, ICML",
        "[4] Lakshminarayanan et al. (2017) Deep Ensembles",
        "[5] Rahimian & Mehrotra (2019) DRO Review",
    ]
    for ref in refs:
        ty = draw_text(c, tx, ty, ref, "Helvetica", 9, TEXT_MED, max_w=body_w)

    ty -= 3 * mm
    # Code link
    draw_rounded_rect(c, tx, ty - 8 * mm, body_w, 8 * mm, 2 * mm, fill=PURPLE_BG)
    c.saveState()
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(PURPLE_DARK)
    c.drawString(tx + 4 * mm, ty - 5.5 * mm, "📂  github.com/pratham-kailasiya/deep-hedging")
    c.restoreState()

    # =========================================================================
    # SAVE
    # =========================================================================
    c.save()
    print(f"✅ PDF saved to: {output_path}")
    print(f"   Dimensions: {PAGE_W/mm:.0f}mm × {PAGE_H/mm:.0f}mm (A0 landscape)")


if __name__ == "__main__":
    generate_poster()
