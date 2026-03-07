# AI Agent Prompt — A0 Academic Poster Design

> **Role:** You are an expert academic poster designer and Figma specialist. Your task is to design a publication-quality A0 research poster using the Figma design tool (or Figma-compatible extension). You will be given the full content, figures, and a reference NeurIPS paper for visual inspiration.

---

## 1. POSTER SPECIFICATIONS

- **Size:** 33 inches (width) × 46 inches (height) — A0 portrait orientation
- **Resolution:** 300 DPI for print-ready output
- **Format:** Export as PDF (vector) and PNG (raster backup)
- **Color mode:** CMYK-safe palette; verify all colors render correctly in print

---

## 2. HEADER BLOCK (Top Strip)

Design a prominent header strip spanning the full poster width containing:

| Field | Value |
|---|---|
| **Title** | Computational and Deep Learning Strategies in Quantitative Finance for Robust Hedging under Market Frictions |
| **Subtitle** | MA-500B Thesis Mid Semester Evaluation and Poster Presentation |
| **Student Name** | Pratham Kailasiya |
| **Enrollment No.** | 21323030 |
| **Programme** | BS-MS Mathematics and Computing |
| **Thesis Supervisor** | Prof. Aditi Gangopadhyay |
| **Department** | Department of Mathematics, Indian Institute of Technology Roorkee (IIT Roorkee) |

- Include the **IIT Roorkee institutional logo** (top-left or top-right).
- Use a clean sans-serif font (e.g., Inter, Helvetica Neue, or Roboto) for the title; keep body text legible at poster viewing distance (~1.5m).

---

## 3. CONTENT STRUCTURE — 10 SECTIONS

Arrange the poster content in a **multi-column layout** (2–3 columns recommended for A0 portrait) following these 10 sections. The detailed content is provided in the attached files: `comprehensive_report.tex`, `cr_part1.tex`, `cr_part2.tex`, `cr_part3.tex`, `cr_part4.tex`. Use `poster_report.tex` as the condensed version for space-constrained sections.

### Section 1: Introduction
- Problem statement, market frictions, motivation for deep learning in hedging
- Keep concise: 3–5 sentences + 1 key equation (P&L definition)

### Section 2: Background and Literature Review
- Classical hedging theory, deep hedging framework, delta bounding
- Brief bullet points referencing key works (Buehler et al. 2019, Kozyra 2019)

### Section 3: Research Contributions
- Overview of 3 novel approaches: **W-DRO-T**, **3SCH**, **RSE**
- Use a visual diagram or icon set to distinguish the three contributions

### Section 4: Mathematical Framework
- Heston model, Euler–Maruyama discretisation, CVaR and Entropic risk definitions
- Include 2–3 key equations; use colored equation boxes for emphasis

### Section 5: Base and Novel Hedging Architectures
- LSTM, Transformer, Signature-based hedger (brief)
- RSE: ensemble equation + affinity matrix table
- 3SCH: mixed loss + annealing schedule
- W-DRO-T: DRO gradient penalty equation + training protocol
- **Figures to include:** Architecture diagrams (Figures 8–12), Training pipelines (Figures 15–17)

### Section 6: Experimental Setup and Statistical Methodology
- Heston parameters table, dataset sizes, training protocol summary
- 10-seed protocol, bootstrap CIs, Holm–Bonferroni correction
- **Figure:** Hyperparameter configuration table

### Section 7: Experimental Results
- Main results table (CVaR, Std, Entropic, Volume, CV%)
- Statistical comparison table (paired tests, Cohen's d)
- **Figures to include:** CVaR comparison with CIs (Fig 18), Multi-metric comparison (Fig 19), Seed consistency (Fig 20), Significance heatmap (Fig 21)

### Section 8: Real Market Validation and Crisis Testing
- SPY and NIFTY CVaR tables across crisis scenarios
- Crisis robustness ratios, P&L volatility analysis
- **Figures to include:** SPY crisis stress testing (Fig 23), NIFTY crisis (Fig 24), Crisis degradation ratio (Fig 25), P&L volatility (Fig 26)

### Section 9: Economic, Regulatory, and Governance Implications
- Capital requirement analysis, transaction cost efficiency, hedge accounting
- Model selection guidelines, governance framework, regulatory compliance
- **Figures to include:** Capital requirement analysis (Fig 29), Pareto frontier (Fig 32), Hedge accounting workflow (Fig 30)

### Section 10: Discussion, Limitations, and Future Work
- When complex vs simple models provide value
- Key limitations (simulation gap, single-asset, data constraints)
- Future directions: online hedging, multi-asset, market impact, rough path signatures

---

## 4. FIGURES

You will be provided with **34 publication-ready figures** (PDF and PNG formats) from the `figures/` directory. Integrate them into the poster following the section assignments above. Key figures to prioritize for visual impact:

| Priority | Figure | Description |
|---|---|---|
| **Must include** | Fig 18 | CVaR comparison with bootstrap CIs |
| **Must include** | Fig 19 | Multi-metric risk comparison (3-panel) |
| **Must include** | Fig 23 | SPY crisis stress testing |
| **Must include** | Fig 29 | Capital requirement analysis (2-panel) |
| **Must include** | Fig 10 | RSE architecture diagram |
| **Must include** | Fig 32 | Pareto frontier: risk vs cost |
| High | Fig 1 | Heston stock paths and variance |
| High | Fig 20 | Seed-by-seed CVaR consistency |
| High | Fig 21 | Statistical significance heatmap |
| Medium | Fig 8–9 | LSTM and Transformer architecture |
| Medium | Fig 15–17 | Training pipeline diagrams |
| Medium | Fig 24 | NIFTY crisis stress testing |

- Scale figures proportionally; ensure axis labels remain legible at print size
- Use consistent figure borders/frames across the poster
- Place figure captions below each figure in smaller font

---

## 5. REFERENCES AND ACKNOWLEDGEMENT

### References
Include a compact references section at the bottom of the poster. Key citations:
- Buehler et al. (2019) — Deep Hedging
- Heston (1993) — Stochastic Volatility
- Blanchet & Murthy (2019) — Wasserstein DRO
- Rockafellar & Uryasev (2000) — CVaR Optimization
- Vaswani et al. (2017) — Transformer
- Bengio et al. (2009) — Curriculum Learning
- Dietterich (2000) — Ensemble Methods
- Esfahani & Kuhn (2018) — Data-Driven DRO
- Lyons (1998) — Rough Paths
- Kozyra (2019) — Curriculum Training for Deep Hedging

Use a numbered compact format: `[1] Author (Year). Title. Venue.`

### Acknowledgement
> The author thanks Prof. Aditi Gangopadhyay for supervision and guidance throughout this thesis work at IIT Roorkee.

---

## 6. GITHUB QR CODE

- Generate a QR code linking to: **https://github.com/prathamkailasiya/deep-hedging/**
- Place it in the **bottom-right corner** of the poster
- Size: approximately 2×2 inches
- Label below: `GitHub Repository` with the URL in small text
- Ensure the QR code has sufficient quiet zone and contrast for scanning

---

## 7. VISUAL DESIGN GUIDELINES — NeurIPS INSPIRATION

You will be provided with a reference NeurIPS conference paper/poster. Use it as **visual inspiration** for:

- **Color palette:** Use a professional, muted academic palette. Suggested primary: deep blue (#1565C0), accent: orange (#E65100), purple (#7B1FA2), green (#2E7D32). Neutral backgrounds: white/light grey (#F5F5F5).
- **Typography hierarchy:** Clear distinction between section headers (bold, 28–36pt), subsection headers (semi-bold, 22–26pt), body text (regular, 16–20pt), captions (12–14pt).
- **Layout grid:** Use a consistent column grid with equal gutters (20–30px at design scale). Align all elements to the grid.
- **Section separators:** Use subtle horizontal rules or colored section header bars (matching the NeurIPS style of clean separation without heavy borders).
- **Equation formatting:** Render equations in LaTeX-quality math font; use light-colored background boxes for key equations.
- **Figure placement:** Figures should be embedded within their relevant sections, not clustered separately. Use consistent padding and alignment.
- **White space:** Maintain generous margins and inter-section spacing. Do not overcrowd — readability at 1.5m distance is critical.
- **Flow:** Content should flow top-to-bottom, left-to-right in columns. Use numbered sections and visual cues (arrows, icons) to guide the reader.

---

## 8. DESIGN TOOL INSTRUCTIONS

- Design in **Figma** (or Figma-compatible tool/extension)
- Set up the canvas at **33 × 46 inches** (2376 × 3312 px at 72 DPI; scale to 9900 × 13800 px at 300 DPI for print)
- Use **Auto Layout** for consistent spacing
- Group elements by section for easy repositioning
- Export final poster as:
  - **PDF** (vector, for print)
  - **PNG** (300 DPI, for digital sharing)

---

## 9. INPUT FILES PROVIDED

You will receive the following files:

| File | Purpose |
|---|---|
| `comprehensive_report.tex` | Main LaTeX file (includes cr_part1-4) |
| `cr_part1.tex` | Sections 1–4: Introduction, Background, Contributions, Math Framework |
| `cr_part2.tex` | Section 5: Base and Novel Hedging Architectures (LSTM, Transformer, RSE, 3SCH, W-DRO-T) |
| `cr_part3.tex` | Sections 6–8: Experimental Setup, Results, Market Validation |
| `cr_part4.tex` | Sections 9–10: Economic/Regulatory/Governance, Discussion/Limitations/Future Work + Appendices |
| `poster_report.tex` | Condensed poster-oriented summary (use for space-constrained areas) |
| `figures/` directory | 34 figures in PDF and PNG format |
| NeurIPS reference paper | Visual layout and design inspiration |

---

## 10. QUALITY CHECKLIST

Before finalizing, verify:

- [ ] All 10 sections are present and properly labeled
- [ ] Header contains: Name, Enrollment No., Title, Supervisor, Department, IIT Roorkee logo
- [ ] At least 6 key figures are included and legible at print size
- [ ] Key equations are rendered in math font with colored boxes
- [ ] Main results table is prominently placed
- [ ] References section is complete (10+ citations)
- [ ] Acknowledgement is included
- [ ] GitHub QR code is scannable and correctly placed
- [ ] Color palette is consistent and CMYK-safe
- [ ] Typography hierarchy is clear and readable at 1.5m
- [ ] Poster dimensions are exactly 33 × 46 inches
- [ ] No text smaller than 12pt at final print size
- [ ] White space is sufficient — poster does not feel cluttered
- [ ] Visual style is inspired by the provided NeurIPS reference
