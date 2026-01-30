# Deep Hedging Academic Paper and Presentation

This directory contains publication-ready LaTeX documents for the deep hedging research project.

## Files

| File | Description |
|------|-------------|
| `paper.tex` | Full academic paper (~25 pages) |
| `slides.tex` | Beamer presentation (35 slides) |
| `appendix.tex` | Technical appendix (included in paper) |
| `bibliography.bib` | BibTeX references |
| `figures/` | Directory for generated figures |

## Compilation

### Paper
```bash
cd /Users/prathamkailasiya/Desktop/Thesis/deep_hedging/paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Slides
```bash
pdflatex slides.tex
```

## Paper Structure

1. **Introduction** - Problem statement, motivation, contributions
2. **Related Work** - Deep hedging, signatures, transformers, RL
3. **Mathematical Framework** - Heston model, risk measures, deep hedging formulation
4. **Model Architectures** - LSTM, Transformer, AttentionLSTM, Signature models
5. **Experiments** - Setup, main results, statistical analysis, ablations
6. **Real Market Validation** - SPY (US) and NIFTY (India) backtests
7. **Economic Analysis** - Capital requirements, hedge accounting, transaction costs
8. **Discussion & Limitations** - When to use each model, future work
9. **Conclusion** - Summary of findings and recommendations

## Presentation Structure (35 Slides)

- **Section 1**: Introduction & Motivation (4 slides)
- **Section 2**: Mathematical Framework (5 slides)
- **Section 3**: Neural Network Architectures (5 slides)
- **Section 4**: Experimental Methodology (2 slides)
- **Section 5**: Results (5 slides)
- **Section 6**: Real Market Validation (3 slides)
- **Section 7**: Economic Implications (4 slides)
- **Section 8**: Discussion & Limitations (3 slides)
- **Section 9**: Conclusion (2 slides)
- **Backup Slides** (3 slides)

## Key Results Summary

| Model | CVaR₉₅ | Improvement vs LSTM | Trading Volume |
|-------|--------|---------------------|----------------|
| LSTM (baseline) | 4.43 ± 0.02 | — | 0.85 ± 0.09 |
| **Transformer** | **4.41 ± 0.03** | **-3.1%** | 0.64 ± 0.10 (-25%) |
| AttentionLSTM | 4.44 ± 0.03 | +0.2% | 0.71 ± 0.13 (-16%) |
| SignatureLSTM | 4.44 ± 0.02 | +0.2% | 0.64 ± 0.09 (-25%) |

## Economic Impact

- **Capital savings**: 11% reduction vs Black-Scholes delta (~$98M per $10B portfolio)
- **Transaction costs**: 24% reduction with Transformer model
- **Hedge accounting**: All models qualify under IAS 39/IFRS 9

## Dependencies

Required LaTeX packages:
- `amsmath`, `amssymb`, `amsthm`
- `tikz`, `pgfplots`
- `booktabs`, `multirow`
- `algorithm`, `algorithmic`
- `hyperref`, `cleveref`
- `natbib`
- `beamer` (for slides)

## Target Venues

- **ML Conferences**: NeurIPS, ICML, ICLR
- **Finance Journals**: Quantitative Finance, Mathematical Finance, JFQA
- **Risk Management**: Journal of Risk, Risk.net

## Contact

For questions about the research or code, please refer to the main repository README.
