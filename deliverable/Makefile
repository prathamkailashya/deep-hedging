# Makefile for Deep Hedging LaTeX Deliverables
# Author: Pratham Kailasiya, IIT Roorkee

LATEX = pdflatex
BIBTEX = bibtex
PAPERS = W-DRO-T_paper 3SCH_paper RSE_paper combined_paper A0_poster

.PHONY: all clean $(PAPERS)

all: $(PAPERS)

$(PAPERS):
	$(LATEX) $@.tex
	$(BIBTEX) $@ || true
	$(LATEX) $@.tex
	$(LATEX) $@.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f *.pdf
