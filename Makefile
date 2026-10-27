# Every figure and number in the paper is produced by a script listed here.
# Convention: experiments/fig_*.py write to paper/figs/, experiments/tab_*.py
# write to results/tables/. EXPERIMENTS runs them in dependency order:
# tab_estimation writes the weekly estimates that fig_ranking reads.

PYTHON ?= python

EXPERIMENTS := tab_audit fig_consequences fig_weekly tab_estimation fig_ranking

.PHONY: data test experiments verify reproduce paper clean

data:
	$(PYTHON) scripts/download/lbnl_chiller.py

test:
	$(PYTHON) -m pytest

experiments:
	@for e in $(EXPERIMENTS); do echo "experiments/$$e.py"; $(PYTHON) "experiments/$$e.py" || exit 1; done

verify:
	$(PYTHON) scripts/verify_numbers.py

reproduce: test experiments verify

paper:
	cd paper && latexmk -pdf main.tex

clean:
	cd paper && latexmk -C main.tex
