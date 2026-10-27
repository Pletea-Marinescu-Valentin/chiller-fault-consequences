# Every figure and number in the paper is produced by a script listed here.
# Convention: experiments/fig_*.py write to paper/figs/, experiments/tab_*.py
# write to results/tables/. EXPERIMENTS runs them in dependency order:
# fig_consequences writes the weekly ground truth that fig_rank_weeks reads,
# tab_estimation writes the weekly estimates that fig_estimation and
# fig_ranking read, and tab_rtu replicates the pipeline on the simulated
# rooftop unit.

PYTHON ?= python

EXPERIMENTS := tab_audit fig_consequences fig_weekly fig_rank_weeks tab_estimation fig_estimation fig_ranking tab_rtu

.PHONY: data test experiments verify reproduce paper clean

data:
	$(PYTHON) scripts/download/lbnl_chiller.py
	$(PYTHON) scripts/download/lbnl_rtu.py

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
