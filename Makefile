# Scale-Recoverability — one-click entrypoints
# ENV selects the conda environment (default: luna).
ENV ?= luna
PY := conda run -n $(ENV) python

.PHONY: env demo test check reproduce data help

help:
	@echo "Targets:"
	@echo "  make env        create conda env from environment/environment.yml (name: $(ENV))"
	@echo "  make demo       run the self-contained analytical benchmark (no data needed)"
	@echo "  make test       run pytest unit tests"
	@echo "  make check      alias for test"
	@echo "  make reproduce  full paper pipeline (requires data + trained artifacts)"
	@echo "  make data       print data fetch instructions"

env:
	conda env create -f environment/environment.yml -n $(ENV)

demo:
	$(PY) applications/paper_main/analyses/_canonical/compute_p0_analytical.py

test:
	$(PY) -m pytest tests/unit -q

check: test

reproduce:
	bash scripts/reproduce_all.sh $(ENV)

data:
	bash scripts/download_data.sh
