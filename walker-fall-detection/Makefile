PY ?= python3

.PHONY: install test eval figures filmstrip all clean

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

eval:
	$(PY) scripts/run_experiments.py

figures:
	$(PY) scripts/make_figures.py

filmstrip:
	$(PY) scripts/make_filmstrip.py

all: test eval figures filmstrip

clean:
	rm -rf results/metrics.json models/*.joblib assets/*.png .pytest_cache
