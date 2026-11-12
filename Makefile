PYTHON ?= python3
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

.PHONY: setup setup-core data embeddings experiments app test lint smoke clean

setup:            ## full install (core + dashboard + vision + dev tools)
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[app,vision,dev]"

setup-core:       ## install without the heavy vision stack
	$(PYTHON) -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[app,dev]"

data:             ## download IndoFashion + Myntra samples from Kaggle (needs ~/.kaggle/kaggle.json)
	$(PY) scripts/fetch_data.py

embeddings:       ## embed sampled images with FashionSigLIP and train the classifier head
	$(PY) scripts/build_embeddings.py

experiments:      ## run E1-E7 and write results/
	$(PY) scripts/run_experiments.py

smoke:            ## quick check that the vision models load and run on bundled images
	$(PY) scripts/smoke_vision.py

app:              ## start the dashboard
	$(PY) -m streamlit run fold_and_place/app/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

clean:
	rm -rf .pytest_cache .ruff_cache build *.egg-info
