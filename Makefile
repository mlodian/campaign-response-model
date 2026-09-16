PY ?= .venv/bin/python
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup prepare train figures evaluate periods pipeline notebook test

setup:        ## create the virtual environment and install pinned dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt -e .

prepare:      ## audit, clean, variable analysis, feature selection, EDA charts
	$(PY) -m campaign_response.pipeline --steps prepare

train:        ## model ladder CV, tune two finalists, pick cut-off, fit final model
	$(PY) -m campaign_response.pipeline --steps train

figures:      ## redraw training-stage charts from saved results
	$(PY) -m campaign_response.pipeline --steps figures

evaluate:     ## test metrics, gains, stability, explanations, campaign economics
	$(PY) -m campaign_response.pipeline --steps evaluate

periods:      ## hold out whole campaign periods; ranking within a period
	$(PY) -m campaign_response.pipeline --steps periods

pipeline:     ## everything, end to end
	$(PY) -m campaign_response.pipeline --steps all

notebook:     ## re-execute the walkthrough notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/campaign_response_walkthrough.ipynb

test:
	$(PY) -m pytest -q
