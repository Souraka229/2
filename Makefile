.PHONY: baseline cv cv-rolling cv-group train all install

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

baseline:
	$(PYTHON) baseline.py

cv-rolling:
	$(PYTHON) cv_rolling.py

cv-group:
	$(PYTHON) cv_group.py

cv: cv-rolling cv-group

train:
	$(PYTHON) run_pipeline.py --final-only

all: baseline cv train
