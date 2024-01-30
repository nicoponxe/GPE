all: help

help:
	@echo ""
	@echo "--------------------------------------------------------------"
	@echo " Please use 'make <target>' where <target> is one of"
	@echo "--------------------------------------------------------------"
	@echo ""
	@echo "  help          to show this message"
	@echo "  install       install the python libraries"
	@echo "  train         train the model"
	@echo "  test          test the model"
	@echo ""

PATH_TO_MAT := /Users/felix/github/GPE/Datasets/

install:
	@pip3 install -r requirements.txt

train:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/GPE.py

test:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/TestGPE.py

.PHONY: mat_2_csv
