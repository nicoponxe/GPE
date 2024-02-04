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
	@echo "  kfold         train and check the model using kfold (k = 1 person)"
	@echo ""

PATH_TO_MAT := /Users/felix/github/GPE/Datasets/

install:
	@pip3 install -r requirements.txt

train:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/GPE.py

test:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/TestGPE.py

kfold:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/KFoldGPE.py

clean:
	@rm -rf person_*.png *.pkl5

.PHONY: mat_2_csv
