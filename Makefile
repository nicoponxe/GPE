all: help

help:
	@echo ""
	@echo "--------------------------------------------------------------"
	@echo " Please use 'make <target>' where <target> is one of"
	@echo "--------------------------------------------------------------"
	@echo ""
	@echo "  help          to show this message"
	@echo "  install       install the python libraries"
	@echo "  kfold         train the model and use kfold (k=1 person)"
	@echo "  train         train the model"
	@echo "  test          test the model"
	@echo ""

PATH_TO_MAT := /Users/felix/github/GPE/Datasets/

install:
	@pip3 install -r requirements.txt

kfold:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/KFoldGPE.py \
		--apply-data-augmentation \
		--include-non-linear-data \
		--include-shank-angles \
		--plot-results
#	--apply-min-max-normalization \
#		--apply-kalman-filter \
#	--include-thigh-angles \

train:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/GPE.py

test:
	@PATH_TO_MAT=$(PATH_TO_MAT) python3 Model/TestGPE.py

clean:
	@rm -rf person_*.png *.pkl5 kalman_*.png

.PHONY: mat_2_csv
