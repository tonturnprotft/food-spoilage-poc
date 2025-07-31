.PHONY: prepare run clean

## Convert DaFiF xlsx → csv & move images
prepare:
	python -m src.pipelines.prepare_dafif --config datasets.yml

## Example: make run DATASET=dafif MODEL=gbm
run:
	python -m src.cli $(DATASET) $(MODEL)

clean:
	rm -rf data/processed models/artifacts
seq:
	python3 -m src.pipelines.make_sequences --dataset=$(DATASET)
calibrate:
	python3 -m src.pipelines.calibrate_threshold --dataset=$(DATASET) --model=$(MODEL)
run_iforest:
	python3 -m src.cli $(DATASET) iforest
index_images:
	python3 -m src.pipelines.index_images --dataset=$(DATASET)
run_cnn:
	python3 -m src.cli $(DATASET) cnn

run_fusion:
	python3 -m src.cli $(DATASET) fusion
