.PHONY: prepare run clean

## Convert DaFiF xlsx → csv & move images
prepare:
	python -m src.pipelines.prepare_dafif --config datasets.yml

## Example: make run DATASET=dafif MODEL=gbm
run:
	python -m src.cli $(DATASET) $(MODEL)

clean:
	rm -rf data/processed models/artifacts
