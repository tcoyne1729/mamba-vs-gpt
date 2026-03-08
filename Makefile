IMAGE = tcoyne1729/cuda12.8-torch-dev:latest

build:
	docker build -t $(IMAGE) .

push:
	docker push $(IMAGE)

train_gpt:
	./runners/run_gpt.sh