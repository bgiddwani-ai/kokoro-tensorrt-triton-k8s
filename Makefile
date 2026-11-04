IMAGE ?= ghcr.io/example/kokoro:latest
BENCHMARK_IMAGE ?= ghcr.io/example/kokoro-benchmark:latest
NAMESPACE ?= kokoro
export IMAGE NAMESPACE

.PHONY: help build push deploy render forward logs down benchmark-image test
help:
	@printf '%s\n' 'make build push IMAGE=<registry>/kokoro:<tag>' 'make deploy IMAGE=<registry>/kokoro:<tag>' 'make forward' 'make logs' 'make down  (keeps model/cache PVC)' 'make test'
build:
	docker build --platform linux/amd64 -t $(IMAGE) .
push:
	docker push $(IMAGE)
deploy render forward logs down:
	@bash scripts/kube.sh $@
benchmark-image:
	docker build --platform linux/amd64 --build-arg SERVER_IMAGE=$(IMAGE) -f Dockerfile.benchmark -t $(BENCHMARK_IMAGE) .
	docker push $(BENCHMARK_IMAGE)
test:
	python3 -m unittest discover -s tests -v
