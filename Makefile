.PHONY: build shell exec test smoke isaac-static isaac-dynamic isaac-occlusion tracking-sam-mt tracking-efficient-tam tracking-mock rviz download

build:
	./scripts/build.sh

shell:
	./scripts/run_container.sh

exec:
	./scripts/exec_container.sh

test:
	./scripts/run_in_container.sh ./scripts/smoke_test.sh

smoke:
	./scripts/smoke_test.sh

isaac-static:
	./scripts/run_isaac.sh static

isaac-dynamic:
	./scripts/run_isaac.sh dynamic

isaac-occlusion:
	./scripts/run_isaac.sh occlusion

tracking-sam-mt:
	./scripts/run_tracking.sh sam_mt

tracking-efficient-tam:
	./scripts/run_tracking.sh efficient_tam

tracking-mock:
	./scripts/run_tracking.sh mock

rviz:
	./scripts/run_rviz.sh

download:
	./scripts/download_checkpoints.sh
