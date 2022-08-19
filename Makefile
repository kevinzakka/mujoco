.PHONY: help build test clean simulate

default: help

help:
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@echo "  build: Build the project"
	@echo "  test: Run the tests"
	@echo "  clean: Delete the build artifacts"
	@echo "  simulate: Run the simulate binary"

build:
	@mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DMUJOCO_HARDEN:BOOL=ON && cmake --build . --config=Release

test:
	cd build && ctest -C Release .

clean:
	rm -rf build

simulate:
	./build/bin/simulate
