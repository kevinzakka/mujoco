# ========================= #
# Settings
# ========================= #

EXEC=simulate

BUILD_DIR:=${CURDIR}/build
INSTALL_DIR=${CURDIR}/mujoco_install
BIN_DIR=$(INSTALL_DIR)/bin
TEST_DIR=$(INSTALL_DIR)/test
DOC_DIR=$(INSTALL_DIR)/doc

# Option default values.
release=0
model=

# Detect OS.
ifeq ($(OS),Windows_NT)
	OS = windows
else
	UNAME := $(shell uname -s)
	ifeq ($(UNAME),Darwin)
		OS = macos
	else ifeq ($(UNAME),Linux)
		OS = linux
	else
    	$(error OS not supported by this Makefile)
	endif
endif

# OS-specific settings.
CMAKE_ARGS:=-G Ninja
ifeq ($(OS),windows)
	CMAKE_ARGS += -DCMAKE_C_COMPILER:STRING=clang -DCMAKE_CXX_COMPILER:STRING=clang++
else ifeq ($(OS),macos)
else ifeq ($(OS),linux)
	CMAKE_ARGS += -DCMAKE_C_COMPILER:STRING=clang -DCMAKE_CXX_COMPILER:STRING=clang++
endif

# Debug/Release settings.
ifeq ($(release),1)
	BUILD_TYPE=Release
else
	BUILD_TYPE=Debug
endif

# ========================= #
# Targets
# ========================= #

.PHONY: help print clean test run install all

default: help

help:
	@echo "Usage: make <target> [<options>]"
	@echo
	@echo "Available targets:"
	@echo "  all:       Build executable (debug mode)"
	@echo "  install:   Install executable (debug mode)"
	@echo "  run:       Build and run the simulate executable (debug mode)"
	@echo "  test:      Run unit tests (debug mode)"
	@echo "  clean:     Delete build and bin directories"
	@echo "  print:     Print Makefile variables for debugging"
	@echo "  help:      Print this message"
	@echo
	@echo "Available options:"
	@echo "  release=1       Execute above targets in release configuration"
	@echo "  model           Absolute path to the model to pass to the simulate executable"

all:
	@mkdir -p ${BUILD_DIR}; \
	cd ${BUILD_DIR}; \
	cmake .. -DCMAKE_BUILD_TYPE:STRING=$(BUILD_TYPE) -DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=ON -DCMAKE_INSTALL_PREFIX:PATH=$(INSTALL_DIR) -DMUJOCO_HARDEN:BOOL=ON $(CMAKE_ARGS); \
	cmake --build . --config=$(BUILD_TYPE)

install: all
	@cd ${BUILD_DIR}; cmake --install .

run: install
	@cd $(BIN_DIR); ./${EXEC} $(model)

clean:
	@rm -rf ${BUILD_DIR} ${INSTALL_DIR}

test: all
	@cd ${BUILD_DIR}; ctest -C $(BUILD_TYPE) --output-on-failure .

print:
	@echo "OS: $(OS)"
	@echo "CMAKE_ARGS: $(CMAKE_ARGS)"