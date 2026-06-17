# Factorization measurement/correctness harness.
# Builds entirely against the PREBUILT libmujoco; touches no MuJoCo source.
#
# Run from factor-opt/ :  make        (build all binaries into build/)
#                         make clean

# Repo root is one level up from factor-opt/.
ROOT     := $(abspath ..)
INSTALL  := $(ROOT)/mujoco_install
INCLUDE  := $(INSTALL)/include
LIBDIR   := $(INSTALL)/lib

CC       := cc
# -I$(INCLUDE) : public headers (mujoco/mujoco.h ...)
# -I$(ROOT)    : internal engine headers as "src/engine/..." (as benchmarks do)
CFLAGS   := -O2 -std=c11 -Wall -Wextra -Werror -I$(INCLUDE) -I$(ROOT) -Ifactor-reference
# Optimized-kernel-only flags: -O3 + native (NEON) auto-vectorization. Applied
# ONLY to factor_optimized.c, never to the harness or reference kernel.
OPTFLAGS := -O3 -march=native -std=c11 -Wall -Wextra -Werror -I$(INCLUDE) -I$(ROOT) -Ifactor-reference
LDFLAGS  := -L$(LIBDIR) -lmujoco -Wl,-rpath,$(LIBDIR)

BUILD    := build
HARNESS  := harness
REF      := factor-reference
OPT      := factor-optimized

BINS := $(BUILD)/dump_corpus \
        $(BUILD)/compare_factor \
        $(BUILD)/certify_factor \
        $(BUILD)/run_factor_ref \
        $(BUILD)/run_factor_opt

.PHONY: all clean
all: $(BINS)

$(BUILD):
	mkdir -p $(BUILD)

# corpus generator (links libmujoco for model load / forward / chol assert)
$(BUILD)/dump_corpus: $(HARNESS)/dump_corpus.c $(HARNESS)/corpus_format.h | $(BUILD)
	$(CC) $(CFLAGS) $(HARNESS)/dump_corpus.c -o $@ $(LDFLAGS)

# comparator: pure data transform, no libmujoco needed
$(BUILD)/compare_factor: $(HARNESS)/compare_factor.c $(HARNESS)/corpus_format.h | $(BUILD)
	$(CC) $(CFLAGS) $(HARNESS)/compare_factor.c -o $@

# certifier: needs libmujoco for the dense Cholesky oracle
$(BUILD)/certify_factor: $(HARNESS)/certify_factor.c $(HARNESS)/corpus_format.h | $(BUILD)
	$(CC) $(CFLAGS) $(HARNESS)/certify_factor.c -o $@ $(LDFLAGS)

# run_factor linked with the REFERENCE kernel
$(BUILD)/run_factor_ref: $(HARNESS)/run_factor.c $(REF)/factor_reference.c \
                         $(REF)/factor_kernel.h $(HARNESS)/corpus_format.h | $(BUILD)
	$(CC) $(CFLAGS) $(HARNESS)/run_factor.c $(REF)/factor_reference.c -o $@ $(LDFLAGS)

# optimized kernel object, built with OPTFLAGS (-O3 -march=native / NEON)
$(BUILD)/factor_optimized.o: $(OPT)/factor_optimized.c $(REF)/factor_kernel.h | $(BUILD)
	$(CC) $(OPTFLAGS) -c $(OPT)/factor_optimized.c -o $@

# run_factor linked with the OPTIMIZED kernel (run_factor.c at harness CFLAGS,
# kernel object at OPTFLAGS)
$(BUILD)/run_factor_opt: $(HARNESS)/run_factor.c $(BUILD)/factor_optimized.o \
                         $(REF)/factor_kernel.h $(HARNESS)/corpus_format.h | $(BUILD)
	$(CC) $(CFLAGS) $(HARNESS)/run_factor.c $(BUILD)/factor_optimized.o -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD)
