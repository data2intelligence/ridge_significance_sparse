# Makefile for compiling and testing Ridge Inference C backends directly

# --- User Configuration ---
# Select backend: 'gsl', 'mkl', or 'all' (default)
# Example: make BACKEND=gsl
# Example: make test_gsl MKLROOT=/path/to/mkl (will only build/test gsl)
BACKEND ?= all

# Environment variables for library locations (optional)
# MKLROOT=/path/to/mkl
# GSL_HOME=/path/to/gsl (Fallback if pkg-config fails)

# --- Tools ---
CC ?= gcc

# Determine OpenMP flag
ifeq ($(findstring icx,$(CC)),icx)
    OPENMP_FLAG = -qopenmp
else ifeq ($(findstring icpx,$(CC)),icpx)
    OPENMP_FLAG = -qopenmp
else
    OPENMP_FLAG = -fopenmp
endif

# Base flags + Auto-dependency generation flags
# *** ADD D_GNU_SOURCE for rand_r etc. ***
CFLAGS_COMMON = -D_GNU_SOURCE -fPIC -O3 -Wall -std=c99 $(OPENMP_FLAG) -I. -MMD -MP
LDFLAGS_COMMON = $(OPENMP_FLAG)
LIBS_COMMON = -lm -ldl

# --- MKL Configuration ---
MKL_CONFIGURED := no
MKL_CPPFLAGS :=
MKL_LDFLAGS :=
MKL_LIBS :=
MKL_SRC = ridge_mkl.c
MKL_OBJ = ridge_mkl.o
MKL_TARGET = main_mkl
MKL_MAIN_SRC = main_mkl.c
MKL_MAIN_OBJ = main_mkl.o

# Check MKLROOT (only if needed by backend selection)
ifeq ($(filter $(BACKEND),mkl all),$(BACKEND))
    ifeq ($(strip $(MKLROOT)),)
        $(info MKL build possibly needed but MKLROOT not set. Skipping MKL configuration.)
    else
        MKLROOT_CLEAN := $(strip $(MKLROOT))
        MKL_BASE := $(shell echo $(MKLROOT) | sed 's/[[:space:]]*$$//')
        MKL_INC_PATH := $(MKL_BASE)/include
        MKL_HEADER_MKLROOT := $(wildcard $(MKL_INC_PATH)/mkl.h)

        ifeq ($(strip $(MKL_HEADER_MKLROOT)),)
            $(warning MKLROOT provided, but mkl.h not found in [$(MKL_INC_PATH)]. Skipping MKL configuration.)
        else
            MKL_LIB_PATH_INTEL64 := $(MKL_BASE)/lib/intel64
            MKL_LIB_PATH_LIB := $(MKL_BASE)/lib
            MKL_LIB_PATH_LIB64 := $(MKL_BASE)/lib64
            MKL_LIB_FILE_MKLROOT := $(firstword $(wildcard $(MKL_LIB_PATH_INTEL64)/libmkl_rt.so $(MKL_LIB_PATH_INTEL64)/libmkl_rt.dylib \
                                                 $(MKL_LIB_PATH_LIB)/libmkl_rt.so $(MKL_LIB_PATH_LIB)/libmkl_rt.dylib \
                                                 $(MKL_LIB_PATH_LIB64)/libmkl_rt.so $(MKL_LIB_PATH_LIB64)/libmkl_rt.dylib))
            ifneq ($(strip $(MKL_LIB_FILE_MKLROOT)),)
                MKL_CONFIGURED := yes
                MKL_LIB_DIR := $(shell dirname '$(MKL_LIB_FILE_MKLROOT)')
                # *** Add HAVE_MKL define here ***
                MKL_CPPFLAGS := -I$(MKL_INC_PATH) -DHAVE_MKL
                MKL_LDFLAGS := -L$(MKL_LIB_DIR) -Wl,-rpath,$(MKL_LIB_DIR)
                MKL_LIBS := -lmkl_rt
                $(info MKL Configured using MKLROOT.)
            else
                $(warning MKLROOT/mkl.h found, but libmkl_rt not found. Skipping MKL configuration.)
            endif
        endif
    endif
endif

# --- GSL Configuration ---
GSL_CONFIGURED := no
GSL_CPPFLAGS :=
GSL_LDFLAGS :=
GSL_LIBS :=
GSL_SRC = ridge_gsl.c
GSL_OBJ = ridge_gsl.o
GSL_TARGET = main_gsl
GSL_MAIN_SRC = main_gsl.c
GSL_MAIN_OBJ = main_gsl.o

# Check GSL (only if needed by backend selection)
ifeq ($(filter $(BACKEND),gsl all),$(BACKEND))
    PKG_CONFIG ?= pkg-config
    ifeq ($(shell $(PKG_CONFIG) --exists gsl && echo yes),yes)
        $(info Found GSL via pkg-config)
        GSL_CONFIGURED := yes
        GSL_CPPFLAGS := $(shell $(PKG_CONFIG) --cflags gsl)
        GSL_LDFLAGS := $(shell $(PKG_CONFIG) --libs-only-L gsl)
        GSL_LIBS := $(shell $(PKG_CONFIG) --libs gsl)
        GSL_LIB_DIRS_PKG := $(shell $(PKG_CONFIG) --libs-only-L gsl | sed 's/-L//g')
        GSL_LDFLAGS += $(foreach dir,$(GSL_LIB_DIRS_PKG),-Wl,-rpath,$(dir))
    else
        $(info GSL not found via pkg-config. Skipping GSL configuration.)
        # Add GSL_HOME fallback here if desired
    endif
endif

# --- Final CFLAGS Definition (Combine ALL potential flags) ---
# This ensures _OPENMP and HAVE_MKL are defined if requested, regardless of BACKEND selection
FINAL_CFLAGS := $(sort $(CFLAGS_COMMON) $(GSL_CPPFLAGS) $(MKL_CPPFLAGS))

# --- Define Build Targets Based on BACKEND and Configuration ---
TARGETS_TO_BUILD :=
LINK_LIBS_GSL :=
LINK_LIBS_MKL :=

ifeq ($(BACKEND),gsl)
    ifeq ($(GSL_CONFIGURED),yes)
        FINAL_OBJECTS := $(GSL_MAIN_OBJ) $(GSL_OBJ)
        FINAL_TARGET := $(GSL_TARGET)
        LINK_LIBS_GSL := $(sort $(LIBS_COMMON) $(GSL_LIBS)) # Libs needed only for GSL link
        $(info Building GSL Backend Target: $(FINAL_TARGET))
    else
        $(error GSL backend requested (BACKEND=gsl) but GSL not found/configured.)
    endif
else ifeq ($(BACKEND),mkl)
    ifeq ($(MKL_CONFIGURED),yes)
        FINAL_OBJECTS := $(MKL_MAIN_OBJ) $(MKL_OBJ)
        FINAL_TARGET := $(MKL_TARGET)
        LINK_LIBS_MKL := $(sort $(LIBS_COMMON) $(MKL_LIBS)) # Libs needed only for MKL link
        $(info Building MKL Backend Target: $(FINAL_TARGET))
    else
        $(error MKL backend requested (BACKEND=mkl) but MKL not found/configured.)
    endif
else # BACKEND=all or default
    $(info Building ALL available backends...)
    ALL_POTENTIAL_OBJECTS := # Define all objects needed for any target
    ifeq ($(GSL_CONFIGURED),yes)
        TARGETS_TO_BUILD += $(GSL_TARGET)
        ALL_POTENTIAL_OBJECTS += $(GSL_MAIN_OBJ) $(GSL_OBJ)
        LINK_LIBS_GSL := $(sort $(LIBS_COMMON) $(GSL_LIBS))
        $(info Will build GSL target: $(GSL_TARGET))
    else
        $(info Skipping GSL target (not configured).)
    endif

    ifeq ($(MKL_CONFIGURED),yes)
        TARGETS_TO_BUILD += $(MKL_TARGET)
        ALL_POTENTIAL_OBJECTS += $(MKL_MAIN_OBJ) $(MKL_OBJ)
        LINK_LIBS_MKL := $(sort $(LIBS_COMMON) $(MKL_LIBS))
        $(info Will build MKL target: $(MKL_TARGET))
    else
         $(info Skipping MKL target (not configured).)
    endif
    # For 'all', FINAL_OBJECTS isn't used directly for linking rules below
    FINAL_TARGET := $(TARGETS_TO_BUILD) # 'all' depends on individual targets
endif

# Dependency files list (all potential .d files)
ALL_OBJS := $(GSL_MAIN_OBJ) $(GSL_OBJ) $(MKL_MAIN_OBJ) $(MKL_OBJ)
DEPS := $(ALL_OBJS:.o=.d)

# --- Targets ---
# Default target 'all': build specific target(s) based on BACKEND
all: $(FINAL_TARGET)

# Specific target rules for direct building
gsl: $(GSL_TARGET)
mkl: $(MKL_TARGET)

# Test targets now depend on specific build targets
test_gsl: $(GSL_TARGET)
	@echo "Running GSL test harness..."
	./$(GSL_TARGET)
	@echo "GSL test harness finished."

test_mkl: $(MKL_TARGET)
	@echo "Running MKL test harness..."
	./$(MKL_TARGET)
	@echo "MKL test harness finished."

# --- Linking Rules ---
# Link GSL target if configured
ifeq ($(GSL_CONFIGURED),yes)
$(GSL_TARGET): $(GSL_MAIN_OBJ) $(GSL_OBJ)
	@echo "Linking test executable $(GSL_TARGET)..."
	@echo "  LDFLAGS = $(LDFLAGS_COMMON) $(GSL_LDFLAGS)"
	@echo "  LIBS = $(LINK_LIBS_GSL)"
	$(CC) $(LDFLAGS_COMMON) $(GSL_LDFLAGS) -o $@ $^ $(LINK_LIBS_GSL)
endif

# Link MKL target if configured
ifeq ($(MKL_CONFIGURED),yes)
$(MKL_TARGET): $(MKL_MAIN_OBJ) $(MKL_OBJ)
	@echo "Linking test executable $(MKL_TARGET)..."
	@echo "  LDFLAGS = $(LDFLAGS_COMMON) $(MKL_LDFLAGS)"
	@echo "  LIBS = $(LINK_LIBS_MKL)"
	$(CC) $(LDFLAGS_COMMON) $(MKL_LDFLAGS) -o $@ $^ $(LINK_LIBS_MKL)
endif

# --- Generic Compilation Rule ---
# Compiles any .c file into a .o file using the unified FINAL_CFLAGS
# Add dependency on corresponding header if it exists
%.o: %.c $(wildcard $(patsubst %.c,%.h,$<))
	@echo "Compiling $< -> $@ ..."
	@echo "  CC = $(CC)"
	@echo "  CFLAGS = $(FINAL_CFLAGS)"
	$(CC) $(FINAL_CFLAGS) -c $< -o $@

# --- Clean Target ---
clean:
	@echo "Cleaning test objects, executables, and dependency files..."
	rm -f $(ALL_OBJS) $(GSL_TARGET) $(MKL_TARGET) $(DEPS) *.o

# --- Info Target ---
info:
	@echo "--- Build Configuration ---"
	@echo "Selected BACKEND = $(BACKEND)"
	@echo "CC = $(CC)"
	@echo "OpenMP Flag = $(OPENMP_FLAG)"
	@echo "--- GSL ---"
	@echo "  Configured: $(GSL_CONFIGURED)"
	@echo "  CPPFLAGS: [$(GSL_CPPFLAGS)]" # Flags specific to GSL
	@echo "  LDFLAGS: [$(GSL_LDFLAGS)]"
	@echo "  LIBS: [$(GSL_LIBS)]"
	@echo "--- MKL ---"
	@echo "  Configured: $(MKL_CONFIGURED)"
	@echo "  CPPFLAGS: [$(MKL_CPPFLAGS)]" # Flags specific to MKL
	@echo "  LDFLAGS: [$(MKL_LDFLAGS)]"
	@echo "  LIBS: [$(MKL_LIBS)]"
	@echo "--- Final Build ---"
	@echo "  CFLAGS (used for .c -> .o): [$(FINAL_CFLAGS)]" # Combined flags used
	@echo "  LDFLAGS (Base): [$(LDFLAGS_COMMON)]"
	@echo "  LIBS (Base): [$(LIBS_COMMON)]"
	@echo "--- Target Info ---"
	@echo "  GSL Target: $(GSL_TARGET) (Requires $(GSL_MAIN_OBJ) $(GSL_OBJ))"
	@echo "  MKL Target: $(MKL_TARGET) (Requires $(MKL_MAIN_OBJ) $(MKL_OBJ))"
	@echo "  Targets Built by 'all': $(TARGETS_TO_BUILD)"


.PHONY: all gsl mkl test_gsl test_mkl clean info

# --- Include Generated Dependencies ---
-include $(DEPS)