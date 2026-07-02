#!/usr/bin/env bash

# Filename: build.sh
#
# Description:
# Compiles the C source files into ENCE_impute.so (an R-loadable shared library)
# via R CMD SHLIB, with OpenMP. Cleans build artefacts before compilation
# and removes object files after a successful build.
#
# To Compile input:
# chmod +x build.sh
# ./build.sh
#
# Author: M. Lane
# Version: 5.0 (Added write betas to file flag)
# Date: 2026-06-16

set -euo pipefail

# Clean directory before compiling
rm -f *.o *.so

# Compile + link with OpenMP.
PKG_CFLAGS="-fopenmp -O2" \
PKG_LIBS="-fopenmp" \
R CMD SHLIB elastic_net_functions.c elastic_net.c ENCE_impute.c elastic_net_R_wrapper.c \
  -o O2_ENCE_impute.so

# Clean up .o files
rm -f *.o

echo "Built O2_ENCE_impute.so"