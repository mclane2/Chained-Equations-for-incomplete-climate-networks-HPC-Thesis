#!/usr/bin/env bash

# Filename: build.sh
#
# Description: 
# Compiles the C source files (elastic_net_functions.c, elastic_net.c, elastic_net_R_wrapper.c) into elastic_net.so (an R-loadable shared library) via R CMD SHLIB. 
# Cleans build artefacts before compilation and removes intermediate object files after a successful build.
#
# To Compile input:
# chmod +x build.sh
# ./build.sh
#
# Author: M. Lane
# Version: 1.0
# Date: 2026-05-15

# Exits on command failure or if any stage of pipeline fails
set -euo pipefail

# Clean directory before compiling
rm -f *.o *.so

# Compiling into .so file
R CMD SHLIB -o elastic_net.so \
    elastic_net_R_wrapper.c \
    elastic_net.c \
    elastic_net_functions.c

# Clean up .o files
rm -f *.o

# Print success message
echo "Built elastic_net.so"