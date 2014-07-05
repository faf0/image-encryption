#!/bin/bash

readonly DATESTR=$(date -u +%Y-%m-%d--%H-%M)
readonly RESULTDIR="../results/"
readonly SRCDIR="../src/"

mkdir -p ${RESULTDIR}
pushd "${SRCDIR}"
make clean
make
popd
./test_overhead > "${RESULTDIR}/${DATESTR}-overhead.csv" 2> "${RESULTDIR}/${DATESTR}-overhead-errors.txt"

