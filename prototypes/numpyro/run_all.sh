#!/bin/bash
# run_all.sh [--bad]: seeds 0-11, three at a time (each runs two chains)
cd "$(dirname "$0")"
suf=""; [ "$1" = "--bad" ] && suf="_bad"
for s in $(seq 0 11); do echo $s; done | xargs -P 3 -I{} sh -c "uv run --with numpyro python run_dataset.py {} $1 > out/log_ds{}$suf.txt 2>&1"
