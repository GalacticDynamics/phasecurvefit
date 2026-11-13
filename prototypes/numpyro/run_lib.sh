#!/bin/sh
# Library sample_track on the known-truth cases, 8 seeds each, one at a time.
for case in helix gap loop vary; do
  for seed in 0 1 2 3 4 5 6 7; do
    uv run python validate_lib.py $case $seed 2>&1 | grep -v Warning | tail -1
  done
done
