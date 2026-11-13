#!/bin/bash
# PR B checks: the vary case (known width and density profiles), and the mock
# stream with velocities tied to the tangent. 3 jobs at a time (shared machine).
cd "$(dirname "$0")"
export PYTHONPATH=.
R="uv run --with numpyro python -W ignore"
( $R mock.py ref --v2 --tan > out_val/log_mock_ref_v2_tan.txt 2>&1
  for s in 0 1 2 3; do $R mock.py $s --v2 --tan > out_val/log_mock_v2_tan_ds$s.txt 2>&1; done ) &
for s in 1 2 3 4 5 6 7 0; do for f in "" "--v2" "--tan --v2"; do
  echo "$R validate.py vary $s $f > out_val/log_vary_${s}_${f// /}.txt 2>&1"
done; done | grep -v "vary 0  >\|vary 0 --v2 >" | xargs -P 2 -I{} sh -c '{}'
wait
echo finished
