#!/bin/bash
# Loose tie (vel_slack = 0.05): mock reference then 4 subsets, alongside loop and
# vary seeds, 3 jobs at a time (shared machine).
cd "$(dirname "$0")"
export PYTHONPATH=.
R="uv run --with numpyro python -W ignore"
( $R mock.py ref --v2 --tan --slack > out_val/log_mock_ref_v2_tan_sl.txt 2>&1
  for s in 0 1 2 3; do $R mock.py $s --v2 --tan --slack > out_val/log_mock_v2_tan_sl_ds$s.txt 2>&1; done ) &
for s in 0 1 2 3 4 5 6 7; do
  echo "$R validate.py loop $s --tan --slack > out_val/log_loop_tan_sl_ds$s.txt 2>&1"
  [ $s -gt 0 ] && echo "$R validate.py vary $s --tan --v2 --slack > out_val/log_vary_tan_v2_sl_ds$s.txt 2>&1"
done | xargs -P 2 -I{} sh -c '{}'
wait
echo finished
