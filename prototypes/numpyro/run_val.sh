#!/bin/bash
# All validation runs, 2 at a time (the machine is shared). NUTS on seeds 0-1, SVI on 0-7.
# Skips runs whose output already has NUTS (or SVI, for SVI-only runs).
cd "$(dirname "$0")"
python3 - > out_val/jobs.txt <<'PY'
import os
import numpy as np
jobs = []  # mock subsets: run by hand (NUTS on the extended model did not mix)
for s in range(8):
    for case, vel in [("helix", ""), ("helix", "--vel"), ("loop", ""), ("loop", "--vel"),
                      ("gap", ""), ("image", ""), ("image_norm", ""), ("image_x5", "")]:
        nuts = "--nuts" if s < 2 and case in ("helix", "loop", "gap", "image") else ""
        tag = f"{case}{'_vel' if vel else ''}_ds{s}"
        f = f"out_val/{tag}.npz"
        if os.path.exists(f) and ("nuts" if nuts else "svi") in np.load(f).files:
            continue
        jobs.append(f"validate.py {case} {s} {vel} {nuts}|{tag}")
print("\n".join(jobs))
PY
while IFS='|' read -r args tag; do
  echo "uv run --with numpyro python $args > out_val/log_$tag.txt 2>&1"
done < out_val/jobs.txt | xargs -P 2 -I{} sh -c '{}'
