#!/usr/bin/env bash
# Reproduce all results. Run from the repo root.
set -e
cd src
python prep.py   > ../results/prep_output.txt
python model.py  > ../results/fit_output.txt
python lopo.py   > ../results/lopo_output.txt
python extrap.py > ../results/extrap_output.txt
python extras.py A B > ../results/extras_AB_output.txt
python extras.py C > ../results/learning_curve_output.txt
