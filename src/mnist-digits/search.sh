#!/bin/sh

./cv.cli.py \
  --neval=30 \
  --nfolds=5 \
  --cv-jobs=5 \
  --jobs=1 \
  --seed=1 \
  --number_of_pos_neg_clauses_per_label=50 \
  --nepochs=300 \
  --states-range="500,2000,20" \
  --threshold-range="5,20,1" \
  --s-range="1.0,6.0"
