#!/usr/bin/python3

"""
Runs hyperparameter search for Tsetlin Machine Classifier employed to solve
MNIST handwritten digits classification problem.

Parameters reported in the original paper:
- number of clauses: 1000 (here it translates to 50 = 1000 / (10 * 2))
- s: 3.0
- threshold: 10
- number of states: 1000
- boost_true_positive_feedback: unknown

The original paper also used 5-fold crossvalidation and 300 epochs. Pixel
values were downscaled to 3-bit range.

Reported accuracy was 95.7 +/- 0.2

"""

N_JOBS = 5

import numpy as np

from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.datasets import load_digits

from tsetlin_tk import TsetlinMachineClassifier


def hyper_objective(train_X, train_y, nfolds, space):
    kwargs = {}
    for k, v in space.items():
        if k in ['boost_true_positive_feedback', 'number_of_states', 'number_of_pos_neg_clauses_per_label', 'threshold']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    pre = Preprocessor(nbits=3)
    clf = Pipeline(steps=[('preprocessor', pre), ('clf', TsetlinMachineClassifier(random_state=1, **kwargs))])

    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=nfolds, random_state=1, shuffle=True)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=N_JOBS, fit_params={'clf__n_iter': 300})

    score = np.mean(scores)

    print('best score: {:.5f}  best params: {}'.format(score, kwargs))
    return -score


def evaluate_hyper(train_X, train_y, objective, neval=500, nfolds=3):
    from hyperopt import fmin, tpe, hp

    space = {
        'boost_true_positive_feedback': hp.choice("x_boost_true_positive_feedback", [0, 1]),
        'number_of_states': hp.quniform("x_number_of_states", 500, 2000, 20),
        'number_of_pos_neg_clauses_per_label': hp.quniform("x_number_of_pos_neg_clauses_per_label", 30, 80, 5),
        'threshold': hp.quniform ('x_threshold', 5, 20, 1),
        's': hp.uniform ('x_s', 1.0, 6.0),
        }

    from functools import partial
    objective_xy = partial(objective, train_X, train_y, nfolds)

    best = fmin(fn=objective_xy,
            space=space,
            algo=tpe.suggest,
            max_evals=neval,
            )
    return best


def main():

    digits = load_digits()
    X, y = digits.data, digits.target

    print('Hyperopt start')
    best = evaluate_hyper(X, y, hyper_objective, neval=30, nfolds=5)
    print('Final best: {}'.format(best))


if __name__ == '__main__':
    main()
