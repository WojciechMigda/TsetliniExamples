import numpy as np

from sklearn.pipeline import Pipeline
from preprocessor import Preprocessor
from sklearn.datasets import load_digits

from tsetlin_tk import TsetlinMachineClassifier


def hyper_objective(train_X, train_y, nfolds, ncvjobs, nepochs,
                    number_of_pos_neg_clauses_per_label, seed, n_jobs,
                    space):
    kwargs = {}
    for k, v in space.items():
        if k in ['boost_true_positive_feedback', 'number_of_states',
                 'threshold']:
            v = int(v)
            pass
        kwargs[k] = v
        pass

    pre = Preprocessor(nbits=3)
    clf = Pipeline(steps=[
            ('preprocessor', pre),
            ('clf', TsetlinMachineClassifier(random_state=seed,
                                             clause_output_tile_size=64,
                                             number_of_pos_neg_clauses_per_label=number_of_pos_neg_clauses_per_label,
                                             n_jobs=n_jobs,
                                             **kwargs))])

    from sklearn.model_selection import StratifiedKFold
    kf = StratifiedKFold(n_splits=nfolds, random_state=seed, shuffle=True)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, train_X, train_y, cv=kf, n_jobs=ncvjobs, fit_params={'clf__n_iter': nepochs})

    score = np.mean(scores)

    print('best score: {:.5f}  best params: {}'.format(score, kwargs))
    return -score


def evaluate_hyper(train_X, train_y, objective,
                   neval, nfolds, ncvjobs,
                   njobs, seed, number_of_pos_neg_clauses_per_label,
                   nepochs, states_range, threshold_range, s_range):
    from hyperopt import fmin, tpe, hp

    states_min, states_max, states_step = map(int, states_range.split(','))
    threshold_min, threshold_max, threshold_step = map(int, threshold_range.split(','))
    s_min, s_max = map(float, s_range.split(','))

    space = {
        'boost_true_positive_feedback': hp.choice("x_boost_true_positive_feedback", [0, 1]),
        'number_of_states': hp.quniform("x_number_of_states", states_min, states_max, states_step),
        'threshold': hp.quniform ('x_threshold', threshold_min, threshold_max, threshold_step),
        's': hp.uniform ('x_s', s_min, s_max),
        }

    from functools import partial
    objective_xy = partial(objective, train_X, train_y,
                           nfolds, ncvjobs, nepochs,
                           number_of_pos_neg_clauses_per_label, seed, njobs)

    best = fmin(fn=objective_xy,
            space=space,
            algo=tpe.suggest,
            max_evals=neval,
            )
    return best


def work(neval,
         nfolds,
         ncvjobs,
         njobs,
         seed,
         number_of_pos_neg_clauses_per_label,
         nepochs,
         states_range,
         threshold_range,
         s_range
    ):

    digits = load_digits()
    X, y = digits.data, digits.target

    print('Hyperopt start')

    best = evaluate_hyper(X, y, hyper_objective,
                          neval=neval, nfolds=nfolds, ncvjobs=ncvjobs,
                          njobs=njobs, seed=seed,
                          number_of_pos_neg_clauses_per_label=number_of_pos_neg_clauses_per_label,
                          nepochs=nepochs,
                          states_range=states_range,
                          threshold_range=threshold_range,
                          s_range=s_range)

    print('Final best: {}'.format(best))

    pass
