#coding=utf8

import sys
from collections import ChainMap

from hpsklearn import HyperoptEstimator,any_classifier,any_regressor

from hyperopt  import tpe

from auto_ml.mopping.switch_log_func import switch_log_func
from auto_ml.prep.costum_error import ParametersError




def _create_core(dataset_dict,max_evals,trail_timeout,method=(None,None),
preprocessing=[],



                                  ex_preprocs=None,

                                  space=None,
                                  loss_fn=None,

                                  verbose=False,
                                  fit_increment=1,
                                  fit_increment_dump_filename=None,
                                  seed=None,
                                  use_partial_fit=False,


                 ):
    regressor,classifier = method

    if regressor is not None and classifier is not None:
        raise ParametersError('regressor and classifier both are not NOne, only receive one type, please set one of them as None!')
    elif regressor is None and classifier is None:
        switch_log_func(,log_func=None)