import numpy as np
from ._scores import scores
from .functional_risk import q_func

def calibrate(evidence, risk, c_global, base_prediction=None):
    evidence = np.asarray(evidence, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 10 or risk.shape != (len(evidence),):
        raise ValueError('Expected evidence [N,10] and risk [N]')
    if not np.isfinite(evidence).all() or np.any(evidence < 0):
        raise ValueError('Evidence must be finite and nonnegative')
    if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
        raise ValueError('Risk must be in [0,1]')
    if not np.isfinite(c_global) or c_global <= 0:
        raise ValueError('c_global must be finite and positive')
    result = scores({'evidence': evidence}, risk, c_global)
    prediction = evidence.argmax(1) if base_prediction is None else np.asarray(base_prediction)
    if prediction.shape != risk.shape or not np.issubdtype(prediction.dtype, np.integer) or np.any((prediction<0)|(prediction>=10)):
        raise ValueError('Invalid original-view prediction')
    # Operational prediction is copied from the base model even in numerical ties.
    result['prediction'] = prediction.copy()
    result['numeric_prediction_changes'] = result['pred_alpha_numeric'] != prediction
    return result

def apply_reference(reference, extracted):
    risk = q_func(reference, extracted, 'L3')
    return calibrate(extracted['evidence'], risk, float(reference['c_global']), extracted['prediction'])
