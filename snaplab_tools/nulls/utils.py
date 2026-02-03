import numpy as np

def get_null_p(observed, null, version='standard', abs=False):
    if abs:
        observed = np.abs(observed)
        null = np.abs(null)

    if version == 'standard':
        if observed >= 0:
            p_val = np.sum(null >= observed) / len(null)
        elif observed <= 0:
            p_val = np.sum(null <= observed) / len(null)
    elif version == 'smallest':
        p_val = np.min([np.sum(null >= observed) / len(null),
                        np.sum(observed >= null) / len(null)])

    return p_val
