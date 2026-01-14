import numpy as np

def get_null_p(observed, null, version='standard', abs=False):
    if abs:
        observed = np.abs(observed)
        null = np.abs(null)

    if version == 'standard':
        p_val = np.sum(null >= observed) / len(null)
    elif version == 'reverse':
        p_val = np.sum(observed >= null) / len(null)
    elif version == 'smallest':
        p_val = np.min([np.sum(null >= observed) / len(null),
                        np.sum(observed >= null) / len(null)])
    elif version == 'absolute':
        p_val = np.sum(np.abs(null) >= np.abs(observed)) / len(null)


    return p_val
