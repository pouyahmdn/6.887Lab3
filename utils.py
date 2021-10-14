import numpy as np


def ewma(seq, alpha, err=1e-3):
    len_fil = np.ceil(np.log(err)/np.log(alpha)).astype(int)
    fil = np.exp(np.arange(len_fil) * np.log(alpha))
    norm_vec = np.empty(len(seq), dtype=float)
    if len_fil >= len(seq):
        norm_vec = np.cumsum(fil)[:len(seq)]
    else:
        norm_vec[:len_fil] = np.cumsum(fil)
        norm_vec[len_fil:] = norm_vec[len_fil-1]
    return np.convolve(seq, fil)[:len(seq)] / norm_vec