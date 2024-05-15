import mpmath as mp
import numpy as np

def boys_kummer(n, x):
    # The Boys function can be expressed in terms of the Kummer confluent hypergeometric fn:
    # Fn(x) = 1/(2n+1)1F1(n + 1/2, n + 3/2, -x)

    return 1.0 / (2.0 * n + 1) * mp.hyp1f1(n + 0.5, n + 1.5, -x)