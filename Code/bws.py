
# Baumgartner-Weis-Schindler statistic and hypothesis testing.
# Copyright (c) Robert Feldt 2013 (robert.feldt@gmail.com)
#
# For license details see the LICENSE.md file in the top dir of the repository
# where you found this file.
#
# The algorithms in this file are based on the definitions of the BWS in the paper:
#
#  Markus Neuhäuser, "Exact tests based on the Baumgartner-WeiB-Schindler statistic - A survey",
#  Statistical Papers, vol. 46, pp. 1-30, Springer Verlag, 2005.

import numpy as np
from scipy.stats import rankdata

# Test if according to the Baumgartner-Weis-Schindler (BWS) statistic we
# can reject the hypothesis that the two set of samples x and y are likely to have
# the same location (mean).


def bws_test_sampled(x, y, numsamples=10000):
    # Get the ranks. The first 1:n are ranks for x, (n+1):(n+m) for y.
    rs = rankdata(np.concatenate([x, y]))

    # call the "bws_test_sampled_from_ranks" function from "FeldtLib"
    return bws_test_sampled_from_ranks(rs, len(x), len(y), numsamples)


def bws_test_sampled_from_ranks(ranks, n, m, numsamples=10000):
    rs = ranks.copy()  # Copy since we will shuffle it below.

    actual_b = bws_statistic_from_ranks(rs, n, m)

    # Array to save the b statistics
    samples = np.zeros(numsamples)
    numlarger = 0

    # Set up for inline sorts.
    ns = np.zeros(n, dtype=np.int64)
    ms = np.zeros(m, dtype=np.int64)
    np1 = n+1

    for i in range(numsamples):
        # Now shuffle the ranks and then ensure they are sorted for each group.
        # Not sure this is actually faster than letting python decide on the copying
        # and gc of intermediate vectors/arrays... ;)
        np.random.shuffle(rs)
        ns = rs[:n]
        ms = rs[np1:]
        ns.sort()
        ms.sort()
        rs[:n] = ns
        rs[np1:] = ms

        # Now we can calculate the bws again and note if it was larger than before.
        samples[i] = bws_statistic_from_ranks(rs, n, m)
        if samples[i] >= actual_b:
            numlarger += 1

    # A larger (or equal) BWS statistic value indicates samples from distributions
    # that are "more apart" and thus that there is a difference between the samples.
    pvalue = numlarger / numsamples

    # , np.mean(samples), np.std(samples))
    return (actual_b, pvalue, numlarger, numsamples)


def bws_statistic(x, y):
    # Get the ranks. The first 1:n are ranks for x, (n+1):end for y.
    rs = rankdata(np.concatenate([x, y]))

    return bws_statistic_from_ranks(rs, len(x), len(y))

# Calculate the BWS statistic from a set of ranks of n values from one sample
# and m values from another sample. The first 1:n rank values in ranks are for
# the first sample and the rest are from the other.


def bws_statistic_from_ranks(ranks, n, m):
    # Pre-calc some values needed in both b calcs.
    nm = n+m
    np1 = n + 1
    mp1 = m + 1

    # Now calc the two b values
    b_x = (1/n) * (n / (m*nm)) * calc_bws_sum(ranks,   1,  n, n, np1, nm, 0)
    b_y = (1/m) * (m / (n*nm)) * calc_bws_sum(ranks, np1, nm, m, mp1, nm, n)

    return (b_x + b_y) / 2


def calc_bws_sum(ranks, mini, maxi, n, np1, nm, subi=0):
    sum = 0.0
    for i in range(mini, maxi+1):
        k = i - subi
        nom = (ranks[i] - (nm/n*k)) ** 2
        denom = (k/np1) * (1 - (k/np1))
        sum += (nom / denom)
    return sum
