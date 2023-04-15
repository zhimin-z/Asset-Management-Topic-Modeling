
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


def calc_bws_sum(ranks, nm):
    size = len(ranks)
    sum = 0
    for i, r in enumerate(ranks):
        k = i + 1
        nom = (r - (nm / size * k)) ** 2
        denom = k / (size + 1) * (1 - k / (size + 1))
        sum += nom / denom
    return sum

# Calculate the BWS statistic from a set of ranks of n values from one sample
# and m values from another sample. The first 1:n rank values in ranks are for
# the first sample and the rest are from the other.


def bws_statistic_from_ranks(ranks, n, m):
    # Pre-calc some values needed in both b calcs.
    nm = n + m

    # Now calc the two b values
    b_x = 1 / (m * nm) * calc_bws_sum(ranks[0:n], nm)
    b_y = 1 / (n * nm) * calc_bws_sum(ranks[n:nm], nm)

    return (b_x + b_y) / 2


def bws_test_sampled_from_ranks(ranks, n, m, numsamples=10000):
    ranks = ranks.copy()  # Copy since we will shuffle it below.

    b_stats = bws_statistic_from_ranks(ranks, n, m)

    # Array to save the b statistics
    samples = np.zeros(numsamples)
    numlarger = 0

    # Set up for inline sorts.
    ns = np.zeros(n, dtype=np.int64)
    ms = np.zeros(m, dtype=np.int64)

    for i in range(numsamples):
        # Now shuffle the ranks and then ensure they are sorted for each group.
        # Not sure this is actually faster than letting python decide on the copying
        # and gc of intermediate vectors/arrays... ;)
        np.random.shuffle(ranks)
        ns = ranks[:n]
        ms = ranks[n+1:]
        ns.sort()
        ms.sort()
        ranks[:n] = ns
        ranks[n+1:] = ms

        # Now we can calculate the bws again and note if it was larger than before.
        samples[i] = bws_statistic_from_ranks(ranks, n, m)
        if samples[i] >= b_stats:
            numlarger += 1

    # A larger (or equal) BWS statistic value indicates samples from distributions
    # that are "more apart" and thus that there is a difference between the samples.
    pvalue = numlarger / numsamples

    return b_stats, pvalue


def bws_test_sampled(x, y, numsamples=10000):
    # Get the ranks. The first 1:n are ranks for x, (n+1):(n+m) for y.
    # ranks are 0-based, but we want 1-based
    rs = np.argsort(np.argsort(np.concatenate([x, y]), kind='stable')) + 1

    return bws_test_sampled_from_ranks(rs, len(x), len(y), numsamples)
