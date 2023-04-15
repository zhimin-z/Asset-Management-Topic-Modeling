
# Baumgartner-Weis-Schindler statistic and hypothesis testing.
# Copyright (c) Robert Feldt 2013 (robert.feldt@gmail.com)
#
# For license details see the LICENSE.md file in the top dir of the repository
# where you found this file.
#
# The algorithms in this file are based on the definitions of the BWS in the paper:
#
# Markus Neuhäuser, "Exact tests based on the Baumgartner-WeiB-Schindler statistic - A survey",
# Statistical Papers, vol. 46, pp. 1-30, Springer Verlag, 2005.

import numpy as np


def calculate_bws_sum(ranks, nm):
    sum = 0
    size = len(ranks)
    for i, r in enumerate(ranks):
        k = i + 1
        nom = (r - (nm / size * k)) ** 2
        denom = k / (size + 1) * (1 - k / (size + 1))
        sum += nom / denom
    return sum


def bws_statistic_from_ranks(ranks, n, m):
    nm = n + m
    
    # Calculate the BWS statistic from a set of ranks of n values from one sample
    # and m values from another sample.
    b_x = calculate_bws_sum(ranks[0:n], nm) / (m * nm)
    b_y = calculate_bws_sum(ranks[n:nm], nm) / (n * nm)

    return (b_x + b_y) * 0.5


def bws_test_sampled_from_ranks(ranks, n, m, numsamples=10000):
    b_stats = bws_statistic_from_ranks(ranks, n, m)

    # Array to save b statistics
    samples = np.zeros(numsamples)
    numlarger = 0

    # Set up for inline sorts.
    ns = np.zeros(n, dtype=np.int64)
    ms = np.zeros(m, dtype=np.int64)

    for i in range(numsamples):
        # Now shuffle ranks and then ensure they are sorted for each group.
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


def bws_test(x, y, numsamples=10000):
    r'''Perform the Baumgartner Weiss Schindler test on two independent samples.

    The Baumgartner Weiss Schindler test is a nonparametric test of 
    the null hypothesis that the distribution underlying sample `x` 
    is the same as the distribution underlying sample `y`. Unlike 
    the Kolmogorv-Smirnov, Wilcoxon, and Cramer-Von Mises tests, 
    the Baumgartner Weiss Schindler test weights the integral by 
    the variance of the difference in CDFs, emphasizing the tails of 
    the distributions, which increases the power of the test for 
    a lot of applications.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    numsamples : int, optional
        The number of samples to use in the test. Default is 10000.
    '''
    # ranks are 0-based, but we want 1-based
    rs = np.argsort(np.argsort(np.concatenate([x, y]), kind='stable')) + 1

    return bws_test_sampled_from_ranks(rs, len(x), len(y), numsamples)
