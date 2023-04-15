import numpy as np
from itertools import permutations
from collections import namedtuple


def calculate_bws_sum(ranks, nm):
    '''Calculate the sum for the Baumgartner-Weiss-Schindler statistic'''
    sum = 0
    size = len(ranks)
    factor = 1 / (size + 1)
    for i, r in enumerate(ranks):
        k = i + 1
        # ranks are 0-based, but we want 1-based
        r = r + 1
        sum += (r - (nm / size * k)) ** 2 / (k * factor * (1 - k * factor))
    return sum


def bws_statistic_from_ranks(ranks, n, m):
    '''Calculate the Baumgartner-Weiss-Schindler statistic from a set of ranks of n values from one sample'''
    nm = n + m

    # Calculate the BWS statistic from a set of ranks of n values from one sample
    # and m values from another sample.
    b_x = calculate_bws_sum(ranks[0:n], nm) / (m * nm)
    b_y = calculate_bws_sum(ranks[n:nm], nm) / (n * nm)
    b = (b_x + b_y) * 0.5

    return b


def bws_test_exact_from_ranks(ranks, n, m):
    b_stats = bws_statistic_from_ranks(ranks, n, m)

    numlarger = 0
    numtotal = 0

    for perm in permutations(ranks):
        perm_ranks = np.concatenate(
            (sorted((perm[:-m])), sorted(list(perm[n:]))))

        # Now we can calculate the bws again and note if it was larger than before.
        sample_stats = bws_statistic_from_ranks(perm_ranks, n, m)

        if sample_stats >= b_stats:
            numlarger += 1

        numtotal += 1

    # A larger (or equal) BWS statistic value indicates samples from distributions
    # that are "more apart" and thus that there is a difference between the samples.
    pvalue = numlarger / numtotal

    return b_stats, pvalue


BWSResult = namedtuple('BWSResult', ('statistic', 'pvalue'))


def bws_test_sampled_from_ranks(ranks, n, m, numsamples):
    '''Random permutation test for the Baumgartner-Weiss-Schindler test.'''
    b_stats = bws_statistic_from_ranks(ranks, n, m)

    # Array to save b statistics
    numlarger = 0

    for _ in range(numsamples):
        # Now shuffle ranks and then ensure they are sorted for each group.
        np.random.shuffle(ranks)
        ranks[:-m] = sorted(ranks[:-m])
        ranks[n:] = sorted(ranks[n:])

        # Now we can calculate the bws again and note if it was larger than before.
        sample_stats = bws_statistic_from_ranks(ranks, n, m)
        numlarger += 1 if sample_stats >= b_stats else 0

    # A larger (or equal) BWS statistic value indicates samples from distributions
    # that are "more apart" and thus that there is a difference between the samples.
    pvalue = numlarger / numsamples

    return b_stats, pvalue


BWSResult = namedtuple('BWSResult', ('statistic', 'pvalue'))


def bws_test(x, y, method='asymptotic', numsamples=1000000):
    r'''Perform the Baumgartner-Weiss-Schindler (BWS) test on two independent samples.

    The Baumgartner-Weiss-Schindler test is a nonparametric test of 
    the null hypothesis that the distribution underlying sample `x` 
    is the same as the distribution underlying sample `y`. Unlike 
    the Kolmogorv-Smirnov, Wilcoxon, and Cramer-Von Mises tests, 
    the Baumgartner-Weiss-Schindler test weights the integral by 
    the variance of the difference in CDFs, emphasizing the tails of 
    the distributions, which increases the power of the test for 
    a lot of applications.

    Parameters
    ----------
    x, y : array-like
        N-d arrays of samples. The arrays must be broadcastable except along
        the dimension given by `axis`.
    numsamples : int, optional
        The number of samples to use in the test. Default is 1000000.
        The closer to the number of permutations, the more accurate the test. 

    Returns
    -------
    res : BWSResult
        An object containing attributes:

        statistic : float
            The Baumgartner-Weiss-Schindler statistic corresponding with sample `x` and `y`.
        pvalue : float
            The associated *p*-value for the chosen `alternative`.
    See Also
    --------
    scipy.stats.wilcoxon, scipy.stats.mannwhitneyu, scipy.stats.ttest_ind

    References
    ----------
    .. [1] Markus Neuhäuser, "Exact tests based on the Baumgartner-WeiB-Schindler statistic - A survey", 
           Statistical Papers, vol. 46, pp. 1-30, Springer Verlag, 2005.

    Examples
    --------
    We follow the example of table 3 in [1]_: Fourteen children were divided randomly into two groups, 
    an experimental group and a control group. For each child the number of correct identifications 
    among 18 new examples was observed.

    >>> control_group_ranks = np.array([1, 2, 3, 4, 6, 7, 8])
    >>> experimental_group_ranks = np.array([5, 9, 10, 11, 12, 13, 14])

    We use the Baumgartner-Weiss-Schindler test to assess whether there is a 
    statistically significant difference in the methods of training preschool children.
    The null hypothesis is that the distribution of first method is the same as 
    the distribution of second one. We decide that a confidence level of 95% is 
    required to reject the null hypothesis in favor of the alternative that 
    the distributions are different.
    Since the number of samples is very small and there are no ties in the
    data, we can compare the observed test statistic against the *exact*
    distribution of the test statistic under the null hypothesis.

    >>> from scipy.stats import bws_test
    >>> B, p = bws_test(males, females, method="exact")
    >>> print(B)
    17.0

    This agrees with :math:`B = 5.132` reported in [1]_. The *p*-value produced
    by `bws_test` agrees with :math:`p = 0.0029` reported in [1]_.

    >>> print(p)
    0.1111111111111111

    It normally takes a long time to calculate the exact *p*-value, so we
    usually approximate it using a random permutation test. Here we use
    `method="asymptotic"` to perform a random permutation test with 1000000 samples.

    >>> from scipy.stats import bws_test
    >>> B, p = bws_test(males, females, method="asymptotic", numsamples=1000000)
    >>> print(p)
    0.1111111111111111

    Under this assumption, the *p*-value would be low enough to reject the
    null hypothesis in favor of the alternative.
    '''
    rs = np.argsort(np.argsort(np.concatenate([x, y]), kind='stable'))
    if method == 'exact':
        b_stats, pvalue = bws_test_exact_from_ranks(rs, len(x), len(y))
    elif method == 'asymptotic':
        b_stats, pvalue = bws_test_sampled_from_ranks(
            rs, len(x), len(y), numsamples)

    return BWSResult(b_stats, pvalue)
