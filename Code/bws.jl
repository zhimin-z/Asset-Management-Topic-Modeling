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
#

export bws_statistic, bws_test_sampled

#############################################################################
## Part I. Calculating the BWS.
#############################################################################

# Calculate the BWS statistic from two samples of values.
function bws_statistic(x, y)
  # Get the ranks. The first 1:n are ranks for x, (n+1):end for y.
  rs = ranks(vcat(x, y))

  FeldtLib.bws_statistic_from_ranks(rs, length(x), length(y))
end

# Test if according to the Baumgartner-Weis-Schindler (BWS) statistic we
# can reject the hypothesis that the two set of samples x and y are likely to have
# the same location (mean).
function bws_test_sampled(x, y, numsamples = 10000)
  # Get the ranks. The first 1:n are ranks for x, (n+1):(n+m) for y.
  rs = ranks(vcat(x, y))

  FeldtLib.bws_test_sampled_from_ranks(rs, length(x), length(y), numsamples)
end

# Calculate the BWS statistic from a set of ranks of n values from one sample
# and m values from another sample. The first 1:n rank values in ranks are for
# the first sample and the rest are from the other.
function bws_statistic_from_ranks(ranks, n, m)
  # Pre-calc some values needed in both b calcs.
  nm = n+m
  np1 = n + 1
  mp1 = m + 1

  # Now calc the two b values
  b_x = (1/n) * (n / (m*nm)) * calc_bws_sum(ranks,   1,  n, n, np1, nm, 0)
  b_y = (1/m) * (m / (n*nm)) * calc_bws_sum(ranks, np1, nm, m, mp1, nm, n)

  (b_x + b_y) / 2
end

function calc_bws_sum(ranks, mini, maxi, n, np1, nm, subi = 0)
  sum = 0.0
  for(i in mini:maxi)
    k = i - subi
    nom = ( ranks[i] - (nm/n*k) )^2
    denom = (k/np1) * (1 - (k/np1))
    sum += (nom / denom)
  end
  sum
end

function bws_test_sampled_from_ranks(ranks, n, m, numsamples = 10000)
  rs = copy(ranks) # Copy since we will shuffle it below.

  actual_b = FeldtLib.bws_statistic_from_ranks(rs, n, m)

  # Array to save the b statistics
  samples = Array(Float64, numsamples)
  numlarger = 0

  # Set up for inline sorts. Not sure if this really speeds things up...
  ns = Array(Int64, n)
  ms = Array(Int64, m)
  np1 = n+1

  for(i in 1:numsamples)
    # Now shuffle the ranks and then ensure they are sorted for each group.
    # Not sure this is actually faster than letting julia decide on the copying
    # and gc of intermediate vectors/arrays... ;)
    shuffle!(rs)
    ns[1:n] = rs[1:n]
    ms[1:m] = rs[np1:end]
    sort!(ns)
    sort!(ms)
    rs[1:n] = ns
    rs[np1:end] = ms

    # Now we can calc the bws again and note if it was larger than before.
    samples[i] = FeldtLib.bws_statistic_from_ranks(rs, n, m)
    if samples[i] >= actual_b
      numlarger += 1
    end
  end

  # A larger (or equal) BWS statistic value indicates samples from distributions
  # that are "more apart" and thus that there is a difference between the samples.
  pvalue = numlarger / numsamples

  (actual_b, pvalue, numlarger, numsamples) # , mean(samples), std(samples))
end

#############################################################################
## Part II. Adding the same interface as used by hypothesis tests in
##          HypothesisTests.jl
#############################################################################
using HypothesisTests

type BWSTest <: HypothesisTests.HypothesisTest
  # Ranks of all values where first 1:n are from one sample and (n+1):end are
  # from the other.
  ranks::Array{Int64,1}
  n::Integer
  m::Integer

  b::Float64 # The BWS statistic itself

  # We save the numlarger and numsamples taken instead of the p-value since
  # we can then add additional iterations later if needed.
  numlarger::Integer
  numsamples::Integer
end
function BWSTest{S <: Number}(x::Vector{S}, y::Vector{S}, numsamples = 10000)
  rs = ranks(vcat(x, y))
  n = length(x)
  m = length(y)
  b, pvalue, numlarger, numsamples = bws_test_sampled_from_ranks(rs, n, m)
  BWSTest(rs, n, m, b, numlarger, numsamples)
end

function pvalue(bwst::BWSTest)
  bwst.numlarger / bwst.numsamples
end