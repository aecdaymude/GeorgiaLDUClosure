# Project:  GeorgiaLDUClosure
# Filename: helper.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
helper: Helper functions for the GeorgiaLDUClosure analysis.
"""

from itertools import product
import pandas as pd
import scipy.stats as ss


def show_df(df, show=True):
    """
    Displays a pandas DataFrame using display() for Jupyter notebooks if and
    only if show is True; otherwise, nothing is displayed.

    :param df: a pandas DataFrame
    :param show: True iff the DataFrame should be displayed
    """
    if show:
        display(df)


def basic_stats(df, measure, agg='sum'):
    """
    Reports the total, sizes, and proportions of the measure by closure status.

    :param df: a pandas DataFrame
    :param measure: a string column name of numeric data to report stats on
    :param agg: a string indicating how to aggregate the measure and groups
    """
    # Group by closure status.
    gb = df.groupby('Closed 2012-2016', as_index=False)

    # Calculate basic stats.
    if agg == 'sum':
        total = df[measure].sum()
        open = gb.get_group(0)[measure].sum()
        closed = gb.get_group(1)[measure].sum()
    elif agg == 'count':
        total = len(df[measure])
        open = len(gb.get_group(0))
        closed = len(gb.get_group(1))

    # Report results.
    print(measure + '\n' + ''.join(['-' for i in range(len(measure))]))
    print('Total:\t\t{:,.3f}'.format(total))
    print('Open:\t\t{:,.3f}\t({:,.3f}%)'.format(open, 100 * open / total))
    print('Closed:\t\t{:,.3f}\t({:,.3f}%)'.format(closed, 100 * closed / total))


def count_stats(df, measure):
    """
    Reports the total, means, medians, minimums, and maximums of the given Count
    measure by closure status and compares their distributions using an
    Independent 2-Sample t-Test and a Mann-Whitney U Rank Test for significance.

    :param df: a pandas DataFrame
    :param measure: a string column name of numeric data to report stats on
    """
    # Group by closure status.
    gb = df.groupby('Closed 2012-2016', as_index=False)
    open = gb.get_group(0)[measure]
    closed = gb.get_group(1)[measure]

    # Perform statistical tests.
    _, pval_t = ss.ttest_ind(open.dropna(), closed.dropna())
    _, pval_mw = ss.mannwhitneyu(open.dropna(), closed.dropna(), method='exact')

    # Report results.
    print(measure + '\n' + ''.join(['-' for i in range(len(measure))]))
    print('Total:\t\t{:,.3f}'.format(df[measure].sum()))
    print('Open:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(open.mean(), open.median(), open.min(), open.max()))
    print('Closed:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(closed.mean(), closed.median(), closed.min(), closed.max()))
    print('2-Samp t-Test:\tpval={:.6f}'.format(pval_t))
    print('Mann-Whit Test:\tpval={:.6f}'.format(pval_mw))


def proportion_stats(df, numer_measure, denom_measure):
    """
    Similar to the above function for Count measures but takes two columns, a
    numerator and a denominator, which together form a Proportion measure.
    Reports the total of the numerator column and the means, medians, minimums,
    and maximums of the Proportion, using an Independent 2-Sample t-Test and a
    Mann-Whitney U Rank Test for significance.

    :param df: a pandas DataFrame
    :param numer_measure: a string column name of numeric data to treat as the
                          Proportion's numerator
    :param denom_measure: a string column name of numeric data to treat as the
                          Proportion's denominator
    """
    # Create a new DataFrame with the additional % column and group by closure.
    prop_df = df.copy()
    prop_df['%'] = 100 * prop_df[numer_measure] / prop_df[denom_measure]
    gb = prop_df.groupby('Closed 2012-2016', as_index=False)
    open = gb.get_group(0)['%']
    closed = gb.get_group(1)['%']

    # Perform statistical tests.
    _, pval_t = ss.ttest_ind(open.dropna(), closed.dropna())
    _, pval_mw = ss.mannwhitneyu(open.dropna(), closed.dropna(), method='exact')

    # Report results.
    print('% ' + numer_measure)
    print(''.join(['-' for i in range(len(numer_measure) + 2)]))
    print('Total:\t\t{:,.3f}'.format(prop_df[numer_measure].sum()))
    print('Open:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(open.mean(), open.median(), open.min(), open.max()))
    print('Closed:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(closed.mean(), closed.median(), closed.min(), closed.max()))
    print('2-Samp t-Test:\tpval={:.6f}'.format(pval_t))
    print('Mann-Whit Test:\tpval={:.6f}'.format(pval_mw))


def odds_ratio(df_ind1, df_ind2):
    """
    Reports the odds ratio and significance of a 2x2 contingency table with
    the given independent variables against LDU closure status.

    :param df_ind1: a pandas DataFrame containing closure status and the first
                    independent variable
    :param df_ind2: a pandas DataFrame containing closure status and the second
                    independent variable
    """


    odds, pval = ss.fisher_exact([[ind1dep1, ind1dep2], [ind2dep1, ind2dep2]])
