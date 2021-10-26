# Project:  GeorgiaLDUClosure
# Filename: helper.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
helper: Helper functions for the GeorgiaLDUClosure analysis.
"""

import itertools
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


def statsbyclosure(df, measure):
    """
    Reports means, medians, minimums, and maximums of the given measure by
    closure status and compares the distributions using an Independent 2-Sample
    t-Test and a Mann-Whitney U Rank Test for significance.

    :param df: a pandas DataFrame containing PCSA data
    :param measure: a string column name to report stats on
    """
    gb = df.groupby('Closed 2012-2016', as_index=False)
    open = gb.get_group(0)[measure]
    closed = gb.get_group(1)[measure]

    _, pval_t = ss.ttest_ind(open.dropna(), closed.dropna())
    _, pval_mw = ss.mannwhitneyu(open.dropna(), closed.dropna(), method='exact')

    print(measure + '\n' + ''.join(['-' for i in range(len(measure))]))
    print('Total:\t\t{:,.3f}'.format(df[measure].sum()))
    print('Open:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(open.mean(), open.median(), open.min(), open.max()))
    print('Closed:\t\tmean={:,.3f}\tmedian={:,.3f}\t({:,.3f}-{:,.3f})'\
          .format(closed.mean(), closed.median(), closed.min(), closed.max()))
    print('2-Samp t-Test:\tpval={:.6f}'.format(pval_t))
    print('Mann-Whit Test:\tpval={:.6f}'.format(pval_mw))
