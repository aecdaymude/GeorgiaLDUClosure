# Project:  GeorgiaLDUClosure
# Filename: helper.py
# Authors:  Joshua J. Daymude (jdaymude@asu.edu).

"""
helper: Helper functions for the GeorgiaLDUClosure analysis.
"""

from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import pandas as pd
import scipy.stats as ss

# Set all plotting to use serif fonts for better matching with the manuscript.
matplotlib.rcParams['font.family'] = 'serif'


def show_df(df, show=True):
    """
    Displays a pandas DataFrame using display() for Jupyter notebooks if and
    only if show is True; otherwise, nothing is displayed.

    :param df: a pandas DataFrame
    :param show: True iff the DataFrame should be displayed
    """
    if show:
        display(df)


def count_stats(df, measure):
    """
    Reports the total, medians, minimums, and maximums of the given Count
    measure by closure status and compares their distributions using a
    Mann-Whitney U test for significance.

    :param df: a pandas DataFrame containing a 'Closed 2012-2016' column
    :param measure: a string column name of numeric data to report stats on
    """
    # Group by closure status.
    gb = df.groupby('Closed 2012-2016', as_index=False)
    open = gb.get_group(0)[measure]
    closed = gb.get_group(1)[measure]

    # Perform statistical test.
    _, pval = ss.mannwhitneyu(open.dropna(), closed.dropna(), method='exact')

    # Report results.
    print(measure + '\n' + ''.join(['-' for i in range(len(measure))]))
    print("{:<12}{:,.3f}".format('Total:', df[measure].sum()))
    print("{:<12}median={:<12,.3f}({:,.3f} - {:,.3f})"\
          .format('Open:', open.median(), open.min(), open.max()))
    print("{:<12}median={:<12,.3f}({:,.3f} - {:,.3f})"\
          .format('Closed:', closed.median(), closed.min(), closed.max()))
    print("{:<12}pval={:.9f}".format('Mann-Whit:', pval))


def proportion_stats(df, numer_measure, denom_measure):
    """
    Similar to the above function for Count measures but takes two columns, a
    numerator and a denominator, which together form a Proportion measure.
    Reports the total of the numerator column and the medians, minimums, and
    maximums of the Proportion, using a Mann-Whitney U test for significance.

    :param df: a pandas DataFrame containing a 'Closed 2012-2016' column
    :param numer_measure: a string column name of numeric data to treat as the
                          Proportion's numerator
    :param denom_measure: a string column name of numeric data to treat as the
                          Proportion's denominator
    """
    # Create a new DataFrame with the additional % column and group by closure.
    prop_df = df.copy()
    prop_df['%'] = prop_df[numer_measure] / prop_df[denom_measure]
    gb = prop_df.groupby('Closed 2012-2016', as_index=False)
    open = gb.get_group(0)['%']
    closed = gb.get_group(1)['%']

    # Perform statistical test.
    _, pval = ss.mannwhitneyu(open.dropna(), closed.dropna(), method='exact')

    # Report results.
    print('% ' + numer_measure)
    print(''.join(['-' for i in range(len(numer_measure) + 2)]))
    print("{:<12}{:,.3f}".format('Total:', prop_df[numer_measure].sum()))
    print("{:<12}median={:<12,.3%}({:,.3%} - {:,.3%})"\
          .format('Open:', open.median(), open.min(), open.max()))
    print("{:<12}median={:<12,.3%}({:,.3%} - {:,.3%})"\
          .format('Closed:', closed.median(), closed.min(), closed.max()))
    print("{:<12}pval={:.9f}".format('Mann-Whit:', pval))


def two_by_two(df, ind1, ind2):
    """
    Reports the values of a 2x2 contingency table with the given independent
    variables against LDU closure status. Used as input to the OpenEpi
    calculator which has both Fisher exact tests and Mantel-Haenszel ratios.

    :param df: a pandas DataFrame containing a 'Closed 2012-2016' column
    :param ind1: a string column name of the first independent variable
    :param ind2: a string column name of the second independent variable
    """
    # Group by closure status.
    gb = df.groupby('Closed 2012-2016')
    open, closed = gb.get_group(0), gb.get_group(1)

    # Print the 2x2 contingency table data.
    k = max(len(ind1), len(ind2)) + len(', Closed:') + 2
    print("{:<{}}{:,.2f}".format(ind1+', Closed:', k, closed[ind1].sum()))
    print("{:<{}}{:,.2f}".format(ind2+', Closed:', k, closed[ind2].sum()))
    print("{:<{}}{:,.2f}".format(ind1+', Open:', k, open[ind1].sum()))
    print("{:<{}}{:,.2f}".format(ind2+', Open:', k, open[ind2].sum()))


def plot_medians(labels, openvals, closedvals, pvals, barwidth=0.25, fmt='{}', \
                 anno=''):
    """
    Plots a grouped bar graph for medians of measures by open and closed LDUs.
    Labels the bars with their corresponding medians and the groups with their
    Mann-Whitney U test p-values.

    :param labels: a list of N string labels for the measures
    :param openvals: a list of N median measure values for open LDUs
    :param closedvals: a list of N median measure values for closed LDUs
    :param pvals: a list of N p-values for the Mann-Whitney tests
    :param barwidth: the width of the bars; should be in (0, 0.5)
    :param fmt: a string format for the bar labels
    :param anno: a string annotation for the plot filename
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300, tight_layout=True)

    # Plot the open/closed median bars.
    x = np.arange(len(labels))
    obars = ax.bar(x - 5*barwidth/8, openvals, barwidth, hatch='\\\\\\', \
                   edgecolor='w', label='LDU Remained Open')
    cbars = ax.bar(x + 5*barwidth/8, closedvals, barwidth, color='firebrick', \
                   label='LDU Closed')

    # Add the bar value labels.
    ax.bar_label(obars, labels=[fmt.format(v) for v in openvals], padding=5)
    ax.bar_label(cbars, labels=[fmt.format(v) for v in closedvals], padding=5)

    # Get the largest y-value for spacing purposes.
    ymax = max(openvals + closedvals)

    # Add the median p-values, bolding and starring them if they're significant.
    for i, pval in enumerate(pvals):
        if pval < 0.05:
            pstr = 'p = {:.3f}*'.format(pval)
            pwgt = 'bold'
        else:
            pstr = 'p = {:.3f}'.format(pval)
            pwgt = 'normal'
        ax.text(x[i], 1.1*ymax, pstr, ha='center', weight=pwgt)

    # Clean up the plot and set labels.
    ax.spines.left.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.set(xticks=x, xticklabels=labels, ylim=[0,1.3*ymax], yticks=[])
    ax.legend()

    # Save the figure.
    fig.patch.set_facecolor('white')
    fig.savefig(osp.join('figs', 'medians' + anno + '.png'))
