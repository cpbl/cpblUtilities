#!/usr/bin/python

import matplotlib.pyplot as plt

def prepare_figure_for_publication(ax=None,
        width_cm=None,
        width_inches=None,
                                   height_cm=None,
        height_inches=None,
        fontsize=9,
        fontsize_labels=None,
        fontsize_legend=None,
                                   fontsize_annotations =None,
        ):
    """
    """
    fuck
    #ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fontsize)
