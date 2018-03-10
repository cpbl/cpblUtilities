#!/usr/bin/python

import matplotlib.pyplot as plt

def prepare_figure_for_publication(ax=None,
        width_cm=None,
        width_inches=None,
                                   height_cm=None,
        height_inches=None,
                                   fontsize=None,
        fontsize_labels=None,
        fontsize_legend=None,
                                   fontsize_annotations =None,
        ):

    """
    One reasonable option for making compact figures like for Science/Nature is to create everything at double scale. 
    This works a little more naturally with Matplotlib's default line/axis/etc sizes.

    Also, if you change sizes of, e.g. xticklabels and x-axis labels after they've been created, they will not necessarily be relocated appropriately.
    So  you can call prepare_figure_for_publication with no ax/fig argument to set up figure defaults
    prior to creating the figure in the first place.

    """

    fig = ax.get_figure() 
    if width_inches:
        fig.set_figwidth(width_inches)
        assert width_cm is None
    if height_inches:
        fig.set_figheight(height_inches)
        assert height_cm is None
    if width_cm:
        fig.set_figwidth(width_cm/2.54)
        assert width_inches is None
    if height_cm:
        fig.set_figheight(height_cm/2.54)
        assert height_inches is None
    
    #ax = plt.subplot(111, xlabel='x', ylabel='y', title='title')
    for item in fig.findobj(plt.Text) + [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
        if fontsize:
            item.set_fontsize(fontsize)

                        
