#!/usr/bin/python

import matplotlib.pyplot as plt

def prepare_figure_for_publication(ax=None,
        width_cm=None,
        width_inches=None,
                                   height_cm=None,
        height_inches=None,
                                   fontsize=None,
        fontsize_labels=None,
        fontsize_ticklabels=None,
        fontsize_legend=None,
                                   fontsize_annotations =None,
                                   TeX = True, # Used for ax=None case (setup)
        ):

    """
    Two ways to use this:
    (1) Before creating a figure, with ax=None
    (2) To fine-tune a figure, using ax

    One reasonable option for making compact figures like for Science/Nature is to create everything at double scale. 
    This works a little more naturally with Matplotlib's default line/axis/etc sizes.

    Also, if you change sizes of, e.g. xticklabels and x-axis labels after they've been created, they will not necessarily be relocated appropriately.
    So  you can call prepare_figure_for_publication with no ax/fig argument to set up figure defaults
    prior to creating the figure in the first place.


Some wisdom on graphics:
 - 2015: How to produce PDFs of a given width, with chosen font size, etc:
   (1) Fix width to journal specifications from the beginning / early. Adjust height as you go, according to preferences for aspect ratio:
    figure(figsize=(11.4/2.54, chosen height))
   (2) Do not use 'bbox_inches="tight"' in savefig('fn.pdf').  Instead, use the subplot_adjust options to manually adjust edges to get the figure content to fit in the PDF output
   (3) Be satisfied with that. If you must get something exactly tight and exactly the right size, you do this in Inkscape. But you cannot scale the content and bbox in the same step. Load PDF, select all, choose the units in the box at the top of the main menu bar, click on the lock htere, set the width.  Then, in File Properties dialog, resize file to content. Save.

    """
    if ax is None: # Set up plot settings, prior to creation fo a figure
        params = { 'axes.labelsize': fontsize_labels if fontsize_labels is not None else fontsize,
                   'font.size': fontsize,
                   'legend.fontsize': fontsize_legend if fontsize_legend is not None else fontsize,
                   'xtick.labelsize': fontsize_ticklabels if fontsize_ticklabels is not None else fontsize_labels if fontsize_labels is not None else fontsize,
                   'ytick.labelsize': fontsize_ticklabels if fontsize_ticklabels is not None else fontsize_labels if fontsize_labels is not None else fontsize,
                   'figure.figsize': (width_inches, height_inches),
            }
        if TeX:
            params.update({
                   'text.usetex': TeX,
                       'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
                'text.latex.unicode': True,
            })
        if not TeX:
            params.update({'text.latex.preamble':''})

        plt.rcParams.update(params)
        return
    
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

def plot_diagonal(xdata=None, ydata=None, ax=None, **args):
    """ Plot a 45-degree line 
    """
    import pandas as pd
    if ax is None: ax = plt.gca()
    #LL = min(min(df[xv]), min(df[yv])), max(max(df[xv]), max(df[yv]))
    if xdata is None and ydata is None:
        xl, yl = ax.get_xlim(), ax.get_ylim()
        LL = max(min(xl), min(yl)),    min(max(xl), max(yl)),
    elif xdata is not None and ydata is None:
        assert isinstance(xdata, pd.DataFrame)
        dd = xdata.dropna()
        LL = dd.min().max(), dd.max().min()
    else:
        assert xdata is not None
        assert ydata is not None
        #if isinstance(xdata, pd.Series): xdata = xdata.vlu
        xl, yl = xdata, ydata
        LL = max(min(xl), min(yl)),    min(max(xl), max(yl)),
    ax.plot(LL, LL,  **args)

                        
def figureFontSetup(uniform=12,figsize='paper', amsmath=True):
    """
This is deprecated. Use prepare_figure_for_publication


Set font size settings for matplotlib figures so that they are reasonable for exporting to PDF to use in publications / presentations..... [different!]
If not for paper, this is not yet useful.


Here are some good sizes for paper:
    figure(468,figsize=(4.6,2)) # in inches
    figureFontSetup(uniform=12) # 12 pt font

    for a subplot(211)

or for a single plot (?)
figure(127,figsize=(4.6,4)) # in inches.  Only works if figure is not open from last run!
        
why does the following not work to deal with the bad bounding-box size problem?!
inkscape -f GSSseries-happyLife-QC-bw.pdf --verb=FitCanvasToDrawing -A tmp.pdf .: Due to inkscape cli sucks! bug.
--> See savefigall for an inkscape implementation.

2012 May: new matplotlib has tight_layout(). But it rejigs all subplots etc.  My inkscape solution is much better, since it doesn't change the layout. Hoewever, it does mean that the original size is not respected! ... Still, my favourite way from now on to make figures is to append the font size setting to the name, ie to make one for a given intended final size, and to do no resaling in LaTeX.  Use tight_layout() if it looks okay, but the inkscape solution in general.
n.b.  a clf() erases size settings on a figure! 

    """
    figsizelookup={'paper':(4.6,4),'quarter':(1.25,1) ,None:None}
    try:
        figsize=figsizelookup[figsize]
    except KeyError,TypeError:
        pass
    params = {#'backend': 'ps',
           'axes.labelsize': 16,
        #'text.fontsize': 14,
        'font.size': 14,
           'legend.fontsize': 10,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True,
           'figure.figsize': figsize
        }
           #'figure.figsize': fig_size}
    if uniform is not None:
        assert isinstance(uniform,int)
        params = {#'backend': 'ps',
           'axes.labelsize': uniform,
            #'text.fontsize': uniform,
           'font.size': uniform,
           'legend.fontsize': uniform,
           'xtick.labelsize': uniform,
           'ytick.labelsize': uniform,
           'text.usetex': True,
           'text.latex.unicode': True,
            'text.latex.preamble':r'\usepackage{amsmath},\usepackage{amssymb}',
           'figure.figsize': figsize
           }
        if not amsmath:
            params.update({'text.latex.preamble':''})

    plt.rcParams.update(params)
    plt.rcParams['text.latex.unicode']=True
    #if figsize:
    #    plt.rcParams[figure.figsize]={'paper':(4.6,4)}[figsize]

    return(params)
