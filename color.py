#!/usr/bin/python
import pandas as pd
import  numpy as np
import pylab as plt
from scipy import interpolate
import matplotlib as mpl

"""
Background: Colormap functions in Matplotlib are confusingly many and a bit spread out. Documentated examples are insufficient. e.g. What does "normalize" mean, precisely, in the cm context?

Here are some ways I might specify the color sequence in a colorbar (ie, what is usually called a colormap, but it's a mapping from [0,1] to colours, not from some arbitrary data).

 -  the name (string) of a built-in colormap
 -  something called a cdict (a dict), which seems the "right" way to efficiently code an arbitrary color sequence [To see one for a built-in colormap, just use plt.cm.datad['jet']. In each triplet, the first number (x) is an index for position in your colormap between 0 and 1; the second and third (y) are often the same value; they are  the amount of a certain primary color. When different, the 2nd is the limit value for y when x comes from below; the third is the limiting y value for x comes from above.
 -  a simple list/sequence of colours, which are assumed to be equally spaced along the color bar (ie [0,1] mapping)

In general, I want to be able to construct color mappings between some *data values* and some colours, and I want to be able to render a colorbar for that mapping (but where I'm not using the image functions, which coordinate colorbars for you).

Moreover, I don't want to be constrained to linear scalings between data values and color indices. Therefore, I want to map the *indices* of data onto the *indices* of colors in a color scheme, so as to maximally make use of the available color variation.  Moreover, I want to have access to these mappings so that I can use the same mapping consistently over several plots.

Therefore, two tools are fundamental to this effort: 

  (1)  First,  assignSegmentedColormapEvenly() optimally / "smartly" assigned colors to data  over the entire range of data, together, or over any number of contiguous subsets of the data, in case you want different color schemes for different ranges of data. For instance, the simplest bipolar scheme may have one color set for >=0 and one for <0.  Thus, this function takes a split-data-at argument to say which data values are the pivots between subranges (use None if there are none), and it takes a specification of the one or more color sequences to apply.  You can specify these color ranges in several ways: 
 -  a list of names (strings) of a built-in colormaps
 -  a list of cdicts
 -  a list of colors that should apply to the data pivot points  (ie with N pivot points, N+2 colours are listed)
 -  a list of color-lists, specifying the sequence of colours making up the color scheme for each subrange (ie with N pivot points, N+1 lists of colours)

 (2) A tool to build a colorbar next to any plot axis in which you've made use of colours (scatter plot, etc). The colorbar shows data values (scaled linearly) and their corresponding colours (which are not in general "linearly" arranged in terms of the underlying color scheme). This tool is addColorbarNonImage()

What about the more straightforward "linear" use of colormaps? The latter function (addColorbarNonImage()) deals also with these. For generating those mappings, however, ie to get a lookup function that goes from data values to colours, or from indexed things (e.g. string names!) to colours, use getIndexedColormap().

n.b. Others have used similar language (nonlinear colormap) and wanted the same thing. (e.g. http://protracted-matter.blogspot.ie/2012/08/nonlinear-colormap-in-matplotlib.html). But they generally fiddle with values in one axis. I want to set up the mapping and use it for various sets of data.

"""
def cdict_to_list_of_colors(cdict,N=256):
    # One has to give new colormaps names; here it is None
    return(plt.cm.colors.LinearSegmentedColormap(None,cdict)(np.arange(N)))

# oh shoot... I already had that:  Or this one is better for discrete values...
def getIndexedColormap(cmap_name,N):
    """
        This will return a sequence of RGBA values, actually an Nx4 ndarray.  I 
    don't think the inclusion of the 4th column hurts anything, but 
    obviously you can use indexing to remove it if you want to.  The last 
    line in the function would change to ...(as below)

    So is this most useful for a small discrete, or categorical, set???

    2014Feb: if a list is given, instead of N, then a dict is returned, which is a lookup from the elements of the list N to colors.
    """
    import numpy as np
    if N.__class__ in [list, np.ndarray, tuple]:
        return(dict(zip(N,getIndexedColormap(cmap_name,len(N)))))
    import numpy as np 
    import matplotlib.cm as cm 
    cmap = cm.get_cmap(cmap_name, N) 
    return cmap(np.arange(N))[:,:-1] 


def assignSplitColormapEvenly(zdata,splitdataat=0.0, RGBpoints=None): # An application of  assignSegmentedColormapEvenly(), with just two regions.
    if RGBpoints is None:
        RGBpoints=[[0.0,0.0,1.0],[0.0,0.0,0],[1.0,0.0,0.0]]  # Blue is low, cutpoint is black, red is high, 
    return(assignSegmentedColormapEvenly(RGBpoints,zdata,splitdataat=[splitdataat]))

def assignSegmentedColormapEvenly(RGBpoints_or_cmaps,zs,splitdataat=None,asDict=False,missing=[1,1,1],Nlevels=None,zlim=None):
    """

Comments need updating. See general introduction at top of file.

    July 2014. This can be called Smart mapping of colours to data. Rather than linearly interpolate, it spreads out the values to represent each colour equally. That is, the mapping is ordinal rather than cardinal.  It does this in each of any number of subranges, so for instance, you can have a polarized colour scale with green for positive, red for negative, black for zero, and make optimal use of both subranges:

    assignSegmentedColormapEvenly([[0.0,0.0,1.0],[0.0,0.0,0],[1.0,0.0,0.0]], mydata,  0.0)

This can also be achieved using the two-segment application of this function, with its default colours.

    assignSplitColormapEvenly(mydata, splitdataat=0.0, RGBpoints=None)

    An even simpler application (replacing my old function for this) is for data with a single colour scheme. This maps data optimally onto a blue-red gradient:

    assignSegmentedColormapEvenly([[0.0,0.0,1.0],[1.0,0.0,0.0]], mydata)

    "missing" color feature not yet implemented

    Nlevels is the value that defaults to N_COL_PER_SEGMENT=256 # This is how detailed our colormaps will always be, except in multiple-segment ones, they'll have this many colours per segment.

    The function returns an interpolation function (mapping data to RGB color triplets) by default, or it can be asked to return a dict (asDict=True).

    So I might use the following to get actual color RGB values for each data point:
    cc=  assignSegmentedColormapEvenly([[0.0,0.0,1.0],[1.0,0.0,0.0]], countries.RMSEs)
    countries['color']=countries.RMSEs.map(cc)


    Comment: If you plot your data using these colours, and want to show a colorbar next to the plot, you need not create any matplotlib colormaps; simply pass the mapping given by this function to my addColorbarNonImage().

    Comment: This uses a lot of memory, since even though it's couched as an interpolation function, it actually includes all the data: it's a lookup rather than an interpolator.  Next task would be to optionally decimate the lookup points (e.g. to just 256 colours), in such a way as to respect the color variation, rather than the data density. How to do that?

    Comment: What my July 24 draft of this did not do is to allow for a full linear segmented colormap WITHIN each range of data. These dicts look, for instance, like plt.cm.datad['jet'] and they give a more complex smooth colour scheme rather than just a gradient between two bounding RGB values. So why not be able to give a dict for each range? In that case, rather than passing N+1 colours, where N is the number of segments, I pass N dicts.
 So there are now two calling forms.

    Comment: For bycolors version,  I should try to make thie result more compact: to do!: create a 256 long colormap first for each segment. But if I do that, then I could rewrite the different versions to all use the same core, in which a list of colors is generated (rather than just a pair), as in the bycolors at the moment.

ie. in general, the function should take specification of the segments' colors as a list, assumed to be equally spaced. This can be generated from cmap's dict, but it is the most general way to specify a list. Absurdly, I've had a hard time getting from the dict to the list, but I guess I could transform that by hand.

I still have a problem now that I get some nans from lookup functions, ie the range doesn't go to -inf.

Comment: My fundamental method now deals with a list of RGBpoints as a list of lists of RGB values. ie each element is a list of RGBs which if spaced equally describe a colormap.  In most cases, the final colour in one element will be the first colour of the next element.  This list of "colorsets" is different from the list of colors, where each color is itself a split point (see ..._bycolors() ). And it's different (but easily derived) from a list of named colormaps, or a list of cdicts.

So lets convert from the various input forms to get to the list of list of RGBpoints, and then run the core code.


I ought to have another option: discretizeColours, which would ensure that there were a smaller (256?) set of colours used in the return lookup for the case of asDict=True.  Or maybe I should have that always happen for asDict=True.

    """
    if Nlevels is None:
        Nlevels=256 # This is how detailed our colormaps will always be, except in multiple-segment ones, they'll have this many colours per segment.
    if isinstance(zs,list): zs=np.array(zs)
        
    splitdataat= []  if splitdataat is None else splitdataat
    splitdataat= splitdataat if splitdataat.__class__ in (list,np.ndarray) else [splitdataat]
    if isinstance(RGBpoints_or_cmaps,dict) or isinstance(RGBpoints_or_cmaps,str):
        RGBpoints_or_cmaps=[RGBpoints_or_cmaps]
    if isinstance(RGBpoints_or_cmaps,(list,np.ndarray)) and  isinstance(RGBpoints_or_cmaps[0],str):
        # Assume they're all strings:
        # Let's turn it into a list of RGB-lists, each of length 256:
        loloRGBs=[getIndexedColormap(acmap,Nlevels) for acmap in RGBpoints_or_cmaps]
    # You can specify a list of cdicts:
    elif isinstance(RGBpoints_or_cmaps,(list,np.ndarray)) and isinstance(RGBpoints_or_cmaps[0],dict):
        loloRGBs=[ cdict_to_list_of_colors(cdict,N=256) for cdict in RGBpoints_or_cmaps]
    # You can specify just a list of RGB values: one per breakpoint
    elif isinstance(RGBpoints_or_cmaps,(list,np.ndarray)) and RGBpoints_or_cmaps[0][0].__class__ in [float,int,np.float64]:
        assert len(RGBpoints_or_cmaps)==len(splitdataat)+2 
        # Reorganize this into a list of pairs of colours which bound the segments: ie duplicate each interior colour which forms a transition/split point:
        loloRGBs=[ np.array([RGBpoints_or_cmaps[ii],RGBpoints_or_cmaps[ii+1]])  for ii in range(len(RGBpoints_or_cmaps)-1) ]
    else:
        oh_oh_whatDidYouPassMe
    return(_assignSegmentedColormapEvenly_bycolorsets(loloRGBs,zs,splitdataat=splitdataat, asDict=asDict,missing=missing, Nlevels=Nlevels   ))


def _assignSegmentedColormapEvenly_bycolorsets(RGBpoints,zs,splitdataat=None,asDict=False,missing=[1,1,1], Nlevels=None):
    """
    In this case, RGBpoints is a list of lists of RGB values. ie each element is a list of RGBs which if spaced equally describe a colormap.  In most cases, the final colour in one element will be the first colour of the next element.  This list of "colorsets" is different from the list of colors, where each color is itself a split point (see ..._bycolors() ).
    One can convert from the latter to the former just by doubling the borders...
    """

    import pylab
    big=1e20 # Using np.inf as limiting x values screws up interp1d

    categorical=False
    assert splitdataat is not None #splitdataat= splitdataat if splitdataat is not None else []
    assert Nlevels is not None
    if splitdataat.__class__ in (int,float,np.float64): splitdataat=[splitdataat]
    splitdataat=[-big]+list(splitdataat)+[big]
    if 0 and isinstance(zs,(list,np.ndarray)) and isinstance(zs[0],str):
        iCategories=range(len(zs))
        szs=range(len(zs))
        asDict=True
        categorical=True
        print 'Initiating categorical mode...'
    else:
        if isinstance(zs,pd.Series): zs=zs.values
        szs=np.sort(np.unique(zs[np.isfinite(zs)]))  # for zz in zs if not isnan(zz)])))  #[szs[0]- dz]  + szs  + [szs[1]+dz])
        dz=szs[-1]-szs[0] 

    ZZ=np.sort(np.unique(zs[np.isfinite(zs)]))
    nSplits=len(splitdataat)
    assert len(RGBpoints)==nSplits-1
    # Now, go through each split region. Build an interpreter from the ordinal *index* of the data to colors

    datagroups=[] # This should end up just being the original data
    dataindices=[]
    interpsegments=[]
    colorgroups=[]
    iito=0
    for ii,fromdata in enumerate(splitdataat[:-1]):
        todata=splitdataat[ii+1]
        if fromdata==todata: 
            print('Empty segment! no data........ Check me...')
            continue
        # Choose data in this segment, and data still to go:
        szs,ZZ=ZZ[ZZ<todata]  ,   ZZ[ZZ>=todata]

        # Get indices to a 256-long set of data spanning this range:
        i256szs=np.linspace(0,len(szs)-1,Nlevels).astype(int)
        # Get those data, ensuring they are unique (n.b. they will stay sorted):
        szs256 = np.unique(szs[i256szs]) 
        NN=len(szs256)

        # Interpolate given list of colours to create a list of colours the same length (NN) as these data:
        iCol=np.linspace(0,len(RGBpoints[ii])-1,  NN)
        colors256=interpolate.interp1d(range(len(RGBpoints[ii])),    np.array(RGBpoints[ii]) ,axis=0)(iCol)

        # Store our 256 (or whatever)-long data and 256-long colours:
        datagroups+=[szs256]
        colorgroups+=[colors256]

    # Now build an interpreter from the actual data value to colours:
    allD,allC=np.concatenate( datagroups),  np.concatenate(colorgroups)
    # By default, extend the mapping to +/- infinity. (With algorithm above, this is also necessary to catch top value)
    #dataToC=interpolate.interp1d(np.concatenate([[-big],allD[1:-1],[big]]) ,allC, axis=0)
    dataToC=interpolate.interp1d(np.concatenate([[-big],allD,[big]]), np.concatenate([[allC[0]], allC,[allC[-1]]]), axis=0)

    if asDict is False:
        return(dataToC) # This is a function; give it a scalar or vector of data, and it gives you a color array.
    # Do we really want to allow calling by this method? Useful for categorical data, but that's not supported anymore/just now...
    # 

    #NEXT: go back to colors. make pd.cut based on the 256 values I know. 
    z2cut = pd.cut(zs,allD)
    allD2cut=pd.cut(allD,allD)
    cut2C =dict([[cut , allC[ii]  ]  for ii, cut in enumerate(allD2cut)])
    z2color=dict(zip( zs,    [cut2C[zc] for zc in z2cut] ))
    if 0: # Don't do this. If someone's getting a dict, they can easily have a default value when they use it.
        z2color[None]=missing
        z2color[np.nan]=missing
    return(z2color)
    ###return(dict(zip(zs,dataToC(zs))))





import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    From:  http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib July 2014

    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def shiftedColorMap_example():
    """ just to show usage of above"""
    biased_data = np.random.random_integers(low=-15, high=5, size=(37,37))

    orig_cmap = matplotlib.cm.coolwarm
    shifted_cmap = shiftedColorMap(orig_cmap, midpoint=0.75, name='shifted')
    shrunk_cmap = shiftedColorMap(orig_cmap, start=0.15, midpoint=0.75, stop=0.85, name='shrunk')

    fig = plt.figure(figsize=(6,6))
    grid = AxesGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5,
                    label_mode="1", share_all=True,
                    cbar_location="right", cbar_mode="each",
                    cbar_size="7%", cbar_pad="2%")

    # normal cmap
    im0 = grid[0].imshow(biased_data, interpolation="none", cmap=orig_cmap)
    grid.cbar_axes[0].colorbar(im0)
    grid[0].set_title('Default behavior (hard to see bias)', fontsize=8)

    im1 = grid[1].imshow(biased_data, interpolation="none", cmap=orig_cmap, vmax=15, vmin=-15)
    grid.cbar_axes[1].colorbar(im1)
    grid[1].set_title('Centered zero manually,\nbut lost upper end of dynamic range', fontsize=8)

    im2 = grid[2].imshow(biased_data, interpolation="none", cmap=shifted_cmap)
    grid.cbar_axes[2].colorbar(im2)
    grid[2].set_title('Recentered cmap with function', fontsize=8)

    im3 = grid[3].imshow(biased_data, interpolation="none", cmap=shrunk_cmap)
    grid.cbar_axes[3].colorbar(im3)
    grid[3].set_title('Recentered cmap with function\nand shrunk range', fontsize=8)

    for ax in grid:
        ax.set_yticks([])
        ax.set_xticks([])


def _colorAssignmentToColormap(lookup,cmapname=None):
    print("This is deprecated. Use the new name.")
    return(_colorAssignmentToColorbarmap(lookup,cmapname=cmapname))
def _colorAssignmentToColorbarmap(d2cDict,cmapname=None):
    """
    2013 Oct: I think I need yet another function. This one takes an assignment between data values and colors, in particular when they are not linearly interpolable from the data values, and generates a new colormap which, if scaled to the data range, would match the color assignment.

    It seems the only way to make a custom colormap is through the piecewise linear method, LinearSegmentedColormap.
    So can I just do this by taking every unique value of data and making a breakpoint at each?! ugh.

    WAIT! IS THIS ONLY USEFUL FOR PLOTTING COLORBARS? Is it used only by addcolorbarnonimage? If so, it should be inside there, no?
    WAIT! I need to use a cumsum of index, not the index of unique(): no? Do a harsher test of this than currently in demo...
    2014Dec: Without dealing with the above questions, I've rewritten this to make it more robust against repeated index values.
    """
    def strictly_increasing(L):
        return all(y-x>1e-16 for x, y in zip(L, L[1:]))
    """
    def strongly_increasing(L):
        return all(y-x>.0001 for x, y in zip(L, L[1:]))
    def thinned_cdict(acdict):
        okaysteps= [[y-x>.0001 for x, y in zip([a for a,b,c in cdict[kk]], [a for a,b,c in cdict[kk]][1:])] for kk in cdict] 
        "
        allokay = 
        tokeep={}
        for kk in acdict:
            tokeep[kk]=[y-x>.0001 for x, y in zip(L, L[1:])]
        "
    """
    if cmapname is None: cmapname='tmpcm'

    allkeys=d2cDict.keys()
    # Drop inifinite values? or explain here why they arise.
    assert all( np.isfinite(allkeys) )

    kmin=min(allkeys)
    kmax=max(allkeys)
    dk=kmax-kmin
    # Put the values in order; and Scale the range to [0,1], using an integer index; 
    lookup3=sorted([[int(1e6*(a-kmin)/dk), a]+list(b)  for a,b in d2cDict.items()])
    # And use the integer index to get rid of duplicates, so as to ensure strictly increasing values:
    z2,ii=np.unique([L[0] for L in lookup3] , return_index=True)
    lookup=  [[a*1.0/1e6,b,c,d,e] for a,b,c,d,e in np.array(lookup3)[ii]]  # Recreate a dict with the subset of distinct values
    lookup[-1][0]=1.0
    assert strictly_increasing([a for a,b,c,d,e in lookup]) # By construction!

    ii,zz,rr,gg,bb=zip(*lookup)
    cccs=['red','green','blue']
    rgb=[rr,gg,bb]
    cdict=dict([ [cc,        np.array([ii,rgb[jj],rgb[jj]]).T]        for jj,cc in enumerate(['red','green','blue'])])

    plt.register_cmap(name=cmapname, data=cdict)
    return(plt.get_cmap(cmapname))


    goiu
    # Careful. unique doesn't work well on floating point arithmetic. So start by rounding all the z values (to 5 decimal places):
    lookup3=[[a]+list(b)  for a,b in d2cDict.items()]
    z2,ii= np.unique(np.round([L[0] for L in lookup3],decimals=5)   , return_index=True)
    lookup=  dict([[a,[b,c,d]] for a,b,c,d in np.array(lookup3)[ii]])  # Recreate a dict with the subset of distinct values

    # lookup should give an RGB 3-tuple for any data value given to it.   Ensure data are floats, not ints:
    zs=1.0 * np.array(  sorted(np.unique(       np.asarray(lookup.keys())[np.isfinite(lookup.keys())]    ))  )

    izs=np.arange(len(zs))*1.0/len(zs)
    izs[-1]=1.0

    # We want the index to be in cardinal values
    izs=zs -min(zs)
    izs=izs/max(izs)

    cdict={'red':[],'green':[],'blue':[]}
    cccs=['red','green','blue']
    for ii,zz in enumerate(zs):
        for ic,cc in enumerate(cccs):
            cdict[cc].append([izs[ii],lookup[zz][ic],lookup[zz][ic]])

    if 1:     # Check validity of cdict. This section is only for debuggin:

        def strictly_increasing(L):
            return all(x<y for x, y in zip(L, L[1:]))
        for kk in cdict:
            assert strictly_increasing([a for a,b,c in cdict[kk]])

    plt.register_cmap(name=cmapname, data=cdict)

    return(plt.get_cmap(cmapname))

def addColorbarNonImage_vDraft(customcmap):
    """ Dec 2014: I found this: https://datasciencelab.wordpress.com/2013/12/21/beautiful-plots-with-pandas-and-matplotlib/
    which might have a nicer (kludged!) way of adding a colorbar.  S/he cites http://stackoverflow.com/questions/8342549/
# matplotlib-add-colorbar-to-a-sequence-of-line-plots.

Actually, this does not do all that mine does, but some of its tricks might be useful to import.
TODO: absorb tricks and delete this function.
    """
    # Create a fake colorbar
    ctb = LinearSegmentedColormap.from_list('custombar', customcmap, N=2048)
    # Trick from http://stackoverflow.com/questions/8342549/
    # matplotlib-add-colorbar-to-a-sequence-of-line-plots
    sm = plt.cm.ScalarMappable(cmap=ctb, norm=plt.normalize(vmin=72, vmax=84))
    # Fake up the array of the scalar mappable
    sm._A = []

    # Set colorbar, aspect ratio
    cbar = plt.colorbar(sm, alpha=0.05, aspect=16, shrink=0.4)
    cbar.solids.set_edgecolor("face")
    # Remove colorbar container frame
    cbar.outline.set_visible(False)
    # Fontsize for colorbar ticklabels
    cbar.ax.tick_params(labelsize=16)
    # Customize colorbar tick labels
    mytks = range(72,86,2)
    cbar.set_ticks(mytks)
    cbar.ax.set_yticklabels([str(a) for a in mytks], alpha=a)

    # Colorbar label, customize fontsize and distance to colorbar
    cbar.set_label('Age expectancy (in years)', alpha=a, 
                   rotation=270, fontsize=20, labelpad=20)
    # Remove color bar tick lines, while keeping the tick labels
    cbarytks = plt.getp(cbar.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)

def addColorbarNonimage(data=None,datarange=None,data2color=None,cmap=None,useaxis=None,ylabel=None,colorbarfilename=None,location=None):
    Sorry_USE_NEW_FORMAT_NEW_NAME
    MUST_not_pass_anything_without_explicit_keyword_in_new_format
#def addColorbarNonimage(mindata,maxdata=None,useaxis=None,ylabel=None,cmap=None,colorbarfilename=None,location=None):
def addColorbarNonImage(data2color=None,data=None,datarange=None,cmap=None,useaxis=None,ylabel=None,colorbarfilename=None,location=None,ticks=None):
    """
    It adds a colorbar on the side to show a third dimension to a plot.

    "location" is not yet shown in docs, below, or in examples. It is passed straight to colobar.make_axes: [`None`|'left'|'right'|'top'|'bottom']

    Returns handle to colorbar object.
     
    colorbarfilename: if given, we want to generate an SVG(?) file containing just the colorbar (probably to add to a geographic/svg map).

    ticks: a list of data values; sets the tick values for the colorbar

    Calling forms: (Always specify keywords explicitly! )
    
    Case 1: Use a matplotlib colormap, named and registered with plt.cm, and spread it out "linearly" over the given range of data.
        Here cmap can be: None (in which case 'jet' is used), a string name of a registered colormap like "hot", or cm.colormap object thingy (CHECK)

        addColorbarNonimage(data=data, cmap=cmap)
        addColorbarNonimage(datarange=[mind,maxd], cmap=cmap)

    Case 2: Supply a mapping from data values to RGB colours; build a custom mpl colormap for this relationship.
        This mapping d2c can be:  (a) a dict or (b) an interp1 function or (c) a pandas Series.
        Either way, this mapping from data values to RGB colours is probably generated with assignSegmentedColormapEvenly() or etc
    
        Case 2a: a dict is known

        addColorbarNonimage(d2cdict)
        addColorbarNonimage(data2color= d2cdict)
        addColorbarNonimage(datarange=[low,high],data2color= d2cdict) # This may not work yet. Restrict the dict to have just the desired range.

        Case 2a: an interp1 function is known

        addColorbarNonimage(data=data, data2color= d2c_interp1)
        addColorbarNonimage(datarange=[mind,maxd], data2color=d2c_interp1)


"""

    import matplotlib as mpl
    import scipy
    def check_if_numeric(a): # Tell a float or numpy float from  arrays, strings
       try:
           float(a)
       except (ValueError,TypeError) as e: 
           return False
       return True
    def get_data_range(data,datarange,data2color):
        if datarange is not None: return(datarange)
        if isinstance(data, (list, np.ndarray, pd.Series)):
            return(min(data),max(data))
        if isinstance(data2color, (list, np.ndarray,pd.Series)):
            return(min(data2color),max(data2color))
        if isinstance(data2color, dict):
            return(min(data2color.keys()),max(data2color.keys()))

    if useaxis is None: useaxis=plt.gca()
    assert data is None or data.__class__ in [list, np.ndarray]
    assert datarange is None or isinstance(datarange,(list, np.ndarray))
    # We need information on the range of data:
    assert data is not None or ( isinstance(datarange,(list, np.ndarray))   and len(datarange)==2 ) or ( isinstance(data2color,(list, np.ndarray))   and len(data2color)==2 ) or isinstance(data2color,dict)
    # Specify either a color sequence (mapped from [0,1]) or a mapping from data values to RBG colours
    assert cmap is None or data2color is None 
    # We can specify cmap in two different ways, or induce the default value (with None)
    assert cmap is None or (isinstance(cmap,basestring) and cmap in plt.colormaps()) or isinstance(cmap,mpl.colors.LinearSegmentedColormap)
    assert data is None or datarange is None # Provide only the data or the range. For the moment, you cannot set a colorbar ylim that is different from the data range.

    mindata,maxdata=get_data_range(data,datarange,data2color)

    # Need to explicitly figure out calling form...
    
    knowncmap=  data2color is None

    if knowncmap: # cmap can be passed as a string or a cmap or an mpl.colors.LinearSegmentedColormap
        if isinstance(cmap,basestring): 
            cmapD=plt.cm.get_cmap(cmap)
        if cmap is None:
            cmapD = mpl.cm.jet
        elif cmap.__class__ in [mpl.colors.LinearSegmentedColormap]:
            cmapD=cmap 

    if not knowncmap:   # Prepare a cmap for if it's not specified. To do that, we need a dict mapping
        cmapName='_tmp'+str(mindata)+str(mindata)
        if isinstance(data2color,dict):
            d2cDict=data2color
        elif isinstance(data2color,pd.Series):
            d2cDict=data2color.to_dict()
        elif isinstance(data2color,scipy.interpolate.interpolate.interp1d):
            if data is None: # Hard-code here to make a colormap using a linspace 256 long.  May have edge value problems?
                dx=maxdata-mindata
                data=np.arange(mindata,maxdata+dx*1.0/256,dx*1.0/256)
            d2cDict=dict([[xx,data2color(xx)] for xx in sorted(data)]) # series? ...
        else:
            raise('Cannot make d2c dict... Stuck')
        cmapD=  _colorAssignmentToColorbarmap(d2cDict,cmapname=cmapName )
        plt.register_cmap(cmap=cmapD)

    cmap=cmapD
        
    # Now, I believe the following is good, although maybe not if there are numerous duplicates in case when a lookup is passed (?)
    cnorm = mpl.colors.Normalize(vmin=mindata,vmax=maxdata)
    cb1 = mpl.colorbar.ColorbarBase(mpl.colorbar.make_axes(useaxis,pad=0,location=location)[0], cmap=cmap,norm=cnorm,orientation='horizontal' if location in ['top','bottom'] else 'vertical')
    if ticks is not None: cb1.set_ticks(ticks)
    cbax=plt.gca()

    if ylabel is not None:
        cbax.set_ylabel(ylabel)
    return(cb1)#cbax)

    """ OLD COMMENTS BELOW ARE TRASH.nov2014

     Below, cmap can be 
            - a dict or an interp1 function, either way a mapping from data values to RGB colours
            - a simple colormap, named and registered with plt.cm. In this case we  assume that it is being spread linearly between the minimum and maximum datavalues. Thus, display the colormap linearly with axis indicating corresponding data values
    addColorbarNonimage(mindata,maxdata,useaxis=None,ylabel=None,cmap=None)
###    addColorbarNonimage(mindata,maxdata=None,useaxis=None,ylabel=None,cmap=None)
    addColorbarNonimage(data,useaxis=None,ylabel=None,cmap=None)
    

        --> Specify a colormap, and
      
    addColorbarNonimage(data2colorLookup,useaxis=None,ylabel=None)

        --> Specify a mapping between data values and RGB 3-tuples.  This can be a nonlinear mapping built from a standard colormap using assignColormapEvenly() .... [ie a dict or a pd.Series lookup]
 I've noticed that some boundary values may not get defined (due to rounding errors?) if a function interp1 is passed.  So let's just reject any values that result in nan:

Comments:    This should get used in my scatterplot functions.

2014: Nov: still not accepting interp1 functions?
    """


def reverse_cdict(cdict):
    """ Take a matplotlib color cdict, and return its mirror image
    """
    cdict_r=dict([[aa,[(1-a,b,c) for a,b,c in bb[::-1]]] for aa,bb in cdict.items()])
    return(cdict_r)










def demoCPBLcolormapFunctions():
    """
I believe the following may be out of date. Moved (nov2014) here fro cpblUtilities.mathgraph but doesn't yet work.

    Agh. I've clearly got too many functions! So try to make sense of them.
    This is only a start; I need comments and differentiation.

    Btw, n.b. assigncolormapevenly takes the output of getindexedcolormap as an argument...

    Maybe should I make a wrapper called assignColormap which takes mode "linear" or "ordinal" for how it stretches out the colors.


    Here are all the cases one needs to deal with:

    (1) - simple: take a given dataset and linearly apply an existing/built-in colormap to it.

        - get a mapping from data values to colours
        - add a colorbar to any figure

    (2) - nonlinear: take a given dataset and a built-in colormap but use the colors to maximum effect by spreading them out of data by ordinal position rather than by cardinal value
        - get a mapping from data values to colours
        - generate a new colormap object/type which is stretched in the same way
        - add a colorbar to any figure

    (3) - segment a colormap, e.g. for polarity: take a given dataset but make a colormap which has different color scheme for different polarity of data values (or at other breakpoints)
        - get a mapping from data values to colours
        - generate a new colormap object/type which is stretched in the same way
        - add a colorbar to any figure

    (4) - take a set of discrete categories, and assign colours to them (in order), using a built-in color scheme /order
    

    The following are used below:
    
addColorbarNonimage
getIndexedColormap
assignColormapEvenly
_colorAssignmentToColormap # Nope! This is now only for use within addColorbarNonimage, since I don't think it's useful except for plotting.
plot_biLinearColorscale

    """
    from cpblUtilities.color import addColorbarNonimage, getIndexedColormap
    import random

    x=np.arange(10,1200,.9)
    y=np.sin(x/50.0)#+random.random()*x

    plt.figure(1)
    plt.clf()
    lclL=linearColormapLookup('hot',y)
    for ii in range(len(y)):    plt.scatter(x[ii],-.5,color=lclL(y[ii]),s=500,linewidths=0,edgecolors='none')
    plt.show()
    ax=plt.gca()

    addColorbarNonimage(min(y),max(y),useaxis=None,ylabel=None, cmap = mpl.cm.hot)
    plt.show()
    
    plt.axes(ax)
    aceInterp=assignColormapEvenly(getIndexedColormap('hot',len(y)),
                             y,asDict=False,missing=[1,1,1]) # T#plt.cm.hot
    aceLookup=assignColormapEvenly(getIndexedColormap('hot',len(y)),
                             y,asDict=True,missing=[1,1,1]) # T#plt.cm.hot
    for ii in range(len(y)):
        plt.scatter(x[ii],.5,color=aceInterp(y[ii]),s=500,linewidths=0,edgecolors='none')
    plt.show()

    addColorbarNonimage(aceLookup,useaxis=ax,ylabel=None) # Use new custom-map feature!

    plt.show()

    # Check that we got it right:
    plt.axes(ax)
    for ii in arange(-.999,.999,.01):    plt.scatter(1400,ii,color=aceInterp(ii),s=500,linewidths=0,edgecolors='none')
    plt.ylim([-1,1])
    plt.show()
    
    plt.figure(2)
    xx=plot_biLinearColorscale(None,y)
    plt.show()

    return


def cpblColorDemos():
    from pylab import close, figure, imshow, colorbar, show,title,hist,gca,subplot
    close('all')

    shiftedColorMap_example()

    mydata=1.0*np.array([4,5,6,3,21,4,6,7,4,3,6,8,8,9,8,7,7,55,-3,-7,-1])

    figure(871)
    subplot(121)
    hist(mydata,bins=50)
    addColorbarNonImage(datarange=[min(mydata),max(mydata)],useaxis=None,ylabel=None,cmap=None)
    subplot(122)
    hist(mydata,bins=50)
    title('addColorbarNonimage(data=mydata)')
    addColorbarNonImage(data=mydata)

    figure(873)
    #We can capture the data-to-color mapping with an interp1 lookup function, as here, or with a dict, as in the next example
    data2colorInterpFcn=assignSplitColormapEvenly(mydata,splitdataat=0.0,)# RGBpoints=None) # An application of  assignSegmentedColormapEvenly(), with just two regions.
    hcb= addColorbarNonImage(data=mydata,data2color=data2colorInterpFcn)
    title('addColorbarNonimage(data=mydata,data2color=assignSplitColormapEvenly(mydata,splitdataat=0.0,) ) ')


    # Make some illustrative fake data:
    x = np.arange(-10,10, 0.1)
    y = np.arange(-10,10, 0.1)
    X, Y = np.meshgrid(x,y)
    Z = X+Y

    x = np.arange(0, np.pi, 0.1)
    y = np.arange(0, 2*np.pi, 0.1)
    X, Y = np.meshgrid(x,y)
    Z = np.cos(X) * np.sin(Y) * 10

    figure(875)
    pc1,pc2= [[0,0,1],[0,.3,0],[1,0,0]], [[1,0,0],[0,.3,0],[0,0,1]] # Red / dark green / blue    (and its reverse)
    pc1,pc2= [[0,0,1],[.1,0,.1],[1,0,0]], [[1,0,0],[.1,0,.1],[0,0,1]] # Red / purple / blue    (and its reverse)
    pc1,pc2= [[0,0,1],[1,1,1],[1,0,0]], [[1,0,0],[1,1,1],[0,0,1]] # Red / white / blue    (and its reverse)  [I want to make the middle transparent, not white!]
    data2color=assignSegmentedColormapEvenly(pc2, mydata,0.0,asDict=True) # asDict=False,missing=[1,1,1], noplot=True, mapname='stwotwo')
    # And note that once I know my data->colors lookup as a dict, I can get from there to a mpl cmap object like this:
    mycmap=_colorAssignmentToColorbarmap(data2color)
    # One can always look inside a colormap object by first *registering it*, and then accessing it by name (in this case, a temporary one)
    # Actually, let's ovewrite the name:
    #mycmap.name= 'test_cmap'
    plt.register_cmap(cmap=mycmap)
    # Check that mpl now knows it:
    mycmap.name in sorted( plt.colormaps() )
    # We can recover the details of a cm with:
    print(plt.cm.datad[mycmap.name])

    subplot(141)
    hist(mydata,bins=50)
    ax=plt.gca()
    hcb=addColorbarNonImage(data=mydata,cmap=mycmap)
    subplot(142)
    plt.title("Don't do this!")
    hist(mydata,bins=50)
    # Here is a gotcha: Remember that a cmap is not a data mapping; it knows nothing about data values (despite the yticks shown on the colorbar!). So addColorbarNonimage cannot show a subset of the full data range if you generated the colormap with the full range. So do NOT do:
    hcb2=addColorbarNonImage(datarange=[min(mydata),5],cmap=mycmap)#,useaxis=ax)
    subplot(143)
    plt.title("Don't do this either!")
    # Instead, you must draw the colorbar using the data range used to make the colormap.  
    # Then, if you want to change the ylim, make sure to use the scaled ylim values, ie in the range 0 to 1.
    # For some reason, that also shrinks the bar and moves it up, with a bug in the border line for it:
    hist(mydata,bins=50)
    hcb3=addColorbarNonImage(data=mydata,cmap=mycmap)#,useaxis=ax)
    hcb3.set_ylim(0, ( 5-min(mydata)) /   (max(mydata)-min(mydata)))

    subplot(144)
    

    # What is the correct way? If I want to have my color mapping resolution go beyond what is shown on the colorbar, I can't just truncate the data I pass when making the colorbar.
    # I think it will be inside addcolorbarnonimage, take the final colormap, create a cdict from it?! and edit the cdict (laborious!) to have the colormap end at the appropriate place. Hard.

    plt.show()
    foiuwer

    plt.imshow(Z, interpolation='nearest', cmap='_tmp-755')
    plt.colorbar()
    hcb=addColorbarNonimage(mydata,cmap=data2colorInterpFcn)
    title('addColorbarNonimage(mydata,cmap=  assignSplitColormapEvenly(mydata,splitdataat=0.0,) ) ')


    plt.show()
    foiuwer

    
    data2colorInterpFcn=assignSplitColormapEvenly(mydata,splitdataat=0.0, RGBpoints=None) # An application of  assignSegmentedColormapEvenly(), with just two regions.
    addColorbarNonimage(mydata,cmap=data2colorInterpFcn,useaxis=None,ylabel=None)

    plt.figure(7777)
    plt.imshow(Z, interpolation='nearest', cmap='_tmp-755')
    plt.colorbar()



    data2colorInterpFcn=assignSegmentedColormapEvenly([[0.0,0.0,1.0],[0.0,0.0,0],[1.0,0.0,0.0]], mydata,  0.0)
    data2colorDict=assignSegmentedColormapEvenly([[0.0,0.0,1.0],[0.0,0.0,0],[1.0,0.0,0.0]], mydata,  0.0,asDict=True)
    data2colorDict=assignSegmentedColormapEvenly(mydata,  splitdataat=0.0,asDict=True)
    addColorbarNonimage(data2colorDict,useaxis=None,ylabel=None)
    fossssssssssssssssart
    #This can also be achieved using the two-segment application of this function, with its default colours.
    assignSplitColormapEvenly(mydata, splitdataat=0.0, RGBpoints=None)

    #An even simpler application (replacing my old function for this) is for data with a single colour scheme. This maps data optimally onto a blue-red gradient:

    assignSegmentedColormapEvenly([[0.0,0.0,1.0],[1.0,0.0,0.0]], mydata)
    foi

    addColorbarNonimage(mindata,maxdata,useaxis=None,ylabel=None,cmap=None)
    addColorbarNonimage(mindata,maxdata=None,useaxis=None,ylabel=None,cmap=None)
    addColorbarNonimage(data,useaxis=None,ylabel=None,cmap=None)

    #    --> Specify a colormap, and assume that it is being spread linearly between the minimum and maximum datavalues. Thus, display the colormap linearly with axis indicating corresponding data values
      
    addColorbarNonimage(data2colorLookup,useaxis=None,ylabel=None)

    #    --> Specify a mapping between data values and RGB 3-tuples.  This can be a nonlinear mapping built from a standard colormap using assignColormapEvenly() .... [ie a dict or a pd.Series lookup]
    #I've noticed that some boundary values may not get defined (due to rounding errors?) if a function interp1 is passed.  So let's just reject any values that result in nan:


#################### CAUTION: linearColormapLookup and assignColormapEvenly are not yet sanctioned to be in this file. Ascertain whether it can be obseleted.

def assignColormapEvenly(cmap,zs,asDict=False,missing=[1,1,1]):
    from cpblUtilities.color import assignSegmentedColormapEvenly
    assert missing== [1,1,1]  #Ignored!!!!!!!
    if cmap is None:
        cmap='jet'
    print('   assignColormapEvenly is now replaced by assignSegmentedColormapEvenly, and is deprecated. ')
    return(assignSegmentedColormapEvenly(cmap,zs,splitdataat=None,asDict=asDict,missing=missing))

def linearColormapLookup(cmap,zs,extendedLimits=None):#,returnformat='function'):
    """
    Right now this takes a string or a LinearSegmentedColormap.
    To reverse a colormap order, just tack "_r" onto the name.

    You have some data. You want to associated a colormap, scaled to the data.
    This returns a lookup/interpolate *function* by default. But it can do a lookup instead (no, not as of 2013Oct. Never used.

    e.g. use: For a id->data dataframe df,
    Just use df.map(linearColormapLookup(cmap,df.values)) if you want a lookup from data to colours, and zs

    2013Dec: also can request to extend the limits to include the top and bottom colour.
    extendedLimits = True
    extendedLimits = [-99999,99999]
    
    """

    returnformat='function'
    import numpy as np 
    import matplotlib.cm as cm 
    #    def make_N_colors(cmap_name, N): 
    #    cmap = cm.get_cmap(cmap_name, N) 
    #    return cmap(np.arange(N))[:,:-1]
    N=100
    if cmap is None:
        cmap='jet'
    if cmap.__class__ not in [mpl.colors.LinearSegmentedColormap ]:
        assert isinstance(cmap,str)
        cmap = cm.get_cmap(cmap, N) 
    cmapLU=cmap(np.arange(N))[:,:-1]
    from scipy import interpolate
    if extendedLimits is None:
        lookupfunc=interpolate.interp1d(np.linspace(min(zs),max(zs),num=N),np.array(cmapLU).T)
    else:
        if extendedLimits is True:
            extendedLimits=[min(zs)-10*(max(zs)-min(zs)),  max(zs)+10*(max(zs)-min(zs))]
        assert len(extendedLimits)==2
        cmapLUe=np.vstack([cmapLU[0],  cmapLU,   cmapLU[-1]])         # Now length N+2
        lookupfunc=interpolate.interp1d(extendedLimits[:1] + np.linspace(min(zs),max(zs),num=N).tolist()+extendedLimits[-1:] ,np.array(cmapLUe).T)

    if returnformat in ['function']:
        return(lookupfunc)
    elif returnformat in ['pandas']:
        return(pd.Series(dict([[zz,lookupfunc(zz)] for zz in zs])))

if __name__ == '__main__':
    import pylab as plt
    import matplotlib as mpl
    cpblColorDemos()
#    demoCPBLcolormapFunctions()    
