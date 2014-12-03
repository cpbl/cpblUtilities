#!/usr/bin/python
from __future__ import division # Python3-style integer division 5/2=2.5, not 3
if 0 and 'not while trying epd': # This was in place 'til May 2012: upgrade to Ubuntu 12.04
    from IPython.Debugger import Tracer; debug_here = Tracer()
try:
    import rpy2.robjects as robjects 
except:
    print('   ((cpblUtils: rpy2 not available on this machine))')
import os
import re
from copy import deepcopy

#import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl # Not yet used, may 2010
# WHAT!? matplotlib.pyplot is not the same as pylab?
#import matplotlib.pyplot as plt
import pylab as plt
import pandas as pd
#from pylab import *
# Why not define these lobally? June 2010
from pylab import figure,plot,ylim,xlim,setp,clf,array,isnan,nan,find,text,isfinite,xlabel,ylabel,title,arange,subplot,gca
NaN=nan
import scipy as sci


cifar={
'green':[.6588,.70588,0],#{168,180,0}
'cyan':[0,0.70196,0.7451],#{0,179,190}
'pink':[0.5098,0,0.31765],#{130,0,81} 
'grey':[0.40392,0.36078,0.32549],#{103,92,83}
'darkgreen':array([91,143,34])/255.,
'lightblue':array([161,222,233])/255.,
'yellow':array([236,227,84])/255.,
'palekhaki':array([181,163,0])/255.,
'black':[0,0,0],
}
"""
Solutions for bounding box / whitespace in output figures:
            plt.subplots_adjust(left  = 0.05,right=1-tiny,bottom=0.1,top=1-tiny) # BRILLIANT!!! USE subplot_tool() to find values!


"""

#print 'fyi: __See figureFontSetup() for plot settings  (cpblUtilMathGraph)'


#####################################################################################
def overplotLinFit(x,y,format=None,color=None,xscalelog=False):
    """
    March 2010: upgrading it so that the fit line doesn't extrapolate beyond the x,y range of data.
    """
    from pylab import log

    if format==None:
        format='k--'
    if color==None:
        color=cifar['grey']
    from pylab import plot,array,xlim,any

    if all(isnan(x)):
        return()
    if 0: # Use quick method, no standard errors
        from rpy2 import r

        ls_fit = r.lsfit(x,y)
        gradient = ls_fit['coefficients']['X']
        yintercept= ls_fit['coefficients']['Intercept']
        xr=array([max(min(x),min(xlim())),min(max(x),max(xlim()))])
        yr=[yintercept+gradient*xr[0],yintercept+gradient*xr[1]]
        plot(xr,yr,format,color=color)
        return(gradient)
    


    if xscalelog:
        x=np.log(x)


    if 1: # FIT A LINE TO THE NON-NAN ELEMENTS, OVERPLOT IT
        from scipy import stats as stats
        from pylab import where,isfinite,logical_and
        ii=where(logical_and(isfinite(x),isfinite(y)))
        ( gradient, yintercept, rrr, twoTailedProb, stderr)=stats.linregress(x[ii],y[ii])
        s=stderr
    else:
        import rpy2.robjects as robjects
        rpy = robjects.r
        #import rpy2 as rpy
        rpy.set_default_mode(rpy.NO_CONVERSION)
        if xscalelog:
            x=np.log(x)
        linear_model = rpy.r.lm(rpy.r("y ~ x"), data = rpy.r.data_frame(x=x, y=y))
        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        gradient=linear_model.as_py()['coefficients']['x']
        yintercept=linear_model.as_py()['coefficients']['(Intercept)']
        s=rpy.r.summary(linear_model)['sigma']
        print '  b: ',gradient
        #{'x': 5.3935773611970212, '(Intercept)': -16.281127993087839}
        print '  sigma: ',s
        #{'terms': <Robj object at 0x0089E240>, 'fstatistic': {'dendf': 2.0, 'value': 2.2088097871524752, 'numdf': 1.0}, 'aliased': {'x': False, '(Intercept)': False}, 'df': [2, 2, 2], 'call': <Robj object at 0x0089E340>, 'residuals': {'1': -9.3064376809571137, '3': -6.9622553363545983, '2': 6.3744808050079511, '4': 9.8942122123037599}, 'adj.r.squared': 0.28720941270437206, 'cov.unscaled': array([[ 2.1286381 , -0.42527178], [-0.42527178, 0.09626979]]), 'r.squared': 0.524806275136248, 'sigma': 11.696414461570097, 'coefficients': array([[-16.28112799, 17.06489672, -0.95407129, 0.44073772], [ 5.39357736, 3.62909012, 1.48620651, 0.27556486]])} 

        """
    import rpy2.robjects as robjects
    r = robjects.r
    # Create the data by passing a Python list to RPy2, which interprets as an R vector
    ctl = robjects.FloatVector(x)
    trt = robjects.FloatVector(y)
    group = r.gl(2, 10, 20, labels = ["Ctl","Trt"])
    weight = ctl + trt
    # RPy2 uses Python dictionary types to index data
    robjects.globalEnv["weight"] = weight
    robjects.globalEnv["group"] = group
    # Run the models
    lm_D9 = r.lm("weight ~ group")
    print(r.anova(lm_D9))
    lm_D90 = r.lm("weight ~ group - 1")
    print(r.summary(lm_D90))

    """


    # Limit extent of the plotted line to the x-extent of non-nan points
    xxx=[xx for ix,xx in enumerate(x) if isfinite(xx) and isfinite(y[ix])]
    lxlim=xlim()
    if xscalelog:
        lxlim=log(lxlim)
    xr=array([max(min(xxx),min(lxlim)),min(max(xxx),max(lxlim))])
    yr=[yintercept+gradient*xr[0],yintercept+gradient*xr[1]]
    if xscalelog:
        xr=np.exp(xr)
    aline=plot(xr,yr,format,color=color)

    # Display fit?
    """
    Here I should add a transAnnotation with the fit info.
    """
    return(aline,gradient,s)


# The following taken from skipper of pystatsmodels... because it allows for errorbars. so incorporate it into mine, above.
def linfit2D(y, x=None, y_unc=None): 
    import numpy as np 
    import scipy.special as ss 
    """
    Fits a line to 2D data, optionally with errors in y. 

    The method is robust to roundoff error. 

    Parameters 
    ---------- 
    y: ndarray 
        Ordinates, any shape. 
    x: ndarray 
        Abcissas, same shape.  Defaults to np.indices(y.length)) 
    y_unc: ndarray 
        Uncertainties in y.  If scalar or 1-element array, applied 
        uniformly to all y values.  [NOT IMPLEMENTED YET!]  Must be 
        positive. 

    Returns 
    ------- 
    a: scalar 
        0 Fitted intercept 
    b: scalar 
        1 Fitted slope 
    a_unc: scalar 
        2 Uncertainty of fitted intercept 
    b_unc: scalar 
        3 Uncertainty of fitted slope 
    chisq: scalar 
        4 Chi-squared 
    prob: scalar 
        5 Probability of finding worse Chi-squared for this model with 
          these uncertainties. 
    covar: ndarray 
        6 Covariance matrix: [[a_unc**2,  covar_ab], 
                              [covar_ab,  b_unc**2]] 
    yfit: ndarray 
        7 Model array calculated for our abcissas 
    
    Notes 
    ----- 
    
    If prob > 0.1, you can believe the fit.  If prob > 0.001 and the 
    errors are not Gaussian, you could believe the fit.  Otherwise 
    do not believe it. 

    See Also 
    -------- 
    Press, et al., Numerical Recipes in C, 2nd ed, section 15.2, 
    or any standard data analysis text. 

    Examples 
    -------- 
    >>> import linfit 

    >>> a = 1. 
    >>> b = 2. 
    >>> nx = 10 
    >>> x = np.arange(10, dtype='float') 
    >>> y = a + b * x 
    >>> y_unc = numpy.ones(nx) 
    >>> y[::2]  += 1 
    >>> y[1::2] -= 1 
    >>> a, b, sa, sb, chisq, prob, covar, yfit = linfit.linfit(y, x, y_unc) 
    >>> print(a, b, sa, sb, chisq, prob, covar, yfit) 
(1.272727272727272, 1.9393939393939394, 0.58775381364525869, 0.11009637651263605, 9.6969696969696937, 0.28694204178663996, array([[ 0.34545455, -0.05454545], 
       [-0.05454545,  0.01212121]]), array([  1.27272727,   3.21212121,   5.15151515,   7.09090909, 
         9.03030303,  10.96969697,  12.90909091,  14.84848485, 
        16.78787879,  18.72727273])) 

    Revisons 
    -------- 
    2007-09-23 0.1  jh@...	Initial version 
    2007-09-25 0.2  jh@...	Fixed bug reported by Kevin Stevenson. 
    2008-10-09 0.3  jh@...	Fixed doc bug. 
    2009-10-01 0.4  jh@...  Updated docstring, imports. 
    """
    # standardize and test inputs 
    if x == None: 
      x = np.indices(y.length, dtype=y.dtype) 
      x.shape = y.shape 

    if y_unc == None: 
        y_unc = np.ones(y.shape, dtype=y.dtype) 

    # NR Eq. 15.2.4 
    ryu2  = 1. / y_unc**2 
    S     = np.sum(1.    * ryu2) 
    Sx    = np.sum(x     * ryu2) 
    Sy    = np.sum(y     * ryu2) 
    # Sxx = np.sum(x**2  * ryu2) # not used in the robust method 
    # Sxy = np.sum(x * y * ryu2) # not used in the robust method 

    # NR Eq. 15.2.15 - 15.2.18 (i.e., the robust method) 
    t = 1. / y_unc * (x - Sx / S) 
    Stt = np.sum(t**2) 

    b = 1. / Stt * np.sum(t * y / y_unc) 
    a = (Sy - Sx * b) / S 

    covab = -Sx / (S * Stt)                  # NR Eq. 15.2.21 

    sa = np.sqrt(1. / S * (1. - Sx * covab)) # NR Eq. 15.2.19 
    sb = np.sqrt(1. / Stt)                   # NR Eq. 15.2.20 

    rab = covab / (sa * sb)                  # NR Eq. 15.2.22 

    covar = np.array([[sa**2, covab], 
                      [covab, sb**2]]) 

    yfit = a + b * x 
    chisq = np.sum( ((y - yfit) / y_unc)**2 ) 

    prob = 1. - ss.gammainc( (y.size - 2.) / 2., chisq / 2.) 

    return a, b, sa, sb, chisq, prob, covar, yfit 



##############################################################################
##############################################################################
#
def cpblOLS(y,xs,betacoefs=False,rhsOnly=None):
    ##########################################################################
    ##########################################################################
    """
    I want to have my OLS routine:
  - take a list of vectors
  - get heteroskedasticity-robust errors back
  - have option to get standardized beta coeffcients rathr than raw b
  - ignore NaNs
 - take sample weights. (WLS)


  So xs can/should be a dict of named vectors.
Actually, here's my preferred calling format: yname,xnames, dataDict
but dataDict should be able to be a list of dicts or a dict of named vectors.


sep 2010: this failed, ie is incompelte, since I can't see how to do weights with it.
    """
    import numpy as np
    import scikits.statsmodels as sm


    # Not yet flexi calling format:
    assert isinstance(y,str)
    #assert isinstance(x,list)
    #assert isinstance(x[0],str)
    assert isinstance(xs,dict)
    dataDict=xs
    Y=dataDict.pop(y) # Remove y from the data to be used for x.
    if not rhsOnly:
        rhsOnly=dataDict.keys()#list(set(x.keys())-set([y]))

    # get data

    if betacoefs:
        X = sm.tools.add_constant(np.column_stack([(xs[kk]-np.mean(finiteValues(xs[kk])))/np.std(finiteValues(xs[kk])) for kk in rhsOnly]))
    else:
        X = sm.tools.add_constant(np.column_stack([xs[kk] for kk in rhsOnly]))
    #beta = np.array([1, 0.1, 10])
    #y = np.dot(X, beta) + np.random.normal(size=nsample)

    # run the regression
    results = sm.OLS(Y, X,weights=Y).fit()
    foiu
    # look at the results
    #print results.summary()

    #and look at `dir(results)` to see some of the results
    #that are available
    return(bs,rses)


##############################################################################
##############################################################################
#
def _obselete_use_savefigall_exportPythonFigToFormats(fileName):
    ##########################################################################
    ##########################################################################
    from pylab import savefig
    for ff in ['pdf','png','eps']:
        # Should still develop this to get rid of borders!
        
        savefig(fileName+'.'+ff,format=ff,transparent=True)

##############################################################################
##############################################################################
#
def figureToInverseVideo(fig=None):
    ##########################################################################
    ##########################################################################
    """
2013 June: backgrounds of boxed text missing. (bbox). Cannot figure out how. 
Added: setp(fff,'facecolor','k') No! Actually, use facecolor='k' option in savefig!
Replaced a bunch of loops with a findobj!
    
    May 2012, ... I've finally made some progress on this. Set figure axis background color, foreground color, and save figure as transparent. Plus, check that black text, black lines .... and then also, harder! other black stuff like patches are dealt with.

sep2012:    Agh. what about stuff drawn in bg colour, eg to over? I want to swtich that to the new bg colour! Not done. Wasneeded for PQ/liberal colour band in rdc/regressoinsQuebec.
    """
    if fig is None:
        fig=plt.gcf()

    for aaa in plt.findobj(fig,mpl.axes.Axes):
        set_backgroundcolor(aaa,'black') # Not None?
        set_foregroundcolor(aaa,'white')

    def checkColour(oo):
        colour=oo.get_color()
        if hasattr(colour, 'shape'): # Is mpl.array!
            assert len(colour) in [3,4]
            colour=list(colour)
        if colour in ['k',(0,0,0),(0,0,0,0),]:
            o.set_color('w')
    def checkFaceEdgeColour(oo):
        colour=oo.get_edgecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_edgecolor('w')
        else:
            ffffooijoweiruiuiuuoiuiu
        colour=oo.get_facecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_facecolor('w')
            if colour in ['w']:#,(0,0,0),(0,0,0,0),]:
                foiofofooffo
                
        else:
            ffffooijoweiruiuiuuoiuiu
            
    for o in fig.findobj(mpl.lines.Line2D):
        checkColour(o)
    for o in fig.findobj(mpl.text.Text):
        checkColour(o)


    # match on arbitrary function 
    def findfaces(x): 
        return hasattr(x, 'set_facecolor') 

    for o in fig.findobj(findfaces): 
        checkFaceEdgeColour(o)

    if 0: # Wow!! All these are obselete, given the above (new 2013 June)!?
        for o in fig.findobj(mpl.patches.Patch):
            checkFaceEdgeColour(o)
        for o in fig.findobj(mpl.collections.PolyCollection):
            checkFaceEdgeColour(o)

        for o in fig.findobj(mpl.patches.Rectangle):
            checkFaceEdgeColour(o)

    # what about figure itself? See also  facecolor of savefig...
    if fig.get_facecolor()[0]==.75:
        fig.set_facecolor('k')
    elif fig.get_facecolor() not in [(0.0, 0.0, 0.0, 1.0)]:
        print('deal with this')
        print fig.get_facecolor()
        deal_with_this


        
    return()

##############################################################################
##############################################################################
#
def figureToGrayscale(fig=None):
    ##########################################################################
    ##########################################################################
    """
    May 2012. Rewriting this from scratch, copied from figureToInverseVideo, after having some luck with the latter.
    This is only just started, though.
    """
    if fig is None:
        fig=plt.gcf()

    if 1: # Note reall
        for aaa in plt.findobj(fig,mpl.axes.Axes):
            set_backgroundcolor(aaa,None) # Not None?
            set_foregroundcolor(aaa,'black')

    def meanColour(oo):
        colour=oo.get_color()
        if hasattr(colour, 'shape'): # Is mpl.array!
            #iopoio
            assert len(colour) in [3,4]
            colour=list(colour)
        if colour not in  ['w','white','k','black',(0,0,0),(0,0,0,0),(1,1,1),(1,1,1,1),]:
            if isinstance(colour,str):
                o.set_color('k')
            #Not finished!!!!
        #Not finished!!!!
    def meanFaceEdgeColour(oo):
        colour=oo.get_edgecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_edgecolor('w')
        else:
            ffffooijoweiruiuiuuoiuiu
        colour=oo.get_facecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_facecolor('w')
        else:
            ffffooijoweiruiuiuuoiuiu
            
    for o in fig.findobj(mpl.lines.Line2D):
        meanColour(o)
    for o in fig.findobj(mpl.text.Text):
        meanColour(o)
    """
    for o in fig.findobj(mpl.patches.Patch):
        meanFaceEdgeColour(o)
    for o in fig.findobj(mpl.collections.PolyCollection):
        meanFaceEdgeColour(o)


    for o in fig.findobj(mpl.patches.Rectangle):
        meanFaceEdgeColour(o)
        """
    return()

##############################################################################
##############################################################################
#
def figureToGrayscaleOld(): # This is a disaster for envelopes, no!? it sets all colors t ogrey!!! needs fixing!  May 2012: delete this when y
    ##########################################################################
    ##########################################################################
    fig=plt.gcf()

    from pylab import mean
    for o in fig.findobj(mpl.patches.Rectangle):
        rgba=o.get_facecolor()
        if not isinstance(rgba,str) and len(rgba)==4:
            gg=mean(rgba[:-1])
            o.set_facecolor([gg,gg,gg,rgba[-1]])

    for o in fig.findobj(mpl.lines.Line2D):
        o.set_color('k')

    # Set all text to black:
    for o in fig.findobj(mpl.text.Text):
        o.set_color('k')


    # Set all envelopes / patch polys to black (this seems to make them appropriately gray if alpha<1 !?)
    setp(fig.findobj(mpl.collections.PolyCollection),'color','k')


    return()

def tightBoundingBoxInkscape(infile,outfile=None,use_xvfb=True):
    """Makes POSIX-specific OS calls. Need xvfb installed. If it fails anyway, could always resort to use_xvfb=False
 """
    if infile.endswith('.pdf'):
        infile=infile[:-4]
    if outfile is None:
        outfile=infile+'-tightbb.pdf'
    if not outfile.endswith('.pdf'):
        outfile+='.pdf'
    usexvfb='xvfb-run  '*use_xvfb  #  # +extension RANDR  : does not work to fix 
    import os
    from cpblUtilities import doSystem
    # 2014: I should redesign this. First, do not change fonts on import. Second, can I skip the svg stage, with modern version?
    doSystem("""
%(XVFB)s inkscape -f %(FN)s.pdf -l %(FN)s_tmp.svg
%(XVFB)s inkscape -f %(FN)s_tmp.svg --verb=FitCanvasToDrawing                                    --verb=FileSave                                    --verb=FileQuit
%(XVFB)s inkscape -f %(FN)s_tmp.svg -A %(OF)s
#rm %(FN)s_tmp.svg
"""%{'XVFB':usexvfb, 'FN':infile,'OF':outfile},   verbose=True,bg='ifserver')

##############################################################################
##############################################################################
#
def savefigall(fn,transparent=True,ifany=None,fig=None,skipIfExists=False,pauseForMissing=True,onlyPNG=False,jpeg=False,jpeghi=False,svg=False,pdf=True,bw=False,FitCanvasToDrawing=False,eps=False,tikz=False,rv=None,facecolor='None'):
    ##########################################################################
    ##########################################################################
    """
    Like savefig, but saves in multiple formats; make this standard

    April 2010: adding transparent option for png only. (right now it's automatic for pdf)

    ifany must be an artist subclass taken by matplotlib.findobj(match=)

Do not forget to use cpblStata's
 (self,figname=None,caption=None,texwidth=None,title=None,onlyPNG=False)
 if you're using a cpbl latex object!

Sept 2010: skipIfExists is a way to skip saving if the file already exists (ie to save time).
    
Sept 2010: Returns written (or existing) file stems.

April 2011: what?! this has evolved to have dissimilar options from my latex class's saveAndIncludeFig() !
      # saveAndIncludeFig(self,figname=None,caption=None,texwidth=None,title=None,onlyPNG=False,rcparams=None,transparent=False):
      # Okay. April 2011: I've hopefully implemented things so can use latex.saveAnd... with the above options.

Nov 2010: Added "bw=False" option. If set to true, saves two sets, one with "-bw" suffix where colour has been turned to grayscale

Dec 2011: okay... when you start specifying fig size, the final boudning box no longer matches the plots. Crazy!
Online says there's a function to fix subplot params, but it doesn't seem to exist in my distro.
Here's an outlandish method: I'm making a very specific option, 'FitCanvasToDrawing', to implement it. It's outlandish because inkcape GUI gets called up and instantiated to do these verbs.


 inkscape -f matplotlibOutput.pdf -l matplotlibOutput.svg
 inkscape -f matplotlibOutput.svg --verb=FitCanvasToDrawing                                    --verb=FileSave                                    --verb=FileQuit
 $ inkscape -f matplotlibOutput.svg -A prettyfigure.pdf
 $ rm matplotlibOutput.svg
 
 
May 2012: Wow, if only the tikz converter matplotlib2tikz worked!! It fails in ways I cannot debug, from a colorbar or an envelope (patch), maybe. grr.

May 2012: adding rv=True and forcing transparent=True (below) !
Is transparent obselete, ie don't I always want it on? Maybe sometimes png was crashing. In 2013, remove it.
rv=True
rv=False
rv="both" and rv=None


June 2013: I still think use of facecolor and transparent in savefig is horribly buggy. I'm getting blue borders around all my figures now, except with rv. :(

"""
    #transparent=True # Huh? 2013 June: commenting this out.

    (root,tail)=os.path.split(fn)
    if bw:
        savefigall(fn,transparent=transparent,ifany=ifany,fig=fig,skipIfExists=skipIfExists,pauseForMissing=pauseForMissing,onlyPNG=onlyPNG,jpeg=jpeg,jpeghi=jpeghi,svg=svg,pdf=pdf,FitCanvasToDrawing=FitCanvasToDrawing,eps=eps,tikz=tikz,rv=rv,facecolor=facecolor)
        if 1: # ahhhh... issues jan 2012. kludge it off:
            figureToGrayscale() 
        savefigall(fn+'-bw',transparent=transparent,ifany=ifany,fig=fig,skipIfExists=skipIfExists,pauseForMissing=pauseForMissing,onlyPNG=onlyPNG,jpeg=jpeg,jpeghi=jpeghi,svg=svg,pdf=pdf,FitCanvasToDrawing=FitCanvasToDrawing,eps=eps,tikz=tikz,rv=rv,facecolor=facecolor)
        return(root+tail)

    if fig is None:
        fig=plt.gcf()
    elif fig.__class__ in [mpl.figure.Figure]:
        figure(fig.number)
    else:
        figure(fig)
    if 0: # I need to use frozen() or deepcopy() to avoid rv and bw from messing up future modified versions of a plot!  But this crashes:
        fig=deepcopy(fig)
    if rv in [True]:
        figureToInverseVideo(fig)
    if rv in ['both',None]: # Do both.
        assert not bw
        # 28 June 2013: trying set transparent=True
        savefigall(fn,transparent=True, #transparent,
                   ifany=ifany,fig=fig,skipIfExists=skipIfExists,pauseForMissing=pauseForMissing,onlyPNG=onlyPNG,jpeg=jpeg,jpeghi=jpeghi,svg=svg,pdf=pdf,rv=False,FitCanvasToDrawing=FitCanvasToDrawing,eps=eps,tikz=tikz,facecolor=facecolor)#facecolor)
        if facecolor is None:
            facecolor='k'
        savefigall(fn+'-rv',transparent=transparent,ifany=ifany,fig=fig,skipIfExists=skipIfExists,pauseForMissing=pauseForMissing,onlyPNG=onlyPNG,jpeg=jpeg,jpeghi=jpeghi,svg=svg,pdf=pdf,rv=True,FitCanvasToDrawing=FitCanvasToDrawing,eps=eps,tikz=tikz,facecolor=facecolor)
        return(root+tail)

    if not root:
        try:
            from cpblDefaults import defaults, paths
            root=paths['graphics']#'/home/cpbl/rdc/workingData/graphics/'#defaults['workingPath']+'graphics/'#'graphicsPath'
        except:
            root='./'#/home/cpbl/rdc/graphicsOuttest/graphics/'#defaults['workingPath']+'graphics/'#'graphicsPath'
    if root and not root.endswith('/'):
        root+='/'

    from cpblUtilities import str2pathname
    tail=str2pathname(tail) # Get rid of any punctuation in the name.


    if skipIfExists and  os.path.exists(root+tail+'.png') and  os.path.exists(root+tail+'.pdf'):
        print '   Skipping production of '+root+tail+'(png/pdf) because it already exists [skipIfExists]'
        return(root+tail)

    if ifany:
        if not plt.findobj(match=ifany):
            print '   savefigall: Empty plot (no %s), so not saving %s.'%(str(ifany),root+tail)
            plt.savefig(root+tail+'.png.FAILED',format='png',facecolor=facecolor)
            if pauseForMissing:
                plt.show()
                from cpblUtilities import cwarning
                cwarning('   savefigall: Empty plot (no %s), so not saving %s.'%(str(ifany),root+tail))
            return(None)

    def stupidOnlyUseGivenArguments(name,transparent=None,facecolor=None): #June 2013
        if transparent and facecolor in ['None','none',None,False]:
            fig.savefig(name,transparent=transparent)
        elif  transparent:
            fig.savefig(name,transparent=transparent,facecolor=facecolor)
        elif  facecolor in ['None','none',None,False]:
            fig.savefig(name)
        else:
            fig.savefig(name,facecolor=facecolor)
        
    print 'Saving graphics: '+root+tail+' (+ext)'
    stupidOnlyUseGivenArguments(root+tail+'.png',transparent=transparent,facecolor=facecolor)

    if eps:
        try: # Damn. Feb 2012 postscript is crashing.
            plt.savefig(root+tail+'.eps',transparent=transparent,facecolor=facecolor)
        except:
            print('*****************\n\n\n\n\nFAILED TO PRODUCE AN EPS\n\n\n\n\n***********')
    if pdf and not onlyPNG: # What?! I don't have to use stupidOnlyUseGivenArguments here? If I did it for PNG, everything's fine?
        plt.savefig(root+tail+'.pdf',transparent=transparent,facecolor=facecolor)
        if FitCanvasToDrawing:
            from cpblUtilities import doSystem
            """
            xvfb is the /dev/null of X servers, if you like... ie normally command-line verbs of inkscape generate a gui, so this prefix hides them:
            """
            print('You need xvfb installed. You need your .config/inkscape/extensions script. March 2014: XRANDR extension missing?! !"AAAAAAAAAAAAAARRRRCH')

            print("""inkscape -f %(fn)s.pdf -l %(fn)sTMPTMPTMP.svg
inkscape -f %(fn)sTMPTMPTMP.svg --verb=FitCanvasToDrawing  --verb=FileSave  --verb=FileQuit
inkscape -f %(fn)sTMPTMPTMP.svg -A %(fn)s-autotrimmed.pdf
 """%{'fn':root+tail})

            tightBoundingBoxInkscape(root+tail,use_xvfb=False) # ARGH March 2014: I'm getting an error from xvfb. So do it with GUI popping up!
       
    if svg:
        plt.savefig(root+tail+'.svg',transparent=transparent,facecolor=facecolor)
    if tikz:
        from matplotlib2tikz import save as tikz_save
        tikz_save(root+tail+'_tikz.tex', figureheight='\\figureheight', figurewidth='\\figurewidth' )
    if jpeg or jpeghi:
        # jpeg Not supported by pylab!!
        #plt.savefig(root+tail+'.jpeg')#transparent=True)
        # 2013: I think jpeg is supported now. But "quality" keyword may not be.
        plt.savefig(root+tail+'.jpeg',transparent=transparent,facecolor=facecolor)
        #os.system('convert '+ root+tail+'.png '+root+tail+'.jpeg'+' &'*('apollo' in os.uname()[1]))
        #if jpeghi:
        #os.system('convert -quality 100 '+ root+tail+'.png '+root+tail+'.jpeg'+' &'*('apollo' in os.uname()[1]))
    return(root+tail+FitCanvasToDrawing*'-autotrimmed')
    if rv:
        assert not bw
        savefigall(fn,transparent=transparent,ifany=ifany,fig=fig,skipIfExists=skipIfExists,pauseForMissing=pauseForMissing,onlyPNG=onlyPNG,jpeg=jpeg,jpeghi=jpeghi,svg=svg,pdf=pdf,FitCanvasToDrawing=FitCanvasToDrawing,eps=eps,tikz=tikz,rv=rv)


def xTicksLogIncome_deprecated_USE_xTicksIncome(nticks=7,natlog=False,tenlog=False,kticks=None):
    deprecated

def xTicksIncome(nticks=7,natlog=False,tenlog=False,kticks=None,dollarsign=True,lnUSA=False):
    """
    Oct 2011: generalise above to deal with non-log?
    
    For any plot in which the abscissa is log10(income) [set log10=True in this case] or log(income) , you can use this to make more readable tick points and labels.

You can specify the income ticks in k$ in kticks.

one of natlog or tenlog should be chose, until 2012 when I can set natlog=True as default, above. [not anymore, unless I make a new log=False argument]

June 2012: agh. lnUSA means that the income is in ln of USA level.

2014 Sept: this can be deprecated by a version of the X-or-y function which follows.
    """
    #assert natlog or tenlog
    assert not natlog or not tenlog
    if  not natlog and not tenlog and not lnUSA:
        assert max(xlim())-min(xlim()) > 10 # Otherwise, it looks like we should have called natlog or tenlog
    base,flog=10.0,np.log10
    if natlog or lnUSA:
        base,flog=np.e,np.log
    if lnUSA: # I think the goal here is to list log 10 fractions? 
        assert not natlog
        assert not tenlog
        xl=pow(base,plt.array(plt.xlim()))
        possibleTicks={.01:r'$\frac{1}{100}$',.0333333:r'$\frac{1}{30}$',.1:r'$\frac{1}{10}$',.33333:r'$\frac{1}{3}$',1:'1',2:'3'} # ,.5:r'$\frac{1}{2}$'
        #{.02:r'$\frac{1}{50}$',.1:r'$\frac{1}{10}$',.5:r'$\frac{1}{2}$',1:'1',2:'2'}
        ticks=sorted([ik for ik in possibleTicks if ik<=xl[1] and ik>=xl[0]])
        plt.setp(plt.gca(),'xticks',[flog(cc) for cc in ticks])
        plt.setp(plt.gca(),'xticklabels',[possibleTicks[cc] for cc in ticks])
    elif natlog or tenlog:
        xl=pow(base,plt.array(plt.xlim()))/1000
        ones=[ik for ik in range(5,200,1) if ik<=xl[1] and ik>=xl[0]]
        fives=[ik for ik in range(5,200,5) if ik<=xl[1] and ik>=xl[0]]
        tens=[ik for ik in range(10,200,10) if ik<=xl[1] and ik>=xl[0]]
        oo=[ones,fives,tens]
        nn=[abs(len(ooo)-nticks) for ooo in oo]
        choice=oo[[inn for inn in range(len(nn)) if nn[inn]==min(nn)][0]]
        if kticks:
            choice=kticks
        plt.setp(plt.gca(),'xticks',[flog(cc*1000.0) for cc in choice])
        plt.setp(plt.gca(),'xticklabels',[r'\$'*dollarsign+(str(cc)+'k' if cc>=1 else '%d'%(cc*1000)) for cc in choice])
    else: #Linear values: just put a dollar sign and take off the thousands.?
        xl=plt.array(plt.xlim())/1000.0
        ones=[ik for ik in range(5,200,1) if ik<=xl[1] and ik>=xl[0]]
        fives=[ik for ik in range(5,200,5) if ik<=xl[1] and ik>=xl[0]]
        tens=[ik for ik in range(10,200,10) if ik<=xl[1] and ik>=xl[0]]
        oo=[ones,fives,tens]
        nn=[abs(len(ooo)-nticks) for ooo in oo]
        choice=oo[[inn for inn in range(len(nn)) if nn[inn]==min(nn)][0]]
        if kticks:
            choice=kticks
        plt.setp(plt.gca(),'xticks',[cc*1000.0 for cc in choice])
        plt.setp(plt.gca(),'xticklabels',[r'\$'*dollarsign+str(cc)+'k' for cc in choice])
    return

def xyTicksIncome(nticks=7,natlog=False,tenlog=False,kticks=None,dollarsign=True,lnUSA=False,XorY='x'):
    """
    2014 Sept: copied from xTicksIncome.
    Not sure how to merge these yet.

    It seems this is rather targeted for US$ amounts. Not, for instance, Korean won!
    """
    if XorY in ['x']:
        XLIM,xticks,xticklabels=plt.xlim,'xticks','xticklabels'
    else:
        XLIM,xticks,xticklabels=plt.ylim,'yticks','yticklabels'

    #assert natlog or tenlog
    assert not natlog or not tenlog
    if  not natlog and not tenlog and not lnUSA:
        assert max(XLIM())-min(XLIM()) > 10 # Otherwise, it looks like we should have called natlog or tenlog
    base,flog=10.0,np.log10
    if natlog or lnUSA:
        base,flog=np.e,np.log
    if lnUSA: # I think the goal here is to list log 10 fractions? 
        assert not natlog
        assert not tenlog
        xl=pow(base,plt.array(XLIM()))
        possibleTicks={.01:r'$\frac{1}{100}$',.0333333:r'$\frac{1}{30}$',.1:r'$\frac{1}{10}$',.33333:r'$\frac{1}{3}$',1:'1',2:'3'} # ,.5:r'$\frac{1}{2}$'
        #{.02:r'$\frac{1}{50}$',.1:r'$\frac{1}{10}$',.5:r'$\frac{1}{2}$',1:'1',2:'2'}
        ticks=sorted([ik for ik in possibleTicks if ik<=xl[1] and ik>=xl[0]])
        plt.setp(plt.gca(),xticks,[flog(cc) for cc in ticks])
        plt.setp(plt.gca(),xticklabels,[possibleTicks[cc] for cc in ticks])
    elif natlog or tenlog:
        xl=pow(base,plt.array(XLIM()))/1000
        ones=[ik for ik in range(5,200,1) if ik<=xl[1] and ik>=xl[0]]
        fives=[ik for ik in range(5,200,5) if ik<=xl[1] and ik>=xl[0]]
        tens=[ik for ik in range(10,200,10) if ik<=xl[1] and ik>=xl[0]]
        oo=[ones,fives,tens]
        nn=[abs(len(ooo)-nticks) for ooo in oo]
        choice=oo[[inn for inn in range(len(nn)) if nn[inn]==min(nn)][0]]
        if kticks:
            choice=kticks
        plt.setp(plt.gca(),xticks,[flog(cc*1000.0) for cc in choice])
        plt.setp(plt.gca(),xticklabels,[r'\$'*dollarsign+(str(cc)+'k' if cc>=1 else '%d'%(cc*1000)) for cc in choice])
    else: #Linear values: just put a dollar sign and take off the thousands.?
        xl=plt.array(XLIM())/1000.0
        ones=[ik for ik in range(5,200,1) if ik<=xl[1] and ik>=xl[0]]
        fives=[ik for ik in range(5,200,5) if ik<=xl[1] and ik>=xl[0]]
        tens=[ik for ik in range(10,200,10) if ik<=xl[1] and ik>=xl[0]]
        oo=[ones,fives,tens]
        nn=[abs(len(ooo)-nticks) for ooo in oo]
        choice=oo[[inn for inn in range(len(nn)) if nn[inn]==min(nn)][0]]
        if kticks:
            choice=kticks
        plt.setp(plt.gca(),xticks,[cc*1000.0 for cc in choice])
        plt.setp(plt.gca(),xticklabels,[r'\$'*dollarsign+str(cc)+'k' for cc in choice])
    return
def yTicksIncome(nticks=7,natlog=False,tenlog=False,kticks=None,dollarsign=True,lnUSA=False):
    return(
        xyTicksIncome(nticks=nticks,natlog=natlog,tenlog=tenlog,kticks=kticks,dollarsign=dollarsign,lnUSA=lnUSA,XorY='y')
    )


def xticksExponentiate(base10=False):
    """
    The x-values are log or log10, but I want the labels to show the unlogged values
    """
    def pow2base10(vv,pos): #'The two args are the value and tick position'
        if int(vv)==vv:
            if vv<2 and vv>=0:
                return(str(int(pow(10,vv))))
            return(r'10$^{%s}$'%int(vv)  )
        return('') # Hide all non-integer values!?


    if base10:
        gca().xaxis.set_major_formatter( mpl.ticker.FuncFormatter(pow2base10))

##############################################################################
##############################################################################
#
def stdev(vals):
    ##########################################################################
    ##########################################################################
    """ the available std does not give 0 if there's only one value
    """
    from pylab import std
    if len(vals)>1:
        return(std(vals))
    elif len(vals)>0:
        return(0.)
    else:
        return (float('nan'))

##############################################################################
##############################################################################
#
from numpy import mean # This one deals with no values!, so overwrite python's default mean..
#def mean(vals):
#    ##########################################################################
#    ##########################################################################
#    """ the available mean cannot deal with no values
#    """
#    import pylab
#    if len(vals)>0:
#        return(np.mean(vals))
#    else:
#        return(float('nan'))
        
##############################################################################
##############################################################################
#
def seMean(vals, ses): # A Weighted mean: weighted by standard errors
    # Takes a simple list of estimates and a simple list of their standard errors.
    # Returns an estimate of the scalar weighted mean and its standard error.
    #
    # - COVARIANCE IS IGNORED (ASSUMED ZERO)
    #
    # - NaNs are DROPPED! before taking mean
    ##########################################################################
    ##########################################################################
    #from pylab import mean #import pylab
    from pylab import sqrt,isnan
    if vals and any(vals):
        vals,ses=[v for v in vals if not isnan(v)],[v for v in ses if not isnan(v)]
        meanSE=sqrt(1.0/sum([1.0/ff/ff for ff in ses]))
        meanVals=sum([vals[ic]/ses[ic]/ses[ic] for ic in range(len(vals))])*meanSE*meanSE
    else:
        meanSE=''#[float('nan')]
        meanVals=''#[float('nan')]
    return(meanVals,meanSE)
##############################################################################
##############################################################################
#
def seSum(x=None, sx=None): # This takes a simple sum across x, i.e. x has length >1.
    ##########################################################################
    ##########################################################################
    #from pylab import mean #import pylab
    from pylab import sqrt,array,isnan,any
    f=sum(x)
    sf=sqrt(sum(array(sx)*array(sx)))
    assert not any(isnan(sf))

    return(f,sf)



##############################################################################
##############################################################################
#
def seProduct(x=None,y=None,sx=None,sy=None,covs=None): 
    ##########################################################################
    ##########################################################################
    """
For f=x*y, with s.e.'s of sx and sy, returns f and approx sf...
# COVARIANCE SO FAR IGNORED!

They need to be floats at the moment.
The arguments are mandatory; I'm making them parameters for easier reading of the call.
"""
    #from matplotlib import *
    x=array(x)
    y=array(y)
    sx=array(sx)
    sy=array(sy)

    f=x*y
    sfoverf=np.sqrt((sx/x)*(sx/x)+(sy/y)*(sy/y))
    sf=sfoverf*abs(f)
    if len(x)==1: # Still to vectorize these!!
        if x==0.0 and y==0.0: # How do I calculate se for this?
            sf=np.sqrt(sx**2+sy**2) # True for any limit of x==y?
        if (sx==0 and x==0) or (sy==0 and y==0):
            f,sf=0,0
        assert isnan(f) or not isnan(sf)
    return(f,sf)
##############################################################################
##############################################################################
#
def seQuotient(x=None,y=None,sx=None,sy=None,covs=None): 
    ##########################################################################
    ##########################################################################
    """
For f=x/y, with s.e.'s of sx and sy, returns f and approx sf...
# COVARIANCE SO FAR IGNORED!

They need to be floats at the moment
"""
    #from matplotlib import *
    x=array(x)
    y=array(y)
    sx=array(sx)
    sy=array(sy)

    f=x/y
    sfoverf=sqrt((sx/x)*(sx/x)+(sy/y)*(sy/y))
    sf=sfoverf*f
    return(f,sf)






def categoryBarPlot(labels,y,skipEmpty=True,sortDecreasing=False,demo=False,labelLoc=None,yerr=None,horiz=True,barColour=None,labelColour=None,grouped=False,stacked=False,animate=False,filename='unnamedCategoryBarPlot',xlabel=None,ylabel=None,title=None,width=None):
    if sortDecreasing:
        print "CAUTION!!!!!!!!!!!!!!!may 2010: sortdecreasing does not sort the labels properly!!!!!!!!!!!!!!!!!!!!!!!!!1"
    """

April 2010: Stupid (bug??). barh does not return handles to the error bars -- it only gives you the rectangles. So I'm writing a new captureLines subroutine.

    Earlier:
    
    Obviously not finished. Jeezus. There's no grouped option. Write a group/stacked bar chart with labels ....

    etc. easy exercise. not sure why not done by others.


Sept 2010: Demo now even fails: on grouped bar plot

horiz: default bar orientation is horizontal. Set to False for vertical bars. Vertical is not yet implemented.


labelLoc modes:
'eitherSideOfZero': put label start or ending on the zero line, depending on whether bar goes pos or neg
'opposite bar or inside': this is the default.
'biggest space': put right/left/inside the bar, whichever has most horizontal space there!

'bestEdge':
'alignLeftAxis':
'eitherSideOfZero':
'fromLeft':
'atRightAxis':
'atRight':
'atLeft':

width: same option as taken by bar() function

    """

    if demo: # Should do a bunch of different formats to demonstrate stacked, errorbars, etc, etc,..
        tableFromXLS=[
        ['', 'USA(SCCBS)', 'Canada (ESC2)', 'Gallup ladder', 'Gallup SWL'],
        ['log(household income)', '0.13', '0.13', '0.34', '0.24'],
        ['visits to family', '0.05', '0.05', '', ''],
        ['number of / contact with close friends', '0.1', '0.06', '', ''],
        ['talk/contact with neighbours', '0.04', '0.05', '', ''],
        ['volunteer groups /memberships', '0.03', '0.01', '', ''],
        ['people can be trusted', '0.08', '0.07', '', ''],
        ['trust neighbours (to return a wallet)', '0.1', '0.07', '', ''],
        ['trust in police (to return a wallet)', '', '0.05', '', ''],
        ['frequency of religious attendance', '0.04', '0.02', '0', '0.03'],
        ['freedom to choose', '', '', '0.08', '0.1'],
        ['perception of corruption (neg)', '', '', '0.09', '0.05'],
        ['donated time', '', '', '0.01', '0.03'],
        ['donated money', '', '', '0.06', '0.05'],
        ['helped a stranger', '', '', '0.03', '0.05'],
        ['safe at night', '', '', '0.03', '0.01'],
        ['importance of religion', '0.03', '0.03', '0', '0.05'],
        ['can count on friends', '', '', '0.09', '0.12'],
        ]

        sccbs=array([tonumeric(tt[1]) for tt in tableFromXLS[1:]])
        esc2=array([tonumeric(tt[2]) for tt in tableFromXLS[1:]])
        ladder=array([tonumeric(tt[3]) for tt in tableFromXLS[1:]])
        swl=array([tonumeric(tt[4]) for tt in tableFromXLS[1:]])
        labels=[tt[0] for tt in tableFromXLS[1:]]



        # Basic demo single bars:
        figure(2)
        categoryBarPlot(labels,swl)#,labelLoc='atRight')#plt.transpose([sccbs,esc2,ladder,swl]))

        figure(5)
        if 0:
            categoryBarPlot(labels,swl,horiz=False)#,labelLoc='atRight')#plt.transpose([sccbs,esc2,ladder,swl]))

        # Grouped
        figure(1)
        categoryBarPlot(labels,plt.transpose([sccbs,esc2,ladder,swl]),grouped=['sccbs','esc2','ladder','swl'])#,labelLoc='atRight')#

        # Animated for presentations
        figure(6)
        yerr=0.2*swl
        categoryBarPlot(labels,swl,yerr=yerr,horiz=True,animate=True)#,labelLoc='atRight')#plt.transpose([sccbs,esc2,ladder,swl]))
        

        return()

    if plt.size(y,0)==len(labels):
        pass
    assert array(y).shape[0]==len(labels) # size(y,0)

    simplevector=len(array(y).shape)==1 # ie not stacked????
    #print 'simplevector',simplevector
    nGroups=len(labels)
    if not simplevector:
        nInGroup=array(y).shape[1]
    assert nGroups==array(y).shape[0]
    if not simplevector:
        if not stacked and not grouped:
            grouped=True
        assert stacked or grouped
        assert not stacked or not grouped

    if skipEmpty and simplevector: # Clean out empty/nan values
        # Not yet made skipEmpty for stacked, grouped.
        ii=plt.where(isfinite(y))
        labels=array(labels)[ii]
        y=array(y)[ii]



    if not barColour and grouped:
        barColour=[cifar['green'],cifar['cyan'],cifar['pink'],cifar['grey'],'r','b','g','k','c','m',cifar['green'],cifar['cyan'],cifar['pink'],cifar['grey'],'r','b','g','k','c','m',cifar['green'],cifar['cyan'],cifar['pink'],cifar['grey'],'r','b','g','k','c','m',cifar['green'],cifar['cyan'],cifar['pink'],cifar['grey'],'r','b','g','k','c','m']

    if not barColour:
        barColour={}
    if isinstance(barColour,dict):
        barColourList=[barColour.get(xx,cifar['cyan']) for xx in labels]
    elif isinstance(barColour,list):
        barColourList=barColour
    else:
        assert 0 # Can't parse barColour


    def unzip(recs):
        return zip(*recs)
    if sortDecreasing and simplevector:
        sss=zip(y,labels)
        sss.sort()# inverse=True  WHY INVERSE FAILES?
        [y,labels]=unzip(sss)


    def str2latexL(ss):
        subs=[
    ['$',r'\$'],
    ['_',r'\_'],
    ['%',r'\%'],
    ['#','\#'],
    ['^','\^'],
    ]
        for asub in subs:
            ss=ss.replace(asub[0],asub[1])
        return(ss)

    ind=arange(len(labels))
    if horiz: # We want to go from top to bottom of page, ie decreasing ordinate
        ind=arange(len(labels)-1,-1,-1)

    if width==None:
        width=1.0
        if labelLoc== 'atRight':
            width=0.8

    # pick TextColour:
    print ' April 2010: set colours porperly'
    if 1: # April 2010!!!!!!!!!!!!!! THIS IS NOT FINISHED YET!!!
        # And choose colour for the text labels:
        if labelColour==None:
            labelColour=[['k'],[cifar['pink']]][int(simplevector)]
        elif isinstance(labelColour,str):
            labelColour=labelColour*len(labels)
            #lcolour=labelColour
        else:
            foiuoi
            lcolour=labelColour[ii]
        #return(lcolour)
        if len(labelColour)==1:
            labelColour=labelColour*len(labels)
    htext=[[] for ff in labelColour]

    if yerr in [None]:
        erb=[[]]*len(y)
    if simplevector:
        yloffset=width/2 # offset of labels from ind

        if horiz==True:
            if 0: # April 2010: do not do them all at once?? so tht I can animate...
                rects=plt.barh(ind,y,color=barColourList,height=width,xerr=yerr,ecolor=cifar['grey'])#,width=width)

            rects=plt.barh(ind,y,color=barColourList,height=width)
            xlrect=plt.xlim()
            erb=[] # list of errorbar handles
            if yerr in [None]:
                erb=[[]]*len(y)
            else:
                for ii in range(len(yerr)):
                    erb+=[plt.errorbar(y[ii],ind[ii]+yloffset, xerr=yerr[ii],ecolor=cifar['grey'],fmt=None)[1:]]
            plt.xlim(xlrect)
            plot([0,0],plt.ylim(),'k-')

            #,xerr=yerr,ecolor=cifar['grey'])#,width=width)
            """
plt.bar(pos,val, align='center')
erb = plt.errorbar(pos, val, yerr=[np.zeros(len(yerr)), yerr], fmt=None)
erb[1][0].set_visible(False) # make lower error cap invisible
plt.xticks(pos, ('Tom', 'Dick', 'Harry', 'Slim', 'Jim'))
errorbar((0+w/2,1+w/2),V,e,elinewidth=3,ecolor='k',fmt=None,capsize=10, mew=5)

plt.show()

"""


        else:
            rects=plt.bar(ind,y,color=barColourList,width=width,yerr=yerr,ecolor=cifar['grey'])#,width=width) cifar['cyan']

    if not simplevector:

        y=array(y)
        assert len(y.shape)==2
        if grouped:
            width=(1.0/(nInGroup+.8)) # So.. leave 0.8 bar's width spacing between groups
            yloffset=width*(nInGroup/2.0) # This isn't quite right!!
        rects=[]
        for iy in range(nInGroup):
            if yerr:
                rects+=[ plt.barh(ind+(nInGroup-1-iy)*width, y[:,iy], height=width,color=barColourList[iy],xerr=yerr[:,iy],ecolor=cifar['grey'])]
            else:
                rects+=[ plt.barh(ind+(nInGroup-1-iy)*width, y[:,iy], height=width,color=barColourList[iy],ecolor=cifar['grey'])]

    if horiz:
        plt.gca().set_yticklabels([])
    else:
        plt.gca().set_xticklabels([])


    'fromLeft'
    tsize1=12*(min(1,16.0/nGroups))

    if labelLoc in ['opposite bar or inside'] and not simplevector: # Default: Place each label in whichever place has more space: just right or left of the error bar, or inside the bar!!!
        if not horiz:
            from cpblUtilities import cwarning
            cwarning('oh-oh... this not programmed ye!!!!!!')#assert horiz # vert mode not coded yet.. just change xlim to ylim..
        tsize1=0.6*18*(min(1,16.0/nGroups))
        for ii in range(len(ind)):#[1:]:
            if not yerr==None:
                ye=yerr[ii]
            else:
                ye=0.0
            ys=array([yyy for yyy in y[ii] if not isnan(yyy)])
            if len(ys)==0:
                continue

            if all([yy<=0 or isnan(yy) for yy in y[ii]]): # For horizontal mode:
                posparams=sorted([[  min(ys-ye)-xlim()[0], min(ys-ye),'right'],
                    [  -max(ys+ye), 0,'right'],
                    [  xlim()[1], 0, 'left'],
                      ],reverse=True)##.sort()##.sort(key=lambda x:x[0])[0]
            else:
                posparams=sorted([[  xlim()[1]-max(ys+ye), max(ys+ye),'left'],
                    [  min(ys-ye), 0,'left'],
                    [  -xlim()[0], 0, 'right'],
                      ],reverse=True)#.sort()##.sort(key=lambda x:x[0])[0]
            pp=[ppp for ppp in posparams if not isnan(ppp[0]) or 1]

            #pp.sort(reverse=True)
            #print labels[ii],pp
            if not pp or isnan(pp[0][1]):
                assert 0
                pp[0]=[nan,xlim()[1],'right']
            pp=pp[0]
            # Don't convert labels with str2latex, as they may already be converted.
            htext[ii]=text(pp[1],ind[ii]+yloffset,' '+labels[ii],verticalalignment='center',horizontalalignment=pp[2],size=tsize1,color=labelColour[ii])
            if 'number of' in labels[ii]:
                assert 1



    if labelLoc in ['opposite bar or inside'] and  simplevector: # Default: Place each label in whichever place has more space: just right or left of the error bar, or inside the bar!!!
        """
        Jan 2010: this used to be the default. But is there a bug? Maybe my intention was the "biggest space" behaviour, which follows. The only change there was to change an xlim()[0] to xlim()[1]... which looks like a bug or??
        """
        if not horiz:
            cwarning('oh-oh... this not programmed ye!!!!!!')#assert horiz # vert mode not coded yet.. just change xlim to ylim..
        #assert horiz # vert mode not coded yet.. just change xlim to ylim..
        for ii in range(len(ind)):#[1:]:
            if not yerr==None:
                ye=yerr[ii]
            else:
                ye=0.0
            if y[ii]<=0: # For horizontal mode:, bar is to the left
                posparams=sorted([[  (y[ii]-ye)-xlim()[0], y[ii]-ye,'right'],
                    [  -(y[ii]+ye), 0,'right'],
                    [  xlim()[1], 0, 'left'],
                      ],reverse=True)##.sort()##.sort(key=lambda x:x[0])[0]
            else: # Bar is to the right
                posparams=sorted([[  xlim()[0]-y[ii]-ye, y[ii]+ye,'left'], # 
                    [  (y[ii]-ye), 0,'left'],
                    [  -xlim()[0], min([0,y[ii]-ye]), 'right'],
                      ],reverse=True)#.sort()##.sort(key=lambda x:x[0])[0]
            pp=posparams
            #pp.sort(reverse=True)
            # Until pylab allows me to right-pad text with spaces:
            pp=pp[0]
            dpad=(xlim()[1]-xlim()[0])/100.0
            padOffset={'right':-dpad,'left':dpad}[pp[2]]
            htext[ii]=text(pp[1]+padOffset,ind[ii]+yloffset,' '+labels[ii],verticalalignment='center',horizontalalignment=pp[2],size=tsize1,color=labelColour[ii])

    print 'NOT DON HEREE::'
    if 1 and labelLoc in [None,'biggest space'] and  simplevector: # : put right/left/inside the bar, whichever has most horizontal space there!
        if not horiz:
            cwarning('oh-oh... this not programmed ye!!!!!!')#assert horiz # vert mode not coded yet.. just change xlim to ylim..
        #assert horiz # vert mode not coded yet.. just change xlim to ylim..
        if horiz:
            for ii in range(len(ind)):#[1:]:
                if not yerr==None:
                    ye=yerr[ii]
                else:
                    ye=0.0
                if y[ii]<=0: # For horizontal mode:, bar is to the left
                    posparams=sorted([[  (y[ii]-ye)-xlim()[0], y[ii]-ye,'right'],
                        [  -(y[ii]+ye), 0,'right'],
                        [  xlim()[1], 0, 'left'],
                          ],reverse=True)##.sort()##.sort(key=lambda x:x[0])[0]
                else: # Bar is to the right
                    posparams=sorted([[  xlim()[1]-y[ii]-ye, y[ii]+ye,'left'], # 
                        [  (y[ii]-ye), 0,'left'],
                        [  -xlim()[0], min([0,y[ii]-ye]), 'right'],
                          ],reverse=True)#.sort()##.sort(key=lambda x:x[0])[0]
                pp=posparams
                #pp.sort(reverse=True)
                # Until pylab allows me to right-pad text with spaces:
                pp=pp[0]
                dpad=(xlim()[1]-xlim()[0])/100.0
                padOffset={'right':-dpad,'left':dpad}[pp[2]]
                htext[ii]=text(pp[1]+padOffset,ind[ii]+yloffset,' '+labels[ii],verticalalignment='center',horizontalalignment=pp[2],size=tsize1,color=labelColour[ii])
        else:
            print "This is not finished!! won't work"
            for ii in range(len(ind)):#[1:]:
                if not xerr==None:
                    xe=xerr[ii]
                else:
                    xe=0.0
                if x[ii]<=0: # For horizontal mode:, bar is to the left
                    posparams=sorted([[  (x[ii]-xe)-ylim()[0], x[ii]-xe,'right'],
                        [  -(x[ii]+xe), 0,'right'],
                        [  ylim()[1], 0, 'left'],
                          ],reverse=True)##.sort()##.sort(key=lambda x:x[0])[0]
                else: # Bar is to the right
                    posparams=sorted([[  ylim()[1]-x[ii]-xe, x[ii]+xe,'left'], # 
                        [  (x[ii]-xe), 0,'left'],
                        [  -ylim()[0], min([0,x[ii]-xe]), 'right'],
                          ],reverse=True)#.sort()##.sort(key=lambda x:x[0])[0]
                pp=posparams
                #pp.sort(reverse=True)
                # Until pylab allows me to right-pad text with spaces:
                pp=pp[0]
                dpad=(xlim()[1]-ylim()[0])/100.0
                padOffset={'right':-dpad,'left':dpad}[pp[2]]
                htext[ii]=text(pp[1]+padOffset,ind[ii]+yloffset,' '+labels[ii],verticalalignment='center',horizontalalignment=pp[2],size=tsize1,color=labelColour[ii])

    if labelLoc=='bestEdge':
        # Place text at chart edge furthest from zero:
        if xlim()[0]<0 and xlim()[1]>=0:
            if -xlim()[0]>xlim()[1]:
                labelLoc='alignLeftAxis'
            if -xlim()[0]<xlim()[1]:
                labelLoc='alignRightAxis'
    if labelLoc=='alignLeftAxis':
        for ii in range(len(ind)):
            htext[ii]=text(xlim()[0],ind[ii]+width/2,' '+labels[ii],verticalalignment='center',horizontalalignment='left',size=tsize1,color=labelColour[ii])
    if labelLoc=='eitherSideOfZero':
        for ii in range(len(ind)):
            htext[ii]=text(0.0,ind[ii]+width/2,' '+labels[ii],verticalalignment='center',horizontalalignment='left'*(y[ii]<=0)+'right'*(y[ii]>0),size=tsize1,color=cifar['pink'])

    if labelLoc in ['fromLeft','atLeftAxis']:
        for ii in range(len(ind)):
            htext[ii]=text(0,ind[ii]+width/2,' '+labels[ii],verticalalignment='center')
    elif labelLoc== 'atRight': # I think this is done.
        for ii in range(nGroups):
            htext[ii]=text(y[ii]+yerr[ii],ind[ii]+yloffset,' '+labels[ii]+' ',verticalalignment='center',horizontalalignment='left',size=tsize1,color=labelColour[ii])
        plt.yticks([])

    elif labelLoc== 'atRightAxis':
        for ii in range(nGroups):
            htext[ii]=text(xlim()[1],ind[ii]+yloffset,' '+labels[ii]+' ',verticalalignment='center',horizontalalignment='right',size=tsize1,color=labelColour[ii])
        plt.yticks([])
    elif labelLoc== 'atLeft':
        plt.yticks(ind+width/2., labels)

    leg=None
    if grouped and 0:
        leg=plt.legend( [rr[0] for rr in rects], grouped)


    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    # Orgainse all the plot objects to they can be accessed by cateogry:

    byBar=[[rects[ii],htext[ii],erb[ii]] for ii in range(len(y))]
    if animate:
        if animate in [True]:
            animate=range(len(y))
        else:
            from cpblUtilities import orderListByRule
            assert isinstance(animate,list)
            animate=orderListByRule(range(len(y)),animate)
        plt.setp([rects,htext,erb],'visible',False)
        for ii,ia in enumerate(animate):
            plt.setp(byBar[ia],'visible',True)
            savefigall(filename+'-%d'%ii)
            
    return({'legend':leg,'bars':rects,'labels':htext,'errors':erb,'byCategory':byBar})



##########################################################################
##########################################################################
#
fNaN=float('nan')
def tonumeric(lists,to='auto',nokeys=False,cNaN=fNaN,skipkeys=None,doNothing=False):
    #
    ##########################################################################
    ##########################################################################
    """ Just a recursive version of float(), almost.
It converts empty sets to NaNs, empty dicts to {}, empty strings to NaNs.
By default it converts strings to floats.
If nokeys is used, it will not affect keys in dicts, though it will still affect their contents.

2009 August: it now considers strings equal to "." to be null, ie nan.

2010 Feb: now able to deal with arrays (rather than lists) of strings?

2010 March: adding new option "skipkeys" which means not to convert any elements of a dict named with something in this list. Also, "doNothing" means return lists unmodified, useful for internal programming, below , in recursion.
    """
    if not skipkeys:
        skipkeys=[]
    if doNothing:
        return(lists)
    
    if lists.__class__ not in [list,str,unicode, int,float]:
        import numpy
        if isinstance(lists,numpy.ndarray): # Cannot use "not lists" expression for arrays, so rest of function  would break
            return(array([tonumeric(ele,to=to,nokeys=nokeys,cNaN=cNaN) for ele in lists]))

    if not lists and lists.__class__ in [list,str,unicode]: return(cNaN)
    if not lists and isinstance(lists,dict): return({})
    if isinstance(lists,int) or isinstance(lists,float): return(lists)
    if lists.__class__ in [str,unicode]:
        try:
            if lists.strip() in ['.','','NaN','nan']:
                return(cNaN)
            if to=='auto':
                if '.' in lists:
                    to=float
                else:
                    to=int
            if to==float:
                return(float(lists))
            elif to==int:
                if '.' in lists:
                    assert abs(float(lists))>=1 # or 5e-8 would fail here?
                    return(int(float(lists)))
                else:
                    return(int(lists))
        except: #  this catches everything right now. ie returns strings intact if they don't look numeric
            return(lists)
            #return(float('nan'))
    elif isinstance(lists,list):
        return([tonumeric(ele,to=to,nokeys=nokeys,cNaN=cNaN) for ele in lists])
    elif isinstance(lists,dict):
        if nokeys:
            return(dict([ (k,tonumeric(lists[k],to=to,nokeys=nokeys,cNaN=cNaN,doNothing=k in skipkeys )) for k in lists.keys()]))
        else:
            return(dict([(tonumeric(k,to=to,nokeys=nokeys,cNaN=cNaN),tonumeric(lists[k],to=to,nokeys=nokeys,cNaN=cNaN,doNothing=k in skipkeys )) for k in lists.keys() ]))




def labelLine(lines):
    """
    Wrap this around (ie feed it hte output of) any plot or errorbar command. In both cases it will put a label at the right end of the line...
    """
    import matplotlib
    aline=lines
    while not aline.__class__== matplotlib.lines.Line2D:
        aline=aline[0]
    alabel=aline._label
    text(aline._x[-1],aline._y[-1],alabel,color=aline.get_color())
    return(lines)

def dfPlotWithEnvelope(df,xv,yv,nse=1.96,linestyle='-',color=None,linecolor=None,facecolor=None,alpha=0.5,label=None,labelson='lines',laxSkipNaNsSE=False,laxSkipNaNsXY=False,skipZeroSE=False,ax=None,laxFail=True,demo=False, lineLabel=None,patchLabel=None):
    """ interface to plotWithEnvelope for pandas DataFrames
    When you specify the name of a yvariable, we will automatically look for variables with prefixes se_ to figure out the upper and lower bounds for the envelopes (confidence intervals).

    If you want to specify labels explicitly (rather than using the column name), you can use label.
    In addition, set labelson ( in ['patches','envelopes','envelope','patch',    ... 'lines','line'] ) to choose whether lines or envelopes are shown in a legend.

    You can specify a list of strings for yv, to plot them all. By default, the column names will be used for labels.
    """
    # 2014June: if given a list for yv but not anything else, loop over them: 
    # 2014 Sept: This is broken! Ignores most other arguments of original call. And doesn't interpret other things sent as lists, like label or colors...  [Partially solved: sep2014]
    from cpblUtilitiesColor import getIndexedColormap

    # Sep 2014: Changed labels. I guess you now canont label both lines and patches! Oops. 
    if patchLabel is not None:
        print(' patchLabel is deprecated. Please use new calling format')
        assert lineLabel is None and label is None
        label=patchLabel
        labelson='patches'
    if lineLabel is not None:
        print(' lineLabel is deprecated. Please use new calling format')
        assert patchLabel is None and label is None
        label=lineLabel
        labelson='lines'
        

    def dodemo():
        df=pd.DataFrame( {'x' : pd.Series([1., 2., 3., 5.]), #, index=['a', 'b', 'c']),
             'y' : pd.Series([1., 2., 3., 4.])}) #, index=['a', 'b', 'c', 'd'])} )
        df['se_y']=df.y*-.01 + .2
        df['z']=-df.y
        df['se_z']=df.se_y
        subplot(2,2,1)
        dfPlotWithEnvelope( df,'x',['y','z'])
        transLegend(loc='center right')
        subplot(2,2,2)
        dfPlotWithEnvelope( df,'x',['y','z'],labelson='envelopes')
        transLegend(loc='best')
        subplot(2,2,3)
        dfPlotWithEnvelope( df,'x',['y','z'],labelson='patch',label=['Upper','Downer'])
        transLegend()
        return()
    if demo is True:
        dodemo()
        return()



    if yv.__class__ == list:
        return( [dfPlotWithEnvelope(df,xv,ayv,nse=1.96,linestyle='-',
                                    color=color if (color is not None and not hasattr(color,'__iter__')) else getIndexedColormap('jet',len(yv))[ii] if color is None else color[ii],
                                    linecolor=None if linecolor is None else linecolor if not hasattr(linecolor,'__iter__') else linecolor[ii], 
                                    facecolor=None,alpha=0.5,
#                                    label=None, if patchLabel is not None and label is None else ayv if label is None else label if  not hasattr(label,'__iter__') else label[ii],
#                                    patchLabel=None if patchLabel is None else patchLabel if not hasattr(patchLabel,'__iter__') else patchLabel[ii], 
#                                    lineLabel=None if lineLabel is None else lineLabel if not hasattr(lineLabel,'__iter__') else lineLabel[ii], #patchLabel=None if patchLabel in [None,False] else   label if label is not None else ayv,
#                                    lineLabel=None if labelson in ['patches','envelopes','envelope','patch'] else ayv if label is None else label if  not hasattr(label,'__iter__') else label[ii],
#                                    patchLabel=None if labelson in ['lines','line'] else ayv if label is None else label if  not hasattr(label,'__iter__') else label[ii],
                                    labelson=labelson,
                                    label= ayv if label is None else label if  not hasattr(label,'__iter__') else label[ii],
                                    laxSkipNaNsSE=laxSkipNaNsSE,laxSkipNaNsXY=laxSkipNaNsXY,skipZeroSE=skipZeroSE,ax=ax,laxFail=laxFail) for ii,ayv in enumerate( yv) ] )

    yLow=df[yv].values - nse * df['se_'+yv].values
    yHigh=df[yv].values + nse * df['se_'+yv].values
    return( plotWithEnvelope( df[xv].values,df[yv].values,yLow=yLow,yHigh=yHigh,linestyle=linestyle,color=color,linecolor=linecolor,facecolor=facecolor,alpha=alpha,label=None,
                              lineLabel=None if labelson in ['patches','envelopes','envelope','patch'] else yv if label is None else label,
                              patchLabel=None if labelson in ['lines','line'] else yv if label is None else label,
laxSkipNaNsSE=laxSkipNaNsSE,laxSkipNaNsXY=laxSkipNaNsXY,skipZeroSE=skipZeroSE,ax=ax,laxFail=laxFail))

def plotWithEnvelope( x,y,yLow=None,yHigh=None,linestyle='-',color=None,linecolor=None,facecolor=None,alpha=0.5,label=None,lineLabel=None,patchLabel=None,laxSkipNaNsSE=False,laxSkipNaNsXY=False,skipZeroSE=False,ax=None,laxFail=True):
    """ For plotting a line with standard errors at each point...
    This makes a patch object that gives the envelope around the line.

yLow and yHigh are the confidence interval lower and upper points.


uhh.. this could have been done with plt.fill() more compactly! [Done]


you can then make a legend (if you've specified lineLabel or patchLabel on each) using "legend()".

    CPBL, Jan 2010

April 2010: rewrote this with pylab.fill() and pylab.poly_between():  Vastly nicer! than old patch() kludge.

April 2010: Added facility for dealing with NaNs. We just excise them rather than splitting the plot.. (this could be an option)

May 2010: laxSkipNaNs says that rather than requiring all values to be defined for all or none of x,y,yLow,yHigh, the function will allow different subsets of each to be defined. So all that will be plotted are those x values for which all are defined (ie non NaN).  Maybe a better behaviour would be to mark points or something where x,y are defined but not yLow,yHigh...?

August 2010: Huh? No explanation for what laxSkipNaNsSE was supposed to do. if lax..XY is turned on, then lax..SE is forced on.  I think what I want is simply to drop all points with SE==NaN if lax is set, right?
Jan 2011: Maybe, but I should also simply skip the envelope whenever all the se's are nan!

Jan 2011: There exists now a fill_between() in matplotlib... I now use that below?

Jan 2013: Try to accomodate a pandas Series for x,y,etc: plotWithEnvelope(DataFrame,xvar,yvar, ???

Feb 2013: Can use format x,y,yHalfWidth instead.

    """
    from pylab import isnan,isfinite,find,where,logical_and,logical_or,logical_not#,any,sqrt,array

    import pandas as pd
    if isinstance(x,pd.DataFrame): 
        fooodles # See function above for df's!

    if yHigh is None: # Assume x,y,yHalfWidth was sent.
        yHigh=y+yLow
        yLow=y-yLow

    if color is not None:
        assert linecolor is None and facecolor is None
        linecolor=color
        facecolor=color
    if linecolor==None and not facecolor==None:
        linecolor=facecolor
    if not linecolor==None and facecolor==None:
        facecolor=linecolor
    if 1: # NO!!! Aug 2010 I need to make this use the next colour in matplotlib's colour sequence... where is that??
        if facecolor==None:
            facecolor='g' 
        if linecolor==None:
            linecolor='g'
    
    if laxSkipNaNsXY: 
        laxSkipNaNsSE=True



    if ax is None:
        ax=plt.gca()

    import pandas
    if isinstance(x,pandas.Series) and isinstance(y,pandas.Series) and isinstance(yLow,pandas.Series) and isinstance(yHigh,pandas.Series):
        x,y,yLow,yHigh=x.values,y.values,yLow.values,yHigh.values


    explainQuit=True
    quitRS='   plotWithEnvelope: Not making any envelope line "%s" '%str(lineLabel)
    if isinstance(x,float) and  isnan(x):
        assert laxFail
        return([],[])
    if not list( x ) and not list( y ) and not list( yLow ) and not list( yHigh):
        print quitRS+' because x and y are empty'
        assert laxFail
        return([],[])
    if all(isnan( x )) and all(isnan( y )) and all(isnan( yLow )) and all(isnan( yHigh)):
        print quitRS+' because x and y are all NaN'
        assert laxFail
        return([],[])
    if any(isfinite( x )) and all( isnan( y )) and all( isnan( yLow )) and all( isnan(yHigh)):
        print quitRS+' because y are all NaN'
        assert laxFail
        return([],[])
    if len(find(isfinite(y)))==1:
        print quitRS+" because there's only one y point to plot!"
        assert laxFail
        return([],[])
    # Deal is NaNs.
    """
What should we allow?? there must be matching between y,yLow,yHigh. But y need not match x. ?


Aug 2010: I'd like to rewrite this whole thing (?)
Should just have an iGood, and add each criterion one by one; can do asserts and notes when something changes.

"""
    xNaN=set(find(isnan(x)))
    yNaN=set(find(isnan(y)))
    seNaN=set(find(logical_or(logical_not(isfinite(yLow)),logical_not(isfinite(yHigh)))))
    bNoSE=all(logical_or(logical_not(isfinite(yLow)),logical_not(isfinite(yHigh))))
    seZero=set(find(logical_or(yLow==y,yHigh==y)))
    iGood=set(range(len(y)))

    assert len(x)==len(y)
    assert len(x)==len(yLow)
    assert len(x)==len(yHigh)
        
    if laxSkipNaNsXY and not bNoSE: # bNoSE means we should entirely skip the envelope
        iGood=iGood-seNaN
    if skipZeroSE:
        iGood=iGood-seZero
    if laxSkipNaNsXY:
        iGood=iGood-xNaN-yNaN
    else:
        assert xNaN==yNaN
    iGood=sorted(list(iGood))
    assert iGood



    # The following logic, new to 2014 Sept, will need refinement. e.g. what if se is zero or bnose
    # After this if,  label  will be ignored. Set patchLable or lineLabel to  something:
    if label not in [None] and lineLabel is None and  patchLabel is None and bNoSE is False:
        lineLabel=label
        #label=None
    #elif label not in [None] and patchLabel not in [None] and lineLabel is None:
    #    lineLabel=label
    #else:
    #    lineLabel=label

    #if label not in [None]:
    #    assert lineLabel in [None] or len(lineLabel)==0
    #    # I am sadly turning the following off for the moment: Sept 2014: I do not understand it, and dfPlotWithEnvelope wants to send a patchlabe...
    #    assert 1 or patchLabel in [None] or len(patchLabel)==0 
    #    lineLabel=label



    # Ahh! How to subscript with a vector??
    vals=deepcopy([x,y,yLow,yHigh])
    xorig,yorig,yLoworig,yHighOrig=deepcopy( (x,y,yLow,yHigh))
    x,y,yLow,yHigh=[[vvv[ii] for ii in iGood] for vvv in vals]
    #x[iGood],y[iGood],yLow[iGood],yHigh[iGood]



    if 0: # Old stuff, sitll needs to be reincorporated with above!!!!!

        if any(isnan(x)) or any(isnan(y)) or any(isnan(yLow)) or any(isnan(yHigh)):
            na,nb,nc,nd=where(isfinite(x)),where(isfinite(y)),where(isfinite(yLow)),where(isfinite(yHigh))
            if laxSkipNaNsSE:
                ne=where(logical_and(logical_and(isfinite(x),isfinite(y)),logical_and(isfinite(yLow),isfinite(yHigh))))
                if not laxSkipNaNsXY:
                    assert set(na[0])==set(nb[0]) # Still require x and y elements to match NaNs
                if not set(nb[0])==set(nc[0]) or not set(nc[0])==set(nd[0]):
                    print 'Warning: laxSkipNaNs is letting go a mismatch .......'
                # Aug 2010: I think above ifs are wrong! What has it got to do with laxSE?
                iGood=ne[0]
            else:
                assert set(na[0])==set(nb[0])
                assert set(nb[0])==set(nc[0])
                assert set(nc[0])==set(nd[0])
                iGood=nb[0]
            # Is below redundant?!/ ancient?
            assert list(na[0]) not in [[]] # For now, don't deal yet with nan's in x..??

            assert list(iGood) not in [[]] # For now, don't deal yet with nan's in x..??

        # --------------- OLD SECTION ABOVE TO BE REINCORPORATED IN TO FURTHER ABOVE --------------


    assert any(isfinite(x)) and  any(isfinite(y))

    hLine=ax.plot(x,y,linestyle=linestyle,color=linecolor,linewidth=2,label=lineLabel)



    #lseX=list(seX)
    #lseY=list(seY)
    #polyXY=plt.array(zip(seX,seY))###
    # was: 

    # fill_between does not support label!! in Matplotlib, yet.
    def fill_between(x, y1, y2=0, ax=None, **kwargs):
        """Plot filled region between `y1` and `y2`.

        This function works exactly the same as matplotlib's fill_between, except
        that it also plots a proxy artist (specifically, a rectangle of 0 size)
        so that it can be added it appears on a legend.
        """
        ax = ax if ax is not None else plt.gca()
        ax.fill_between(x, y1, y2, **kwargs)
        p = plt.Rectangle((0, 0), 0, 0, **kwargs)
        ax.add_patch(p)
        return p



    if 0: # Obselete now that fill_between exists:
        xs, ys = ax.poly_between(x, yLow, yHigh)
        if bNoSE:
            envelopePatch=None
        else:
            envelopePatch=ax.fill(xs,ys,facecolor=facecolor,alpha=alpha,linewidth=0,label=patchLabel)# edgecolor=None, does not work!! So use linewidth instead?
    if bNoSE:
        envelopePatch=None
    else:
        envelopePatch=fill_between(x, yLow, yHigh,ax=ax,facecolor=facecolor,alpha=alpha,linewidth=0,label=patchLabel)# edgecolor=None, does not work!! So use linewidth instead?




        #if patchLabel is not None:
        #    envelopePatch=ax.fill_between(x, yLow, yHigh,facecolor=facecolor,alpha=alpha,linewidth=0,label=patchLabel)# edgecolor=None, does not work!! So use linewidth instead?
        #    ouoiuoiu
        #else:
        #    envelopePatch=ax.fill_between(x, yLow, yHigh,facecolor=facecolor,alpha=alpha,linewidth=0)# edgecolor=None, does not work!! So use linewidth instead?



    #seY=yLow+yHigh[::-1]
    #seX=x+x[::-1]

    #polyXY=plt.transpose(plt.array([seX,seY]))

    #envelopePatch=mpl.patches.Polygon(polyXY,True,facecolor=facecolor,alpha=alpha,linewidth=0,label=patchLabel)# edgecolor=None, does not work!! So use linewidth instead?
    #ax.add_patch(envelopePatch)

    return(hLine,envelopePatch)


def axisNearlyTight(ax=None):
    from pylab import axis,getp,xlim,ylim,gca,exp,log
    if ax==None:
        ax=gca()
    plt.axes(ax)
    plt.axis('tight') # Sigh... this seems not to work well.
    if getp(ax,'xscale')=='log':
        xr=max(xlim())/min(xlim())
        xlim(min(xlim())/exp(log(xr)/20.0),max(xlim())*exp(log(xr)/20.0))
    else:
        xr=max(xlim())-min(xlim())
        xlim(min(xlim())-xr/20.0,max(xlim())+xr/20.0)
    if getp(ax,'yscale')=='log':
        yr=max(ylim())/min(ylim())
        ylim(min(ylim())/exp(log(yr)/20.0),max(ylim())*exp(log(yr)/20.0))
    else:
        yr=max(ylim())-min(ylim())
        ylim(min(ylim())-yr/20.0,max(ylim())+yr/20.0)


def figureFontSetup(uniform=12,figsize='paper'):
    """
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

2012 May: new matplotlib has tight_layout(). But it rejigs all subplots etc.  My inkscape solution is much better, since it doesn't change the layout. Hoewever, it does mean that the original size is not respected! Sigh... Still, my favourite way from now on to make figures is to append the font size setting to the name, ie to make one for a given intended final size, and to do no resaling in LaTeX.  Use tight_layout() if it looks okay, but the inkscape solution in general.
n.b.  a clf() erases size settings on a figure! 

    """
    figsizelookup={'paper':(4.6,4),'quarter':(1.25,1) ,None:None}
    params = {#'backend': 'ps',
           'axes.labelsize': 16,
           'text.fontsize': 14,
           'legend.fontsize': 10,
           'xtick.labelsize': 16,
           'ytick.labelsize': 16,
           'text.usetex': True,
           'text.usetex.unicode': True,
           'figure.figsize': figsizelookup.get(figsize,figsize)
        }
           #'figure.figsize': fig_size}
    if uniform is not None:
        assert isinstance(uniform,int)
        params = {#'backend': 'ps',
           'axes.labelsize': uniform,
           'text.fontsize': uniform,
           'legend.fontsize': uniform,
           'xtick.labelsize': uniform,
           'ytick.labelsize': uniform,
           'text.usetex': True,
           'text.usetex.unicode': True,
           'figure.figsize': figsizelookup.get(figsize,figsize)
           }
    plt.rcParams.update(params)
    plt.rcParams['text.latex.unicode']=True
    #if figsize:
    #    plt.rcParams[figure.figsize]={'paper':(4.6,4)}[figsize]

    return(params)


def addSignatureToPlot():
    """
    my translegend can now also add annotations [annotate] at the bottom (or top!?) of the legend.
    But this is nice; you can include newlines etc in the text.
    Could add option for box, and options for placement.
    """
    Dx=max(xlim())-min(xlim())
    Dy=max(ylim())-min(ylim())
    th=plt.text(max(xlim())-Dx/100.0,min(ylim())+Dy/50.0,'C. Barrington-Leigh\nMcGill');
    plt.setp(th,'fontsize',7,'verticalalignment','bottom','horizontalalignment','right')
    return(th)


##########################################################################
##########################################################################
#
class Position:
    #
    ##########################################################################
    ##########################################################################
    """ This is about as much GIS as I really need here: calculate great circle distance between two points on a sphere. This is done through the subtraction operation (-) for this clas. """

    def __init__(self, longitude, latitude):
        import math
        'init position with longitude/latitude coordinates'
        llx = math.radians(longitude)
        lly = math.radians(latitude)
        self.x = math.sin(llx) * math.cos(lly)
        self.y = math.cos(llx) * math.cos(lly)
        self.z = math.sin(lly)

    def __sub__(self, other):
        import math
        'get distance in km between two positions'
        d = self.x * other.x + self.y * other.y + self.z * other.z
        if d < -1:
            d = -1
        if d > 1:
            d = 1
        km = math.acos(d) / math.pi * 20000.0
        return km

    #def __mul__(self, other): # Scalar multiply?
    #    import math
    #    assert isinstance(other,float)
    #    return(self.__init(self.x*other,self.y*other))

    # Yikes! This needs to be able to return lat, lon, etc.. much more needed.



def cpblScatter(df,x,y,z=None,markersize=20,cmap=None,vmin=None,vmax=None,labels=None,nse=1,ax=None,fig=None,clearfig=False,marker='o',labelfontsize=None):#,xlog=False,ylog=False,): #ids=None,xlab=None,ylab=None,fname=None,,xlog=False,byRegion=None,labelFunction=None,labelFunctionArgs=None,fitEachRegion=False,regionOrder=None,legendMode=None,markerMode=None,subsetIndices=None,forceUpdate=True,labels=None,nearlyTight=True,color=None): #markersize=10         
    """
    2013: Simplify. If you have groups, it's easiest to plot with groupby
    z would be values to turn into colours (with cmap), or the name of the field in df to use for that.

    For simplicity, let's say everything has to be a column name of the df for now.

    See regressionsQuebec animation for a good example of use.

    Use    addColorbarNonimage(df.year.min(), df.year.max(),useaxis=None,ylabel=None) to separately add a colorbar to the result.

    """
    assert isinstance(df,pd.DataFrame)

    #if ids is None and len(df.index)>0:
    #    ids=[str(ii) for ii in df.index]
    #elif isinstance(ids,str):
    #    ids=df[ids].values
    #if z is not None: #isinstance(color,str):
    #    assert color is None
    #    #color=df[color].values
    if isinstance(markersize,str):
        markersize=df[markersize].values
    if 0 and labels is None and df.index not in [None,[]] and len(df)<20: # Only default to labelling if there are few
        labels=[str(LL) for LL in df.index]
    elif labels is not None and labels.__class__  is str:
        labels=df[labels].values
    #plot(df[x].values,df[y].values,'.')
    #return()
    if z is None:
        z='b'
    elif isinstance(z,str) and z in df:
            z=df[z].values
    elif isinstance(z,str):
        pass
    elif isinstance(z,int):
        pass
    #z=np.ones(len(df))*z
    if markersize is None:
        markersize=20 # plt.scatter() default
    if labelfontsize is None:
        labelfontsize=14

    if fig is not None:
        plt.figure(fig)
    if clearfig:
        plt.clf()
    if ax is None:
        ax=plt.gca()

    xx=x if not isinstance(x,str) else df[x].values 
    yy=y if not isinstance(y,str) else df[y].values 

    # #################################### ERROR BARS
    xse,yse=None,None#df[x].values*np.nan,df[x].values*np.nan
    eb=[]
    if isinstance(x,str):
        if 'se'+x in df:
            xse=nse*df['se'+x].values#,nse*df['se'+y].values
        elif 'se_'+x in df:# and 'se_'+y in df:
                xse=nse*df['se_'+x].values#,nse*df['se_'+y].values
    if isinstance(y,str):
        if 'se'+y in df:
            yse=nse*df['se'+y].values
        elif 'se_'+y in df:
            yse=nse*df['se_'+y].values
    if xse is not None or yse is not None:
        eb=ax.errorbar(df[x].values,df[y].values,yerr=yse,xerr=xse,fmt=None,ecolor='0.85',zorder=-99) #,markersize=markersize)
    # Do errorbars start at edges of marker, or at center?
    

    # #################################### SCATTER PLOT: FILLED CIRCLES WITH SIZE AND COLOUR
    sc=ax.scatter(xx,yy, s=markersize, c=z, marker=marker, cmap=cmap, norm=None,   vmin=vmin, vmax=vmax, alpha=None, linewidths=None,      verts=None)#, **kwargs)


    if isinstance(x,str):  xlabel(x)
    if isinstance(y,str):  ylabel(y)


        
    # #################################### COLOR BAR
    # Do it yourself in calling routine: Just call cb=plt.colorbar(sc). Or use my function, below somewhere.... Below is a nice example:
    #    axtmp=fig.add_axes((.2,.1,.7,.8),label='dummy')  # Fractional units. Make it square, overlaying the right end of first axis
    #    hcb=plt.colorbar(hh['sc'],aspect=10) #norm=cnorm,
    #    delaxes(axtmp)


    # #################################### LABELS
    # This could be substituted with passed function. But this one makes sure labels at right end of plot go on the left of points, and vice versa.
    lb=[]
    if labels is not None:
        Dx=max(xlim())-min(xlim())
        Dy=max(ylim())-min(ylim())
        for xx,yy,L in zip(df[x].values,df[y].values,labels):
            if xx>= min(xlim())+.8*Dx:
                offset=-Dx/100.0
                ha='right'
            else:
                offset=Dx/100
                ha='left'
            lb.append(ax.text(xx+offset,yy+Dy/100,L+'   ',rotation='horizontal',horizontalalignment=ha,fontsize=labelfontsize,verticalalignment='center',color='k'))

                    
    return({'markers':sc,'eb':eb,'labels':lb})

def cpblScatterPlotDF(df,x,y,nse=1,ids=None,xlab=None,ylab=None,fname=None,fig=None,xlog=False,byRegion=None,labelFunction=None,labelFunctionArgs=None,fitEachRegion=False,regionOrder=None,legendMode=None,markersize=None,markerMode=None,subsetIndices=None,forceUpdate=True,labels=None,nearlyTight=True,color=None): #markersize=10         
    assert isinstance(df,pd.DataFrame)
    print('Deprecated. Switch over to cpblScatter... May2013. If doing groups, make new function that calls cpblScatter more than once.')
    sorry
    xse,yse=df[x].values*np.nan,df[x].values*np.nan
    if 'se'+x in df:
        xse=nse*df['se'+x].values#,nse*df['se'+y].values
        
    elif 'se_'+x in df:# and 'se_'+y in df:
        xse=nse*df['se_'+x].values#,nse*df['se_'+y].values
    if 'se'+y in df:
        yse=nse*df['se'+y].values
    elif 'se_'+y in df:
        yse=nse*df['se_'+y].values
    if ids is None and len(df.index)>0:
        ids=[str(ii) for ii in df.index]
    elif isinstance(ids,str):
        ids=df[ids].values
    if xlab is None:
        xlab=x
    if ylab is None:
        ylab=y
    if isinstance(color,str):
        color=df[color].values
    if isinstance(markersize,str):
        markersize=df[markersize].values
    if labels is None and df.index not in [None,[]]:
        labels=[str(LL) for LL in df.index]
    elif labels.__class__  is str:
        labels=df[labels].values
    #plot(df[x].values,df[y].values,'.')
    #return()


    plt.figure(fig)
    plt.scatter(df[x].values,df[y].values, s=markersize, c=color, marker='o', cmap=None, norm=None,
        vmin=None, vmax=None, alpha=None, linewidths=None,
        verts=None)#, **kwargs)
    ihih



    
    return(
    cpblScatterPlot(
df[x].values,df[y].values,
ids=ids,
xlab=xlab,
ylab=ylab,
fname=fname,
fig=fig,
xlog=xlog,
byRegion=byRegion,
labelFunction=labelFunction,
labelFunctionArgs=labelFunctionArgs,
fitEachRegion=fitEachRegion,
regionOrder=regionOrder,
legendMode=legendMode,
markersize=markersize,
markerMode=markerMode,
subsetIndices=subsetIndices,
forceUpdate=forceUpdate,
xse=xse,
yse=yse,
labels=labels,
nearlyTight=True,color=color))

##########################################################################
##########################################################################
#    This function absorbed scatterPlotSEs, eventually.
def cpblScatterPlot(
# Common arguments:
x,y,
# Deprecated:
fn=None,
# arguments for by-region group-coloured plots:
ids=None,xlab=None,ylab=None,fname=None,fig=None,xlog=False,byRegion=None,labelFunction=None,labelFunctionArgs=None,fitEachRegion=False,regionOrder=None,legendMode=None,markersize=None,markerMode=None,subsetIndices=None,forceUpdate=True,         
# arguments for error-bar plots (from old scatterPlotSE)
xse=None,yse=None,labels=None,nearlyTight=True,color=None): #markersize=10
    #
    ##########################################################################
    ##########################################################################
    """ 
Feb 2010: Adding "byRegion" option: pass the region name for each country

March 2010: Moved to cpblUtilitiesMathGraph from regressionsGallup (may need generalising!?)
    # So this is for Gallup...

Hmmm. trying to generalise, so:

    if labelFunction(x,y,ids,color=None, labelFunctionArgsDict=None) is specified, it will be used to label nations within each region.


    fitEachRegion: turn this on to add a temporary regression line to each subsample, and one to the overall set.

    March 2010: Getting rid of logx and making an xlog, ie just a flag that the x value is a log value already.  If so, this is noted in the filename and  reflected in the axis scaling. Prior to now, this function used to make both non-log and log versions whenever the defunct logx was True.


I used to have two scatter plot functions. One that makes SEs and one that does it by regions. Now, there is ACTIONMODE to do different things, in some cases, depending on what is passed.

LAlignMode:
nnMLB
MLT

April 2010: Automatically choose this based on density of points in four quadrants? Argh, am I reinventing the auto legend placements??
So far I haven't done the following at the beginning... hmmm. (elementwise logic)
    xx=x[find(~isnan(x) & ~isnan(y))]
    yy=x[find(~isnan(x) & ~isnan(y))]


September 2010: added z=None. z data will be represented in symbol size, at least for the currently-displayed region...  At first, z data will need to be in the actual right units of size, not some arbitrary scale. well, maybe it should be renamed o be symbolsize (yes) for now.

markerMode: This will specify various behaviour about when/whether dots or scaled-size scatterplot symbols are shown. Default behaviour for starters (Sept 2010):  when symbolSize data are provided, they are only displayed for each region when it's being highlighted. ie simple dots are shown the rest of the time. Also, the symbolsizes are just relative, and are all rescaled. Also, they are rescaled for EACH GROUP, not for the whole set. Note tht markerMode can be a list of strings to specify various settings.
Here are some options
['hiddenInGlobalView',  # Agh.. no, right now this doesn't work, cause i want hidden in dots view but not hidden in final, with lines and annotations view.
'rescaledByGroup' or 'rescaledGlobal' or 'absoluteScale']
]
I also need to make a "labelsMode" or get the country labels working somehow.

Hm, did I know about scatter() when I wrote this?

Sept 2010: forceUpdate=True: By default, this function always recreates plots. But if forceUdpate=False, it will skip actually saving the plots if it already exists!

Sept 2010: TO DO!!! incorporate the cpblUtilitiesLabelPlacement code!
In general, it seems the label placement is so customised/kludged, it may be better to just use the labelfunction as much as possible...


Sept 2010: Now returns a list of filenames (or stems) if filenames are given. Otherwise, it returns a list of handles to stuff plotted (???)[latter not tested2013]

Sep 2010:  This used to capitalise axis labels in the byregion mode.  I'm taking that off. Can put it back in as an option, but then need to not capitalise stuff inside $...$. !

2013 Feb: 

2013 May: Rewrite this using matplotlib's (new?) scatterplot. I just need to add errorbars, etc.
    """

    from cpblUtilities import uniqueInOrder
    from math import sqrt
    import pylab as plt
    from pylab import xlim,ylim,arrow,errorbar,figure,savefig,setp,getp,ion,show,array,gca,log10

    if not fname:
        fnameTitle='test'
    try: # June 2011: Not sure why fnameTitle is not available now...
        fnameTitle
    except:
        print '       caution: using '+fname+' in place of fnameTitle (undefined) (loc2)'
        fnameTitle=fname
    if fig==None:

        fign=    plt.mod(sum([ord(f)^2 for f in fnameTitle]),1000) 
        fig = plt.figure(fign)
        #plt.setp(fig,'figheight',11.0125,'figwidth', 8.1125)
        fig.clear()
        ax = fig.add_subplot(111)
    else:
        figure(fig)
        ax=plt.gca()
        
    if ylab==None:
        ylab=''
    if xlab==None:
        xlab=''
    def dummyLabelFunction(x,y,ids,args=None,color=None):
        return([])
    if labelFunction==None:
        labelFunction=dummyLabelFunction


    if xlog==True:
        xlog='-logx'
    else:
        assert xlog in [False, None]
        xlog=''
    if fn is not None:
        assert fname is None
        fname=fn
        print " WARNING! USE OF fn IS DEPRECATED IN CPBLSCATTERPLOT. USE FNAME"
    fDir=None
    if not fname is None:
        fDir,fname= os.path.split(fname)
    if not fDir:
        from cpblDefaults import paths
        fDir=paths['graphics']#'/home/cpbl/gallup/graphicsOut/'
    else:
        fDir+='/'
    if markersize in [None]:
        markersize=10 # This is needed for s.e. version..
        if 0:  #Agh, no! The following fails in one of the kinds of scatter plots! (se)
            markersize=[None for _ in x]
        assert markerMode in [None]
    #if markersize.__class__ in [int,float]:
    #    markersize=array([markersize for xx in x])

    if markerMode in [None]:
        markerMode=['hiddenInGlobalView','rescaledByGroup']
    assert not ('rescaledByGroup' in markerMode and  'rescaledGlobally' in markerMode)
    assert not ('absoluteScale' in markerMode and ('rescaledByGroup' in markerMode or  'rescaledGlobally' in markerMode))



    ACTIONMODE=[]
    if xse is not None or yse is not None:
        ACTIONMODE+=['errorbars']
    if byRegion:
        ACTIONMODE+=['by region']
    if not ACTIONMODE:
        ACTIONMODE+=['errorbars']

    outfns=[] #Collect list of graphics files written.

    def safelen(stuff): # return len if it's not in [None, [], array([])]:
        if stuff in [None,[]]:
            return(0)
        if stuff.__class__ in [int,float]:
            return(1)
        return(array(stuff).size)# isinstance(a, numpy.ndarray)

        
    assert len(x)==len(y)
    origLen=len(x)
    x=array(x)
    y=array(y)
    if 'by region' in ACTIONMODE:
        assert len(x)==len(ids)
        ids=array(ids)
    if subsetIndices not in [None]:#.any():
        x=x[subsetIndices]
        y=y[subsetIndices]
        if 'by region' in ACTIONMODE:
            byRegion=array(byRegion)[subsetIndices]
        if safelen(ids):
            ids=ids[subsetIndices]
        if safelen(markersize)==origLen:
            markersize=array(markersize)[subsetIndices]
        if xse is not None:
            xse=xse[subsetIndices]
        if yse is not None:
            yse=yse[subsetIndices]
    if isinstance(labels,dict): # So labels is a lookup by ids. Convert to a list.
        assert safelen(ids)
        labels=[labels[anid] for anid in ids]
    if safelen(labels):
        assert not labelFunction or labelFunction  == dummyLabelFunction
    if not labelFunction  == dummyLabelFunction:
        assert safelen(ids)

    if 'errorbars' in ACTIONMODE:
        """ This is my Sept 2010 kludge to start merging these functions."""

        """
        def _scatterPlotSE(x,y,xse,yse,fname=None,labels=None,nearlyTight=True,color=None,markersize=10):


        NearlyTight sets the xlim to just slightly bigger than tight.


    N.B.!!!!! I now have two scatter plot functions. One that makes SEs and one that does it by regions.
    Shouldn't I combine these????
    yes. They're now both in cpblScatterPlot()

        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $ BEGIN:  Scatter plot with STANDARD ERRORS
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        """

        if safelen(labels):##labels and labels.size:
            labels=array(labels)
        else:
            labels=array([])


        if 0: # I'm removing his june 2012. what isf i've set a fig or axis?
           figureFontSetup()


        #LL=0.1*(max(xlim())-min(xlim()))*min([sss for sss in ses if not isnan(sss)])

        if color is None:
            color='k'#['k' for xx in x]

        plabels,bplabels,perrorbars=[],[],[]
        secolor='0.85' # ie a shade of gray.  secolor='gray'
        # If xse, yse =None,Non, following will not barf, but nothing plotted [No longer true! 2013. we should barf if no se in this ACTIONMODE ,no?]

        if color.__class__ in [np.array,np.ndarray] or color.__class__ is str and len(color)>1:#isinstance(color,list) or isinstance(color,np.array):
            for ixx,xx in enumerate(x):
                if xse is None and yse is None:
                    perrorbars+=ax.plot(xx,y[ixx],'o',markersize=markersize,color=color[ixx],label=labels[ixx])
                else:
                    perrorbars+=ax.errorbar(xx,y[ixx],yerr=yse[ixx],xerr=xse[ixx],fmt='o',ecolor=secolor,markersize=10,color=color[ixx],label=labels[ixx])
        else:
            perrorbars+=ax.errorbar(x,y,yerr=yse,xerr=xse,fmt='o',ecolor=secolor,markersize=markersize,color=color)
            adot=ax.plot(x,y,'.',markersize=10,color=color[0])
            setp(adot,'visible',False)


        rotateLabels=False
        #print '+++++++++++++++++++++++++',fnameTitle
        if 0 and "LadderVLadder" in fnameTitle:
            oiuoiu
        if 0 and any(perrorbars):
            setp(adot,'visible',False)
        ## else:
        ##     adot=[]
        ##     for ixx,xx in enumerate(x):
        ##         adot+=ax.plot(xx,y[ixx],'.',markersize=10,label=labels[ixx],color=color[ixx])

        if labelFunction is not dummyLabelFunction:
            plabels=labelFunction(x,y,ids,args=labelFunctionArgs)#,color=rcolors[ir])#,label=aregion)
        elif  safelen(labels):#.size: # ie if not empty
            """
            I can make the labels go along the lines, but I'd need to convert dy and dx to more physical units, since the axes are not in image mode.
            """
            for iix in range(len(x)):#if 1:#len(coefsByCRuid.keys())<20:#postPlotArgs['geoOrder'])<20:
                #geoLabels=dict(provinceNames)
                xi,yi=x[iix],y[iix]
                # Bug in matplotlib: text fails if NaN:
                if isnan(xi) or isnan(yi):
                    continue
                if not isnan(xi) and not isnan(yi):
                    if rotateLabels:
                        plabels+=[plt.text(xi,yi,namesByCRuid[CRuid],rotation=plt.arctan2(dy,dx)*57.0)]
                    else:
                        plabels+=[plt.text(xi+.01,yi,labels[iix]+'   ',rotation='horizontal',horizontalalignment='right',fontsize=14,verticalalignment='center',color='k')]#color[0])]#plt.getp(adot[0],'color'))]

                if 'betterLabels':
                    Dx=max(xlim())-min(xlim())
                    Dy=max(ylim())-min(ylim())
                    if xi>= min(xlim())+.8*Dx:
                        offset=-Dx/100.0
                        ha='right'
                    else:
                        offset=Dx/100
                        ha='left'

                    bplabels+=[plt.text(xi+offset,yi+Dy/100.0,labels[iix]+'   ',rotation='horizontal',horizontalalignment=ha,fontsize=14,verticalalignment='center',color='k')]#color[0])]#plt.getp(adot[0],'color'))]
                    plt.setp(plabels,'visible',False)


        if rotateLabels:
            plt.axis('image')
                #horizontalalignment='right',fontsize=8,verticalalignment='center',


        #plt.ylabel(yName.upper(),size=20)
        #plt.xlabel(xName.upper(),size=20)

        if 0: # Sorry..!! July 2011 I'm cutting out this kludge from some specific thing?!. oh.. no.. it's a bug in pylab that I'm hoping will disappear at some point??
                # Uhhh... fix an annoying bug in text placed with whitespace at end:
                xr=max(xlim())-min(xlim())
                [label.set_position(label.get_position()+plt.array([-xr/12.0,0])) for label in plabels]

        #figure(fign)
        if nearlyTight:
            plt.axis('tight') # Sigh... this seems not to work well.
            xr=max(xlim())-min(xlim())
            yr=max(ylim())-min(ylim())
            xlim(min(xlim())-xr/20.0,max(xlim())+xr/20.0)
            ylim(min(ylim())-yr/20.0,max(ylim())+yr/20.0)


        #return(plabels,[])
        """
    for ic=1:length(names)
        name=names{ic};
        Dx=max(xlim)-min(xlim);
        Dy=max(ylim)-min(ylim);
        if x(ic)>= min(xlim)+.8*Dx,
            offset=-Dx/100; ha='right';
            %offset=Dx/100; ha='left';
        else
            offset=Dx/100; ha='left';%.006
        end % if

        if length(name)>1,
            th=text(x(ic)+offset,y(ic)+.01,name);%[upper(name(1)) lower(name(2:end))]);
            set(th,'fontsize',10,'verticalalignment','middle','horizontalalignment',ha)
            if th,        ths=[ths th]; end %if
        end %if
        fprintf(' Labelling "%s"\n',name);

        if settings.signature,
            th=text(max(xlim)-Dx/100,min(ylim)+Dy/50,'C. Barrington-Leigh, UBC');
            set(th,'fontsize',7,'verticalalignment','bottom','horizontalalignment','right')
        end % if
    end %for

    """

        #_scatterPlotSE(x,y,xse,yse,fname,labels,nearlyTight,color,markersize)
        xlabel(xlab)
        ylabel(ylab)
    if 'by region' in ACTIONMODE:
        """
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $ BEGIN:  Scatter plot with REGION GROUPS
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        """
        # These are hardcoded or Gallup, so need fixing/parameterising for general case!
        limsDefault={
        'SWLylim':(2,8.5),
        'INCFRACxlim':  [0.15,12],
        'INCFRACxlimLOG':(0.007,2),
        'POWERxlimLOG':(.2,10.5),
            }
        lims=limsDefault

    ##         if 'INCOME' in xlab.upper() and 'FRACTION' in xlab.upper():
    ##             xlim(INCFRACxlimLOG)
    ##         if 'POWER PER CAPITA' in xlab.upper():
    ##             xlim(POWERxlimLOG)
    ##         if 'SATISFACTION' in ylab.upper():
    ##             ylim(SWLylim)



        adot=plot(x,y,'.',hold=True)
        setp(adot,'visible',False)

        if xlog:
            setp(gca(),'xscale','log')
        axisNearlyTight()
        if 'SATISFACTION' in ylab.upper():
            ylim(lims['SWLylim'])
        if xlog:
            if 'INCOME' in xlab.upper() and 'FRACTION' in xlab.upper():
                xlim(lims['INCFRACxlimLOG'])
            if 'POWER PER CAPITA' in xlab.upper():
                xlim(lims['POWERxlimLOG'])
            if 'SATISFACTION' in ylab.upper():
                ylim(lims['SWLylim'])
                #INCFRACxlim=  [0.15,12]


        hDbyRegion,hLbyRegion,fitsByRegion,bigMarkersByRegion={},{},{},{}
        """'cifar['cyan']'# 'CIFARdarkgreen' 'cifar['green']'# 'cifar['grey']' 'CIFARlightblue' 'CIFARpalekhaki' 'cifar['pink']'# 'CIFARyellow'
        """
        rcolors=['r','b','g','k','m','c','y']+[cifar[kk] for kk in cifar]+['r','b','g','k','m','c','y']+[cifar[kk] for kk in cifar]+['r','b','g','k','m','c','y']+[cifar[kk] for kk in cifar]+['r','b','g','k','m','c','y']+[cifar[kk] for kk in cifar]

        if regionOrder:
            regs=regionOrder
        else:
            regs=uniqueInOrder([kk for kk in byRegion if not (isinstance(kk,float) and isnan(kk))], drop= ['98','99','']+['nan']+[nan])
            regs=uniqueInOrder( byRegion, drop= ['98','99','']+['nan']+[nan,NaN])#dropNaN=True)
        #assert regs or not uniqueInOrder(byRegion)
        assert not len(regs)==0


        # Now evaluate where to put legend based on density of points:
        xx=x[find(~isnan(x) & ~isnan(y))]
        yy=y[find(~isnan(x) & ~isnan(y))]
        if xlog:
            LL=xx<=sqrt(xlim()[0]*xlim()[1])
        else:
            LL=xx<=mean(xlim())
        TT=yy>= mean(ylim())
        quadrants=[len(find(LL * TT)), len(find((~ LL) * TT)),
                   len(find(LL * (~ TT))), len(find((~ LL) * (~ TT))), ]
        wquadrant=[0,0,
                   0,0]
        # Automatically choose legendMode:
        assert not legendMode # unused now.
        if not legendMode:
            LAlignMode='auto'
            iLA=sorted( enumerate(quadrants),key=lambda z:z[1])[0][0] #['LLT','RRT', 'LLB','RRB']
        if xlog:
            xspace=pow(xlim()[1]/xlim()[0],(1.0/50))
            xbase=[min(xlim())*xspace,     max(xlim())/xspace,
                   min(xlim())*xspace,   max(xlim())/xspace  ]
        else:
            xspace=(max(xlim())-min(xlim()))/50.0
            xbase=[min(xlim())+xspace,     max(xlim())-xspace,
                   min(xlim())+xspace,   max(xlim())-xspace,  ]
        yspace=(max(ylim())-min(ylim()))/50.0
        ybase=[max(ylim())-yspace, max(ylim())-yspace,
               min(ylim())+yspace, min(ylim())+yspace ]
        ' 0.32 was for the Gallup incomes/SWL plots...  ie 7 regions??'
        yincrement=array([-0.32, -0.32,
                    +0.32, +0.32])/.32*(max(ylim())-min(ylim()))/2.0/len(uniqueInOrder(regs))
                     #+0.32, +0.32])/.32*(max(ylim())-min(ylim()))/15.0
        """ Preposterous. This should depend on ylim(). 2011nov: adding in affect versions. so generalising from 0.32 (also done below, where it's hardcoded for not in auto mode.
        oh, wait, it's not .32 anymore. it was already scaled. ah, but it should not depend on nregions! it should just depend on the font size vs plt size...
        """
        # 2011 Nov: following simple formula works nicely!
        yincrement=array([-1, -1,
                          +1, +1])*yspace*50/20
        

        horizontalalignment=['left', 'right',
                             'left', 'right']
        verticalalignment=['top','top',
                             'bottom','bottom']
        for ir,aregion in enumerate(regs):
            inRegion=find(array(byRegion) == aregion)
            #if lastMarkers:
            #    plt.setp(lastMarkers,'visible',False)#fitsByRegion[aregion]        
            hDbyRegion[aregion]= plt.plot(x[inRegion],y[inRegion],'.',hold=True,color=rcolors[ir],label=aregion)

            if safelen(markersize)==safelen(x) and any(isfinite(array(markersize)[inRegion])): # This if line excludes Nones and NaNs , respectively.
                z=array(markersize)[inRegion]#array(markersize)[inRegion]
                #if z.__class__ in [int,float]:
                    #z=array([markersize for xx in x])
                if 'rescaledByGroup' in markerMode:
                    aa,bb=min(z),max(z)
                    z=((z/bb)-(aa/bb)+.1)*100
                if 'rescaledGlobally' in markerMode:
                    aa,bb=min(markersize),max(markersize)
                    z=((z/bb)-(aa/bb)+.1)*100
                if 'absoluteScale' in markerMode:
                    z=array(markersize[inRegion])
                bigMarkersByRegion[aregion]= plt.scatter(x[inRegion],y[inRegion],s=z,marker='o',hold=True,c=rcolors[ir],label=aregion)
            else:
                print '  Not using large markers here becuase markersize was not specified for each point'
                hDbyRegion[aregion]= plot(x[inRegion],y[inRegion],'.',hold=True,color=rcolors[ir],label=aregion)


            if labelFunction:
                hLbyRegion[aregion]=labelFunction(x[inRegion],y[inRegion],ids[inRegion],args={'xoff':0},color=rcolors[ir])#,label=aregion)
                setp(hLbyRegion[aregion],'label',aregion)
            if 'doRegionLabelsAsWell-horrid-kludge-nov2011':
            # else: # Huh? 2011 Nov: it looks like above labels countries, while below writes region name. Why skip this if labelFunction exists?
                spf=0.32 # KLUDGE!! vertical spacing factor. This was .32 for 2010 plots.
                """ Let's say there is space for 20 labels in a reasonable font (size 14 now???), top to bottom.
                so if "yspace" is already designed to be yrange/50, let
                spf=yspace*50/20
                """
                spf=yspace*50.0/20.0 #Oh, see above for auto case (yincrement).

                if LAlignMode=='auto':
                    hLbyRegion[aregion]+=[text(xbase[iLA],ybase[iLA]+ir*yincrement[iLA],aregion,color=rcolors[ir], verticalalignment=verticalalignment[iLA],horizontalalignment=horizontalalignment[iLA])]
                elif xlog and LAlignMode=='MLB': #middle, left-aligned, bottom
                    hLbyRegion[aregion]+=[text(sqrt(xlim()[0]*xlim()[1]),min(ylim())+yspace+ir*spf,aregion,color=rcolors[ir])]
                elif not xlog and LAlignMode=='MLB': #middle, left-aligned, bottom
                    hLbyRegion[aregion]+=[text(mean(xlim()),             min(ylim())+yspace+ir*spf,aregion,color=rcolors[ir])]
                elif xlog and LAlignMode=='MLT': #middle, left-aligned, top
                    hLbyRegion[aregion]+=[text(sqrt(xlim()[0]*xlim()[1]),max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top')]
                elif not xlog and LAlignMode=='MLT': #middle, left-aligned, top
                    hLbyRegion[aregion]+=[text(mean(xlim()),             max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top')]
                elif xlog and LAlignMode=='RRB': #middle, left-aligned, bottom
                    hLbyRegion[aregion]+=[text(max(xlim())/xspace,min(ylim())+yspace+ir*spf,aregion,color=rcolors[ir],horizontalalignment='right')]
                elif not xlog and LAlignMode=='RRB': #middle, left-aligned, bottom
                    hLbyRegion[aregion]+=[text(max(xlim())-xspace,             min(ylim())+yspace+ir*spf,aregion,color=rcolors[ir],horizontalalignment='right')]
                elif xlog and LAlignMode=='RRT': #middle, left-aligned, top
                    hLbyRegion[aregion]+=[text(max(xlim())/xspace,max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top',horizontalalignment='right')]
                elif not xlog and LAlignMode=='RRT': #middle, left-aligned, top
                    hLbyRegion[aregion]+=[text(max(xlim())-xspace,             max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top',horizontalalignment='right')]
                elif xlog and LAlignMode=='LLT': #left, left-aligned, top
                    hLbyRegion[aregion]+=[text(xspace*min(xlim()),max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top')]
                elif not xlog and LAlignMode=='LLT': #left, left-aligned, top
                    hLbyRegion[aregion]+=[text(xspace+min(xlim()),             max(ylim())-yspace-ir*spf,aregion,color=rcolors[ir],verticalalignment='top')]
                else:
                    assert 0 # Don't know LAlignMode
                    ###print 'aaaaaaaaaaaaaaaaaaaaahhhhhhh this fails on fedora!!!!! '
            if any(isfinite(x[inRegion])) and any(isfinite(y[inRegion])) :#'fails on fedora only':
                fitsByRegion[aregion],gradient,s=overplotLinFit(x[inRegion],y[inRegion],format='-',color=rcolors[ir],xscalelog=xlog)
                assert fitsByRegion[aregion]
            else:
                fitsByRegion[aregion],gradient,s=[],[],[]
        #legend([hLbyRegion[kk][0] for kk in regs])

        #labs=labelFunction(x,y,ids,xoff=0)
        #plot(x,y,'.',hold=True)
        setp(adot,'visible',False)
        del adot
        xlabel(xlab.replace('KW/','kW/')) # .upper()
        ylabel(ylab)  #.upper()
        if 'SATISFACTION' in ylab.upper():
            ylim(lims['SWLylim'])
        from cpblUtilities import flattenList
        #allFits=flattenList([fitsByRegion[kk] for kk in fitsByRegion])
        #print allFits
        setp(fitsByRegion.values(),'visible',False)
        setp(hLbyRegion.values(),'visible',False)
        setp(bigMarkersByRegion.values(),'visible',False)

        #savefigall('/home/cpbl/gallup/graphicsOut/'+fname+'-8')#,transparent=transparent)
        for ir,region in enumerate(regs):
            if fname:
                outfns.append(savefigall(fDir+fname+xlog+'-%d'%(0+ir),skipIfExists=not forceUpdate))#(len(regs)-ir))#,transparent=transparent)
            setp(fitsByRegion.values(),'visible',False)
            #region=regs[len(regs)-ir-1]
            #setp(fitsByRegion.values(),'visible',False)
            setp(bigMarkersByRegion.values(),'visible',False)
            setp(hLbyRegion.values(),'visible',False)
            setp(fitsByRegion[region],'visible',True)
            setp(hLbyRegion[region],'visible',True)
            setp(bigMarkersByRegion.get(region,[]),'visible',True) # For case when, e.g. markers show population size
            ###setp(fitsByRegion[region],'visible',False)

        if fname:
            outfns.append(savefigall(fDir+fname+xlog+'-%d'%(0+len(regs)),skipIfExists=not forceUpdate))#(len(regs)-ir))#,transparent=transparent)
        setp(fitsByRegion.values(),'visible',False)
        setp(hLbyRegion.values(),'visible',True)
        setp(bigMarkersByRegion.values(),'visible',True)
        if  'fails on fedora only': #It won't anymore, though the python stats linreg causes a warning... obselete?!
            fitsByRegion['all'],gradient,s=overplotLinFit(x,y,format='-',color=[0.5,0.5,0.5],xscalelog=xlog)
        if fname:
            outfns.append(savefigall(fDir+fname+xlog+'-%d'%(1+len(regs)),skipIfExists=not forceUpdate))
        setp(fitsByRegion.values(),'visible',False)
        if fname:
            outfns.append(savefigall(fDir+fname+xlog+'-%d'%(2+len(regs)),skipIfExists=not forceUpdate))
        try: # June 2011: Not sure why fnameTitle is not available now...
            fnameTitle
        except:
            print '       caution: using '+fname+' in place of fnameTitle (undefined) '
            fnameTitle=fname
        title(fnameTitle+xlog)


    if fname and not 'by region' in ACTIONMODE:
        outfns.append(savefigall(fDir+fname,skipIfExists=not forceUpdate))
        title(fname)

    if fname:
        return(outfns)
    else:
        return(perrorbars)

##     #savefig(fDir+fname+'.png',transparent=transparent)
##     #savefig(fDir+fname+'.pdf',transparent=transparent)

##     # Make log version too?
##     if 'INCOME' in xlab.upper() or logx:
##         clf()
##         dummy=plot(x,y,'.',hold=True)
##         axisNearlyTight()
##         if 'SATISFACTION' in ylab.upper():
##             ylim(SWLylim)
##         if 0:
##             setp(fitsByRegion.values(),'visible',False)
##             setp(hLbyRegion.values(),'visible',False)
##             # "del" below does NOT get rid of them! So do two lines above.
##             del hLbyRegion
##             del fitsByRegion
##         setp(gca(),'xscale','log')
##         axisNearlyTight()
##         if 'INCOME' in xlab.upper() and 'FRACTION' in xlab.upper():
##             xlim(INCFRACxlimLOG)
##         if 'POWER PER CAPITA' in xlab.upper():
##             xlim(POWERxlimLOG)
##         if 'SATISFACTION' in ylab.upper():
##             ylim(SWLylim)
##             #INCFRACxlim=  [0.15,12]


##         hDbyRegion,hLbyRegion,fitsByRegion={},{},{}
##         rcolors=['r','b','g','k','m','c','y']+[cifar[kk] for kk in cifar]
##         ncolor=0
##         x=array(x)
##         y=array(y)
##         ids=array(ids)
##         #regs=[kk for kk in uniqueInOrder(byRegion) if kk not in ['98','99','']]
##         for aregion in regs:
##             inRegion=find(array(byRegion) == aregion)
##             hDbyRegion[aregion]= plot(x[inRegion],y[inRegion],'.',hold=True,color=rcolors[ncolor],label=aregion)
##             hLbyRegion[aregion]=labelFunction(x[inRegion],y[inRegion],ids[inRegion],xoff=0,color=rcolors[ncolor])#,label=aregion)
##             setp(hLbyRegion[aregion],'label',aregion)
##             if 0:#hLbyRegion[aregion]:
##                 hLbyRegion[aregion].set_label(aregion)
##             text(sqrt(xlim()[0]*xlim()[1]),min(ylim())+.1+ncolor*.32,aregion,color=rcolors[ncolor])
##             ncolor+=1
##             fitsByRegion[aregion],gradient,s=overplotLinFit(x[inRegion],y[inRegion],format='-',color='grey')

##         #labelFunction(x,y,ids,xoff=0)


##         for ir,region in enumerate(regs):
##             savefigall(fDir+fname+'-logx-%d'%(1+ir))#(len(regs)-ir))#,transparent=transparent)
##             setp(fitsByRegion.values(),'visible',False)
##             #region=regs[len(regs)-ir-1]
##             #setp(fitsByRegion.values(),'visible',False)
##             setp(fitsByRegion[region],'visible',True)
##             setp(hLbyRegion[region],'visible',True)
##             ###setp(fitsByRegion[region],'visible',False)

##         savefigall(fDir+fname+'-logx-%d'%(1+len(regs)))#(len(regs)-ir))#,transparent=transparent)
##         setp(fitsByRegion.values(),'visible',False)
##         fitsByRegion['all'],gradient,s=overplotLinFit(x,y,format='-',color=[0.5,0.5,0.5])
##         savefigall(fDir+fname+'-logx-%d'%(2+len(regs)))
##         setp(fitsByRegion.values(),'visible',False)
##         savefigall(fDir+fname+'-logx-%d'%(3+len(regs)))


##         #savefig(fDir+fname+'_logx.png',transparent=transparent)
##         #savefig(fDir+fname+'_logx.pdf',transparent=transparent)

##     #savefig(fDir+fname+'_t.png',transparent=transparent)

scatterPlotByRegion=cpblScatterPlot
dfscatterplot=cpblScatter

from math import modf, floor
def quantile(x, q,  qtype = 7, issorted = False):
    """

I want something like this but which admits weights for the data!

    
    Args:
       x - input data
       q - quantile
       qtype - algorithm
       issorted- True if x already sorted.
 
    Compute quantiles from input array x given q.For median,
    specify q=0.5.
 
    References:
       http://reference.wolfram.com/mathematica/ref/Quantile.html
       http://wiki.r-project.org/rwiki/doku.php?id=rdoc:stats:quantile
 
    Author:
	Ernesto P.Adorio Ph.D.
	UP Extension Program in Pampanga, Clark Field.
    """
    if not issorted:
        y = sorted(x)
    else:
        y = x
    if not (1 <= qtype <= 9): 
       return None  # error!
 
    # Parameters for the Hyndman and Fan algorithm
    abcd = [(0,   0, 1, 0), # inverse empirical distrib.function., R type 1
            (0.5, 0, 1, 0), # similar to type 1, averaged, R type 2
            (0.5, 0, 0, 0), # nearest order statistic,(SAS) R type 3
 
            (0,   0, 0, 1), # California linear interpolation, R type 4
            (0.5, 0, 0, 1), # hydrologists method, R type 5
            (0,   1, 0, 1), # mean-based estimate(Weibull method), (SPSS,Minitab), type 6 
            (1,  -1, 0, 1), # mode-based method,(S, S-Plus), R type 7
            (1.0/3, 1.0/3, 0, 1), # median-unbiased ,  R type 8
            (3/8.0, 0.25, 0, 1)   # normal-unbiased, R type 9.
           ]
 
    a, b, c, d = abcd[qtype-1]
    n = len(x)
    g, j = modf( a + (n+b) * q -1)
    if j < 0:
        return y[0]
    elif j > n:
        return y[n]
 
    j = int(floor(j))
    if g ==  0:
       return y[j]
    else:
       return y[j] + (y[j+1]- y[j])* (c + d * g)    
 
def quantileTest():
    x = [11.4, 17.3, 21.3, 25.9, 40.1, 50.5, 60.0, 70.0, 75]
 
    for qtype in range(1,10):
        print qtype, quantile(x, 0.35, qtype)

def transAnnotation_draftOld(comments,pos=None,loc=None):
    """ I think this is waiting to find the algorithm for optimal-positioning. [oct 2011]
    I suppose in the mean time you could just use it as a shorthand for putting things in corners?
    But it's hardly worth functionalising it. Just copy the code!

    """
    leftB,rightB=plt.xlim()
    botB,topB=plt.ylim()
    dX=rightB-leftB
    dY=topB-botB
    plt.gca().text(rightB-.025*dX,topB-.025*dY,waves[wave]+'\n'+r'$\pm$2 s.e. shown'+0*('Chris Barrington-Leigh\n                         McGill'), ha='right', va="top", size=14,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    
def transAnnotation_stillNoGood(comments,pos=None):
    """
    Attempt again Nov 2011. Use a single, blank item to make a legend. Then remove that item from the legend and add in the coments. This makes use of matplotlib's auto placement
    """
    from pylab import xlim,ylim,flatten,gca,legend
    # Start with a dummy legend entry:
    dummyh=plot(mean(xlim()),mean(ylim()),'.',label=' xx')
    lh=legend(dummyh,loc='best')
    oiuoiu
    plt.setp(dummyh,'visible',False)
    # Now, modify the text content of legend
    from matplotlib.offsetbox import TextArea, VPacker 
    fontsize=None
    if isinstance(comments,str):
        comments=[comments]
    if lh:
        fontsize=lh.get_texts()[0].get_fontsize()
        # else!! Warning: if lh==None, there were no lines plotted.. hmm.
        legendcomment=TextArea('\n'.join(comments), textprops=dict(size=fontsize)) 
        lh._legend_box = VPacker(pad=5,
                                   sep=0, 
                                   children=[legendcomment],  # ie omit: lh._legend_box,
                                   align="left")  # Or should it be centre?
        lh._legend_box.set_figure(plt.gcf())
        plt.draw() # Do i need this?
    return()

def transAnnotation(comments,loc=None,removeOldLegends=True,title=None,titlesize=None,box=True,rv=False,color=None):
    """ Above Still no good! No automatic placement. Maybe use
    matplotlib.legend.Legend._find_best_position(self, width, height, renderer, consider=None)  ??

    oops. Adding "loc" for sensible compatibility! I started with pos for no good reason

    oops! Why am I not using axes coordinates?? why would i use data coords? without taking into account log, etc....
    """
    #assert pos is None # June 2012: phase it out. comment this out to bypass..
    #assert not pos or not loc
    #if pos:
    #    loc=pos
    if loc is None:
        loc=''
    if not ' this function is not written yet. make it use the fine_best_position....':
        plt.setp(plt.findobj(plt.gcf(),lambda x: isinstance(x,mpl.text.Text ) and x._label=='cpblAnnotation'),'visible',False)
    """
    gca().text(.252,-.13,'Chris Barrington-Leigh\n                         McGill', ha='left', va="center", size=7,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        gca().text(.328,-.13,r'$\pm$1 s.e. shown', ha='right', va="center", size=7,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
"""

    # I guess I'm going to need to specify loc for this option. grrr.

    if titlesize is None: # Should really use textsize, but not imple yet
        titlesize=20

    xl,yl=xlim(),ylim()
    dx,dy=xl[1]-xl[0],yl[1]-yl[0]
    tlx,tly,ha=xl[0]+.1*dx,  yl[1]-.1*dy,'left' # top left 
    if loc.lower() in ['northeast','ne','topright']:
        tlx,tly,ha=xl[1]-.05*dx,  yl[1]-.1*dy,'right' # top right
    if loc.lower() in ['southwest','sw']:
        tlx,tly,ha=xl[0]+.1*dx,  yl[0]+.1*dy,'left' # bottmleft 
    if loc and loc.lower() in ['southeast','se','bottom right','lower right']:
        tlx,tly,ha=xl[1]-.1*dx,  yl[0]+.1*dy,'right' # bottmright
    if isinstance(loc,list):
        assert len(loc)==3
        tlx,tly,ha=loc
    if title:
        comments=title+'\n'+comments
    bbox=None
    if box:
        bbox=dict(boxstyle="round", fc='k' if rv else "w", ec="0.5", alpha=0.9)

    # Test. rewrite them all like this?
    if loc and loc.lower() in ['southeast','se','bottom right','lower right']:
        tt=gca().text(.95,.05, comments, ha='right', va="bottom", size=titlesize,  bbox=bbox,label='cpblAnnotation',transform=gca().transAxes,color=color)
    else:
        tt=gca().text(tlx, tly, comments, ha=ha, va="center", size=titlesize,  bbox=bbox,label='cpblAnnotation',color=color)


    print """this is not finished. and no auto placement!. Actually, you shoul djust use a one-line boxed text, e.g.:

                gca().text(.252,-.13,'Chris Barrington-Leigh\n                         McGill', ha='left', va="center", size=7,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
        gca().text(.328,-.13,r'$\pm$1 s.e. shown', ha='right', va="center", size=7,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
"""


    return(tt)#(transLegend(comments=comments,showOnly=[],loc=loc))

# 2014: For three years, transLegend hasn't been working properly. Retire it! and just use these wrappers around the normal legend call:
def addCommentToLegend(comments,lh):
    # 2014. Updated method. From Jae-Joon Lee
    if lh is None: 
        print(' *FAILING* to add "%s" to legend, since legend does not exist'%comments)
        return(None)
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(comments)
    box = lh._legend_box 
    # STILL TO DO!!! Make new txt width the same as box width
    # Maybe this will be useful: http://abitofpythonabitofastronomy.blogspot.ca/2010/01/customized-legend.html
    box.get_children().append(txt)
    box.set_figure(box.figure)
    return(lh)
def transbg(lh): # Make a legend transparent (This could learn to treat other objects, too: figures/axes?)
    if lh is None: return(None)
    lh.get_frame().set_alpha(0.5)
    assert lh is not None
    return(lh)

def transLegend(comments=None,title=None,loc='best',bbox_to_anchor=None,ncol=1):
    """
    Some features of now-retired transLegend2013, previously transLegend(), not yet implemented.


This can be deprecated, though, in favour of this approach:
        lh=legend(loc='lower left',ncol=2)
        assert lh is not None
        addCommentToLegend(r'Bands show 90% confidence intervals', transbg(lh))
    """
#    lh=plt.legend(loc=loc)
    lh=plt.legend(fancybox=True,shadow=False,title=title,loc=loc,bbox_to_anchor=bbox_to_anchor,ncol=ncol)
    lh2=transbg(lh)
    if comments:
        addCommentToLegend(comments,lh2)
    return(lh2)


def transLegend2013(figlegend=False,comments=None,removeOldLegends=True,showOnly=None,title=None,loc='best',titlesize=None,bbox_to_anchor=None,ncol=1): # Make a legend and, if it's not empty, set some transparency properties.
    """
    Started out simply making a translucent legend (somewhat safely).

    If comments (extra lines in legend) supplied, then return value gives them too in a tuple. Comments come AFTER evverything else. The title gives stuff to go before the main content of the legend..

    removeOldLegends gets rid of one if there's already


    May 2010: how the hell do i restrict which objects get listed in the legend??????????? it says this is not supported?!?!
    showOnly takes a possibly nested list of objects. only these are shown.. Hm. But what about the order?! damn!

    Can I use this purely to make annotations? Or should I have a separate one to do that? how about transLegend(showOnly=[],comments='hello')
    Hm, no. I should just make a box when there's nothing but annotations, so that the annotations can start flush left. Not done. See boxstyle below...


Sept 2010: Figured out how to add comments nicely!!! So edit above notes [oh? i've not got comments only done yet.]
So, title goes on top; comments go below.

Need to do some error checking here to deal with pylab bugs and fragility: e.g. if tex mode not set, warnings ..?

    April 2011: Agh! I've broken my own work.
    showOnly=None: should work like legend(), using the best guess (all) of labelled objects.
    showOnly=[]: should avoid showing any legend entries. This just should produce a comment box...?

Annotations:
 you can set loc to be a 3-vector of x,y,h-alignment.

n.b. if you want to include handles from more than one axis, eg. axes made with twinx(), just specify the handles with "showonly"

Dec 2011: A new bug cropped up with Ubuntu 11.10: now if I specify comments in this function, the legend appears off-screen, or out o faxes, anyway. Argh.

    """

    from pylab import xlim,ylim,flatten,gca

    import os #duhh... this bug is due to mixing ubuntu versions??? delete .matplotlib? 
    if 0: #os.uname()[2][0]=='3':
        from pylab import legend
        lh=legend()
        lh.get_frame().set_alpha(0.5)
        return

    #assert showOnly != []
    if showOnly==[] and not comments==None: # Just do an annotation!!!
        transAnnotation(comments,loc=loc,title=title,titlesize=titlesize)
        return()


    commentsH=[]
    if comments and isinstance(comments,str):
        comments=[comments]
    #if comments:
    #    if isinstance(comments,str):
    #        comments=[comments]
    #    for astr in comments:
    #        commentsH+=plt.plot([xlim()[0],xlim()[0]],[ylim()[0],ylim()[0]],'-',color=[1,1,1],visible=False,label=astr)

    if 0 and showOnly in [[],None] and commments is None and title is None:
        print 'No transLegend unless you specify showonly?! or title or comments'
        return
    if 0 and showOnly==None: # No, don't do this! legend() makes the best choice, ie of how to deal with groups of lines?? The following gives redundant entries.
        #import matplotlib.text as text
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle, Shadow, FancyBboxPatch
        #from matplotlib.collections import LineCollection, RegularPolyCollection,  CircleCollection

        showOnly=plt.gcf().findobj(Line2D)#[h for h in self.legendHandles if isinstance(h, Line2D)]#[xxx for xxx in plt.gcf().findobj(lambda x:hasattr(x, 'label')) if xxx not in plt.gcf().findobj(text.Text)]






    if showOnly not in [[],None]:
        from cpblUtilities import flattenList
        showOnly=[xx for xx in flatten(showOnly) if xx.get_label()]+[xx for xx in flatten(commentsH) if xx.get_label()] 

        showLabels=[oob.get_label() for oob in showOnly]

        if removeOldLegends: # Possibly I could actually delete the object using o.remove()
            plt.gca().legend_ = None         # Remove existing legend
            plt.draw(  )                 # Remove existing legend


    if figlegend:
        1/0 # No, don't use that. maybe I can move it outside the axes if desired, though.
        lh=plt.figlegend(fancybox=True,shadow=False,loc=loc)
    else: # June 2010: Bug in pylab: loc=0 fails with log y axis? well, not in trivial cases. hm.
        if 0 and loc==None:
            loc=0
        if showOnly is None:
            lh=plt.legend(fancybox=True,shadow=False,title=title,loc=loc,bbox_to_anchor=bbox_to_anchor,ncol=ncol)
        elif showOnly not in [[]]:
            # # Feb 2011: need? horrific bloody kludge here to eliminate envelopes. matplotlib bug??
            #for ixx,xx in enumerate(showOnly):
            #    lh=plt.legend([xx],[showLabels[ixx]])#,fancybox=True,shadow=False,title=title,loc=loc)#loc=0,
            lh=plt.legend(showOnly,showLabels,fancybox=True,shadow=False,title=title,loc=loc,ncol=ncol)#loc=0,
                
        else:
            print """Do not know yet how to make a legend without lines (ie benefit from auto placement), and so I don't have a comment-only function as far as I can tell."""

            assert not showOnly is None # Not sure this is right. apr2011
            # The following does nothing. This is broken. I posted April 2011 on google groups
            return()
            lh=plt.legend(plt.gcf(),[],[],fancybox=True,shadow=False,title=title,loc=loc,ncol=ncol)#loc=0,
    if lh:
        lh.get_frame().set_alpha(0.5)

    if comments:
        from matplotlib.offsetbox import TextArea, VPacker 
        fontsize=None    
        if lh:
            fontsize=lh.get_texts()[0].get_fontsize()
            # else!! Warning: if lh==None, there were no lines plotted.. hmm.
            if 1: # Bloody hell. late 2011, this stopped working with a pylab update.
                legendcomment=TextArea('\n'.join(comments), textprops=dict(size=fontsize)) 
                lh._legend_box = VPacker(pad=5, 
                                           sep=0, 
                                           children=[lh._legend_box,legendcomment], 
                                           align="left")  # Or should it be centre?
                lh._legend_box.set_figure(plt.gcf())
                plt.draw() # Do i need this?
            else:
                print '_)))))))))) Bug in matplotlib from 2011: Therefore dropping comments for legend: ',comments

    if lh and  title and titlesize:
        fontsize=lh.get_texts()[0].get_fontsize()
        plt.setp(plt.getp(lh,'title'),'fontsize',titlesize)

    if comments:
        return(lh,commentsH)
    else:
        return(lh)







def labelAbscissaSurveyYears(surveys,yGood=None,years=None,addQuebecHistory=False,clearOld=True,fontsize=None,fgcolor='k'):
    """
    This is rather too specialised to be in this module, but where else? to make sure I find it easily.

    Mark cycle of GSS or other survey on a plot for which x scale is year (time).

    years could be figured out (if None) from survey name

    yGood is either a vector of y values the same length as surveys, or a boolean vector of same length. It is used to show only surveys that have good data.

    if addHistory, it ONLY adds history notes to the x axis
    """
    from pylab import ylim
    # Label cycles
    if years==[]:  # Since defaults not programmed here yet.
        return()
    assert years # Since defaults not programmed here yet.
    # Ensure yGood is boolean vector; may have come as floating, ie the y data for each *possible* year/survey.
    if yGood==None:
        yGood=[True for yy in years]
    if not all([yy==True or yy==False for yy in yGood]):
        yGood=[isfinite(yy) for yy in yGood]

    if clearOld:
        oos=plt.findobj(plt.gca(),lambda o:o.get_label()=='labelAbscissaSurveyYears')
        while oos: # This is a bit awkward. It's just to deal with duplicates! Weird.
            oos[-1].remove()  # FINALLY!! I have found out how to remove an object from a plot!!
            oos=plt.findobj(plt.gca(),lambda o:o.get_label()=='labelAbscissaSurveyYears')

    if not addQuebecHistory:
        for ixx,xx in enumerate(years): # this just uses years of last province!
            if yGood[ixx]:
                plt.text(xx,min(ylim()),surveys[ixx].replace('GSSp','GSS'),horizontalalignment='center',verticalalignment='bottom',rotation=90,label='labelAbscissaSurveyYears',fontsize=fontsize,color=fgcolor)

    if addQuebecHistory:
        """ For presentation:  Should also label the folllowing (as another overlay version!! Make rest of figure fade, and overlay arrows...
        """
        history=[[1965,'1960s Quiet Revolution'],
                 [1970,'October Crisis'],
                 [1977,'Bill 101'],
                 [1980,'Referendum on sovereignty-association'],
                 [1982,'Canada Act Constitution'],
                 [1987,'Meech Lake Accord ...'],
                 [1992,'... to Charlottetown Accord'],
                 [1995,'Second referendum on sovereignty'],
                 ]
        for ayear,aevent in history:
            plt.text(ayear,min(ylim()),aevent,horizontalalignment='center',verticalalignment='bottom',rotation=90,color=fgcolor)


# I THINK FOLLOWING NOT USED ANYWHERE. PYGREP AND TRASH IT.
def empiricalcdf(data, method='Hazen'):
	    """Return the empirical cdf.
            [copied from scipi...]
hih? trash

	    Methods available:
	        Hazen:       (i-0.5)/N
	            Weibull:     i/(N+1)
	        Chegodayev:  (i-.3)/(N+.4)
	        Cunnane:     (i-.4)/(N+.2)
	        Gringorten:  (i-.44)/(N+.12)
	        California:  (i-1)/N
	   
    Where i goes from 1 to N.
	    """
	   
	    i = np.argsort(np.argsort(data)) + 1.
	    N = len(data)
	    method = method.lower()
	    if method == 'hazen':
	        cdf = (i-0.5)/N
	    elif method == 'weibull':
                cdf = i/(N+1.)
	    elif method == 'california':
	        cdf = (i-1.)/N
	    elif method == 'chegodayev':
	        cdf = (i-.3)/(N+.4)
	    elif method == 'cunnane':
	        cdf = (i-.4)/(N+.2)
	    elif method == 'gringorten':
	        cdf = (i-.44)/(N+.12)
	    else:
	        raise 'Unknown method. Choose among Weibull, Hazen, Chegodayev, Cunnane, Gringorten and California.'
	   
	    return cdf

def plotQuantilesYvsX(x,y,w,nSegments=20,nquantiles=None,yAreDiscrete=False,loessKernelWidth=.45,skiploess=False,decimateLoessTo=1000):
    """
    March 2013: not yet pandas-ready...
    I'm having problems with loess. Maybe I should trap errors running rpy_loess

    March 2012: Oops. Should be using runmed in R: the most robust median smoothing possible...
    
    CPBL 2012 Feb
    Order the data by x. Split up the data into nSegments, based on weights (even split of quantiles, using weights? Or even split of points...?) For each range, calculate various quantiles (and mean and sem?). Plot them.
    So this is  meant to look like a set of nested envelopes (ie thick shaded lines/regions) showing various interquantile ranges.

    if y takes on discrete values, we don't want precise quantile values, because they'll be too discrete, themselves. Instead, we can take averages of quantile bands, hoping that at least sometimes these include a mix of more than one value. (still, they often won't!)

    So:

    yAreDiscrete: if False, plots bands bounded by quantile values, e.g. one envelope would be bounded by the 25th and 75th %iles.  If True, plots mean values within quantile bands, e.g. one envelope would be bounded by the mean of 20%-30% %ile and the 70th-80th%ile..

    x,y,w: vectors of floats


    Feb 2012: So I should really incorporate the other things I've done of a similar nature, e.g. this s hould also use

   rpy_loess(x,y,w=None,nPts=None,plotTitle=None,kernelwidth=0.75,qBands=None,OLDFORMAT=False)

   This actually shows quantile bands, continuously!  [not yet included!]
   
   and it should also cacluate the means as it goes along. [maybe done: elements mean,sem for each segment]

   and it should also do the loess from Stata. I used this for OrdinalGWP 2012 paper.

   And my other rolling y(x) (no other controls) functions?

   the problem with this plan is that I might want to use a different number of segments for means vs quantiles... Hm, not sure why, actually.

    """
    from cpblUtilities import dsetset,finiteValues
    if x in [None,[],0]:
        import random
        N=20000
        respondents=[dict(x=random.uniform(0,10),y=random.normalvariate (.2,.1),w=random.uniform(.1,.8)) for ii in range(N)]
    else:
        N=len(x)
        respondents=[dict(x=x[ii], y=y[ii],w=w[ii]) for ii in range(N) if isfinite(y[ii]) and isfinite(x[ii])]
    """
    x=[random.uniform(0,10) for ii in range(20000)]
    y=[random.paretovariate(.2) for xx in x]
    w=[random.uniform(.1,.8) for for xx in x]
    """
    if N>100000:
        print 'Starting plotQuantilesYvsS...'

    respondents.sort(key=lambda z: z['x'])
    print 'Sorted complete list...'

    if skiploess is True:
        rpyloess=None
    else:
        xx,yy,ww=finiteValues(x,y,w) # This is done within rpy_loess, no?
        import rpy2# for error code
        try: 
            rpyloess=rpy_loess(xx,yy,w=ww,nPts=None,plotTitle=None,kernelwidth=loessKernelWidth,qBands=[.1,.25,.5,.75,.9],OLDFORMAT=False,decimateTo=decimateLoessTo)
        except rpy2.rinterface.RRuntimeError:
            print("RPYLOESS FAILED .... Let's continue without any loess..")
            rpyloess=False # Should probably return None, but this is for now to indicate a Failure.

    # First version used sortDictsIntoQuantiles to choose the x-based groups for calculation. This took forever for some reason. #:( Just do it manually here:

    
    sd=sortDictsIntoQuantiles(respondents,'x',1,nQuantiles=nSegments,approxN=None)
    qSets={}
    for segment in sorted(sd.keys()):
        if N>100000:
            print 'Starting segment ',segment
        rds=sorted(sd[segment],key=lambda z:z['y']) # This would fail if there were NaNs for 'y' values !!!! yikes.
        xx=[rr['x'] for rr in rds]
        yy=[rr['y'] for rr in rds]
        ww=[rr['w'] for rr in rds]
        dsetset(qSets,[segment,'x'],np.average(xx,weights=ww))

        if not yAreDiscrete:
            weightedQuantileGenerator=weightedQuantile(yy,ww)
            for q in [.10,.25,.45,.5,.55,.75,.9]:
                dsetset(qSets,[segment,q],   weightedQuantile(yy,ww,q) )
        if yAreDiscrete:
            """ Careful. Right now, there are big gaps in quantile values when """
            qmids=[.1,.25,.5,.75,.9]
            qys=sortDictsIntoQuantiles(sd[segment],'y','w',nQuantiles=1,approxN=None)[0.0] # Create 'rankY' element in each.
            for qm in qmids:
                dq=.05
                inrangeR=[rr for rr in sd[segment] if rr['filledrankY']>=qm-dq and rr['filledrankY']<qm+dq]
                dsetset(qSets,[segment,qm],np.average([rr['y'] for rr in inrangeR],weights=[rr['w'] for rr in inrangeR]))
        # Regardless of wether y are discrete, also calculate the piecewise mean and sem:
        
        dsetset(qSets,[segment,'mean'],np.average([rr['y'] for rr in sd[segment]],weights=[rr['w'] for rr in sd[segment]]))
        dsetset(qSets,[segment,'sem'],wtsem([rr['y'] for rr in sd[segment]],w=[rr['w'] for rr in sd[segment]]))

    sdk=sorted(sd.keys())
    Xs=[qSets[xx]['x'] for xx in sdk]

    #figure(nSegments) # Leave this up to calling function.
    plt.fill_between(Xs, [qSets[xx][0.1] for xx in sdk], [qSets[xx][0.9] for xx in sdk] ,alpha=.3)
    plt.fill_between(Xs, [qSets[xx][0.25] for xx in sdk], [qSets[xx][0.75] for xx in sdk] ,alpha=.3)
    if not yAreDiscrete:
        plt.fill_between(Xs, [qSets[xx][0.45] for xx in sdk], [qSets[xx][0.55] for xx in sdk] ,alpha=.3)
    # Median:
    plot(Xs, [qSets[xx][0.5] for xx in sdk],'r')


    return({'qSets':qSets,'loess':rpyloess})
    

def weightedQuantile(qv,weights,test=False,method=None):
    """
    cpbl March 2013

    Gives ranks for values qv, when each value has a sample (probability) weight in weights.

    There are different ways to calculate quantile: we could assign the total weight to the left of a given entry, or we could assign each to the midpoint of the space between the left and right neighbours.
    method=left: quantile is proportion of distribution with strictly lower value
    method=middle: repeated values are assigned the mid point between their collective range in the distribution
    method=continuous: repeated values are given different quantiles, in order to make a more continuous distribution

    Accepts lists, numpy arrays, or a pandas dataframe(?).
    """
    import numpy as np
    # From stack exchange? No, from wikipedia

    if method is None:
        method="middle"

    if test:
        for qq,ww in [
          [  [100, 80, 70, 80], [1,1,1,2],]
        ]:
            for mm in ['left','middle','continuous']:
                print qq,ww,mm
                print "             -->", weightedQuantile(qq,ww,method=mm)
        return
    
    # We accept lists
    if isinstance(qv,list): qv=np.array(qv)*1.0
    if isinstance(weights,list): weights=np.array(weights)*1.0

    # We accept data frames (pandas)
    usePandas=isinstance(qv,pd.core.series.Series) or isinstance(qv,pd.DataFrame) # In latter case, need to check for single dimension!
    if usePandas:
        qv=qv.values
        weights=weights.values

    # We accept numpy arrays
    assert isinstance(qv,np.ndarray)
    assert isinstance(weights,np.ndarray)

 

    if method=="continuous":  # This uses whatever ordering exists to rank ties, ie falsely assigning an order to identical values
        sindex=np.argsort(qv)
        cs=np.cumsum(np.array(weights)[sindex])
        cs=(cs[:-1]/cs[-1]).tolist()
        qq=[0]+cs # Return value, returned below
        q=np.arange(len(qv))*np.nan #nan*qv
        q[sindex]=qq


    if method in ['left',"middle"]:
       #finiteIndex=np.logical_and(np.isfinite(qv),np.isfinite(weights))
        finiteIndex=np.logical_and(pd.notnull(qv),pd.notnull(weights))
        af,wf=qv[finiteIndex],weights[finiteIndex]
        if not len(af):
                print('   %d missing values in weightedQuantile: skipping'%len(qv))
                return(np.nan)
        if not len(wf):
                print('   %d missing weights in weightedQuantile: skipping'%len(weights))
                return(np.nan)
        if 0:
            print('    %d values, %d weights in weightedQuantile: proceeding'%(len(qv),len(weights)))

        sort_indx = np.argsort(af)

        tmp_weights = wf[sort_indx]
        cum_weights = np.cumsum(tmp_weights)
        # Normalise weights
        tmp_weights = tmp_weights/cum_weights[-1]
        cum_weights = cum_weights/cum_weights[-1]

        if method=="middle":
        # To get it completely right,  we should assign for x_i:   
        # q= cum sum up to the previous entry in the list (i-1) plus 1/2 of x_i's weight.
            qSorted= np.append(0, cum_weights) + 0.5 * np.append(tmp_weights,0)
        elif method=="left": 
            def weighted(sortedv,sortedw):
                cumulated_weight = 0
                effective_cumulated_weight = 0
                prev_val = None
                for vv,ww in zip(sortedv,sortedw):
                    if prev_val != vv:
                        effective_cumulated_weight = cumulated_weight
                    #print '        : ',vv,ww,':',effective_cumulated_weight, cumulated_weight
                    yield 0*ww + effective_cumulated_weight
                    prev_val = vv
                    cumulated_weight += ww 
            qSorted=[w for w in weighted(qv[sort_indx],tmp_weights)]
        q=np.arange(len(qv))*np.nan #nan*qv
        q[find(finiteIndex)[sort_indx]]=qSorted
        #print method,':',qv,qv[sort_indx],'-->',qSorted,'-->',q
    if usePandas:
        return(pd.Series(q))
    return(q)

    # Formula from stats.stackexchange.com post.
    s_vals = [0.0];
    for ii in range(1,N):
        s_vals.append( ii*tmp_weights[ii] + (N-1)*cu_weights[ii-1])
    s_vals = np.asarray(s_vals)

 

def weightedQuantile_g_deprec(x,w,q=None):
    """
    return an x-value giving quantile q of x, where individuals/samples are weighted by w
    """
    CDF=np.cumsum(w)*1.0/sum(w)
    from scipy import interpolate
    if q is None: # Return a function!
        return(interpolate.interp1d(array(CDF),array(x)))
    else: # Return a value for quantile q
        return(interpolate.interp1d(array(CDF),array(x))(q))
    #return(interpolate.interp1d(np.linspace(min(zs),max(zs),num=100),array(cmapLU).T))


def sortDictsIntoQuantiles(dicts,sortkey,weightkey,nQuantiles=None,approxN=None):
    """
    Generate quantile groups for data with sample weights. The data should be in the (inefficient!) format of a list of dicts.
    CPBL, 4 August 2010

    Aug 2010: Also, add a ranking for each respondent into her dict. This seems to be a curious beast... I'm going to put the lowest at 0 and the highest at 1. Yet I'm going to distribute the rest according to weights....... hmm

dicts: list of dicts. Each dict contains one respondent (sample), with named variables and their values. Should all have identical format. Horribly inefficient data structure!  Could maybe offer an alternative, where these variables have been vectorized! Indeed, all I need is y and w, and then this could return a dict of indices rather than sorting... So the behaviour should be quite different depending on whether first argument is a list of dicts or a dict of named vectors (not yet done).

sortkey: the variable whose order is being ranked into quantiles

weightkey: sample weights. or the number 1, to signify no weights.

nQuantiles=None: Either nQuantiles or approxN or neither can be specified. This tells how many quantiles to calculate. They'll be evenly spaced from 0 to 1.

approxN=None:  nQuantiles can be calculated based on how many samples there are, and requiring that roughly approxN or more samples are going to be in each quantile (ie would be if weights were uniform).

In August 2010 I made another CDF-related function, to calculate non-parametric confidence intervals.

Feb 2012: Still haven't made alternate calling form. But more to the point, when y values are discrete, there are gaps in the quantile values assigned.
 So now, it creates a "filledrankY" as well as "rankY". The former has more continuous support; the latter gives the same rank to everyone in groups with the same y value.
 Oops. I should have used the "adding jitter to create unique ordering of points for rpy loess" method here, rather than what I have done. :(   ---> To Do: replace the method! [feb 2012] --> [ ]
    """
    assert approxN==None or nQuantiles==None

    finiteSubset=[dd for dd in dicts if isfinite(dd[sortkey])]
    finiteSubset.sort(key=lambda x: x[sortkey])
    y=array([respondent[sortkey] for respondent in finiteSubset])
    if weightkey==1:
        w=[1 for ii in y]
    else:
        w=array([respondent[weightkey] for respondent in finiteSubset])
    if approxN==None:
        approxN=20
    if nQuantiles==None:
        # Restrict number of quantiles to be between 2 and 25
        nQuantiles=max(2,min(25,floor(len(y)/approxN)))
        #nQuantiles=min(25,floor(len(y)/approxN))
    if 'verbose':
        print 'Using %d quantiles ...'%nQuantiles
    pQtl=(1.0+1.0*arange(nQuantiles))/nQuantiles
    assert len(pQtl)==nQuantiles
    assert all(isfinite(w))
    CDF=np.cumsum(w)*1.0/sum(w)
    pQtl=(1.0*arange(nQuantiles))/nQuantiles
    # Now assign respondents into the pQtl quantiles categories (ie dict:
    byQtl={}
    used=len(finiteSubset)
    for qq in pQtl[::-1]:
        iQ=find(CDF[:used]>=qq)
        byQtl[qq]=[finiteSubset[iir] for iir in iQ]
        used=min(iQ)
  
    if 'verbose':
        print 'Done the breaking into groups.... Now starting on assigning rankings. '
    
    # Also, assign to each respondent (sample) a ranking, ranging from 0 to 1, but taking into account weights (a bit weird?)
    """
 I guess the way to do that is actually the same as above, except use n_unique(y) as the number of bins?
Algorithm:
assign rank to each, from 0 to 1, ignoring that some y values are the same. Then average over those with common y values.

Okay, but ALSO provide the un-averaged version, in case I need a more continuous support over the quantiles range [0,1]. Call this the filledRank.

(1) Stretch the CDF to go from 0 to 1:
    """
    CDFscaled=CDF/(max(CDF)-min(CDF))
    CDFscaled=CDFscaled-min(CDFscaled)
    """
    (2) Now average rank over duplicates in y
    """
    from scipy import stats
    keyname='rank'+sortkey[0].upper()+sortkey[1:]
    for isample,asample in enumerate(finiteSubset):
        asample[keyname]=CDFscaled[isample]
        asample['filled'+keyname]=CDFscaled[isample]
    from dictTrees import dictTree
    byY=dictTree(finiteSubset,[sortkey])
    for kk in byY:
        sameY=byY[kk]
        if len(sameY)>1:
            for dd in sameY:
                # WHat!!!!!!!!! there is no weighted standard error of sample mean with sample weights in scipy???? Following needs weights!!!!11
                #debugprint( '       WHat!!!!!!!!! there is no weighted standard error of sample mean with sample weights in scipy???? Following needs weights!!!!11')
                dd[keyname]=np.mean([aresp[keyname] for aresp in sameY])  # Feb 2012: can use np.average.
                # NO! THIS WOULD NOT BE VERY INFORMATIVE OF ANYTHING... dd['se'+keyname]=stats.sem([aresp[keyname] for aresp in sameY])

    return(byQtl)

def scaledHist(rawData,bins=None,histtype=None,weights=None,color=None,label=None):
    """
    I find it hard scale a histogram so its max value is 1. So make a function to do this using a kludge.

    So normed arg not allowed in this call, unlike for hist()
    """
    if weights==None:
        weights=array([1.0 for xx in rawData])
    [a,b]=plt.histogram(rawData,bins=bins,weights=weights)
    weights=weights/max(a*1.0)
    return(plt.hist(rawData,bins=bins,histtype=histtype,weights=weights,color=color,label=label))

def colouredHistByVar(adf,pvar='nearbyDEADEND',cvar=None,bins=40,fig=None,clearfig=False,width=None,cvarRange=None,cmap='jet',Ncolors=256):#,saveas=None): 
    #pvardesc='Nearby fraction of dead-ends'
    """
    Works so far only on dataframes.
    show histogram of one variable.  But build it up by color based on another variable. e.g. colour oculd show year.
    For now, since colours show year, the build-it-up variable has discrete values only.
    N.B. Null values must be dropped before calling this. !

    You can specify bins to be an explicit list of breakpoints.
    dfn.cvar1[dfn.cvar1<1900]=1900

    You can specify explicit range for the color scale if you dont want it to simply span the data range.


    Returns handles to the main and colorbar axes.
    
    Issues: this overwrites 'cvar1' if it exists in the df.
    This function is really slow when cvar is continuous. Should discretize it!! [Fixed!]
    This code still refers to "years" etc: names should be generalized?

2014July: trying to add rolling mean option for smoothing!

    """
    if fig is not None:
        figure(fig)
    if clearfig:
        clf()
    dfn=adf
    if not len(dfn):
        print('Skipping a histogram with no data...')
        return([],[],None)
    if not any(pd.notnull(adf[pvar])):
        print('Skipping a histogram with no main data...')
        return([],[],None)
    if not any(pd.notnull(adf[cvar])):
        print('Skipping a histogram with no colour data...')
        return([],[],None)
    assert cvar is not None #Just use hist() otherwise!
    from cpblUtilitiesColor import assignSegmentedColormapEvenly
    colorsf= assignSegmentedColormapEvenly(cmap,dfn[cvar] if cvarRange is None else cvarRange ,asDict=False,missing=[1,1,1],Nlevels=Ncolors)
    # Using asDict=True is a nice way to discretize the data to 256 levels (spaced by quantile)
    colors= assignSegmentedColormapEvenly(cmap,dfn[cvar] if cvarRange is None else cvarRange ,asDict=True,missing=[1,1,1],Nlevels=Ncolors)
    # Convert to hex??
    #colors=dict([[a,mpl.colors.rgb2hex(b)] for a,b in  colors.items()])

    dfn['cvar1']=dfn[cvar]

    aa,bb=np.histogram(dfn[pvar],bins=bins)
    #allyears=sorted(dfn.cvar1.unique())
    if  len(dfn.cvar1.unique()) > 300:
        print('     discretizing data...'),
        # This is too many levels, ie too many histograms stacked on top of each other!! Let's discretize this variable.  Pandas has qcut() function to do this, now. (I can't just use the colour scheme, since it's continuous)
        dfn['color']=dfn.cvar1.map(lambda ff:colors[ff])#.map(mpl.colors.rgb2hex)
        # Discretize data to means within each colour:
        dfn['cvar2']=dfn.cvar1
        assert 'color' in dfn
        def fun(ff):
            ff['cvar1']=ff['cvar1'].dropna().mean()
            assert 'color' in ff
            return(ff)
        dfn3 = dfn.groupby('color').apply(fun)#.reset_index()#lambda x: x.mean())
        assert 'color' in dfn3
        dfn=dfn3
        print('     done')

        #allyears=sorted(dfn.cvar1.unique())
    else:
        dfn['color']=dfn.cvar1.map(colorsf)#.map(mpl.colors.rgb2hex)
    

    def customSmoothing(layerDepth,bins):
        Nbox=3
        a2_a=pd.stats.moments.rolling_mean(layerDepth[np.where(bins[1:]<1950)], Nbox)
        a2_b=np.array(layerDepth[np.where(bins[1:]>=1950)])
        layerDepth=np.concatenate([a2_a,a2_b])
        return(layerDepth[Nbox-1:],bins[Nbox-1:])
        
    sofar=0
    for yy,adf in dfn.groupby('cvar1'):#allyears:
        #a1,b1=np.histogram(dfn[dfn.cvar1==yy][pvar],bins=bb)
        a1,b1=np.histogram(adf[pvar],bins=bb)
        # June2014: I had to add the .values now, yet this worked before?!:
        #plt.hist(dfn[dfn.cvar1==yy][pvar].values,bins=bb,bottom=sofar,edgecolor='none',color=colors[yy])#,width=width)
        if 0 and customSmoothing is not None:
            a1,b1=customSmoothing(a1,b1)
        # Adam (2014, sprawl project) wanted to smooth layers over a subrange of the x axis. But then we have to plot ourselves, rather than using hist(). I'm abandoning this for now.
        # These are all the same color. so

        plt.hist(adf[pvar].values,bins=bb,bottom=sofar,edgecolor='none',color=adf.color.values[0])#,width=width)
        sofar+=a1
    ylabel('Number')# of streets (edges)')
    xlabel(pvar)
    ax=plt.gca()
    from cpblUtilitiesColor import addColorbarNonImage
    cbax=addColorbarNonImage(data2color=colors,ylabel=cvar.replace('_',' '))#'YEAR')   # min(allyears),max(allyears)
    plt.axes(ax)
    return(ax,cbax,colorsf)
    countyname=dfn.county.unique()[0] if len(dfn.county.unique())==1 else 'Entire USA'
    gca().text(.8*max(plt.xlim()),.8*max(plt.ylim()),countyname, ha='left', va="center", size=12,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9),horizontalalignment='right')
    #    gca().text(.328,-.13,r'$\pm$1 s.e. shown', ha='right', va="center", size=7,  bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))
    if saveas:
        plt.savefig(saveas+'.pdf')

            
def analyseErrorDistributionNonParametric(errorList,weights=None,twoSidedP=[.68,.95,.99],visualise=False):
    """
    Pretty specific at the momen; can be generalised. Returns a set of confidence intervals for two-sided ranges in an empirical disteibution.
    #####NOPE! of errors. ie it's specific to "errors" only in that I'm interested in deviations around zero.

    .eg. values for p=.9 igves the the 5th and 95th quantile estimates.
    
    """
    assert weights==None # No one has written a weighted quantile function?! Ah, use rpy hmisc's wtd.quantile. Or port it.
    errorList.sort()
    outset={}
    for pp in twoSidedP:
        outset[pp]=[quantile(errorList, (1.0-pp)/2.0,issorted=True),quantile(errorList, 0.5+pp/2.0,issorted=True)]

    if visualise:
        plt.close('all') # debug!!!!!
        figure(876)
        plotPDFandCDF(errorList,label='error/E',CDFlabel='hm',color='g',normColor='r',absCDF=True,weights=weights)
        subplot(211)
        for kk in outset:
            plot(outset[kk],[kk,kk],'rx')
    return(outset)


def plotPDFandCDF(yy,label=None,CDFlabel=None,color=None,normColor=None,absCDF=True,weights=None):
    """
    For now, this assumes zero-centred distribution of data, and plots PDF on top and CDF on bottom axes of current figure. It adds Gaussian curve fits by default, and generates labels but leaves the legend for someone else to call, since it can be used multiple times on the same axis.
    It ignores/hides long tails! How to do this? Hm, at moment by looking at only the middle 75% of the range? or by looking at only 3 sigma, whicever is smaller.
    It does not allow weights yet!!!!
    At the moment, the CDF, unlike the PDF, displays the distribution of absolute values! So this is not the integral of the PDF. This is appropriate for look at a distribution of errors, my first application.
    
    CPBL Aug 2010
    """
    from pylab import sqrt
    import scipy
    from scipy.stats import distributions
    assert absCDF==True and weights==None # Missing features!
    yy=array(sorted(finiteValues(yy)))
    # Find width of core of distribution:
    def noTailsStdDev(yyy):
        """
        just use the inner 2 sigma to fit a Gaussian
        Assume mean zero!

        Actually, also look at inner 75%ile, and if that's smaller, take it, since sigma must be driven by hug outliers.
        """
        tailsLim=plt.std(yyy)
        sigmaFit=plt.std(yyy[find(abs(yyy)<2.0*tailsLim)])

        # Check alternative method:
        n=len(yyy)
        if n>10:
            yyy.sort()
            sigmaFit=min(sigmaFit,plt.std(yyy[int(n*0.125):int(n*.875)]))
        return(sigmaFit)

    # Assume zero-centred!
    
    fitSig=noTailsStdDev(yy)
    xx=arange(-5*fitSig,5.0*fitSig,fitSig/500.0) # For plotting fits
    tailsLim=fitSig*5.0
    bins=arange(-tailsLim,tailsLim,tailsLim/sqrt(len(yy))) # for histogram
    absbins=arange(0,2.0*tailsLim,2.0*tailsLim/sqrt(len(yy))) # for CDF of abs(yy)

    
    # PDF 
    subplot(211)

    scaledHist(yy[find(abs(yy)<tailsLim)],bins=bins,histtype='step',label=label,color=color)
    fitSig=noTailsStdDev(yy)
    aGauss=distributions.norm.pdf(xx,scale=fitSig)
    plot(xx,aGauss/max(aGauss),color+':',label=r'fit ($\sigma=$%.02f)'%fitSig,hold=True)


    if not normColor==None:
        # A sigma=1 normal curve for comparison
        xxn=arange(-6,6,100)
        aGauss=distributions.norm.pdf(xxn)
        plt.axis('tight')
        plot(xxn,aGauss/max(aGauss),normColor,label=r'normal ($\sigma$=1)')

    ylabel('Relative frequency')
    setp(gca(),'yticklabels',[])
    #xlabel(r'$\varepsilon/\sigma$')

    subplot(212) # CDF of absolute values



    nyy=abs(yy)
    num_bins = len(xx)#sqrt(len(nyy))
    counts, bin_edges = np.histogram(nyy, bins=absbins, normed=True)
    cdf = np.cumsum(counts)  # cdf not normalized, despite above
    scale = 1.0/cdf[-1]
    ncdf = scale * cdf
    plot(bin_edges[1:], ncdf,color,label=CDFlabel,hold=True)

    if not normColor==None:
        xxn=arange(0,6,100)
        plot(xxn, (distributions.norm.cdf(xxn)-0.5)*2.0 ,normColor,label=r'normal ($\sigma$=1)')


    return()


def finiteValues(ay,by=None,cy=None,dy=None):
    """
Trivial shorthand: return an array of only the elements that are not NaN.
Various error checking missing...

CPBL August 2010

Sept 2010: If a second argument is given, then only elements of two vectors where both are finite are returned!

"""

    if by==None:
        return(array(ay)[find(isfinite(array(ay)))])
    if cy==None:
        ii=plt.logical_and(isfinite(array(ay)),isfinite(array(by)))
        return(array(ay)[ii],array(by)[ii])
    if dy==None:
        ii=plt.logical_and(    plt.logical_and(isfinite(array(ay)),isfinite(array(by))),   isfinite(array(cy)))
        return(array(ay)[ii],array(by)[ii], array(cy)[ii])
    foooodl




#%%%%%%%%%%%%%%%%% There are no weighted mean, weighted se of mean in numpy?!
# So, use following.
def wtvar(X, W, method = "R"):
    """
    is this weighted variance??? uhhh. no...
    """
    sumW = sum(W) 
    if X.size==0:
        return(nan)
    if method == "nist": 
        xbarwt = sum([w * x for w,x in zip(W, X)])/sumW # fixed.2009.03.07, divisor added. 
        Np = sum([ 1 if (w != 0) else 0 for w in W]) 
        D = sumW * (Np-1.0)/Np 
        return sum([w * (x - xbarwt)**2 for w,x in zip(W,X)])/D 
    else: # default is R 
        sumW2 = sum([w **2 for w in W]) 
        xbarwt = sum([(w * x) for (w,x) in zip(W, X)])/sumW 
        return sum([(w * (x - xbarwt)**2) for (w,x) in zip(W, X)])* sumW/(sumW**2 - sumW2)

#def wtsem(X, W, method = "R"): 
#    thevar=wtvar(X, W, method = "R")
def wtsem(a, w=None,axis=0, bootstrap=False): #
    """
    Returns the standard error of the mean, assuming normal distribution, for weighted data.
    """
    if w is None:
        w=np.ones(len(a))
    if not bootstrap:
        #a, axis, w = _chk_asarray(a, axis,w)
        a=np.asarray(a)
        w=np.asarray(w)
        #n = a.count(axis=axis)
        #s = a.std(axis=axis,ddof=0) / ma.sqrt(n-1)
        assert axis==0 # Hm, not done yet.
        s = np.sqrt(wtvar(a,w) / (a.size-1) )
        return s
    else:
        a=np.asarray(a)
        # Bootstrap version?? (not used, just for thought.. waste of space, then.)
        """
        Not clear how to deal with weights when boostrapping. weight sampling based on them? renomralise for each sample draw (done here..)
        """
        # 2013: replace following import * with specifics
        # from cpblUtilities import 
        w = w / w.sum()
    #Initialize:
        n = 10000
        ma = np.zeros(n)
        ssize=a.size
    #Save mean of each bootstrap sample:
        for i in range(n):
            idx = np.random.randint(0, ssize,ssize)
            sw=w[idx]/w[idx].sum()
            ma[i] = np.dot(a[idx], sw)

    #Estimate of Error in mean:
        #print ma.std(), wtsem(a,w)
        return(ma.std())

def sample_wr(population, k):# Not used yet!! 2011 Nov. But apparently very fast.
    """Chooses k random elements (with replacement) from a population
    For sampling without replacment, see random.sample
    """
    n = len(population)
    _random, _int = random.random, int  # speed hack
    return [_int(_random() * n) for i in itertools.repeat(None, k)]


def discreteSEbootstrap(popDistx=None,y=None,nTrials=100):
    """ Nov 2011: compare simple s.e. with bootstrap for a discrete distribution
    oops. Do I already have this, above?

    If no args are given, it will use demo and make plots
    """

    import random
    from pylab import array,find,arange,cumsum,zeros,hist,show,figure,mean,ones,std,sqrt,bar
    if popDistx is None:
        # Characterise survey sample
        popDist=[[1,234],[2,300],[3,800],[4,600]]
        x,y=zip(*popDist)
    elif y is None:
        x,y=zip(*popDist)
    else:
        x=popDistx

    cdf=array(list(y)+[sum(y)])*1.0/sum(y)
    cdf=cumsum(array(y)*1.0/sum(y))

    # For simplicity, create fake original sample (!)
    survey=[]
    for iix,xx in enumerate(x):
        survey+=list(ones(y[iix])*xx)

    print 'Survey mean and std and stderr:',mean(survey),std(survey),std(survey)/sqrt(len(survey)-1)
    print 'Alt: survey mean: ',wtsem(survey)
    print 'already-programmed bootstrap ',wtsem(survey,bootstrap=True)

    sampleSize=sum([pd[1] for pd in popDist])
    nTrials=1000
    means=[]
    for ii in range(nTrials):
        means.append(mean(sample_wr(survey,len(survey))))
        if 0: #alternative method:
            choice=random.choice 
            asample = [choice(survey) for _ in xrange(len(survey))]
            means.append(mean(asample))

    if popDistx is None:
        figure(23)
        (hn,hb,hp)=hist(means,normed=True)
        bar(x,array(y)*1.0/sum(y)*max(hn),color='r',alpha=.5)
        show()
        
    print 'Bootstrap mean and std of these means:',mean(means),std(means)



# And numpy's average() cannot deal with empty sets!!
def wtmean(a, axis=None, weights=None, returned=False):
    """
    This is numpy.average, augmented to be able to take emtpy sets.
    Return the weighted average of array over the specified axis.

    Parameters
    ----------
    a : array_like
        Data to be averaged.
    axis : int, optional
        Axis along which to average `a`. If `None`, averaging is done over the
        entire array irrespective of its shape.
    weights : array_like, optional
        The importance that each datum has in the computation of the average.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If `weights=None`, then all data in `a` are assumed to have a
        weight equal to one.
    returned : bool, optional
        Default is `False`. If `True`, the tuple (`average`, `sum_of_weights`)
        is returned, otherwise only the average is returned.  Note that
        if `weights=None`, `sum_of_weights` is equivalent to the number of
        elements over which the average is taken.

    Returns
    -------
    average, [sum_of_weights] : {array_type, double}
        Return the average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `Float`
        if `a` is of integer type, otherwise it is of the same type as `a`.
        `sum_of_weights` is of the same type as `average`.

    Raises
    ------
    ZeroDivisionError
        When all weights along axis are zero. See `numpy.ma.average` for a
        version robust to this type of error.
    TypeError
        When the length of 1D `weights` is not the same as the shape of `a`
        along axis.

    See Also
    --------
    ma.average : average for masked arrays

    Examples
    --------
    >>> data = range(1,5)
    >>> data
    [1, 2, 3, 4]
    >>> np.average(data)
    2.5
    >>> np.average(range(1,11), weights=range(10,0,-1))
    4.0

    """
    if not isinstance(a, np.matrix) :
        a = np.asarray(a)

    if a.size==0: # Added by CPBL, Sep2010
        return(nan)
    elif weights is None :
        avg = a.mean(axis)
        scl = avg.dtype.type(a.size/avg.size)
    else :
        a = a + 0.0
        wgt = np.array(weights, dtype=a.dtype, copy=0)

        # Sanity checks
        if a.shape != wgt.shape :
            if axis is None :
                raise TypeError, "Axis must be specified when shapes of a and weights differ."
            if wgt.ndim != 1 :
                raise TypeError, "1D weights expected when shapes of a and weights differ."
            if wgt.shape[0] != a.shape[axis] :
                raise ValueError, "Length of weights not compatible with specified axis."

            # setup wgt to broadcast along axis
            wgt = np.array(wgt, copy=0, ndmin=a.ndim).swapaxes(-1,axis)

        scl = wgt.sum(axis=axis)
        if (scl == 0.0).any():
            raise ZeroDivisionError, "Weights sum to zero, can't be normalized"

        avg = np.multiply(a,wgt).sum(axis)/scl

    if returned:
        scl = np.multiply(avg,0) + scl
        return avg, scl
    else:
        return avg


def LambertW(z):
  import numpy
  from math import log,sqrt,exp

  """debugprint( "Should in new version get: scipy.special.lambertw")"""
  " Lambert W function, principal branch "
  if z.__class__ in [numpy.ndarray]:
      return(array([LambertW(zz) for zz in z]))
  if z.__class__ in [list]:
      return([LambertW(zz) for zz in z])
  eps=4.0e-16
  em1=0.3678794411714423215955237701614608
  if z<-em1:
    print >>stderr,'LambertW.py: bad argument %g, exiting.'%z
    exit(1)
  if 0.0==z: return 0.0
  if z<-em1+1e-4:
    q=z+em1
    r=sqrt(q)
    q2=q*q
    q3=q2*q
    return\
     -1.0\
     +2.331643981597124203363536062168*r\
     -1.812187885639363490240191647568*q\
     +1.936631114492359755363277457668*r*q\
     -2.353551201881614516821543561516*q2\
     +3.066858901050631912893148922704*r*q2\
     -4.175335600258177138854984177460*q3\
     +5.858023729874774148815053846119*r*q3\
     -8.401032217523977370984161688514*q3*q
  if z<1.0:
    p=sqrt(2.0*(2.7182818284590452353602874713526625*z+1.0))
    w=-1.0+p*(1.0+p*(-0.333333333333333333333+p*0.152777777777777777777777))
  else:
    w=log(z)
  if z>3.0: w-=log(w)
  for i in xrange(10):
    e=exp(w)
    t=w*e-z
    p=w+1.0
    t/=e*p-0.5*(p+1.0)*t/p
    w-=t
    if abs(t)<eps*(1.0+abs(w)): return w


# """ file variance.py tranlator Dr. Ernesto P. Adorio UP at Clark Field. License GNu Free documentation license(wikipedia license) """   def svar(X, method = 0): """ Computes the sample variance using various methods explained in the reference.   method: 0 - basic two-pass, the default. 1 - single pass 2 - compenstated 3 - welford Reference: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance """ if not(0 < method < 3): method = 0 n = len(X) if method == 0: # basic two-pass. xbar = sum(X) / float(n) return sum([(x - xbar)**2 for x in X])/(n -1.0)   elif method == 1: # single pass. S = 0.0 SS = 0.0 for x in X: S += x SS += x*x xbar = S/float(n) return (SS -n * xbar * xbar) / (n -1.0)   elif method == 2: # compensated algorithm. xbar = mean(X)   sum2 = 0 sumc = 0 for x in X: sum2 = sum2 + (x - mean)^2 sumc = sumc + (x - xbar) return (sum2 - sumc^2/n)/(n - 1)   elif method == 3: # Welford algorithm n = 0 mean = 0 M2 = 0   for x in X: n = n + 1 delta = x - mean mean = mean + delta/n M2 = M2 + delta*(x - mean) # This expression uses the new value return (M2/(n - 1.0))   def weightedsvar(X, W, method = 3): # Computes the weighted variance using weight vector W. # Welford algorithm. n = 0 for x, weight in zip(X,W): if n==0: n = 1 mean = x S = 0 sumweight = weight else: n = n + 1 temp = weight + sumweight S = S + sumweight*weight*(x-mean)^2 / temp mean = mean + (x-mean)*weight / temp sumweight = temp return S * n / ((n-1) * sumweight)   def pvar(X, method = 0): """ Computes the population variance. """ n = len(X) return svar(X, method)*(n-1.0)/ n   def weightedpvar(X, W, method = 3): # Wait a while for the other methods. n = len(X) return weightedsvar(X, W) * (n-1.0)/n   def sdev(X, method = 0): return sqrt(svar(X, method))   def pdev(X, method = 0): return sqrt(pvar(X, method)   # R names for the sample variances and std. deviations. var = svar sd = sdev   Sxx = svar(X) Sxx = svar Sx = sd   def cov(X,Y): # covariance of X and Y vectors with same length. n = len(X) xbar = sum(X) / n ybar = sum(Y) / n return sum([(x-xbar)*(y-ybar) for x,y in zip(X,Y)])/(n-1)   Sxy = cov Sx = sdev   def cor(X,Y): # correlation coefficient of X and Y. return(cov(X,Y)/(Sx(X)*Sx(Y))
# 
# Read more: http://adorio-research.org/wordpress/?p=242#ixzz0z5CKmn1K

def rpy_semipar(XX,y,nonparvars=None):
    """
    NOT WRITTEN YET.
    
    XX is a dict of observations with named-variable vectors!
    y is dependent variable
    nonparvars is a list (or string) with names of nonparametric variables

    library(SemiPar)

        robjects.globalenv["x"]=robjects.FloatVector(x)
    robjects.globalenv["y"]=robjects.FloatVector(y)
    if w is not None:
        robjects.globalenv["w"]=robjects.FloatVector(y)

    """
    fit=robjects.r("""

# CPBL  Jan 2011: agh. I'll use this if it works, but:    The is not a robust procedure. For robust quantile smoothing look at the cobs package on CRAN. Also look at Roger Koencker's package, quantreg, for more quantile smoothing.
    
# This code relies on the rollapply function from the "zoo" package.  My thanks goes to Achim Zeileis and Gabor Grothendieck for their work on the package.
Quantile.loess<- function(Y, X = NULL,
							number.of.splits = NULL,
							window.size = 20,
							percent.of.overlap.between.two.windows = NULL,
							the.distance.between.each.window = NULL,
							the.quant = .95,
							window.alignment = c("center"),
							window.function = function(x) {quantile(x, the.quant)},
							# If you wish to use this with a running average instead of a running quantile, you could simply use:
							# window.function = mean,
							...)
{
	# input: Y and X, and smothing parameters
	# output: new y and x
 
	# Extra parameter "..." goes to the loess
 
	# window.size ==  the number of observation in the window (not the window length!)
 
	# "number.of.splits" will override "window.size"
	# let's compute the window.size:
	if(!is.null(number.of.splits)) {window.size <- ceiling(length(Y)/number.of.splits)}
 
	# If the.distance.between.each.window is not specified, let's make the distances fully distinct
	if(is.null(the.distance.between.each.window)) {the.distance.between.each.window <- window.size}
 
	# If percent.of.overlap.between.windows is not null, it will override the.distance.between.each.window
	if(!is.null(percent.of.overlap.between.two.windows))
		{
			the.distance.between.each.window <- window.size * (1-percent.of.overlap.between.two.windows)
		}
 
 
 
	# loading zoo
	if(!require(zoo))
	{
		print("zoo is not installed - please install it.")
		install.packages("zoo")
	}
 
 
	if(is.null(X)) {X <- index(Y)} # if we don't have any X, then Y must be ordered, in which case, we can use the indexes of Y as X.
 
	# creating our new X and Y
	zoo.Y <- zoo(x = Y, order.by = X)
	#zoo.X <- attributes(zoo.Y)$index
 
	new.Y <- rollapply(zoo.Y, width = window.size,
								FUN = window.function,
								by = the.distance.between.each.window,
								align = window.alignment)
	new.X <- attributes(new.Y)$index
	new.Y.loess <- loess(new.Y~new.X, family = "sym",...)$fitted
 
	return(list(y = new.Y, x = new.X, y.loess = new.Y.loess))
}










    
    #require(graphics)
    #x <- c(%s)
    # y <- c(%s)
     lo <- loess(y~x,span=%f)# I haven't figured out how to skip the second argument yet. Or I can create a data frame. then add: ,weights)

     #plot(x,y)
    """%(','.join([str(xx) for xx in x]),','.join([str(xx) for xx in y]), kernelwidth) +"""
#     lines(predict(lo), col='red', lwd=2)
# Create a 99-length quantile vector:
#xl <- seq(0.01,.99,.01)
# Or a 100-length vector from x: 

xl <- seq(min(x),max(x), (max(x) - min(x))/%d)"""%(nPts-1)+"""

# Plot result, with data:
#lines(xl, predict(lo,xl), col='red', lwd=2)

fit <- predict(lo,xl)

"""+'\n'.join(["""q%02d <- Quantile.loess(y,x,the.quant=%.2f)
"""%(qq*100.0,qq) for qq in qBands])+"""
""")
#q75 <- Quantile.loess(y,x,the.quant=.75)
#q25 <- Quantile.loess(y,x,the.quant=.25)
#""")


def rpyloessDF(df,xname,yname):
    """
    Not finished?!
    """
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate() # Some crazy trick to do magical conversions.
    # rpy2 only works with a structured or record array, not a pandas dataframe
    if 1:
        rd=df[[xname,yname]].to_records()
        ##links2=links[['FOURWAYPLUS','nearbyDEADEND','nearbyFOURWAY','nearbyMEAN_LEGS','realpriceBTU',]+('clusterID' in links)*['clusterID']].to_records()
    if 0: # Demo from online fails, march 2013.
        import rpy2.robjects.lib.ggplot2 as ggplot2
        #import rpy2.robjects as ro
        from rpy2.robjects.packages import importr
        base = importr('base')

        datasets = importr('datasets')
        mtcars = datasets.mtcars
        gp = ggplot2.ggplot(mtcars)

        pp = gp + \
         ggplot2.aes_string(x='wt', y='mpg') + \
         ggplot2.geom_point() + \
         ggplot2.stat_smooth(method = 'loess')
        pp.plot()

    fit=ro.r.loess('%s ~ %s'%(xname,yname),data=rd)
    fit_summ=ro.r.summary(fit)
    #the two delegators rx and rx2, representing the R functions [ and [[ respectively.
    print fit_summ.rx2('coefficients')   # coefficients are same as from Stata
    # Need to use sandwich package to get robust s.e.
    ro.r.library("sandwich")
    print ('HC1 ERRORS:')
    print ro.r.vcovHC(fit,type="HC1")
    if 'clusterID' in links:
        print ('ARELLANO ERRORS (ie clustered):')
        print ro.r.vcovHC(fit,type="HC1",cluster='clusterID',method='arellano')
    # s.e. are the sqrt of the diagonals in the V-COV matrix:
    print sqrt(diag(ro.r.vcovHC(fit,type="HC1"))) # Identical to Stata.
    print sqrt(diag(ro.r.NeweyWest(fit)))
    # Now, how to get the coefs and s.e.s (and p-values) back to python in a nice way?



    fitxts=ro.r.lm('FOURWAYPLUS ~ nearbyDEADEND*realpriceBTU+ nearbyFOURWAY*realpriceBTU+ nearbyMEAN_LEGS*realpriceBTU',data=links2)
    fitxts_summ=ro.r.summary(fitxts)
    print fitxts_summ.rx2('coefficients')   # coefficients are same as from Stata
    # Need to use sandwich package to get robust s.e.
    ro.r.library("sandwich")
    print ro.r.vcovHC(fitxts,type="HC1")
    # s.e. are the sqrt of the diagonals in the V-COV matrix:
    print sqrt(diag(ro.r.vcovHC(fitxts,type="HC1"))) # Identical to Stata.
    print sqrt(diag(ro.r.NeweyWest(fitxts)))
    # Now, how to get the coefs and s.e.s back to python in a nice way?

    # Reproduce pca in Stata
    links2=links[['FOURWAYPLUS','DEADEND','MEAN_LEGS']].to_records()
    links2=links2[['FOURWAYPLUS','DEADEND','MEAN_LEGS']]  # remove TLID
    pca=ro.r.princomp(links2,cor=True)
    print pca.rx2('loadings')   #looks good - same as Stata

def pdlowess(df,xv,yv):
    """
    Returns x,y arrays of the lowess fit. xv and yv are strings.
    June 2013: first draft.
    June 2013: SHould automatically decimate data if something large gets passed (I just got a Memory Error on a 0.4TB RAM machine!)
    """
    if len(df)>1000:
        import random
        rows = random.sample(df.index, 1000)
        df = df.ix[rows]

    import statsmodels.api as sm
    lowess = sm.nonparametric.lowess
    dfnn=df.dropna(subset=[xv,yv])
    print('Lowess...')
    z=lowess(dfnn[yv],dfnn[xv])
    x,y=zip(*z)
    return(np.array(x),np.array(y))

def rpy_loess(x,y,w=None,nPts=None,plotTitle=None,kernelwidth=0.75,qBands=None,OLDFORMAT=False,decimateTo=None):
    """
    Jan 2011.
    Return loess smoothed values for y(x).
    Return 25th and 75th percentile values (by default), smoothed across x.

    Return loess smoothed standard deviation...? Not sure how to convert this to a standard error.  If I did a boxcar average and the x-points are evenly distributed I might be able to calculate standard error.

    It returns version of the loess for both the original points and 

    2011 April: Added qBands (to generalise from 0.25, .75) and OLDFORMAT. Latter is since I now return just one xvector...

    May 2011: it has started seg faulting a bunch. argh!

It also makes a plot??

UQ: Who told me non-parametric is the way of the future: it was at the reception at some museumish thing with a hummer artwork.
UQ: Ask him how to get loess with s.e.?  [See lpoly!]


decimateTo: if the number of points is greater than this, just resample the data to end up with this many points!  This makes loess more tolerably fast.

2013 March: "RRuntimeError: Error in simpleLoess(y, x, w, span, degree, parametric, drop.square, normalize,  :   NA/NaN/Inf in foreign function call (arg 1)".  StackExchange suggests I need to use family="gaussian" somewhere?  Wrap everything in error-trapping?

    2013 June: There's a loess in statsmodels now; see my new pdlowess. Not sure which to go with...
    """
    import time
    if qBands is None:
        qBands=[0.25,0.75]
    print ' '+time.asctime()+ ' Initiating loess through R (mean,'+','.join(['%02d%%ile'%(100.0*qq) for qq in qBands])+')...'


#    import rpy2.robjects as robjects
    if nPts is None:
        nPts=100


    if not len(x):
        print 'No x data, but made it to LOESS anyway. Whoops! Returning dummies'
        if OLDFORMAT: # Retired April 2011, ie for gallup compatibility only.
            return(
            dict(xfit=[],yfit=[],
                 q25=dict(x=[],y=[],fit=[]),
                 q75=dict(x=[],y=[],fit=[])
                   )
                )
        else:
            outDict=dict(xfit=[],yfit=[],qFits={})
            if qBands:
                outDict['xq']=[]
            for qq in qBands:
                outDict['y%02d'%(qq*100)]=[]
            return(outDict)

    from cpblUtilities import uniqueInOrder,finiteValues
    if not  len(x) == len(uniqueInOrder(x)): # zoo package in R fails otherwise?
        print '    Adding jitter to create unique ordering of points for rpy loess!!!', plotTitle
        import random
        x=list(array(x)+array([random.random()*1.0e-9 for ii in range(len(x))]))

    # LOESS is slow. Decimate data down to an easy size:
    if not decimateTo is None and len(x)>decimateTo:
        from random import sample
        print 'Decimating data for LOESS to only %d  out of %d points!'%(decimateTo,len(x))
        ii=sample(range(len(x)),decimateTo)
        x,y,w=list(array(x)[ii]),list(array(y)[ii]),list(array(w)[ii])
        from cpblUtilities import finiteValues
    # Hm... I need to avoid NaNs for R:  [caution: adding this 2012 feb]
    if w is not None and len(w)==len(x):
        x,y,w=finiteValues(x,y,w)
    else:
        assert w is None
        x,y=finiteValues(x,y)


    #robjects.r(
    '''
       f <- function(r, verbose=FALSE) {
                if (verbose) {
                    cat("I am calling f().\n")
                }
                2 * pi * r
            }
            f(3)
    '''
    #r_f = robjects.globalenv['f']

    #x,y=[1,2,3,4,5],[4,5,6,7,4]
    robjects.globalenv["x"]=robjects.FloatVector(x)
    robjects.globalenv["y"]=robjects.FloatVector(y)
    if w is not None:
        robjects.globalenv["w"]=robjects.FloatVector(y)
    print 'Calling R...'

    fit=robjects.r("""

# CPBL  Jan 2011: agh. I'll use this if it works, but:    The is not a robust procedure. For robust quantile smoothing look at the cobs package on CRAN. Also look at Roger Koencker's package, quantreg, for more quantile smoothing.
    
# This code relies on the rollapply function from the "zoo" package.  My thanks goes to Achim Zeileis and Gabor Grothendieck for their work on the package.
Quantile.loess<- function(Y, X = NULL,
							number.of.splits = NULL,
							window.size = 20,
							percent.of.overlap.between.two.windows = NULL,
							the.distance.between.each.window = NULL,
							the.quant = .95,
							window.alignment = c("center"),
							window.function = function(x) {quantile(x, the.quant)},
                                                        # na.rm=TRUE, # Do i want this?
							# If you wish to use this with a running average instead of a running quantile, you could simply use:
							# window.function = mean,
							...)
{
	# input: Y and X, and smothing parameters
	# output: new y and x
 
	# Extra parameter "..." goes to the loess
 
	# window.size ==  the number of observation in the window (not the window length!)
 
	# "number.of.splits" will override "window.size"
	# let's compute the window.size:
	if(!is.null(number.of.splits)) {window.size <- ceiling(length(Y)/number.of.splits)}
 
	# If the.distance.between.each.window is not specified, let's make the distances fully distinct
	if(is.null(the.distance.between.each.window)) {the.distance.between.each.window <- window.size}
 
	# If percent.of.overlap.between.windows is not null, it will override the.distance.between.each.window
	if(!is.null(percent.of.overlap.between.two.windows))
		{
			the.distance.between.each.window <- window.size * (1-percent.of.overlap.between.two.windows)
		}
 
 
 
	# loading zoo
	if(!require(zoo))
	{
		print("zoo is not installed - please install it.")
		install.packages("zoo")
	}
 
 
	if(is.null(X)) {X <- index(Y)} # if we don't have any X, then Y must be ordered, in which case, we can use the indexes of Y as X.
 
	# creating our new X and Y
	zoo.Y <- zoo(x = Y, order.by = X)
	#zoo.X <- attributes(zoo.Y)$index
 
	new.Y <- rollapply(zoo.Y, width = window.size,
								FUN = window.function,
								by = the.distance.between.each.window,
								align = window.alignment)
	new.X <- attributes(new.Y)$index
	new.Y.loess <- loess(new.Y~new.X, family = "sym",...)$fitted
 
	return(list(y = new.Y, x = new.X, y.loess = new.Y.loess))
}

    #require(graphics)
    #x <- c(%s)
    # y <- c(%s)
     lo <- loess(y~x,span=%f)# I haven't figured out how to skip the second argument yet. Or I can create a data frame. then add: ,weights)

     #plot(x,y)
    """%(','.join([str(xx) for xx in x]),','.join([str(xx) for xx in y]), kernelwidth) +"""
#     lines(predict(lo), col='red', lwd=2)
# Create a 99-length quantile vector:
#xl <- seq(0.01,.99,.01)
# Or a 100-length vector from x: 

xl <- seq(min(x),max(x), (max(x) - min(x))/%d)"""%(nPts-1)+"""

# Plot result, with data:
#lines(xl, predict(lo,xl), col='red', lwd=2)

fit <- predict(lo,xl)

"""+'\n'.join(["""q%02d <- Quantile.loess(y,x,the.quant=%.2f)
"""%(qq*100.0,qq) for qq in qBands])+"""
""")
#q75 <- Quantile.loess(y,x,the.quant=.75)
#q25 <- Quantile.loess(y,x,the.quant=.25)
#""")



    fit1=array(robjects.globalenv["fit"])
    xl=array(robjects.globalenv["xl"])
    #q25=array(robjects.globalenv["q25"])

    # It seems q25,q75 (or whatever qBands were calculated) contain various things.
    qFits={}
    for qq in qBands:
        qname='q%02d'%(qq*100.0)
        qx,qy,qyfit=list(robjects.globalenv[qname].rx2('x')),list(robjects.globalenv[qname].rx2('y')),list(robjects.globalenv[qname].rx2('y.loess'))
        qFits[qq]={'x':qx,'y':qy,'yfit':qyfit}
        assert qx==qFits[qBands[0]]['x']
        #q25x,q25y,q25fit=list(robjects.globalenv['q25'].rx2('x')),list(robjects.globalenv['q25'].rx2('y')),list(robjects.globalenv['q25'].rx2('y.loess'))
        #q75x,q75y,q75fit=list(robjects.globalenv['q75'].rx2('x')),list(robjects.globalenv['q75'].rx2('y')),list(robjects.globalenv['q75'].rx2('y.loess'))
    #assert q25x==q75x

    #figure(22)
    #plot(q25x,q25y,q75x,q75y,q25x,q25fit,q75x,q75fit)
    print "Sorry -- Plotting seems not updated for new variables dict. Skipping it"
    if 0 and plotTitle and .25 in qBands and .75 in qBands: # Since I haven't generalised to other qBands yet
        figure(23)
        clf()
        #
        if 0:
            plot(x,y,'r.')
        plot(q25x,q25fit,'m',q75x,q75fit,'m')
        if 0:
            plot(q25x,q25y,'g',q75x,q75y,'g')

        # Add envelope:
        #@xs, ys = plt.fill_between(q25x, q25fit,q75fit)
        envelopePatch=plt.fill_between(q25x, q25fit,q75fit,facecolor='green',alpha=.5,)#,linewidth=0,label=patchLabel)# edgecolor=None, does not work!! So use line
        plot(xl,fit1,'b--',label='loess fit')

        # Show whatever the other thing that Quantile.loess is returning:
        #plot(q25x,q25y,q75x,q75y,q25x,q25fit,q75x,q75fit)
        title(plotTitle)
        plt.show()

    if OLDFORMAT: # Retired April 2011, ie for gallup compatibility only.
        return(
        dict(xfit=xl,yfit=fit1,
             q25=dict(x=q25x,y=q25y,fit=q25fit),
             q75=dict(x=q75x,y=q75y,fit=q75fit)
               )
            )
    else:
        outDict=dict(xfit=xl,yfit=fit1,qFits=qFits)
        if qBands:
            outDict['xq']=qFits[qBands[0]]['x']
        for qq in qBands:
            outDict['y%02d'%(qq*100)]=qFits[qq]['yfit']
        outDict['decimateTo']=decimateTo
        return(outDict)

    """
 print(lm_D9.rclass)
[1] "lm"
Here the resulting object is a list structure, as either inspecting the data structure or reading the R man pages for lm would tell us. Checking its element names is then trivial:

>>> print(lm_D9.names)
 [1] "coefficients"  "residuals"     "effects"       "rank"
 [5] "fitted.values" "assign"        "qr"            "df.residual"
 [9] "contrasts"     "xlevels"       "call"          "terms"
[13] "model"
And so is extracting a particular element:

>>> print(lm_D9.rx2('coefficients'))
(Intercept)    groupTrt
      5.032      -0.371
or

>>> print(lm_D9.rx('coefficients'))

"""

    robjects.r('''
       f <- function(x,y, verbose=FALSE) {
                if (verbose) {
                    cat("I am calling f().\n")
                }
                2 * pi * r
            }
            f(3)
    ''')


    """# aha!! A quanitle loess will give me what I want for sigmas!! just show an envelope of interquartile range!!
 # fitting the Quantile LOESS
source("http://www.r-statistics.com/wp-content/uploads/2010/04/Quantile.loess_.r.txt")
QL <- Quantile.loess(Y = Ozone.2, X = Temp.2, 
							the.quant = .95,
							window.size = 10,
							window.alignment = c("center"))
points(QL$y.loess ~ QL$x, type = "l", col = "green")
 
legend("topleft",legend = c("95% Quantile regression", "95% Quantile LOESS"), fill = c("red","green"))

"""



    
    

    
def OBSELETE_assignColormapEvenly(cmap,zs,asDict=False,missing=[1,1,1]):
    from cpblUtilitiesColor import assignSegmentedColormapEvenly
    assert missing== [1,1,1]  #Ignored!!!!!!!
    if cmap is None:
        cmap='jet'
    print('   assignColormapEvenly is now replaced by assignSegmentedColormapEvenly, and is deprecated. ')
    return(assignSegmentedColormapEvenly(cmap,zs,splitdataat=None,asDict=asDict,missing=missing))
    """

Nice: Use a colormap, supplied rather explicitly (or, now, by its name), to evenly cover the given range of data. In default mode, it  returns a function, using interp. 
In asDict mode, returns a lookup table of values to RGB colour list. 

Rather than linearly map a colormap to the min and max data values, this assigns a colormap just based on ordinal position in the data, ie it spreads the colours out nicely across the data range.




(Is "floor" correct?)

If you're instead wanting to colour a set of categories (possibly ordered), use instead getIndexedColormap()? Hm, May 2012. maybe not.. Indexes to your categories might work better for this one, since it returns a lookup. What aabout if z is a list of strings, do categorical? Done!! If list of strings is passed, asDict is true, ...! 

See also (very nice!) def plot_biLinearColorscale(rgbBotMidTop,datavals) in cpblUtilitiesMath

Also, matplotlib.cm has lots of utility functions that I'm not using.

What? What does evenly mean?? Can I just look at the end points and spread the values out linearly? Or was this supposed to spread them out evenly across the colours? Ahh... I think the latter. So I should revert htis function to a much older version and make a new linear one.

July 2014 N.B. If you're starting from your own set of colours, rather than a built-in colormap, then you can use the new assignSegmentedColormapEvenly function.

July 2014: That new function now does everything, except for dealing properly with "missing", so this one is deprecated.

    """
    import pylab
    categorical=False
    if isinstance(zs,list) and isinstance(zs[0],str):
        iCategories=range(len(zs))
        szs=range(len(zs))
        asDict=True
        categorical=True
        print 'Initiating categorical mode...'
    else:
        #szs=deepcopy([zz for zz in zs if not isnan(zz)])
        #szs.sort()
        # Find length of non-nan, unique values.
        #from cpblUtilities import uniqueInOrder#,finiteValues
        if isinstance(zs,pd.Series): zs=zs.values
        szs=np.sort(np.unique(zs[np.isfinite(zs)]))  # for zz in zs if not isnan(zz)])))  #[szs[0]- dz]  + szs  + [szs[1]+dz])
        dz=szs[-1]-szs[0] 

    #assert asDict
    # Ie (above): revert this to old version, or retire it...
    
    if isinstance(cmap,str): # cmap can be pass as a string or a cmap
        cmap=getIndexedColormap(cmap,len(szs)) ##plt.get_cmap(cmap)
    if cmap is None:
        cmap=getIndexedColormap('jet',len(szs))
    
        #cmap = plt.cm.jet # use 'hot' colormap
        if 0:
            fooo
            cmap=tonumeric([LL.strip().split('\t') for LL in open('mcool.tsv','rt').readlines()])[::16]
            from pylab import jet
            cmap=jet()
            #colormap(jet)

    #assert cmap
    
    #ccs=[cmap[int(floor(iz*len(cmap)*1.0/len(szs)))] for iz in range(len(szs))]
    lencmap=len(cmap)#cmap.N
    ccs=[cmap[int(floor(iz*lencmap*1.0/len(szs)))] for iz in range(len(szs))]
    if categorical:
        return(dict(zip(zs+[None],ccs+[missing])))
    if asDict:#deprecatd:
        return(dict(zip(szs.tolist()+[pylab.nan],ccs+[missing])))
    else: # Return a lookup function!!
        from scipy import interpolate
        ccs2=np.concatenate((array(ccs[:1]),array(ccs),array(ccs[:1])), axis=0)
        szs2= np.concatenate((szs[:1]-dz, szs, szs[-1:]+dz))
        return(interpolate.interp1d(szs2.T,ccs2.T)) # A lookup function for z values!
        #return(interpolate.interp1d(array(szs).T,array(ccs).T)) # A lookup function for z values!


            
def biLinearColorscale(rgbBotMidTop,datavals,splitdataat=None,noplot=False, mapname='TwoLinear'):
    """
    rgbBotMidTop: Specify the bottom, middle, and top RGB values of a colormap. 

    datavals: You can pass your entire set of datavals (must be more than 3 long), if you want a data-to-color lookup. In this case, you need also to specify splitdataat.  
    datavals: Alternatively, just give the bottom and top data val, or the bottom, middle, and top data vals.

    splitdataat: some middle data value at which the color break occurs the linear extremes of the data.  Cannot be specified if datavals has length three, since in this case the middle value is used.   If not specified, and len(datavals) is not 3, then the mean of the max and min of datavals is used.

    noplot: If this is false, function produces a colorbar with the given map and tickmarks for the given data extremes, mapped linearly to the colormap. 

2013 Nov: Return value changed. It's now always a 3-tuple, with None's for n/a values. cmap, data-to-color lookup, figure.
    
#####It returns a tuple consisting of the figure, the colormap (plt.cm.get_cmap(mapname)) and  a data-to-color lookup (pandas Series). 
###If datavals is only two or three in length, the data-to-color lookup is abset.###  If data are no, in which case it returns  
    
Also, matplotlib.cm has lots of utility functions that I'm not using...

Obselete:#####2013oct: If you just want the colormap (e.g. to pass to colorize_svg), use noplot=True and you'll get back the cmap.  [!? cmap is not good enough... I want the data mapping.

mapname: this is for defining a name for the result, so that pylab knows about that color scheme. It's not for making use of an existing one.
    """
    if not rgbBotMidTop: # Offer a default example: green red
        rgbBotMidTop=[[0.5,0.0,0.0],[1.0,1.0,.9],[0.0,0.5,0.0]]
    rgbB,rgbM,rgbT=rgbBotMidTop #=[[rgb1],[],[]]
    """
    cdict = {'red':((0.0,  .5, .5),
               (0.5,  1.0, 1.0),
               (1.0,  0.0, 0.0)),

     'green': ((0.0,  0.0, 0.0),
               (0.5, 1.0, 1.0),
               (1.0,  0.5, 0.5)),

     'blue':  ((0.0,  0.0, 0.0),
               (0.5,  .9, .9),
               (1.0,  0.0, 0.0))}
      """
    cdict = {'red':((0.0,  rgbB[0],rgbB[0]),
               (0.5,  rgbM[0],rgbM[0]),
               (1.0,  rgbT[0],rgbT[0])),

     'green': ((0.0,  rgbB[1],rgbB[1]),
               (0.5,  rgbM[1],rgbM[1]),
               (1.0,  rgbT[1],rgbT[1])),

     'blue': ((0.0,  rgbB[2],rgbB[2]),
               (0.5,  rgbM[2],rgbM[2]),
               (1.0,  rgbT[2],rgbT[2]))         }
    #import numpy as np
    #import pylab as plt
    #from pylab import *
    #import matplotlib as mpl
    #import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    greenred= LinearSegmentedColormap(mapname, cdict)

    plt.register_cmap(name=mapname, data=cdict) # optional lut kwarg

    # Now look at the data vals.
    minD,maxD=min(datavals),max(datavals)
    midD=mean([minD,maxD])
    if len(datavals)==3:
        midD=datavals[1]
        assert splitdataat is None
    if splitdataat is not None:
        midD=splitdataat
    
    if noplot and len(datavals)==2:
        return(greenred,None,None)
    if noplot: # We were passed an array of the actual data
        # Return two things: the colormap (though it can be retrieved by name as well) and the lookup.
        return(greenred,  sci.interpolate.interp1d(        [minD,midD,maxD],        [array(rgbB),array(rgbM),array(rgbT)], axis=0) ,None)
        #before minD, I had: 1.0/2.0*(minD+maxD)        


    fig=plt.gcf()#plt.figure(1)
    plt.clf()
    #cnorm = mpl.colors.Normalize(vmin=min(datavals),vmax=max(datavals))
    #####minD,maxD=min(datavals),max(datavals)
    hi=plt.imshow([[minD,maxD],[maxD,minD]], interpolation='nearest',cmap=mapname)
    hax=gca()
    #plt.colormap('GreenRed')
    setp(hax,'visible',False)
    #cb1 = mpl.colorbar.ColorbarBase(mpl.colorbar.make_axes(hax,pad=0)[0], cmap=greenred,norm=cnorm,orientation='vertical',format='%d')#,ticks=range(len(allCohorts)))#uniqu
    hcb=plt.colorbar(aspect=6) #norm=cnorm,

    # Also return a lookup function that can be used with interpolation to give colour for any country value:
    import scipy.interpolate as sp
    #print  [minD,1.0/2.0*(minD+maxD),maxD]
    print  [minD,midD,maxD]
    #print  [minD,1.0/2.0*(minD+maxD),maxD]
    print [array(rgbB),array(rgbM),array(rgbT)]
    return(greenred, sci.interpolate.interp1d(        [minD,midD,maxD],        [array(rgbB),array(rgbM),array(rgbT)], axis=0) ,fig)
    #    return(fig,sci.interpolate.interp1d(        [minD,1.0/2.0*sum(minD+maxD),maxD],        [array(rgbB),array(rgbM),array(rgbT)], axis=0) )

# Old name for it (now it returns lookup as well as generating plot...
plot_biLinearColorscale=biLinearColorscale


##############################################################################
##############################################################################
#
def gAnnotate():
    ##########################################################################
    ##########################################################################
    """ Interactively add an arrow; produce code for it. This is just a development/coding tool.
    Nov 2011. """
    xy1,xy2=plt.ginput(2)
    plt.gca().annotate('tyui',xy=xy2,xytext=xy1,xycoords='data',arrowprops=dict(shrink=0.05,width=.2,color='c',headwidth=5))
    print "gca().annotate('labelh',xy=",str(xy2),",xytext=",str(xy1),",xycoords='data',arrowprops=dict(shrink=0.05,width=.2,color='c',headwidth=5))"
    plt.draw()
def gannotate():
    gAnnotate()






def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
 
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """
 
    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))
 
    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points
 
    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
 
    # Build lower hull 
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
 
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
 
    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list. 
    return lower[:-1] + upper[:-1]
 
 
# Example: convex hull of a 10-by-10 grid.
#assert convex_hull([(i/10, i%10) for i in range(100)]) == [(0, 0), (9, 0), (9, 9), (0, 9)]

##############################################################################
##############################################################################
#
def tmp2_convex_hull(points, graphic=True, smidgen=0.0075):
    ##########################################################################
    ##########################################################################

    '''Taken by cpbl from scipy cookbook, dec 2011:
    
    Calculate subset of points that make a convex hull around points

Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

:Parameters:
    points : ndarray (2 x m)
        array of points for which to find hull
    graphic : bool
        use pylab to show progress?
    smidgen : float
        offset for graphic number labels - useful values depend on your data range

:Returns:
    hull_points : ndarray (2 x n)
        convex hull surrounding points
'''

    import numpy as n, pylab as p, time

    def _angle_to_point(point, centre):
        '''calculate angle in 2-D between points and x axis'''
        delta = point - centre
        res = n.arctan(delta[1] / delta[0])
        if delta[0] < 0:
            res += n.pi
        return res


    def _draw_triangle(p1, p2, p3, **kwargs):
        tmp = n.vstack((p1,p2,p3))
        x,y = [x[0] for x in zip(tmp.transpose())]
        p.fill(x,y, **kwargs)
        #time.sleep(0.2)


    def area_of_triangle(p1, p2, p3):
        '''calculate area of any triangle given co-ordinates of the corners'''
        return n.linalg.norm(n.cross((p2 - p1), (p3 - p1)))/2.



    if graphic:
        p.clf()
        p.plot(points[0], points[1], 'ro')
    n_pts = points.shape[1]
    assert(n_pts > 5)
    centre = points.mean(1)
    if graphic: p.plot((centre[0],),(centre[1],),'bo')
    angles = n.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:,angles.argsort()]
    if graphic:
        for i in xrange(n_pts):
            p.text(pts_ord[0,i] + smidgen, pts_ord[1,i] + smidgen, \
                   '%d' % i)
    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        if graphic: p.gca().patches = []
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i],     pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], \
                                   pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i],     pts[(i + 2) % n_pts])
            if graphic:
                _draw_triangle(centre, pts[i], pts[(i + 1) % n_pts], \
                               facecolor='blue', alpha = 0.2)
                _draw_triangle(centre, pts[(i + 1) % n_pts], \
                               pts[(i + 2) % n_pts], \
                               facecolor='green', alpha = 0.2)
                _draw_triangle(centre, pts[i], pts[(i + 2) % n_pts], \
                               facecolor='red', alpha = 0.2)
            if Aij + Ajk < Aik:
                if graphic: p.plot((pts[i + 1][0],),(pts[i + 1][1],),'go')
                del pts[i+1]
            i += 1
            n_pts = len(pts)
        k += 1
    return n.asarray(pts)


def xTicksAreDates(years=False):
    """
    Copied from matplotlib example
    uhhhh.. this is COMPLETE GARBAGE so far. not succeeded in 
    """
    import datetime
    #import numpy as np
    import matplotlib
    #import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.mlab as mlab
    import matplotlib.cbook as cbook


    fig=plt.gcf()
    ax=plt.gca()
    if years:
        from matplotlib.ticker import MultipleLocator, FormatStrFormatter
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        # 2013: or should it be:
        #import matplotlib.dates as mdates
        #    plt.gca().format_xdata = mdates.DateFormatter('%Y')
        fig.autofmt_xdate()
        
        return()

    years    = mdates.YearLocator()   # every year
    months   = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    # load a numpy record array from yahoo csv data with fields date,
    # open, close, volume, adj_close from the mpl-data/example directory.
    # The record array stores python datetime.date as an object array in
    # the date column
    #datafile = cbook.get_sample_data('goog.npy')
    #r = np.load(datafile).view(np.recarray)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot(r.date, r.adj_close)

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    uiu
    if 0:
        datemin = datetime.date(r.date.min().year, 1, 1)
        datemax = datetime.date(r.date.max().year+1, 1, 1)
        ax.set_xlim(datemin, datemax)

    # format the coords message box
    def price(x): return '$%1.2f'%x
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax.format_ydata = price
    ax.grid(True)

    # rotates and right aligns the x labels, and moves the bottom of the
    # axes up to make room for them
    fig.autofmt_xdate()

    return()


def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,                                                             
         tick labels, axis labels, title and legend                                                                                   
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)


def set_backgroundcolor(ax, color):
     '''Sets the background color of the current axes (and legend).                                                                   
         Use 'None' (with quotes) for transparent. To get transparent                                                                 
         background on saved figures, use:                                                                                            
         pp.savefig("fig1.svg", transparent=True)                                                                                     
     '''
     ax.patch.set_facecolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)




def weightedMeanSE(arrin, weights_in, inputmean=None, calcerr=True, sdev=False):
    """
    NAME:
      wmom()
      
    PURPOSE:
      Calculate the weighted mean, error, and optionally standard deviation of
      an input array.  If calcerr==False, error is calculated assuming the weights are
      1/err^2, but if you send calcerr=True this assumption is dropped and the
      error is determined from the weighted scatter.

    CALLING SEQUENCE:
     wmean,werr = wmom(arr, weights, inputmean=None, calcerr=False, sdev=False)
    
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean: 
          An input mean value, around which them mean is calculated.
      calcerr=False: 
          Calculate the weighted error.  By default the error is calculated as
          1/sqrt( weights.sum() ).  If calcerr=True it is calculated as sqrt(
          (w**2 * (arr-mean)**2).sum() )/weights.sum()
      sdev=False: 
          If True, also return the weighted standard deviation as a third
          element in the tuple.

    OUTPUTS:
      wmean, werr: A tuple of the weighted mean and error. If sdev=True the
         tuple will also contain sdev: wmean,werr,wsdev

    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU
      I renamed this from wmom to weightedMeanSE and changed the calcerr default to True!

   """

    # no copy made if they are already arrays
    arr = np.array(arrin, ndmin=1, copy=False)
    
    # Weights is forced to be type double. All resulting calculations
    # will also be double
    weights = np.array(weights_in, ndmin=1, dtype='f8', copy=False)
  
    wtot = weights.sum()
        
    # user has input a mean value
    if inputmean is None:
        wmean = ( weights*arr ).sum()/wtot
    else:
        wmean=float(inputmean)

    # how should error be calculated?
    if calcerr:
        werr2 = ( weights**2 * (arr-wmean)**2 ).sum()
        werr = np.sqrt( werr2 )/wtot
    else:
        werr = 1.0/np.sqrt(wtot)

    # should output include the weighted standard deviation?
    if sdev:
        wvar = ( weights*(arr-wmean)**2 ).sum()/wtot
        wsdev = np.sqrt(wvar)
        return wmean,werr,wsdev
    else:
        return wmean,werr

def weightedMeanSE_pandas(df,varNames,weightVar='weight'):
    """
    Example use: aggregate by age:
    means=df.groupby('age').apply(lambda adf: weightedMeanSE_pandas(adf,tomean,weightVar='cw')).reset_index()
    """
    outs={}
    varPrefix=''
    for mv in varNames:
        X,W=finiteValues(df[mv].values,df[weightVar].values)
        mu,se=weightedMeanSE(X,W) # Gives same as Stata!
        # Values need to be vectors(lists) for the conversion to DataFrame, it seems.
        outs.update({varPrefix+mv: mu,   varPrefix+'se_'+mv: se})
    return(pd.Series(outs)) # Return this as a Series; this avoids adding a new index in groupby().apply


##############################################################################
##############################################################################
#
def weightedMeansByGroup(pandasDF,meansOf,byGroup=None,weightVar='weight',varPrefix=''):# varsByQuantile=None,suffix='',skipPlots=True,rankfileprefix=None,ginifileprefix=None,returnFilenamesOnly=False,forceUpdate=False,groupNames=None,ginisOf=None,loadAll=None,parallelSafe=None):
    #
    ##########################################################################
    ##########################################################################
    """
2013 Feb: I'm adapting this from weightedQuantilesByGroup.

I've finally found something that gives the same s.e. as Stata. 
The list of what does NOT is very long: my ( wtmean,wtsem,wtvar), above. 
statsmodels.WLS gives robust standard errors, I guess, but they ignore the weights!! ie give same as stata without any weights. Wow, how poor.  But the code someone gave me on a list in 2010 works.

    This returns another DataFrame with appropriately named columns.

My related functions: 2013July I need a weighted moment by group in recodeGallup.

2014June :(Usually you can get by without this function?:     means=df.groupby('age').apply(lambda adf: weightedMeanSE_pandas(adf,tomean,weightVar='cw')) )

    """

    df=pandasDF
    #from scipy import stats
    if isinstance(byGroup,str):
        byGroup=byGroup.split(' ')
    if isinstance(meansOf,str):
        meansOf=meansOf.split(' ')
    assert all([mm in df for mm in meansOf])
    assert all([mm in df for mm in byGroup])
    grouped=df.groupby(byGroup)

    newdf=grouped.apply(weightedMeanSE_pandas,meansOf,weightVar)
    if varPrefix:
        newdf.columns=[varPrefix+cc for cc in newdf.columns]
    return(newdf)


##############################################################################
##############################################################################
#
def weightedMomentsByGroup(pandasDF,momentsOf,momfunc,byGroup=None,weightVar='weight',varPrefix=''):
    #
    ##########################################################################
    ##########################################################################
    """
2013 To calculate, for example, the mean log income (using momfunc=np.log)

So far, does not report any S.E. 



Sample usage: sprawl/cpblUtilitiesMathGraph.py:    newdf=grouped.apply(weightedMeanSE_pandas,meansOf,weightVar)

"""

    df=pandasDF
    if isinstance(byGroup,str):
        byGroup=byGroup.split(' ')
    if isinstance(momentsOf,str):
        momentsOf=momentsOf.split(' ')
    assert all([mm in df for mm in momentsOf])
    assert all([mm in df for mm in byGroup])
    grouped=df.groupby(byGroup)

    #def weightedMoment(arrin, weights_in):#, inputmean=None, calcerr=True, sdev=False):
    
        
    def weightedMoment_pandas(df,varNames,weightVar='weight'):
        outs={}
        varPrefix=''
        for mv in varNames:
            X,W=finiteValues(df[mv].values,df[weightVar].values)
            mu= (W * X.map(momfunc)).sum()/W.sum()
            #mu,se=weightedMoment(X,W) # Gives same as Stata!
            # Values need to be vectors(lists) for the conversion to DataFrame, it seems.
            ninininin
            outs.update({varPrefix+mv: mu,   varPrefix+'se_'+mv: se})
        return(pd.Series(outs)) # Return this as a Series; this avoids adding a new index in groupby().apply

    newdf=grouped.apply(weightedMoment_pandas,momentsOf,weightVar)
    if varPrefix:
        newdf.columns=[varPrefix+cc for cc in newdf.columns]
    return(newdf)




##############################################################################
##############################################################################
#
def weightedQuantilesByGroup(pandasDF,quantilesOf,byGroup=None,weightVar='weight',varPrefix='qtl_', varsByQuantile=None):#,suffix='',skipPlots=True,rankfileprefix=None,ginifileprefix=None,returnFilenamesOnly=False,forceUpdate=False,groupNames=None,ginisOf=None,loadAll=None,parallelSafe=None):
    #
    ##########################################################################
    ##########################################################################
    """

2013 Jan: This is derived from cpblStata's generateRankingData() for stata data, but this one takes pandas DataFrame instead. And we don't include ginis! (ugh). And leave plotting to a separate function, since we could return data.

e.g.: generateRankingData(pandasDF,'income', varsByQuantile=None,byGroup='year PRuid',weightVar='weight',suffix='',skipPlots=True,rankfileprefix=None,returnFilenamesOnly=False,forceUpdate=False,groupNames=None,parallelSafe=None):

As for the "varsByQuantile", those can easily be done in pandas using cut and groupby etc... not done yet.

This no longer creates files. It returns an augmented DataFrame.
Does not allow more than one variable for quantilesOf.
    """

    df=pandasDF
    from scipy import stats
    assert quantilesOf in df
    if isinstance(byGroup,str):
        byGroup=byGroup.split(' ')

    import numpy as np

    newvar=varPrefix+quantilesOf
    df[newvar]=np.nan
    def getq(adf):
        # If I remove the .values from the following, it fails to preserve order.
        ww=weightedQuantile(adf[quantilesOf].values,adf[weightVar].values)
        adf[newvar]=ww
        assert ww is np.nan or len(ww)==len(adf)
        return(adf)

    print 'Calculating quantiles...', #,end=' ')
    withquantiles=df.groupby(byGroup,group_keys=False).apply(getq)
    print(' [Done]')
    return(withquantiles)
    # 2013 Feb. Also calculate varsByQuantile, if desired.
    if varsByQuantile==None:
        varsByQuantile==[]
    assert all(vbq in df for vbq in varsByQuantile)
    assert not varsByQuantile
    if 0: # NOT WRITTEN YET!!!!!!!!!!!!!!!!!!!!!!!
        for iv,vname in enumerate(varsByQuantile+[quantilesOf]):
            # Use values with weights:
            vvww=[  finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]),
				   array([respondent[weightVar] for respondent in byQtl[qtl]])
				   ) for qtl in pQtl]

            #qtlStats['uw_'+vname]=[np.mean(
            #            finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]))
            # )                    for qtl in pQtl]
            qtlStats[vname]=[wtmean(vv,weights=ww) for vv,ww in vvww]
            #qtlStats['uw_se'+vname]=[stats.sem(
            #            finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]))
            #            )             for qtl in pQtl]
            qtlStats['se'+vname]=[wtsem(vv,ww) for vv,ww in vvww]

	    # Ugly kludge:
	    if vname in ['SWL','lifeToday']:

                vvall,wwall=finiteValues(array([respondent[vname] for respondent in groupDfinite]),
				   array([respondent[weightVar] for respondent in groupDfinite]))
		from pylab import histogram,array
                qtlStats['hist'+vname]=histogram(vvall,bins=-0.5+array([0,1,2,3,4,5,6,7,8,9,10,11]),weights=wwall)


	    # Shall I also calculate Gini here? It seems it may be much faster than Stata's version. #:(, Though I won't have a standard error for it.
	    if doGini and (ginisOf is None or vname in ginisOf):
                # n.b. I don't just want the ones with finite rankVar. So go back to groupD:
                xxV=array([respondent[vname] for respondent in groupD])
		macroInequalities[agroup]['gini'+vname]= cpblGini(weightD,xxV)


		#print "             %s=%s: Gini=%f"%(byGroup,agroup,inequality.Gini)

            # ne=where(logical_and(logical_and(isfinite(x),isfinite(y)),logical_and(isfinite(yLow),isfinite(yHigh))))


            #vQtl=array([stats.mean(finiteValues(
            #            vv[find(logical_and(y<=yQtl[iq] , y>=([min(y)]+yQtl)[iq]))]      )) for iq in range(len(yQtl))])
            #sevQtl=array([stats.sem(finiteValues(
            #            vv[find(logical_and(y<=yQtl[iq] , y>=([min(y)]+yQtl)[iq]))]      )) for iq in range(len(yQtl))])

























    return(withquantiles) 





    if 0:
        def assignQs(x,w):#adf, xv,wv)
            from scipy import interpolate
            import numpy as np
            #w,x=adf[wv],adf[xv]
            CDF=np.cumsum(w)*1.0/sum(w)
            # interp1d returns a function...
            qinterp=interpolate.interp1d(np.array(CDF),np.array(x))
            return([np.nan if np.isnan(xi) else qinterp(xi) for xi in x])
        # else: # Return a value for quantile q
        #            return(interpolate.interp1d(array(CDF),array(x))(q))

    #quantiles=df.groupby(byGroup).apply(lambda adf: assignQs(adf[quantilesOf],adf[weightVar]))








    bb=quantiles0[quantiles0['PRuid']==24]
    plt.plot(bb['qtl_lnHHincome'],bb['lnHHincome'],'.')   
    plt.show()
    iuiui
    # as_index=False makes it so that the eventual returned value is not grouped!
    print 'Calculating quantiles...', #,end=' ')
    quantiles=df.groupby(byGroup, as_index=False).apply(lambda adf: weightedQuantile(adf[quantilesOf],adf[weightVar]))
    print(' [Done]')





    quantilesi=df.groupby(byGroup, group_keys=False).apply(lambda adf: weightedQuantile(adf[quantilesOf],adf[weightVar]))

    xdf=df.groupby(byGroup).transform(lambda adf: weightedQuantile(adf[quantilesOf],adf[weightVar]))
    #df.merge(quantilesi
    ###links2=links.merge(pd.DataFrame(fuelByStateYear),how='left',left_on=['MIN_AGE','state'],right_on=['year','state'])



    fooo
    # OLD FUNCTION BELOW

    from pylab import figure,plot,show,clf,arange,floor,array,find,logical_and,where,isfinite,xlabel,ylabel,cumsum,subplot,rcParams
    rcParams.update({'text.usetex': False,}) #Grrr. need it for plusminus sign, but can't deal with all foreign characters in country and region names?!
    import numpy as np
    from cpblUtilities import plotWithEnvelope,transLegend,savefigall,sortDictsIntoQuantiles,finiteValues,shelfSave,shelfLoad
    # Because numpy and scipy don't have basic weight option in mean, sem !!!
    from cpblUtilities import wtmean,wtsem,wtvar
    from inequality import ineq,cpblGini



    if byGroup==None:
        byGroup=''
    if varsByQuantile==None:
        varsByQuantile==[]
    if suffix:
        suffix='-'+suffix
    assert isinstance(byGroup,str)
    #tsvFile=WP+stripWPdta(stataFile)+'-qtlInput'+suffix+'.tsv'
    microQuantFile=WP+stripWPdta(stataFile)+'-qtlData'+suffix+'.tsv'
    macroQuantFileShelf=WP+stripWPdta(stataFile)+'-qtlData-'+byGroup+suffix+'.pyshelf'
    macroQuantFile=WP+stripWPdta(stataFile)+'-qtlData-'+byGroup+suffix+'.tsv'
    macroGiniFile=WP+stripWPdta(stataFile)+'-gini-'+byGroup+suffix+'.tsv'
    plotfileprefix=WP+'graphics/TMPRANK'
    if rankfileprefix:
        microQuantFile=rankfileprefix+'-'+byGroup+'.tsv'
        macroQuantFileShelf=rankfileprefix+'-'+byGroup+'.pyshelf'
        macroQuantFile=rankfileprefix+'-'+byGroup+'.tsv'
	plotfileprefix=WP+'graphics/'+stripWPdta(rankfileprefix)+byGroup
    if ginifileprefix:
	    macroGiniFile=ginifileprefix+'-'+byGroup+'.tsv'
    if not fileOlderThan([microQuantFile,macroQuantFileShelf]+doGini*[macroGiniFile],WPdta(stataFile)) and not forceUpdate:
        print '    (Skipping generateRankingData; no need to update %s/%s from %s...)'%(microQuantFile,macroQuantFileShelf,stataFile)
        return(os.path.splitext(microQuantFile)[0],os.path.splitext(macroQuantFileShelf)[0])
        #return(microQuantFile,macroQuantFileShelf)

    # Suffix is used in following to ensure that different calls to this function get the correct result exported from Stata, etc, (see notes in fcn below).
    # Caution! if
    onlyVars=None
    if not loadAll:
        onlyVars=' '.join(uniqueInOrder(inVars+[byGroup, quantilesOf]+varsByQuantile+[weightVar]))
    # If parallelSafe, Make the following force-updated, to avoid using shelve/shelf files simultanously by different processes!!
    dddT=loadStataDataForPlotting(stataFile,onlyVars=onlyVars,treeKeys=[byGroup],forceUpdate='parallel' if parallelSafe else forceUpdate,suffix=suffix)#vectors=True)#False,forceUpdate=False,singletLeaves=False):

    # Testing functionality aug 2012 to make this robust to weight variable not existing for all in dataset:
    for kk in dddT:
        plen=len(dddT[kk])
        dddT[kk]=[rrrr for rrrr in dddT[kk] if isfinite(rrrr[weightVar])]
        if not len(dddT[kk])==plen:
            print('CAUTION: I found and ditched some (%d/%d) individuals without weight %s for group %s in generateRankingData'%(plen-len(dddT[kk]),plen,weightVar,kk))

    if 0:
        from dictTrees import dictTree
        kk=ddd.keys()
        #for byKey in byGroup
        print 'Sorting by key...'
        dddT=dictTree([dict([[akey,ddd[akey][irow]] for akey in kk]) for irow in range(len(ddd[kk[0]]))],[byGroup])

    # Now.. Order these and assign ranking (between 0 and 1):  This should take into account the weights, properly.
    print '%d elements have no group (%s).'%(len(dddT.get('',[])),byGroup)
    rankGroups=[]
    macroStats=[]
    macroInequalities={}
    if not skipPlots:
        figure(126)
        clf()
        figure(124)
    for agroup in sorted(dddT.keys()):#.keys()[0:10]:
        if not agroup:
            continue
        groupD=dddT[agroup]
        weightD=array([respondent[weightVar] for respondent in groupD])
        groupDfinite=[xx for xx in groupD if isfinite(xx[quantilesOf]) ]
        # Hm, does the following fail if I include the nan's!?
        groupDfinite.sort(key=lambda x:x[quantilesOf])
	if doGini:
            macroInequalities[agroup]={byGroup:agroup}

        if 0: # I'm eliminating the following, unweighted ranking for now.
            if len(groupDfinite)==0:
                continue
            if len(groupDfinite)==1:
                groupDfinite[0]['rank'+quantilesOf]=0.5
            else:
                for iRank,respondent in enumerate(groupDfinite):
                    # THIS IS WRONG!!!!!!!!!! IT IGNORES WEIGHT. I SHOULD BE USING WEIGHTED RANK. I DO THIS BELOW. CANNOT FIND scipy ROUTINE TO DO QUANTILES WITH SAMPLE WEIGHTS.
                    respondent['rank'+quantilesOf]=iRank*1.0/(len(groupDfinite)-1)
                    x=array([respondent['rank'+quantilesOf] for respondent in groupDfinite])
        y=array([respondent[quantilesOf] for respondent in groupDfinite])
        w=array([respondent[weightVar] for respondent in groupDfinite])


        # Now, I also need to section these up into groups, in order to calculate other variables by quantile. How to do this? I could use a kernel smoothing, to estimate y(I), where, e.g. y is SWB and I is income.  OR I could calculate quantiles. e.g. qtlY(I) would be the mean y amongst all those in the ith quantile. I'll do the latter. This means that curves will NOT represent y(I), since it's mean(y) but i<I.
        minN=20
        nQuantiles=min(25,floor(len(y)/minN))
        pQtl=(1.0+1.0*arange(nQuantiles))/nQuantiles
        assert len(pQtl)==nQuantiles

        assert all(isfinite(w))  # Really? Couldn't I make this robust... [aug2012: okay, i have, above, by modifying ddTT]

        # Use my nifty sort-into-quantiles function
        minN=20
        if len(y)<minN/2:
            print ' SKIPPING '+agroup+' with only %d respondents...'%len(y)
            continue
        nQuantiles=max(2,min(25,floor(len(y)/minN)))
        # The following function ALSO fills in a new element of the weighted rank of each individual.
        byQtl=sortDictsIntoQuantiles(groupD,sortkey=quantilesOf,weightkey=weightVar,approxN=25,)#nQuantiles=min(25,floor(len(y)/minN)))
        pQtl=sorted(byQtl.keys())
        print '   Quantiles: parsing for group %s=%20s,\t with %d respondents,\t with %d having rank variable;\t therefore using %d quantiles...'%(byGroup,agroup,len(groupDfinite),len(finiteValues(y)),len(pQtl))


        # So since sortDictsIntoQ... filled in individual ranks, I can now plot these:
        x=array([respondent['rank'+quantilesOf[0].upper()+quantilesOf[1:]] for respondent in groupDfinite])
        if not skipPlots:
            figure(126)
            clf()
            subplot(121)
            plot(y,x,hold=True)
            xlabel(substitutedNames(quantilesOf))
            ylabel('Quantile')
	    print 'More up to date plots are made by a custom function using the .shelf data, in regressionsInequality'

        #print [stats.mean([gg['lnHHincome'] for gg in byQtl[qq]])  for qq in pQtl]
        #print [stats.mean([gg['lifeToday'] for gg in byQtl[qq]])  for qq in pQtl]


        # Cool! That worked nicely, and is even quite efficient.

        # I wonder how byQtl.keys() compares with the unweighted measure below...    (uses approximately quantile unbiased (Cunnane) parameters)
        yQtl2=stats.mstats.mquantiles(y, prob=pQtl, alphap=0.40000000000000002, betap=0.40000000000000002, axis=None, limit=())


        # Now calculate weighted means for variables of interest within each quantile group:
        qtlStats={'qtl':pQtl,'group':agroup}
        # Also save in the output any variables which are uniform within this group (ie markers of a group in which this is a subgroup):
        if 0:
            for vvv in [vv for vv in inVars if vv not in [byGroup]]:
                if all(array([respondent[vvv] for respondent in groupDfinite])==groupDfinite[0][vvv]): # ah, this variable is uniform within the group
                    qtlStats[vvv]=groupDfinite[0][vvv]

        qtlStats['n']=[ len(
                        finiteValues(array([respondent[quantilesOf] for respondent in byQtl[qtl]]))
                        )             for qtl in pQtl]
        for iv,vname in enumerate(varsByQuantile+[quantilesOf]):
            # Use values with weights:
            vvww=[  finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]),
				   array([respondent[weightVar] for respondent in byQtl[qtl]])
				   ) for qtl in pQtl]

            #qtlStats['uw_'+vname]=[np.mean(
            #            finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]))
            # )                    for qtl in pQtl]
            qtlStats[vname]=[wtmean(vv,weights=ww) for vv,ww in vvww]
            #qtlStats['uw_se'+vname]=[stats.sem(
            #            finiteValues(array([respondent[vname] for respondent in byQtl[qtl]]))
            #            )             for qtl in pQtl]
            qtlStats['se'+vname]=[wtsem(vv,ww) for vv,ww in vvww]

	    # Ugly kludge:
	    if vname in ['SWL','lifeToday']:

                vvall,wwall=finiteValues(array([respondent[vname] for respondent in groupDfinite]),
				   array([respondent[weightVar] for respondent in groupDfinite]))
		from pylab import histogram,array
                qtlStats['hist'+vname]=histogram(vvall,bins=-0.5+array([0,1,2,3,4,5,6,7,8,9,10,11]),weights=wwall)


	    # Shall I also calculate Gini here? It seems it may be much faster than Stata's version. #:(, Though I won't have a standard error for it.
	    if doGini and (ginisOf is None or vname in ginisOf):
                # n.b. I don't just want the ones with finite rankVar. So go back to groupD:
                xxV=array([respondent[vname] for respondent in groupD])
		macroInequalities[agroup]['gini'+vname]= cpblGini(weightD,xxV)


		#print "             %s=%s: Gini=%f"%(byGroup,agroup,inequality.Gini)

            # ne=where(logical_and(logical_and(isfinite(x),isfinite(y)),logical_and(isfinite(yLow),isfinite(yHigh))))


            #vQtl=array([stats.mean(finiteValues(
            #            vv[find(logical_and(y<=yQtl[iq] , y>=([min(y)]+yQtl)[iq]))]      )) for iq in range(len(yQtl))])
            #sevQtl=array([stats.sem(finiteValues(
            #            vv[find(logical_and(y<=yQtl[iq] , y>=([min(y)]+yQtl)[iq]))]      )) for iq in range(len(yQtl))])


            if (not skipPlots) and vname in varsByQuantile:
                figure(126)
                subplot(122)
                colors='rgbckm'
                vQtl= array(qtlStats[vname])
                sevQtl= array(qtlStats['se'+vname])
                pQtl=array(pQtl)
                plotWithEnvelope(pQtl,vQtl,vQtl+sevQtl,vQtl-sevQtl,linestyle='.-',linecolor=None,facecolor=colors[iv],alpha=0.5,label=None,lineLabel=None,patchLabel=vname,laxSkipNaNsSE=True,laxSkipNaNsXY=True,ax=None,skipZeroSE=True) # Why do I seem to need both lax flags?
                plot(pQtl,vQtl,'.',color=colors[iv],alpha=0.5)
                xlabel(substitutedNames(quantilesOf) +' quantile')

            ##ylabel(vname)
        from cpblUtilities import str2pathname
        if not skipPlots:
            transLegend(comments=[groupNames.get(agroup,agroup),r'$\pm$1s.e.'],loc='lower right')
            savefigall(plotfileprefix+'-'+str2pathname(agroup))
        rankGroups+=groupDfinite
        macroStats+=[qtlStats]


	if 0*'doRankCoefficients':
		groupVectors=dict([[kk,[gd[kk] for gd in groupDfinite ]] for kk in groupDfinite[0].keys()])
		from cpblUtilities import cpblOLS
		x=cpblOLS('lifeToday',groupVectors,rhsOnly=[ 'rankHHincome'],betacoefs=False,weights=groupVectors['weight'])
		foioi

        # assert not 'afg: Kabul' in agroup
        # Add the quantile info for this group to the data. Also, compile the summary stats for it.

#[, 0.25, 0.5, 0.75]
        # Centre a series of quantiles
        """
	No. Create 20 quantiles. Assign. if none there, weight nearest?

	e.g. 1  2 10 13


	scipy.stats.mstats.mquantiles

	scipy.stats.mstats.mquantiles(data, prob=[, 0.25, 0.5, 0.75], alphap=0.40000000000000002, betap=0.40000000000000002, axis=None, limit=())

	"""


    from cpblUtilities import dictToTsv
    dictToTsv(rankGroups,microQuantFile)
    tsv2dta(microQuantFile)
    if doGini:
        dictToTsv(macroInequalities.values(),macroGiniFile)
	tsv2dta(macroGiniFile)

    shelfSave(macroQuantFileShelf,macroStats)
    if 0: # whoooo... i think this was totally misguided. it's not a macro file..
        dictToTsv(macroStats,macroQuantFile)
        tsv2dta(macroQuantFile)

    #vectorsToTsv(qtlStats,macroQuantFile)
    #tsv2dta(macroQuantFile)

    #inequality,redundancy,equality,variation,thesum,absolute=ineq(zip(popn,wealth))

    return(os.path.splitext(microQuantFile)[0],os.path.splitext(macroQuantFileShelf)[0])
    #return(microQuantFile,macroQuantFileShelf)



def surveyMeans():
    """
    My Nth attempt over the years to find the right way to efficiently calculate group means or conditional/subset means or etc from Stata Data, when there are sample (probability) weights on observations and when I want standard errors of means.
    2013 Feb: Do it with a WLS using python, applied to a DataFrame.
    oh.. actually, wait on this until Apollo is fixed to have statsmodels (for WLS).
    In the mean time, I'm implementing the same idea using regression in Stata. See cpblStata. 

    Yes, I could have done this much more nicely all along using regression in Stata. However, that doesn't also get me other stats when I want them. I now use dataframes in pandas and groupby; see/use  weightedMeansByGroup().
    """




##############################################################################
##############################################################################
#
def parallelCollapseCode_nostata(stataFile,listOfDicts=None,mergeAllVars=None,parallelSafe=None,forceUpdate=False,combinedOutputName=''):
    # 
    ##########################################################################
    ##########################################################################
    """
    Rather than doing all collapses at once, waste RAM like crazy (big server) and do them piecewise! This is also useful for being able to add a mean or two to the list of meaned variables without redoing all of them.
ie the plan is to load up a Stata file, and then use collapse to create some means, but not worry about whether these are all the means we want. 

There could be trouble if this runs more than one instance at once?

Also, in order it easy to generate a single do file, in general, this function must not execute. Rather, it will return a list of statacodes (one for each collapse) and one other stata code (for merging)...

each dict in listOfDicts has structure:
dict(
stataFile:'', # only if the single argument, stataFile, is not given
by:'',
weight:'weight',
postLoadCode:'',
varsToMean:'',
meanPrefix:'',
meanSuffix:'',
makeSEs:False,
savefile:None
)
default values will be filled in where absent.
For the time being, "if" clauses can be done with the postLoadCode...

AGHGH Dec 2012: I drafted this, but it's not tested and not used anywhere!!, since it turns out I probably want to stick with my mechanisms in masterPrepareSurveys.py for recodeCCHS.  Nice idea, though.

Two ways to call:

parallelCollapses(listOfDicts)   # with "stataFile" field in each Dict.
parallelCollapses(stataFile,listOfDicts)

2012 Dec: This will look for the set of unique collapse variables, and merge the means into one file for each set of collapse vars.
So, with that feature, I may want to replace my normal CR-means functions with this general one.
Though... shouldn't I be doing all this in pandas?

2013 Feb: implemented SEs. Btw, the right way to do means and se's is just using a regression. :|
	"""


    import pandas as pd
    

    if listOfDicts is None:
        assert isinstance(stataFile,list) or isinstance(stataFile,dict) 
	listOfDicts=stataFile
    else:
	assert not any('stataFile' in ad for ad in stataFile)
        for ad in listOfDicts:
           ad['stataFile']=stataFile


    if isinstance(listOfDicts,list):
       stataOut=[parallelCollapseCode(ad,parallelSafe=parallelSafe,forceUpdate=forceUpdate)[0] for ad in listOfDicts]
       combines=[]
       outfiles=[]
       mergeVars=[]

       for uby in uniqueInOrder([ sf['by'] for sf  in listOfDicts]):
           ld=[df for df in listOfDicts if df['by']==uby]
	   assert len(uniqueInOrder([LL['stataFile'] for LL in ld]))==1 # Ensure they're all from the same source file. So we can use that for naming the output, along with a provided outputsuffix and the collapse vars. [no, not anymore]
	   outfiles+=[WP+str2pathname(combinedOutputName+'-'+uby)]
			   #ld[0]['stataFile']+'-'+outputSuffix+'-'+'_'.join(fromToList))]
	   combines+=[stataLoad(ld[0]['savefile'])+'\n'.join(stataMerge(uby,sf['savefile']) for sf in ld[1:] if sf)+"""
           """+stataSave(outfiles[-1])]
	   mergeVars+=[uby]

       if 0 and mergeAllVars:
	       combine=stataLoad(listOfDicts[0]['savefile'])+'\n'.join(stataMerge(mergeAllVars,sf['savefile']) for sf in listOfDicts[1:] if sf)
       return(stataOut,combines,outfiles,mergeVars) 

    cd=listOfDicts
    assert isinstance(cd,dict)
    stataFile=cd['stataFile']
    commonPrefix=cd.get('meanPrefix','')
    commonSuffix=cd.get('meanSuffix','')

    fromToList=[vv for vv in cd.get('varsToMean','').strip().split(' ') if vv]
    assert fromToList
    fromtos=' '.join(['%s%s%s=%s'%(commonPrefix,ft,commonSuffix,ft) for ft in fromToList])
    if cd.get('makeSEs',True):
	    fromtos+=' (semean) '+' '.join(['%sse_%s%s=%s'%(commonPrefix,ft,commonSuffix,ft) for ft in fromToList])
    ###allcollapsedvars=' '+' '.join(['%s%s'%(commonPrefix,ft) for ft in fromToList])+' '
    weight=cd.get('weight','weight')
    bykey=cd.get('by')
    assert bykey
    savefile=cd.get('savefile',stataFile+'-'+commonPrefix+commonSuffix+'-'+'_'.join(fromToList))
    cd['savefile']=savefile # Update this field, since the recursion top will look.

    stataout=stataLoad(stataFile)+'\n'+listOfDicts.get('postLoadCode','')+"""

        collapse  """+fromtos+ ' [pw='+weight+'],  by('+bykey+""") fast

"""+stataSave(savefile)

    #assert stataout*(forceUpdate or fileOlderThan(WPdta(savefile),WPdta(stataFile)))
    return(stataout*(forceUpdate or fileOlderThan(WPdta(savefile),WPdta(stataFile))),'','')



def imagesToVideo(filelist): # Not written yet, but it would go something like this:
    #filelist=sorted([ff for ff in os.listdir(mapPath) if 'historical_'+fips in ff and ff.endswith('.jpg')])
    os.system('cd %s && rm tmpln_*'%mapPath)
    # In one shell line, this would be: i=0; for f in *.jpg; do ln -s $f $(printf "tmp%04d.jpeg" $i); i=$((i+1)); done
    for iff, ff in enumerate(filelist):   # Need to rename files to have consecutive zero-based integers.
        print('ln -s %s tmpln_%s_%05d.jpg'%(ff,fips,iff))
        os.system('cd %s && ln -s %s tmpln_%s_%05d.jpg'%(mapPath,ff,fips,iff))
    print('ffmpeg -y -r 10 -b 65536k -i tmpln_%s_%%05d.jpg %s'%(fips,mp4))
    os.system('cd %s && ffmpeg -y -r 10 -b 4800 -i tmpln_%s_%%05d.jpg %s'%(mapPath,fips,mp4))
    os.system('cd %s && rm tmpln_%s_*.jpg'%(mapPath,fips)) # Clean up symbolic links
    # -r is frame Rate in Hz.
    # -y : don't confirm overwriting output
    return


def drawStack(left,ys,width=1,facecolor=None,label=None,ax=None):
    """
     Make a single stack in a stacked barplot.  Mostly, matplotlib builds all the categories (stacks) at once. But I want different colour schemes for each stack, for David MacKay-style energy demand/supply displays, 2014.

Example usage:
            DD=dd.ix[[PR]][demandsInOrder]
#            PP=justren.ix[[PR]]

            PP=justren.ix[[PR]][recolsinorder]
            # Choose separate colormap for the two stacks of bars:
            colorlookup=getIndexedColormap('cool', recolsinorder)
            colorlookup.update(getIndexedColormap('spring', DD.columns.values))

            # New March 2014: Build bar plot by hand, using new drawStack() function not with pandas' stacked bar:
            # This lets me specify colour immediately, and to draw top to bottom (like legend)
            # It also lets me manipulate (reorder, etc) the legend more easily, since I haven't messed 
            from cpblUtilities import drawStack

            #pdemands=[0]+list(DD.ix[PR].values.cumsum())
            dnames= DD.columns.values#ix[PR].index.values
            rnames= PP.columns.values
            width=0.8
            ax=gca()
            ax.add_patch(Rectangle((1,1),1,1,color='r',label='---A-------',alpha=0))
            h1=drawStack(0,DD.ix[PR].values,width,[colorlookup[LL] for LL in dnames],dnames)
            ax.add_patch(Rectangle((1,1),1,1,color='k',label='----B------',alpha=0))
            h2=drawStack(1,PP.ix[PR].values,width,[colorlookup[LL] for LL in rnames],rnames)
            ax.add_patch(Rectangle((1,1),1,1,color='g',label='----C-----',alpha=0))
            ax.set_xlim([-.15,1+width+.15])
            ax.set_xticks([width/2,1+width/2])
            ax.set_xticklabels(['Current demand','Renewable Potential'])
            ylabel({'P':'Power (TWh/year)','percap':'Power (kWh/year/person)'}[pmode])
            title(longnames[PR])



"""
    bottoms=[0]+list(np.cumsum(ys))
    if ax is None: ax=gca()
    hh=[]
    for ii in range(len(ys))[::-1]:
        assert not ys[ii]<0
        if ys[ii]>0: # Don't plot zero-size plates
            hh+=[ax.bar(left,
                        ys[ii],
                        bottom=bottoms[ii],#sum([0]+list(ys[:ii])), #data_stack[i-1],
                        color=facecolor[ii],#cols[i],
                        edgecolor='k',#edgeCols[i],
                        width=width,
                        linewidth=0.5,
                        label=label[ii],
                        #                   align='center'
                        )]
    return(hh)



if __name__ == '__main__':
    from cpblStata import *
    demoCPBLcolormapFunctions()
    hu
    weightedQuantile(0,0,test=True)
    hu

    rpyloessDF(0,'x','y')#df,xname,yname)
    hgyygygyygyg

    cchs=loadStataDataNumpy(WP+'masterCCHS20092010')
    wqby=weightedQuantilesByGroup(cchs,'lnHHincome', varsByQuantile=None,byGroup=['year','PRuid'])#,weightVar='weight',suffix='',skipPlots=True,rankfileprefix=None,ginifilepref
    from cpblUtilities import *    
    if 0:
        categoryBarPlot([],[],demo=True)
        show()
    plotQuantilesYvsX(0,0,0)#x,y,w,nSegments=10,nquantiles=None)
    fff


    """
2013 OCT: CAN I COMBINE SVGs SIDE BY SIDE WITH SOMETHING LIKE THIS: (assuming no labels collision)

<code>
from lxml import etree

files_to_combine = ['file1.svg','file2.svg']
output_file = open('output.svg','w')

base_svg_text = '''
<svg
    xmlns="http://www.w3.org/2000/svg"
    xmlns:svg="http://www.w3.org/2000/svg"
    xmlns:xlink="http://www.w3.org/1999/xlink"
    >
</svg>
'''

base_svg = etree.fromstring(base_svg_text)

for filename in files_to_combine:
    text = open(part_filename,'r').read()
    for element in etree.fromstring(text):
        base_svg.append(element)

output_file.write(etree.tostring(base_svg, pretty_print=True))
</code>

    """

