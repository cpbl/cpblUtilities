#!/usr/bin/python
import sys,os
import numpy as np
import cpblUtilities.textables as cpblt
def test_chooseSFormat():
# ff,conditionalWrapper=['',''],lowCutoff=None,lowCutoffOOM=True,convertStrings=False,highCutoff=1e8,noTeX=False,threeSigDigs=False,se=None, leadingZeros=False):
    """ This chooses a reasonable number of significant figures for a numerical value in a results table...
    for LaTeX format.
    It takes "." to be a NaN, which it represents with a blank.
    It aims not to show more than three significant figures.
    It deals nicely with negatives.

     If conditionalWrapper is supplied, it is only given, enclosing the output, if the output is not empty.
If, instead of conditionalWrapper, "convertStrings" is supplied, then strings will be converted to floats or ints.
Error handling for conversions is not yet done.

If lowCutoff is supplied, smaller numbers than it will be shown as "zero".
If lowCutoffOOM in [True,'log'], we will use "<10^-x" rather than "zero" in above. It will, awkwardly, report $-<$10$^{-6}$ for -1e-7.
If lowCutoffOOM is some other string, then it will be used for small numbers. E.g, provide lowCutoffOOM='$<$10$^{-6}$' to show that for numbers smaller than the lowCutoff. Note that in this case the output is not sensitive to sign.

If highCutoff is supplied, larger numbers will be shown as "big". May 2011: reducing this from 1e6 to 1e8, since it was binding on sample sizes.

se is the standard error. you can just specify that for smarter choices about sig digs to show...

2014-03 adding leadingZeros: In regression tables, I don't want them. But in general I might.
"""
    assertVals=[
        [   1382.797202,'1383',{},''],
        [   1382,'1400',dict(sigdigs=2),''],
        [   1382,'1380',dict(sigdigs=3),''],
        [   1382,'1382',{},''],
        [   1382,'1382',dict(sigdigs=5),''],
       ['4','4',{},''],
       ['.','',{},''],
       [4,'4',{},''],
       [0.4,'.40',{},''],
       [123456789098,'big',{},''],
       [0.00049239847,'.0005',{},''],
        ]+[
       ['4','4',dict(threeSigDigs=True),''],
       [4,'4',dict(threeSigDigs=True),''],
       [0.4,'.400',dict(threeSigDigs=True),''],
       [0.00049239847,'.0005',dict(threeSigDigs=True),' Sic??'],
        ]+[
       [0.0,  '0',dict(lowCutoff=1e-5), ''],
       [0,  '0',dict(lowCutoff=1e-5), ''],
       ['4','4',dict(lowCutoff=1e-5), ''],
       [4,'4',dict(lowCutoff=1e-5), ''],
       [0.4,'.40',dict(lowCutoff=1e-5), ''],
        [0.00049239847,'.0005',dict(lowCutoff=1e-5), ''],
       [-1e-9,  '-$<$10$^{-9}$',dict(lowCutoff=1e-5),' Is this right? '],
       [-np.inf,'big',dict(lowCutoff=1e-5), ''],
      [.0001,'.0001',dict(lowCutoff=1e-5), ''],
        ]
    for fff,sss,ddd,ccc in assertVals:
        outs=cpblt.chooseSFormat(fff, **ddd)
        print('chooseSFormat('+str(fff)+',\t\t'+str(ddd)+') == \t'+outs+'          '+ccc)
        assert outs==sss


        """        

       [4123,'4123'],
       [4123456789,'big'],
        [4000000,'4000000'],],
        print 

    if lowCutoff==None:
        lowCutoff==1.0e-99 # Sometimes "None" is explicitly passed to invoke default value.
    import numpy#from numpy import ndarray
    if isinstance(ff,list) or isinstance(ff,numpy.ndarray): # Deal with lists
        return([chooseSFormat(fff,conditionalWrapper=conditionalWrapper,lowCutoff=lowCutoff,convertStrings=convertStrings,threeSigDigs=threeSigDigs) for fff in ff])
    if ff=='': # Leave blanks unchanged
        return('')
    if ff=='.': # lone dots can mean NaN to Stata
        return('')
    if not isinstance(ff,int) and not isinstance(ff,float) and not convertStrings:
        return(conditionalWrapper[0]+str(ff)+conditionalWrapper[1])
    if isinstance(ff,basestring):
        #print "converting ",ff," to num:",
        if '.' in ff:
            ff=float(ff)
        else:
            ff=int(ff)
        #print '--> ',ff
    aa=abs(ff)
    if aa>highCutoff:
        return('big')
    if not aa>=0: # ie is "nan"
        return('')
    ss='%.1g'%ff
    if aa<lowCutoff:
        ss='0'
        if lowCutoffOOM in [True,'log']:
            negexp=int(np.ceil(np.log10(aa)))
            ss='-'*(ff<0)+ r'$<$10$^{%d}$'%negexp
        elif isinstance(lowCutoffOOM,basestring):
            ss=lowCutoffOOM
    if aa>=0.0001:
        ss=('%.4f'%ff)
    if aa>=0.001:
        ss=('%.3f'%ff)
    if aa>=0.01:
        ss='%.3f'%ff
    if aa>=0.1:
        ss='%.2f'%ff
    if threeSigDigs and aa>=0.1:
        ss='%.3f'%ff
    if aa>2.0:
        ss='%.1f'%ff
    if aa>10.0:
        ss='%.1f'%ff
    if aa>100.0:
        ss='%.0f'%ff
    if ss[0:2]=='0.' and not leadingZeros:
        ss=ss[1:]
    if ss[0:3]=='-0.' and not leadingZeros:
        ss='-'+ss[2:]
    if ss[0]=='-' and not noTeX:
        ss='$-$'+ss[1:]

    # Override all this for integers:
    if isinstance(ff,int):
        ss='$-$'*(ff<0)+str(abs(ff))


    return(conditionalWrapper[0]+ss+conditionalWrapper[1])

    """


def test_all():
    import sys
    current_module = sys.modules[__name__]
    import inspect
    all_functions = [(nn,onefunc) for nn,onefunc in inspect.getmembers(current_module, inspect.isfunction) if nn not in ['test_all','__nonzero__'] and 'SKIP_ME' not in nn]
    for nn,onefunc in all_functions:
        print("======= NEXT TEST:%s:  %s ==================="%(__file__,nn))
        onefunc()
    
if __name__ == "__main__":
    test_all()
