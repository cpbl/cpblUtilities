#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'cpbl'

import numpy as np
import pandas as pd
#from cpblUtilities.textables  import chooseSFormat

# The reason I wrote textables originally was to format tables of statistical results.
# This module has tools for formatting estimates and their standard errors into LaTeX
# that will go into a table.

# 201804: There's a major redundancy right now. The following methods are here for historical reasons: formatPairedRow, formatPairedRow_DataFrame, and latexFormatEstimateWithPvalue.  But I am embarking on rewriting these to be more testable and readable.

# Recent use cases: coal_analysis

# To do: Allow se to be passed in df_format_estimate_column_with_latex_significance

def test_pvalue_to_latex_significance():
    assert pvalue_to_latex_significance(2.394832234, .0013984) == '\\wrapSigOnePercent{2.4}'
    assert pvalue_to_latex_significance([1.1234567,2.394832234], [.8,.0013984]) == ['1.12', '\\wrapSigOnePercent{2.4}']
    df= pd.DataFrame({
        'mean': [3.1234567,2.394832234],
        'p': [.8,.00013984]})
    assert pvalue_to_latex_significance(df['mean'], df.p) == ['3.1', '\\wrapSigOneThousandth{2.4}']
    #assert pvalue_to_latex_significance(2.394832234, .0013984, sigdigs=3) == '\\wrapSigOnePercent{2.39}'

    
def pvalue_to_latex_significance(estimate, pvalue, **vararg):
    """ Given a p-value, and a floating-point estimate, generate a LaTeX string to colour-code the formatted estimate according to its (p-value) significance.
    
    estimate: scalar or vector float
    pvalue:   scale or vector (of same length as estimate) float

    See test_(methodname) method for examples.
    """

    if not np.isscalar(estimate):
        assert len(estimate) == len(pvalue)
        return [pvalue_to_latex_significance(e,p, **vararg) for e,p in zip(estimate, pvalue)]

    if pd.isnull(pvalue):
        return( r'$\cdot$')

    significanceString=(['']+[tt[0] for tt in significanceTable if pvalue<= tt[2]*1.0/100.0])[-1]

    return (significanceString +
            chooseSFormat(estimate,
                          **dict([(k,v) for k,v in vararg.items() if k in ['sigdigs']])
            )+
            '}' * (not not significanceString))
"""    
    if significanceString and yesGrey:
            significanceString=r'\agg'+significanceString[1:]
    if not significanceString and yesGrey:
            significanceString=r'\aggc{'
    if yesGrey:
        greyString=r'\aggc'
"""
        
    
def test_df_format_estimate_column_with_latex_significance():
    df= pd.DataFrame({
        'mean': [3.1234567,2.394832234, 7.65432],
        'p': [.8,.00013984, .01]})
    odf = df_format_estimate_column_with_latex_significance(df, 'mean','p')
    assert (odf.columns==['mean']).all()
    assert odf['mean'].tolist() == ['3.1', '\\wrapSigOneThousandth{2.4}', '\\wrapSigOnePercent{7.7}']
    
    
def df_format_estimate_column_with_latex_significance(df,
                                                      est_col,
                                                      p_col=None,
                                                      se_col=None,
                                                      replace_estimate_column=True,
                                                      drop_p_column=True):
    """
    Either the p-value (p_col) or the standard error (se_col) must be given.
    The column names are passed. They can be strings or tuples (for MultiIndex columns)
    Both columns should be float type.

    By default, the p-value column is dropped, and the est_col is replaced by a string version, with LaTeX formatting to denote the p-value significance.
    """
    s_est = pvalue_to_latex_significance(df[est_col], df[p_col])
    if replace_estimate_column:
        df[est_col] = s_est
    else:
        must_supply_prefix
    if drop_p_column:
        df.drop(columns=[p_col], inplace=True)
    return df

    
    #pvalue_to_latex_significance(estimate, pvalue, **vararg)

    
###########################################################################################
###
def chooseSFormat(ff,
                  conditionalWrapper=['', ''],
                  lowCutoff=None,
                  lowCutoffOOM=True,
                  convertStrings=False,
                  highCutoff=1e8,
                  noTeX=False,
                  sigdigs=4,
                  threeSigDigs=False,
                  se=None,
                  leadingZeros=False):
    ###
    #######################################################################################
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

To do: New parameter sigdigs only implemented for integers so far. And it needs to be reconciled with threeSigDigs (which should become deprecated).  threeSigDigs is really about precision; it corresponds closely/directly to decimal places.

   - N.B.: see new (2016Aug)   format_to_n_sigfigs and round_to_n_sigfigs in mathgraph.py, which should probably be used here.
   The latter  rounds correctly but does not do the displaying part.




"""
    if lowCutoff == None:
        lowCutoff == 1.0e-99  # Sometimes "None" is explicitly passed to invoke default value.
    import numpy  #from numpy import ndarray
    if isinstance(ff, list) or isinstance(ff,
                                          numpy.ndarray):  # Deal with lists
        return ([
            chooseSFormat(
                fff,
                conditionalWrapper=conditionalWrapper,
                lowCutoff=lowCutoff,
                convertStrings=convertStrings,
                sigdigs=sigdigs,
                threeSigDigs=threeSigDigs) for fff in ff
        ])
    if ff == '':  # Leave blanks unchanged
        return ('')
    if ff == '.':  # lone dots can mean NaN to Stata
        return ('')
    if not isinstance(ff, int) and not isinstance(
            ff, float) and not convertStrings:
        return (conditionalWrapper[0] + str(ff) + conditionalWrapper[1])
    if isinstance(ff, basestring):
        #print "converting ",ff," to num:",
        if '.' in ff:
            ff = float(ff)
        else:
            ff = int(ff)
        #print '--> ',ff
    aa = abs(ff)
    if aa > highCutoff:
        return ('big')
    if not aa >= 0:  # ie is "nan"
        return ('')
    ss = '%.1g' % ff
    if aa < lowCutoff:
        ss = '0'
        if lowCutoffOOM in [True, 'log'] and not aa == 0:
            negexp = int(np.ceil(np.log10(aa)))
            ss = '-' * bool(ff < 0) + r'$<$10$^{%d}$' % negexp
        elif isinstance(lowCutoffOOM, basestring):
            ss = lowCutoffOOM
    else:
        if aa >= 0.0001:
            ss = ('%.4f' % ff)
        if aa >= 0.001:
            ss = ('%.3f' % ff)
        if aa >= 0.01:
            ss = '%.3f' % ff
        if aa >= 0.1:
            ss = '%.2f' % ff
        if threeSigDigs and aa >= 0.1:
            ss = '%.3f' % ff
        if aa > 2.0:
            ss = '%.1f' % ff
        if aa > 10.0:
            ss = '%.1f' % ff
        if aa > 100.0:
            ss = '%.0f' % ff
        if ss[0:2] == '0.' and not leadingZeros:
            ss = ss[1:]
        if ss[0:3] == '-0.' and not leadingZeros:
            ss = '-' + ss[2:]
        if ss[0] == '-' and not noTeX:
            ss = '$-$' + ss[1:]

    # Override all this for integers:
    if isinstance(ff, int) and not ff == 0:
        round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1))
        ff = round_to_n(ff, sigdigs)
        #if ff>10**sigdigs:
        #    ff=int(np.round(ff % (10**sigdigs)))* (10**sigdigs)
        ss = '$-$' * bool(ff < 0) + str(abs(ff))

    return (conditionalWrapper[0] + ss + conditionalWrapper[1])


###########################################################################################
###
def latexFormatEstimateWithPvalue(x,
                                  pval=None,
                                  allowZeroSE=None,
                                  tstat=False,
                                  gray=False,
                                  convertStrings=True,
                                  threeSigDigs=None):
    ###
    #######################################################################################
    """
    This is supposed to encapsulate the colour/etc formatting for a single value and, optionally, its standard error or t-stat or etc. (take it out of formatpairedrow?)
    It's rather closely connected to cpbl's pystata package and latex_tables package, which produce statistical tables.

    It'll do the chooseSformat as well.

    x: a float (the estimate). There's no need to inform this method of the standard error, because that does not get coloured in cpbl's stat table format.


    May 2011.
    Still needs to learn about tratios and calculate own p...! if tstat==True
    """
    from pystata import significanceTable
    from textables import chooseSFormat
    yesGrey=gray
    if isinstance(x,list) or isinstance(x,tuple):
        assert len(x)==2
        est,ses= x # Primary estimate, and secondary t-stat/se/p-value
        singlet=False
    else:
        singlet=True
        est=x ###chooseSFormat(x,convertStrings=convertStrings,threeSigDigs=threeSigDigs)

    assert isinstance(pval,float) or pval in [] # For now, require p to be passed!

    if 0 and ses<1e-10 and not allowZeroSE:
        pair[0]=''
        pair[1]='' # Added Aug 2009... is not this right? It covers the (0,0) case.
    if pval not in [None,fNaN]: # This is if we specified p-values directly: then don't calculate it from t-stat, etc!
        significanceString=(['']+[tt[0] for tt in significanceTable if pval<= tt[2]*1.0/100.0])[-1]

    if significanceString and yesGrey:
            significanceString=r'\agg'+significanceString[1:]
    if not significanceString and yesGrey:
            significanceString=r'\aggc{'
    if yesGrey:
        greyString=r'\aggc'
    if singlet:
        return(significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs)+'}'*(not not significanceString))

    return([significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs)+'}'*(not not significanceString),
           significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs,conditionalWrapper=[r'\coefp{','}'])])




###########################################################################################
###
def latexFormatEstimateWithPvalue(x,
                                  pval=None,
                                  allowZeroSE=None,
                                  tstat=False,
                                  gray=False,
                                  convertStrings=True,
                                  threeSigDigs=None):
    ### Why is this in pystata.py? Becuase it needs significanceTable.
    #######################################################################################
    """
    This is supposed to encapsulate the colour/etc formatting for a single value and, optionally, its standard error or t-stat or etc. (take it out of formatpairedrow?)

    It'll do the chooseSformat as well.

    May 2011.
    Still needs to learn about tratios and calculate own p...! if tstat==True
    """
    yesGrey = gray
    if isinstance(x, list):
        assert len(x) == 2
        est, ses = x  # Primary estimate, and secondary t-stat/se/p-value
        singlet = False
    else:
        singlet = True
        est = x  ###chooseSFormat(x,convertStrings=convertStrings,threeSigDigs=threeSigDigs)

    assert isinstance(pval, float) or pval in [
    ]  # For now, require p to be passed!

    if 0 and ses < 1e-10 and not allowZeroSE:
        pair[0] = ''
        pair[1] = ''  # Added Aug 2009... is not this right? It covers the (0,0) case.
    if pval not in [
            None, fNaN
    ]:  # This is if we specified p-values directly: then don't calculate it from t-stat, etc!
        significanceString = (
            [''] +
            [tt[0] for tt in significanceTable if pval <= tt[2] * 1.0 / 100.0]
        )[-1]

    if significanceString and yesGrey:
        significanceString = r'\agg' + significanceString[1:]
    if not significanceString and yesGrey:
        significanceString = r'\aggc{'
    if yesGrey:
        greyString = r'\aggc'
    if singlet:
        return (significanceString + chooseSFormat(
            est, convertStrings=convertStrings, threeSigDigs=threeSigDigs) +
                '}' * (not not significanceString))
    # By default, I don't think we want the standard errors/etc to contain the colour/significance formatting.
    if 0:
        return ([
            significanceString + chooseSFormat(
                est, convertStrings=convertStrings,
                threeSigDigs=threeSigDigs) + '}' *
            (not not significanceString), significanceString + chooseSFormat(
                ses,
                convertStrings=convertStrings,
                threeSigDigs=threeSigDigs,
                conditionalWrapper=[
                    r'\coefp{',
                    '}' + '}' * (not not significanceString),
                ])
        ])
    else:
        return ([
            significanceString + chooseSFormat(
                est, convertStrings=convertStrings, threeSigDigs=threeSigDigs)
            + '}' * (not not significanceString), chooseSFormat(
                ses,
                convertStrings=convertStrings,
                threeSigDigs=threeSigDigs,
                conditionalWrapper=[
                    r'\coefp{',
                    '}',
                ])
        ])

def formatPairedRow_DataFrame(df, est_col, se_col, pvalue_col=None, prefix=None,
                              drop_original_columns = False,
                              **varargs):
    """
    The name of this function is not great, but it simply applies formatPairedRow to two columns in a dataframe, returning the df with two new formatted string columns. The formatting is for use in cpblTables.

2017: What about a tool to take every other column and stick them as alternate rows? See interleave_columns_as_rows in cpblutils
(See #4 in pystata github.)
    
    est_col and se_col are column names. They can be strings or (for a MultiIndex column) tuples 

    est_col and se_col can be tuples that reference a MultiIndex column. In that case, prefix would need to have the same length as est_col (ie the number of levels in the MultiIndex)

    There are two calling forms (just like for formatPairedRow):

    formatPairedRow_DataFrame(df, e, s):   Specify est_col and se_col in order to infer p-values from the standard errors.

    formatPairedRow_DataFrame(df, e, s, p):   In addition, specify the p-value to be used to format the estimate (e).

    """
    #assert df[est_col].notnull().all()
    #assert df[est_col].notnull().all()
    #a,b = formatPairedRow([df[est_col].fillna('.').values.tolist(), df[se_col].fillna('.').values.tolist()])
    if isinstance(est_col, tuple) or isinstance(se_col, tuple) or isinstance(pvalue_col, tuple):
        assert  isinstance(est_col, tuple) and isinstance(se_col, tuple) 
        assert len(est_col) == len(se_col)

    # For p-value not specified:
    if pvalue_col is None:
        s1, s2 = formatPairedRow(
            [df[est_col].values.tolist(), df[se_col].values.tolist()], **varargs)
    else: # if p-value is specified (different calling form)
        s1, s2 = formatPairedRow(
            df[est_col].values.tolist(), df[se_col].values.tolist(), df[pvalue_col].values.tolist(), **varargs)
            
    if prefix is None and isinstance(est_col,basestring):
            prefix_est_col, prefix_se_col = 's'+est_col, 's'+se_col
            assert prefix_est_col not in df
            assert prefix_se_col not in df
            
    if prefix is None and isinstance(est_col, tuple):
            prefix = tuple([''] * (len(est_col)-1)) + ('s',)
            prefix_est_col = tuple([''.join(a) for a in zip(prefix, est_col)])
            prefix_se_col = tuple([''.join(a) for a in zip(prefix, se_col)])

            
    df[prefix_est_col] = s1
    df[prefix_se_col] = s2
    return (df)


###########################################################################################
###
def formatPairedRow(pair,
                    pValues=None,
                    greycells=None,
                    modelsAsRows=None,
                    varsAsRows=None,
                    allowZeroSE=False):
    ###
    #######################################################################################
    """ August 2009.

    Takes a pair of entries. First in each row of pair is a label; typicall that in the second row (label for s.e.'s) is blank. Returns pair of lists of LaTeX strings.


Takes care of significance marking; grey shading, and formatting row/column headers. (maybe.. not implemented yet.  For the latter, it would need to know whether rows or cols..)

pairedRows were conventionally one model? one variable?

greycells: can list column numbers? can be True ie for all in this model.

needs to be in charge of making row/model header grey too, if greycells==True  [done!]

    Dec 2009: allowZeroSE: set this to true when the values passed are sums, etc, which may be exact (is SE=standard error  is 0).


? April 2010. Trying to get multirow to work with colortbl.  Why was I not using rowcolor{} ?? Try to implement that now/here... hm.. no, kludging it in the calling function for now?


May 2011: can now also send an array of pvalues. This avoids calculating p categories from t-stats. Also, there may not be t-stats, as in the case of suestTests: ie test results.

May 2011: I'm trying to remove some of the logic from here to a utility, latexFormatEstimateWithPvalue(x,pval=None,allowZeroSE=None,tstat=False,gray=False,convertStrings=True,threeSigDigs=None)
    """

    from pylab import isnan
    if not greycells:
        greycells = []
    if greycells == True:
        greycells = range(len(pair[0]))

    outpair = deepcopy(pair)
    coefs = pair[0]
    ses = pair[1]

    def isitnegative(obj):
        if not isinstance(obj, float): return False
        return obj < 0

    assert not any([isitnegative(ss) for ss in ses])

    # If it's not regression coefficients, we may want to specify the p-values (and thus stars/colours) directly, rather than calculating t values here). Indeed, aren't p-values usually available, in which case they should be used anyway? hmm.
    if pValues is None:
        pValues = [None for xx in coefs]
    assert len(pValues) == len(coefs)

    # Is following redundant?!
    significanceString, greyString = [[] for i in range(len(pair[0]))
                                      ], [[] for i in range(len(pair[0]))]
    for i in range(len(pair[0])):
        significanceString[i], greyString[i] = '', ''
    for icol in range(len(pair[0])):
        yesGrey = icol in greycells or greycells == True
        if isinstance(coefs[icol], basestring) or isinstance(
                coefs[icol], unicode) or isnan(coefs[icol]):
            if coefs[icol] in ['nan', fNaN] or (isinstance(coefs[icol], float)
                                                and isnan(coefs[icol])):
                outpair[0][icol], outpair[1][
                    icol] = r'\aggc' * yesGrey, r'\aggc' * yesGrey
            else:
                outpair[0][icol], outpair[1][icol] = coefs[
                    icol] + r'\aggc' * yesGrey, ses[icol] + r'\aggc' * yesGrey
            continue
        # So we have floats
        # Aug 2012: Agh.. It's in principle possible to have an int, with 1e-16 s.e., etc.
        assert isinstance(outpair[0][icol], float) or isinstance(
            outpair[0][icol], int)
        assert isinstance(outpair[1][icol], float) or isinstance(
            outpair[0][icol], int)

        # Safety: so far this is only because of line "test 0= sumofincomecoefficients": when that fails, right now a value is anyway written to the output file (Aug 2008); this needs fixing. In the mean time, junk coefs with zero tolerance:
        if ses[icol] < 1e-10 and not allowZeroSE:  #==0 or : # Added sept 2008...
            pair[0][icol] = ''
            pair[1][
                icol] = ''  # Added Aug 2009... is not this right? It covers the (0,0) case.
        if isinstance(coefs[icol], float) and not str(
                coefs[icol]) == 'nan':  # and not ses[icol]==0:
            tratio = abs(coefs[icol] / ses[icol])
            # Ahh. May 2010: why is there a space below, rather than a nothing? Changning it.
            #significanceString[icol]=([' ']+[tt[0] for tt in significanceTable if tratio>= tt[1]])[-1]
            significanceString[icol] = (
                [''] + [tt[0] for tt in significanceTable if tratio >= tt[1]]
            )[-1]

        if pValues[icol] not in [
                None, fNaN
        ]:  # This is if we specified p-values directly: then don't calculate it from t-stat, etc!
            significanceString[icol] = ([''] + [
                tt[0] for tt in significanceTable
                if pValues[icol] <= tt[2] * 1.0 / 100.0
            ])[-1]

        if significanceString[icol] and yesGrey:
            significanceString[icol] = r'\agg' + significanceString[icol][1:]
        if not significanceString[icol] and yesGrey:
            significanceString[icol] = r'\aggc{'
        if yesGrey:
            greyString[icol] = r'\aggc'

        # Changing line below: May 2011. Why did I put a ' ']
        #outpair[0][icol]= significanceString[icol]+chooseSFormat(coefs[icol])+'}'*(not not significanceString[icol])

        outpair[0][icol] = significanceString[icol] + chooseSFormat(
            coefs[icol]) + '}' * (not not significanceString[icol])
        #debugprint( 'out0=',        outpair[0][icol],
        #'   signStr=',(not not significanceString[icol]),significanceString[icol],
        #'   tratio=',tratio,coefs[icol],ses[icol])

        # This catches a 2017 bug:
        ###NO!assert all([isitnull(outpair[0][ii]) == isitnull(coefs[ii])  for ii in range(len(outpair[0]))])

        if ses[icol] < 1e-10 and allowZeroSE:  # Added Dec 2009
            ses[icol] = 0

        outpair[1][icol] = chooseSFormat(
            ses[icol],
            conditionalWrapper=[r'\coefse{', '}']) + greyString[icol]
        # Shoot. May 2011, following line suddently started failing (regressionsQuebec.py: nested models). Can't fiure it out. I'm disabling this for the moment!! I guess try some asserts instead.
        # Okay, I fixed it by rephrasing using "in": [may 2011]
        # May 2011: Following FAILS to detect nan's. need to use isinstnace float and isnan as another option. Also, this should not be triggered for suestTests.
        if ses[icol] in ['', float('nan')] and not coefs[icol] in [
                '', float('nan')
        ]:  #isnan(ses[icol]) and not isnan(coefs[icol]):
            outpair[0][icol] = chooseSFormat(coefs[icol])
            outpair[1][icol] = '?'
            print ' CAUTION!! VERY STRANGE CASE OF nan FOR s.e. WHILE FLOAT FOR COEF '
            assert 0
    return (outpair)

#below: a t-distribution with infinitely-many degrees of freedom is a normal distribution...
TstatsSignificanceTable = [
    [-0.1, 0.00001],  # Backstop
    [0.0, 0.00001],
    [0.5, 0.674],
    [0.6, 0.841],
    [0.7, 1.036],
    [0.80, 1.28155],
    [0.90, 1.64485],
    [0.95, 1.95996],
    [0.96, 2.054],
    [0.98, 2.32635],
    [0.99, 2.57583],
    [0.995, 2.80703],
    [0.998, 3.09023],
    [0.999, 3.29052],
]
significanceTable = [
    # Latex format, t-ratio, tolerance percentage
    [r'', 0, 100.0],
    [r'\signifTenPercent', 1.64485, 10.0],
    [r'\signifFivePercent', 1.95996, 5.0],
    [r'\signifOnePercent', 2.57583, 1.0],
    [r'\wrapSigOneThousandth{', 3.291, 0.1],
    #r'', 999999999999999999999999999
]
significanceTable = [
    # Latex format, t-ratio, tolerance percentage
    [r'', 0, 100.0],
    [r'\wrapSigTenPercent{', 1.64485, 10.0],
    [r'\wrapSigFivePercent{', 1.95996, 5.0],
    [r'\wrapSigOnePercent{', 2.57583, 1.0],
    [r'\wrapSigOneThousandth{', 3.291, 0.1],
    #r'', 999999999999999999999999999
]
tsvSignificanceTable=[# for spreadsheet (tsv/csv) format: ie just text
    # stars, t-ratio, tolerance percentage
    [r'',0  ]  ,
    [r'*',  1.64485, 10],
    [r'**',  1.95996, 5],
    [r'***',  2.57583, 1],
    [r'****',  3.291,0.1],
    #r'', 999999999999999999999999999
]
"""
    Here are t-stats for one-sided and two sided.
    One Sided	75%	80%	85%	90%	95%	97.5%	99%	99.5%	99.75%	99.9%	99.95%
    Two Sided	50%	60%	70%	80%	90%	95%	98%	99%	99.5%	99.8%	99.9%
               0.674	0.842	1.036	1.282	1.645	1.960	2.326	2.576	2.807	3.090	3.291

    """

# this below is obselete?
# Maybe the following "significanceLevels" could be set externally by a call to this object/module.
significanceLevels = [
    90, 95, 99
]  # This is used for the command below, as well as for the colour/star replacement and corresponding legend!
significanceLevels = [
    100 - ss[2] for ss in significanceTable[1:]
]  # This is used for the command below, as well as for the colour/star replacement and corresponding legend!


if __name__ == '__main__':
    test_pvalue_to_latex_significance()    
    test_df_format_estimate_column_with_latex_significance()
    print ('All tests passed')
