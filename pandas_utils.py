#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
import os
import re
from copy import deepcopy
import time
"""
import sys
import pandas as pd
import numpy as np
from cpblUtilities.mathgraph import weightedMeanSE

def str2df(tss):
    """ Read a tab-separated string as though it's a text file.
The first line must contain the column names.
    """
    if sys.version_info[0] < 3: 
        from StringIO import StringIO
    else:
        from io import StringIO
    return pd.read_table(StringIO(tss.strip('\n')))

def rectangularize_index(df):
    # Create a rectangular index of index columns:
    fofofo
    blank =pd.DataFrame(index=pd.MultiIndex.from_product([
        sorted(df[yearvar].unique()), sorted(df[groupvar].unique())], names= [yearvar,groupvar]))
    dfsq = blank.join(df.set_index([yearvar, groupvar]))[cols].reset_index()
    

def first_differences_by_group(df, groupvar,yearvar):
    """ This may be esoteric, but: rectangularize an index, and take first differences within each group.  

    Takes a DataFrame; returns the same but with extra columns and possibly new rows with NaN values.
    """
    # Assume index has been reset. Here are the data columns:
    cols = [cc for cc in df.columns if cc not in [groupvar,yearvar]]
    # Create a rectangular index of all group and year values:
    blank =pd.DataFrame(index=pd.MultiIndex.from_product([
        sorted(df[yearvar].unique()), sorted(df[groupvar].unique())], names= [yearvar,groupvar]))
    dfsq = blank.join(df.set_index([yearvar, groupvar]))[cols].reset_index()
    # Next, we can simply take differences:
    dfsq.sort_values([groupvar,yearvar], inplace=True)
    for var in cols:
        dfsq['d_'+var] =dfsq.groupby([groupvar])[var].transform(lambda x: x.diff())
    return dfsq

def correlation_table(df, texfile=None):
    """
    Generates a correlation table of columns, and a table of standard errors, and of p-values,  and creates a cpblTables LaTeX output table.
    """
    
    #construct two arrays, one of the correlation and the other of the p-vals
    rho = df.corr()
    pval = np.zeros([df.shape[1],df.shape[1]])
    for i in range(df.shape[1]): # rows are the number of rows in the matrix.
        for j in range(df.shape[1]):
            JonI        = pd.ols(y=df.icol(i), x=df.icol(j), intercept=True)
            pval[i,j]  = JonI.f_stat['p-value']


def weightedMeanSE_pandas(df,varNames=None ,weightVar='weight', uniform_sample=None, aggstats=None, as_df=False,
                          sem_name = 'se_mean',
                          statname='statname', # default name for index column
):
    """
    uniform_sample=True drops NaNs for all variables at once (ie keeps only rows with finite values for all columns). Otherwise, NaNs are dropped column-by-columnn.

    varNames: columns to aggregate

    aggstats: by default, aggstats is ['mean'] and produces columns with the original name and with a 'se_' suffixe for the standard error of the mean.  Other stats not yet implemented

    An alternative output format, in which a multiindex is used instead of prefixes, can be had with...

    Example use: 

    # aggregate by age:
    means=df.groupby('age').apply(lambda adf: weightedMeanSE_pandas(adf,tomean,weightVar='cw')).reset_index()
    
    # Aggregate all variables except weight:
    weightedMeanSE_pandas(df,weightVar='cw')


N.B. Can I use  following instead?
    from statsmodels.stats.weightstats import DescrStatsW
    weighted_stats = DescrStatsW(dfdna[xv], weights=dfdna['Population'])#, ddof=0) 

N.B.: If I want to use the unbiased version of variance, will need to specify whether the weights are frequency or reliablity weights.

    """
    assert aggstats is None or all([ags in ['mean'] for ags in aggstats])
    outs={}
    if weightVar is None:
        weightVar= '___tmpweight'
        df[weightVar] = 1
        
    if varNames is None: varNames = df.columns
    if isinstance(varNames,basestring):
        varNames=[varNames]
    if uniform_sample:
        df=df.copy()
        df = df[varNames+[weightVar]].dropna() # Should this be: df2=df[np.isfinite(df[[mv,weightVar]]).all(axis=1)]

    varPrefix=''
    for mv in varNames:
        df2=df[np.isfinite(df[[mv,weightVar]]).all(axis=1)]
        mu,se=weightedMeanSE(df2[mv], df2[weightVar]) # Gives same as Stata
        variance = np.average( (df2[mv]-mu)**2, weights = df2[weightVar])
        # Values need to be vectors(lists) for the conversion to DataFrame, it seems.
        if as_df:
            res1={'mean': mu,   sem_name: se, 'N': len(df2), 'min': df2[mv].min(), 'max': df2[mv].max(), 'std': np.sqrt(variance) }
            #res = zip(*{'mean': mu,   'sem': se, 'N': len(df2), 'min': df2[mv].min(), 'max_': df2[mv].max() }.items())
            outs.update({mv:pd.DataFrame(res1, index = [mv])})
        else:
            outs.update({varPrefix+mv: mu,
                         varPrefix+'se_'+mv: se,
                         varPrefix+'N_'+mv: len(df2),
                         varPrefix+'min_'+mv: df2[mv].min(),
                         varPrefix+'max_'+mv: df2[mv].max(),
                         varPrefix+'std_'+mv: np.sqrt(variance)             })
    if weightVar == '___tmpweight':
        df = df[[cc for cc in df.columns if not cc==weightVar]]
    if as_df:
        odf= pd.concat(outs.values())
        odf.index.rename('statname', inplace=True)
        return odf
    else:
        return(pd.Series(outs)) # Return this as a Series; this avoids adding a new index in groupby().apply

def test_weightedMeansByGroup():
    import pandas.util.testing as tm; tm.N = 3
    # Not written yet
    fofofo
    weightedMeansByGroup(pandasDF,meansOf=None,byGroup=None,weightVar='weight',varPrefix='',)
        
##############################################################################
##############################################################################
#
def weightedMeansByGroup(pandasDF,meansOf=None,byGroup=None,weightVar='weight',varPrefix='',
                         groups_to_columns = False,
                         wide = False):# varsByQuantile=None,suffix='',skipPlots=True,rankfileprefix=None,ginifileprefix=None,returnFilenamesOnly=False,forceUpdate=False,groupNames=None,ginisOf=None,loadAll=None,parallelSafe=None):
    #
    ##########################################################################
    ##########################################################################
    """
    This returns another DataFrame with appropriately named columns.

    groups_to_columns = True  induces as_df=True in weightedMeanSE_pandas., and restacks so that the rows are the variables given in "meansOf", and the columns are a multiindex of groups and statistics (mean, se_mean, std, N, min, max).  You may want to do something like .reorder_levels([1,2,0], axis=1).sortlevel(level=0, axis=1, sort_remaining=True) to the result.

    wide=True: This returns the results in a simple wide format, with no row index

This gives the same s.e. as Stata.  (The list of what does NOT is very long: my ( wtmean,wtsem,wtvar), above.)
statsmodels.WLS gives robust standard errors, I guess, but they ignore the weights! ie give same as stata without any weights. Wow, how poor.  But the code someone gave me on a list in 2010 works.

2013 Feb: I'm adapting this from weightedQuantilesByGroup.
My related functions: 2013July I need a weighted moment by group in recodeGallup.

2014June (Usually you can get by without this function?:     means=df.groupby('age').apply(lambda adf: weightedMeanSE_pandas(adf,tomean,weightVar='cw')) )

    """
    df=pandasDF
    #from scipy import stats
    if isinstance(byGroup,str):
        byGroup=byGroup.split(' ')
    if isinstance(meansOf,str):
        meansOf=meansOf.split(' ')
    if meansOf is None:
        meansOf = [cc for cc in pandasDF.columns if cc not in byGroup+[weightVar]]

    assert all([mm in df for mm in meansOf])
    if byGroup is None:
        raise(Error('Just use weightedMeanSE_pandas if there are no groups to average over'))

    assert all([mm in df for mm in byGroup])
    grouped=df.groupby(byGroup)

    if groups_to_columns or wide:
        newdf = grouped.apply(weightedMeanSE_pandas, meansOf, weightVar,  None, None, True) #.unstack(level=[0, 1])
        if wide:
            newdf.reset_index(inplace=True)
        if groups_to_columns:
            def rotate(l):
                return l[1:]+l[:1]
            # This puts the groups at the top of the columns multiindex, and has all stats together for each group:
            newdf = newdf.unstack(level= range(len(byGroup))).reorder_levels(rotate(range(1+len(byGroup))), axis=1).sortlevel(level=0, axis=1, sort_remaining=True)
            print('NOOoooooo! sortlevel is deprecated')
    else:
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



Sample usage: sprawl/cpblUtilities.mathgraph.py:    newdf=grouped.apply(weightedMeanSE_pandas,meansOf,weightVar)

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

2013 Jan: This is derived from pystata's generateRankingData() for stata data, but this one takes pandas DataFrame instead. And we don't include ginis! (ugh). And leave plotting to a separate function, since we could return data.

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



def weightedSTD_pandas(df,varNames,weightVar='weight'):
    """
    Adapted from above, but not tested versus stata...
    Example use: aggregate by age:
    means=df.groupby('age').apply(lambda adf: weightedSTD_pandas(adf,tomean,weightVar='cw')).reset_index()

    from statsmodels.stats.weightstats import DescrStatsW
    outs={}
    if isinstance(varNames,basestring):
        varNames=[varNames]
    varPrefix=''
    for mv in varNames:
        df2=df[np.isfinite(df[[mv,weightVar]]).all(axis=1)]
        mu,se=weightedMeanSE(df2[mv], df2[weightVar]) # Gives same as Stata!
        weighted_stats = df[xv].apply(DescrStatsW,df.Population']]., weights=weights, ddof=0)
    

        # Values need to be vectors(lists) for the conversion to DataFrame, it seems.
        outs.update({varPrefix+mv: mu,   varPrefix+'se_'+mv: se})
    return(pd.Series(outs)) # Return this as a Series; this avoids adding a new index in groupby().apply
    """

    










