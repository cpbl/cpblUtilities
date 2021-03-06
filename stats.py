#!/usr/bin/python
import statsmodels.formula.api as sm
import pandas as pd
import numpy as np

def df_impute_values_ols(adf,outvar,model,  verbose=True):
    """Specify a Pandas DataFrame with some null (eg. np.nan) values in column <outvar>.
    Specify a string model (in statsmodels format, which is like R) to use to predict them when they are missing. Nonlinear transformations can be specified in this string.

    e.g.: model='  x1 + np.sin(x1) + I((x1-5)**2) '

    At the moment, this uses OLS, so outvar should be continuous. 

    Two dfs are returned: one containing just the updated rows and a
    subset of columns, and version of the incoming DataFrame with some
    null values filled in (those that have the model variables) will
    be returned, using single imputation.

    AAh!! This bug https://github.com/statsmodels/statsmodels/issues/2171 wasted me hours. :( The latest release of Statsmodels is over a year old...
    So this is written in order to avoid ANY NaN's in the modeldf. That should be less necessary in future versions.

    To do: 
    - Add plots to  verbose mode 
    - (It should have a discrete or at least binary outvar model option)

    Issues:
    - the "horrid kluge" line below will give trouble if there are column names that are part of other column names.

    """
    formula=outvar+' ~ '+model
    rhsv=[vv for vv in adf.columns if vv in model] # This is a horrid kluge! Ne
    updateIndex= adf[pd.isnull(adf[outvar]) ] [rhsv].dropna().index
    modeldf=adf[[outvar]+rhsv].dropna()
    results=sm.ols(formula, data=modeldf).fit()
    if verbose:
        print    results.summary()
    newvals=adf[pd.isnull(adf[outvar])][rhsv].dropna()
    newvals[outvar] = results.predict(newvals)
    adf.loc[updateIndex,outvar]=newvals[outvar]
    if verbose:
        print(' %d rows updated for %s'%(len(newvals),outvar))
    return(newvals, adf)


def test_df_impute_values_ols():
    # Find missing values and fill them in:
    df = pd.DataFrame({"A": [10, 20, 30, 324, 2353, np.nan],
                       "B": [20, 30, 10, 100, 2332, 2332],
                       "C": [0, np.nan, 120, 11, 2, 2 ]})
    newv,df2=df_impute_values_ols(df,'A',' B + C ',  verbose=True)
    print df2
    assert df2.iloc[-1]['A']==2357.5427562610648
    assert df2.size==18

    # Can we handle some missing values which also have missing predictors?
    df = pd.DataFrame({"A": [10, 20, 30,     324, 2353, np.nan, np.nan],
                       "B": [20, 30, 10,     100, 2332, 2332,   2332],
                       "C": [0, np.nan, 120, 11,   2,    2,     np.nan ]})
    newv,df2=df_impute_values_ols(df,'A',' B + C + I(C**2) ',  verbose=True)
    print df2

    assert pd.isnull(  df2.iloc[-1]['A'] )
    assert  df2.iloc[-2]['A'] == 2352.999999999995



