#!/usr/bin/python
import statsmodels.formula.api as sm
import pandas as pd
import numpy as np

def df_impute_values_ols(adf,outvar,model,  verbose=True):
    """
    Specify a Pandas DataFrame with some null (eg. np.nan) values in column <outvar>.
    Specify a string model (in statsmodels format, which is like R) to use to predict them when they are missing. Nonlinear transformations can be specified in this string.

    e.g.: model='  x1 + np.sin(x1) + I((x1-5)**2) '

    A version of df with some null values filled in (those that have the model variables) will be returned, using single imputation.
    At the moment, this uses OLS, so outvar should be continuous.

    AAh!! This bug https://github.com/statsmodels/statsmodels/issues/2171 wasted me hours. :( The latest release of Statsmodels is over a year old...
    So this is written in order to avoid ANY NaN's in the modeldf. That should be less necessary in future versions.

    To do: Add a verbose mode with plots 

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
    return(adf)
