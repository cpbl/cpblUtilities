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


def str2df(tss):
    """ Read a tab-separated string as though it's a text file.
The first line must contain the column names.
    """
    if sys.version_info[0] < 3: 
        from StringIO import StringIO
    else:
        from io import StringIO
    return pd.read_table(StringIO(tss.strip('\n')))

def df_first_differences_by_group(df, groupvar,yearvar):
    """ This may be esoteric, but: rectangularize an index, and take first differences within each group.
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
