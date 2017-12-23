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
