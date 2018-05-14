#!/usr/bin/python

# Actually, I think each method in mathgraph.py should have a test_method() in mathgraph.py. This module here should simply call all those tests.
from cpblUtilities.mathgraph import *


df = pd.DataFrame({"A": [10, 3,4,5,6,7,20, 30, 324, 2353, np.nan],
                   "B": [20, 4,3,6,5,7.1,30, 10, 100, 2332, 2332],
                   'w':[14, 10,10,10,10,10,    14,20,14,14,14,],
                    })

if 0: 
    df1=df.dropna()
    plot(df1.A,df1.B,'.')
    dfOverplotLinFit(df1,'A','B',label=None)#
if 0: 
    import seaborn as sns
    tips = sns.load_dataset("tips")
    dfOverplotLinFit(tips, 'total_bill','tip',label=None)#

x= np.random.randn(100)
df2=pd.DataFrame({"A": x,
                  "B":  x + np.random.randn(100) + 10})
plt.plot(df2.A,df2.B,'.')
dfOverplotLinFit(df2,'A','B',label=None)#
plt.show()

# This was supposed to be for developing weighted, with ci visualisation. Not done.
