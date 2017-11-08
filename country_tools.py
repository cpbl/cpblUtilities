#!/usr/bin/python
"""
Various tools for use with osm analysis

"""
import os, sys #, platform, time, psutil
import pandas as pd
import numpy as np


class country_tools():
    def __init__(self, forceUpdate=False):
        """
        allc = pd.read_csv(paths['otherinput']+'CountryData/raw.githubusercontent.com_lukes_ISO-3166-Countries-with-Regional-Codes_master_all_all.csv')

         # https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv
         # https://github.com/lukes/ISO-3166-Countries-with-Regional-Codes/blob/master/all/all.csv ie https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv, now in CountryData as raw.githubusercontent.com_lukes_ISO-3166-Countries-with-Regional-Codes_master_all_all.csv

        """
        self.forceUpdate=forceUpdate
        SP='/tmp/' # paths['scratch']
        self.useful_sites={'lukes':{'tmpfile':SP+'countryTools_lukes_tmp.pandas', 'url': 'https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv',
                                    'rename':{'region':'region5', 'sub-region':'region22', 'name':'countryName'}},
                           # N.B. The following includes six languages for all countries,  not every country's official language
                     'multilang':{'tmpfile':SP+'countryTools_gitdatasets_tmp.pandas', 'url': 'https://raw.githubusercontent.com/datasets/country-codes/master/data/country-codes.csv',
                                  'rename':{}}}

        return
    def update_downloads(self):
        for kk,dd in self.useful_sites.items():
            if self.forceUpdate or not os.path.exists(dd['tmpfile']):
                # Keep Pandas from translating Namibia's "NA" to NaN:
                tdf = pd.read_csv(dd['url'],na_filter = False).to_pickle(dd['tmpfile']) #, encoding='utf8'
                print('   Loaded '+ dd['url'])
    def load_and_rename_cols(self,key):
        """ Load one of the files in useful_sites, and apply the column renamings
        """
        self.update_downloads()
        df = pd.read_pickle(self.useful_sites[key]['tmpfile'])
        df.rename(columns = self.useful_sites[key]['rename'], inplace=True)
        return df
        
    def get_country_list(self, columns=None):
        """ columns can be used to imply which info you need. See the useful_sites "rename"s for column names
        """
        df = self.load_and_rename_cols('lukes')
        assert df[pd.isnull(df.region5)].empty
        ###df = df[pd.notnull(df.region)]
        df['iso2'] =df['iso_3166-2'].str[-2:].values
        df['ISOalpha3'] =df['alpha-3']
        if columns is not None:
            assert all([cc in df for cc in columns])
            df=df[columns]

        #df2=self.load_and_rename_cols('multilang')
        return df


# See / integrate country2... in osmTools.py
"""
 country2ISOLookup(): could use multilang one! 
"""

################################################################################################
if __name__ == '__main__':
################################################################################################
    # Test everything
    cc = country_tools(forceUpdate = True)
    cc.get_country_list()
    
