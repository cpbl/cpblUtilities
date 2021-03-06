#!/usr/bin/python
"""
Various tools for use with OSM and other country indices / matching.

Also see cpbl's whr2017 for some WHR-related matching.

"""
import os, sys #, platform, time, psutil
import pandas as pd
import numpy as np
#from .cpblUtilities_config import paths
from ..cpblUtilities_config import defaults
paths = defaults['paths']

__local_input_path__ = os.path.dirname(__file__)+'/' 
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
    @staticmethod
    def country2ISOLookup():
        """
        Returns dict of dicts
        cname2ISO is country name to ISO alpha 3 lookup
        ISO2cname is the reverse
        ISO2shortName is a consistent dict of short names (they might not be officials)
            which also strip out non-ASCII128 characters (sorry, Cote d'Ivoire...)

        This could easily be adapted to return WB codes as well
        """
        import pandas as pd
        lookup = pd.read_table(os.path.dirname(__file__)+'/nationsonline.org.tsv',         # Are these OSM-oriented??
                               dtype={'ISO3digit=UN-M49-Numerical':'str'},
                               encoding='utf8')
        lookup['ISO3digit'] =lookup['ISO3digit=UN-M49-Numerical']
        oldCPBLlookup_unused_here_so_far = pd.read_table(os.path.dirname(__file__)+'/countrycode_main.tsv',
                               dtype={'ISO3digit':'str'},
                               skiprows =3)
        # The following, since we append them to the end, probably has the effect of overwriting country names to these simpler ones in ISO2cname, etc.
        # Another way to do this would be to keep creating new columns of alternative names, and then when doing a name index, use them all, but otherwise have a preferred one (the shortest!?)
        equivalents = zip(*[LL.split('\t') for LL in """
Hong Kong, SAR China	Hong Kong
Macao, SAR China	Macao
Iran, Islamic Republic of	Iran
Macedonia, Republic of	Macedonia
Micronesia, Federated States of	Micronesia
Taiwan, Republic of China	Taiwan
Tanzania, United Republic of	Tanzania
Korea (North)	North Korea
Korea (South)	South Korea
Palestinian Territory	Palestine
Russian Federation	Russia
Syrian Arab Republic (Syria)	Syria
United Kingdom	Great Britain
Venezuela (Bolivarian Republic)	Venezuela""".strip().split('\n')])
        eq=pd.DataFrame({'countryname': equivalents[0], 'newname':equivalents[1]})
        edf = lookup[lookup.countryname.isin(eq.countryname)].merge(eq)
        edf['countryname']= edf['newname']
        lookup = pd.concat([lookup, edf[lookup.columns] ] )

        cname2ISO = lookup.set_index('countryname').ISOalpha3.to_dict()

        ISO2cname = lookup.set_index('ISOalpha3').countryname.to_dict()
        ISO2cname['ALL'] = 'World'
        ISOalpha2ISOdigit = lookup.set_index('ISOalpha3').ISO3digit.to_dict()
        ISOdigit2ISOalpha = lookup.set_index('ISO3digit').ISOalpha3.to_dict()

        lookup = pd.read_table(os.path.dirname(__file__)+'/shortNames.tsv', dtype={'ISO3digit':'str'})
        ISOalpha2shortName = lookup.set_index('ISOalpha3').shortName.to_dict()
        ISOalpha2shortName['ALL'] = 'World'

        # Add in some alternate names manually (different variants of country name)
        # these are only used in going TO iso from the country name
        # We could also do some automatically, based on country names with parentheses in them.
        # Most of these should be added instead as equivalents, above.
        cname2ISO.update({'The Bahamas': 'BHS', 'United Republic of Tanzania': 'TZA', 'Ivory Coast': 'CIV', 
                          'Republic of Serbia': 'SRB', 'Guinea Bissau': 'GNB', 'Iran': 'IRN', 'Democratic Republic of the Congo': 'COD',
                          'Republic of Congo': 'COG',  'Bolivia': 'BOL', 'South Korea': 'KOR', 'Laos': 'LAO',
                          'Brunei': 'BRN', 'East Timor': 'TLS', 'Vietnam': 'VNM',  'North Korea': 'PRK', 'Moldova': 'MDA', 'Vatican City': 'VAT',
                          'Macedonia': 'MKD', 'United Kingdom': 'GBR', 'Tanzania':'TZA', 'Cape Verde':'CPV', 'Reunion':'REU', 'Falkland Islands':'FLK', 
                          'Micronesia':'FSM', 'United States':'USA'})
        for kk,vv in cname2ISO.items(): cname2ISO[kk.lower()] = vv
        
        
        return {'cname2ISO':cname2ISO, 'ISO2cname': ISO2cname, 'ISOalpha2ISOdigit': ISOalpha2ISOdigit, 'ISOdigit2ISOalpha': ISOdigit2ISOalpha, 'ISOalpha2shortName':ISOalpha2shortName }
    
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

    @staticmethod
    def get_Gallup_country_lookups(verbose=True):
        """ Kosovo is the only GWP country not matched to a 3-letter ISO code. Let's ignore it.
        """
        dfr = pd.read_table(__local_input_path__+'GallupWorldPoll-region-country.tsv').rename(columns={'country':'rcountry'})
        dfr['lccountry'] = dfr.rcountry.str.lower()
        dfr = dfr.set_index('lccountry')
        dfw = pd.read_table(__local_input_path__+'GallupWorldPoll-WP5-defs-2016.tsv').rename(columns={'country':'wcountry'})
        dfw['lccountry'] = dfw.wcountry.str.lower()
        dfw = dfw.set_index('lccountry')
        wp5s = pd.read_table(__local_input_path__ +'countrycode_main.tsv',  skiprows=3).set_index('country_GWP3_wp5')
        wp5s = wp5s[['countryCode_GWP3_wp5', 'countryCode_ISO3','country_bestShortName','country_bestName','twoletter_AlexShultz_svg']]
        df= wp5s.join(dfr).join(dfw).rename(columns = {'countryCode_ISO3':'ISO',})
        df.index.name = 'country'
        assert 'South Africa'.lower() in dfr.rcountry
        assert 'South Africa'.lower() in df.index


        # Now several checks:
        # Did regions get their ISO?
        problems = {
            ' Published WHR country lacks an ISO: ': df[pd.notnull(df.rcountry) & pd.isnull(df.ISO)][['ISO','countryCode_GWP3_wp5','WP5','rcountry']],
            ' Published WHR country lacks a WP5: ': df[pd.notnull(df.rcountry) & pd.isnull(df.WP5)],
            ' Published WHR country lacks a map code: ': df[pd.notnull(df.rcountry) & pd.isnull(df.twoletter_AlexShultz_svg)],
            ' Old Gallup micro country lacks an ISO in my master lookup: ': df[pd.notnull(df.countryCode_GWP3_wp5) & pd.isnull(df.ISO)][['ISO','countryCode_GWP3_wp5','WP5','wcountry']],
            ' 2016 Gallup micro country lacks an ISO in my master lookup: ': df[pd.notnull(df.WP5) & pd.isnull(df.ISO)][['ISO','countryCode_GWP3_wp5','WP5','wcountry']],
        }

        if verbose:
            for tt,dd in problems.items():
                if not dd.empty:
                    print('\n\n -- country_tools WARNING: '+tt)
                    print dd
        return df.reset_index()


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
    
