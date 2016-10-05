#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
A few tools for reading config files for projects. In each case,
there's a config-template.cfg and a config.cfg in the project's own
repo space, and possible also one of each in the local folder at
runtime.  Each project has a projectname-config.py which makes use of these.

See OSM, RDC, Gallup, etc projects for sample config-template.cfg file layouts, which match the config_file_structure dict specified as documented below.

An example of the config_file_structure value is:

config_file_structure={
    'paths': [
        'working',
        'input',
        'graphics',
        'outputdata',
        'output',
        'tex',
        'scratch',
        'bin',
        ],
    'defaults': [
        ('rdc',bool),
        'mode',
        ],
    'server': [
        ('parallel',bool),
        ('manycoreCPU',bool),
        ('islinux',bool),
        'stataVersion', # e.g. 'linux12'
    ],
    'gallup': [
        'MAX_WP5',
        'MAX_WAVE',
        'version',
        'GWPdataVersion',
        'GWPrawdataDir',
        ],
    }


An example config-template.cfg file is then:

[paths]
working = /home/foo

[server]
parallel = True

"""
import os,copy
# Following seems amazing. Shouldn't I just get these from cpblUtilities_config? 
from .utilities import dsetset,dgetget,merge_dictionaries, read_hierarchy_of_config_files,readConfigFile
def __readConfigFile(inpath, config_file_structure):
    """
    """
    import ConfigParser
    config = ConfigParser.SafeConfigParser({'pwd': os.getcwd(),'cwd': os.getcwd()})
    config.read(inpath)
    outdict={}
    for section in     config_file_structure:
        if config.has_section(section):
            for option in config_file_structure[section]:
                if config.has_option(section,option  if isinstance(option,str) else option[0]):
                    if isinstance(option,str):
                        dsetset(outdict,(section,option), config.get(section,option))
                    elif option[1]==bool:
                        dsetset(outdict,(section,option[0]), config.getboolean(section,option[0]))
                    elif option[1]==int:
                        dsetset(outdict,(section,option[0]), config.getint(section,option[0]))
                    elif option[1]==float:
                        dsetset(outdict,(section,option[0]), config.getfloat(section,option[0]))
                    elif option[1]=='commasep':
                        dsetset(outdict,(section,option[0]), config.get(section,option[0]).split(','))
                    else:
                        raise('Do not know config value type '+str(option[1]))
    return(outdict)

def tmp_read_hierarchy_of_config_files(files,config_file_structure, verbose=True):
    """
    Reads a sequence of config files, successively updating a dict of config settings.
    Returns the dict.

    if verbose is True, it also reports file was the last to set each setting.

    Note that there is also a verboseSource feature in merge_dictionaries, which reports updating as it goes, but this is less useful than the verbose behaviour given here.
    """
    configDict={}
    configDictOrigins={}
    def setOrigin(filename,adict):
        for kk in adict:
            if isinstance(adict[kk],dict):
                setOrigin(filename,adict[kk])
            else:
                adict[kk]=filename
    def reportOrigin(adict, vdict):
        for kk in adict:
            if isinstance(adict[kk],dict):
                reportOrigin(adict[kk], vdict[kk])
            else:
                print(kk+'\t = \t'+str(vdict[kk])+' :\t (from '+adict[kk]+ ')')
    for ff in files:
        if os.path.exists(ff):
            newConfigDict=readConfigFile(ff,config_file_structure)
            if verbose:
                newConfigOrigins=copy.deepcopy(newConfigDict)
                setOrigin(ff,newConfigOrigins)
                configDictOrigins=merge_dictionaries(configDictOrigins,newConfigOrigins)
            configDict=merge_dictionaries(configDict,newConfigDict, verboseSource=False) #False if not verbose else ff) #bool(configDict))

    
    if not configDict:
        raise Exception("Cannot find config[-template].cfg file in "+', '.join(files))
    if verbose:
        reportOrigin(configDictOrigins, configDict)
    return configDict





