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
import os
from .utilities import dsetset,dgetget,merge_dictionaries
def readConfigFile(inpath, config_file_structure):
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

def read_hierarchy_of_config_files(files,config_file_structure):
    configDict={}
    for ff in files:
        if os.path.exists(ff):
            newConfigDict=readConfigFile(ff,config_file_structure)
            configDict=merge_dictionaries(configDict,newConfigDict, verboseSource=False) #bool(configDict))


    if not configDict:
        raise Exception("Cannot find config[-template].cfg file in "+', '.join(files))
    return configDict
