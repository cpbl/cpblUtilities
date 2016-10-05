#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, re, sys, copy
"""This module provides dicts paths and defaults, which contain any parameters needed by multiple other modules.
These parameters are generally unchanging over time, but may vary from one installation/environment to another.

There are four places these config settings could be set. In order of priority:

(1) local config.cfg file
(2) local config-template.cfg file
(3) cpblUtilities source folder config.cfg file
(4) cpblUtilities source folder config-template.cfg file


 - Specify structure of file.
 - Load cascaded values from config files.
 - Then rearrange as we need to put them into the dict arrays paths and defaults.

Note that some key functions located in other modules of cpblUtilities are reproduced
here, in order to avoid circular dependencies. e.g. dsetset, dgetget,
and even a version of read_hierarchy_of_config_files()

"""
UTILS_config_file_structure={
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
    }



# The file config-template.cfg contains an example of a file which should be renamed config.cfg
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

def _notgeneral_readConfigFile(inpath):
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
                    elif option[1]=='commasep':
                        dsetset(outdict,(section,option[0]), config.get(section,option[0]).split(','))
    return(outdict)

################################################################################################
################################################################################################
def dsetset(adict,keys,avalue):
    ############################################################################################
    ############################################################################################
    """
    July 2011: making the converse of dgetget... but I think rather more efficient!
    This sets the value of a nested dict, ensuring that the sublevels exist.
    adict must exist and be a dict, of course.
    """
    if len(keys)>1:
        if keys[0] not in adict:
            adict[keys[0]]={}
        dsetset(adict[keys[0]],keys[1:],avalue)
    else:
        adict[keys[0]]=avalue
        return

################################################################################################
################################################################################################
def dgetget(adict,keys,defaultvalue,*args):
    ############################################################################################
    ############################################################################################
    """
    July 2011: rewriting degetget, using recursion, and conforming only to the newer format in which a  list of the keys is passed.
    Much more efficient than the old version which took nargs!
    adict must exist and be a dict, of course.

    *args is vestigial, for backwards compatibility. It should not be used.
    """
    # Backwards compatibility: Ancient dgetgetold(adict, key1, key2, key3=None,key4=None,key5=None,key6=None,keyn=None):
    if not isinstance(keys,list):
        keylist=[keys,defaultvalue]+list(args)
        #keylist=keylist[:min([ii  for ii in range(len(keylist)) if keylist[ii] is None])]
        keylist, defaultvalue= keylist[:-1] ,keylist[-1]
        return( dgetget(adict,keylist,defaultvalue))
    #  
        return( dgetgetOLD(adict,keys,defaultvalue,key3=key3,key4=key4,key5=key5,key6=key6,keyn=keyn))

    # New, recursive algorithm, which takes a list of keys as second argument:
    if keys[0] not in adict:
        return(defaultvalue)
    if len(keys)==1:
        return(adict[keys[0]])
    return(dgetget(adict[keys[0]],keys[1:],defaultvalue))

def merge_dictionaries(default,update, verboseSource=False, allow_new_keys=True):
    """Given two dictionaries, this deep copies 'default' but updates it with any
    matching keys from 'update'.

    allow_new_keys = False ensures that only keys in the default are taken (updated) from the update.

If not False, verboseSource must be a string, which denotes the updating source file description
    """
    result=copy.deepcopy(default)
    for key in update:
        if key not in default:
            if allow_new_keys:
                result[key]=update[key]
            else:
                print("WARNING: configuration merge_dictionaries got an update, but\
 that key doesn't exist in the default config settings. key=%s"%key)
                continue
        if type(update[key])==dict:
            result[key]=merge_dictionaries(result[key],update[key], verboseSource=verboseSource)
        else:
            result[key]=update[key]
            if verboseSource:
                print('   Using '+verboseSource+' config value for: '+key)
    if 0:
        print('-------')
        print default
        print update
        print result
    return result


def _co_read_hierarchy_of_config_files(files):
    """
    There is a more general version of this in configtools, which is used by other modules. But I can't use that one because these modules would be cross-dependent
    
    """
    configDict={}
    for ff in files:
        if os.path.exists(ff):
            newConfigDict=readConfigFile(ff)
            configDict=merge_dictionaries(configDict,newConfigDict, verboseSource=False) #bool(configDict))


    if not configDict:
        raise Exception("Cannot find config[-template].cfg file in "+', '.join(files))
    return configDict

def read_hierarchy_of_config_files(files,config_file_structure, verbose=True):
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


def main():
    """
    """
    localConfigFile=os.getcwd()+'/config.cfg'
    localConfigTemplateFile=os.getcwd()+'/config-template.cfg'
    repoPath=os.path.abspath(os.path.dirname(__file__ if __file__ is not None else '.'))

    if 0: 
        # Change directory to the bin folder, ie location of this module. That way, we always have the config.cfg file as local, which means other utlilities using config.cfg will find the right one.
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        if 'cpblUtilities' not in os.getcwd():
            os.chdir(dir_path)

    repoFile=(repoPath if repoPath else '.')+'/config.cfg'
    repoTemplateFile=(repoPath if repoPath else '.')+'/config-template.cfg'


    print('cpblUtilities setting defaults:')
    merged_dictionary=read_hierarchy_of_config_files([
        repoTemplateFile,
        repoFile,
        localConfigTemplateFile,
        localConfigFile,
    ], UTILS_config_file_structure)

    # Now impose our structure
    defaults=dict([[kk,vv] for kk,vv in merged_dictionary.items() if kk in ['rdc','mode']])
    defaults.update(dict(paths=merged_dictionary['paths'],
                  ))
    defaults['stata']={'paths':copy.deepcopy(defaults['paths'])}
    return(defaults)
defaults=main()
paths=defaults['paths']
if 'python_utils_path' in paths:
    sys.path.append(paths['python_utils_path'])

