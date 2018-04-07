#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This is a poorly-organized collection of utilities of general, errr, utility.

Configuration: See cpblUtilities-config.py.  In short, copy config-template.cfg to config.cfg and edit it.

 A config.cfg file should be used to set up folders to be used by cpblUtilities. The configuration procedure is as follows:

  (1) If config.cfg exists locally, it will be used.

  (2) Otherwise, cpblUtilities will look for a config.cfg file in its own (the cpblUtilities repository) folder.  

"""
from .cpblUtilities_config import *

import os
import re
from copy import deepcopy
import sys
import time


def debugprint(a='',b='',c='',d='',f='',g='',h='',i='',j='',k='',l='',m='',n=''):
    #print 'DEBUG -- '+str([a,b,c,d,f,g,h,i,j,k,l,m,n])
    pass
    return

def toYearFraction(date):
    """ Convert a date to a decimal number of years
    It takes a datetime object, I tink. """
    import datetime as dt
    import time
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    if isinstance(date,list) and len(date)==3:
        date=dt.date(date[0],date[1],date[2])

    year = date.year
    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction




#---function to grab a web page---#000000#FFFFFF--------------------------------
def wget(url,binaryMode=0):
    """Grab a page and extract certain infos"""
    import urllib
    import urllib2
    import re  # Regular expressions
    from time import sleep
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = { 'User-Agent' : user_agent }
    req = urllib2.Request(url, None, headers)
    htmlSource=[]
    while not htmlSource:
        try:
            htmlSource = urllib2.urlopen(req).read()#,headers)
            # sock = urllib.urlopen(url)
        except: # urllib2.HTTPError, e:
            print('Caught an error reading URL...HTTP?!')
            htmlSource=[]
            sleep(5)
            #if e.code == 401:
            #    dlog('HTTP ERROR!!: not authorized')
            #elif e.code == 404:
            #    dlog('HTTP ERROR!!: not found')
            #elif e.code == 503:
            #    dlog('HTTP ERROR!!: service unavailable')
            #else:
            #    dlog('HTTP ERROR!!: unknown error: ')
        if not htmlSource:
            print('Failed to open url'+url)
##     sock=[]
##     while not sock:
##         try:
##             sock = urllib2.urlopen(req)
##             # sock = urllib.urlopen(url)
##         except urllib2.HTTPError, e:
##             sock=[]
##             if e.code == 401:
##                 dlog('HTTP ERROR!!: not authorized')
##             elif e.code == 404:
##                 dlog('HTTP ERROR!!: not found')
##             elif e.code == 503:
##                 dlog('HTTP ERROR!!: service unavailable')
##             else:
##                 dlog('HTTP ERROR!!: unknown error: ')
##         if not sock:
##             dlog('Failed to open url'+url)
##     htmlSource = sock.read()
    #sock.close()
    if not binaryMode:
        htmlSource=re.sub("\n","",str(htmlSource)).replace('\r','')  # Remove newlines and ^M's
        # Remove all newlines and all tabs from html. Later I use tabs for CSV
        return(str(htmlSource).expandtabs())
    else:
        return(htmlSource)


###########################################################################################
###
def doSystem(acommand,verbose=False,bg=None):
    ###
    #######################################################################################
    """
    2011/2: Added some facility for background launching.
    """
    if bg=='ifserver':
        if 'apollo' in os.uname()[1]:
            bg=True
        else:
            bg=False
    if bg in [None,False]:
        debugprint( '  Calling system: %s'%acommand,)
        if verbose:
            print( '  Calling system: %s'%acommand,)
        for oneline in acommand.split('\n'):
            os.system(oneline)
        debugprint( ' ... Done system call')
        return
    if bg is True:
        if verbose:
            print( '  Calling system: %s in background'%acommand,)
        import tempfile
        fn=tempfile.NamedTemporaryFile('w',delete=False)
        fn.write(acommand+'\n')
        os.system('nohup nice /bin/bash '+fn.name+' &')
        #import subprocess
        #subprocess.Popen(["nohup", "python", "test.py"])
        #os.system()



####################################################################################
# Aug 2010: You may still consider using shelve instead, but  I've just simplifed this so that it does work now.
#     See comments for what's missing.
####################################################################################
def dictToTsv(dicts,tsvFile,snan='',skipCheckStructure=False):
    """
Assumptions:
- all elements are dicts with identical structure (!) [May 2011: No, this is only relied upon if skipCheckStructure is True]
- NaN's should be stored as empty entries. Maybe this causes trouble for the first row?
Problems:
- does not yet insert a second row containing format information.


"""
    outfile=open(tsvFile,'w')
    keys=dicts[0].keys()
    from pylab import flatten
    if not skipCheckStructure: # Find superset of keys
        keys=uniqueInOrder([xx for xx in flatten([dd.keys() for dd in dicts])])
    else: # Or rely on every dict being identical.
        keys=dd[0].keys()

    if ''=='Create second line of file: indicates format types':
        fieldTypes=[]
        for k in keys:
            if isinstance(dicts[0][k],float):
                fieldTypes+=r'%f'
            elif isinstance(dicts[0][k],int):
                fieldTypes+=r'%d'
            else:
                fieldTypes+=r'%s'
        outfile.write(''.join(fieldTypes)+'\n')
    outfile.write('\t'.join(keys)+'\n')
    if ''=='Create second line of file: indicates format types': # Right now this would NOT have the feature of dealing with NaNs
        for v in dicts:
            outfile.write(''.join(fieldTypes).replace(r'%','\t' r'%')[1:]%([v[k] for k in keys])+'\n')
    else:
        for v in dicts:
            row=[]
            for k in keys:
                row+=[str(v.get(k,''))]
                if row[-1]=='nan' and isinstance(v[k],float) and not isfinite(v[k]):
                    row[-1]=snan
                # "nan" should never be written...
                #assert not isinstance(v[k],float) or  isfinite(v[k]) or row[-1]==snan
            outfile.write('\t'.join(row) + '\n')
            assert 'countryLong' not in tsvFile
            ###outfile.write('\t'.join([str(v.get(k,'')) for k in keys])+'\n')
            """
    def table2tsv(filename,table):#,utf8=False):
        fout=open( filename,'wt')
        mk=sorted(table[0].keys())
        fout.write(('\t'.join([k for k in mk])+'\n'+\
     '\n'.join(['\t'.join([nm.get(k,'') for k in mk]) for nm in table])+'\n').encode('utf8'))
        fout.close()
        """

    return(outfile.close())

# Convert a matrix (ie list of lists) into a list of dicts, given a list of keys for the columns. This is standard for csv/CSV files. It's so short that I'm not making it into a fucntion:
#def csvToDicts(matrix,keys):
#    listOfDicts=[dict(keys,row) for row in matrix]
#
# Okay: I changed my mind: Here is a function to read a tab-CSV, convert to dicts, and if asked even convert numbers to numbers.
#
# Read entire csv file into list of lines, each of which is a list of cells:
#
# 2007Aug: it seems not to convert to a dict by keys yet. This is one more line. If the key is "idnum",         dates=dict([(RR['idnum'],RR) for RR in dateslist]), where dates list is returned from this fucntion

##############################################################################
##############################################################################
#
def tsvToDict(filename,keyRow=0,dataRow=1,formatRow=None,vectors=False,replaceHeaders=[],headerFormats=[],splitBy='\t',isNaN=['','no data/no data','no data','.','*','NaN'],NaN=None,utf8=True,treeKeys=None,sort=False,allowShortRows=False,singletLeaves=False):#[['',['','']]]): # Row counting starts at 0    #  Needs a "keepCols"?
    ##########################################################################
    ##########################################################################

    """
    This is much more fully-featured than "importSpreadsheet", which needs integrating with this...

    Read a CPBL-style tab-separated-value (_tsv/csv/CSV) file into a list of dicts:
    keyRow can be either a row number (0 based) or a list of column header strings.
    Format row could be any of: 2 or  ['f','d','s'] or '%f%d%s'
      i.e. the second row of the file could specify a format string, or ..etc.
    header Formats is a zip-style list of headers and corresponding formats, e.g. [('fieldone','s'),('field8','d')]

9 Aug 2006: The tsv program has grown in features. It also now assumes the file is in UTF-8 format... This is what I guess I output from OpenOffice. That means that any high characters are stored with a high prefix; typically they are thus two bytes long rather than one. This is sometimes visible in emacs or etc.


    Sep2007: I've added a feature to capture Macintosh format files and load them in all at once, to fix them, rather than line by line. Oh, don't be silly: I read the whole file anyway.
    Hours of effort to make this function work with troublesome cases (final blank line, extra key fields, etc)
        # Hm. It looks here like the formats for replaceHeaders are: [[newname,oldname],[anothernewname,anotheroldname]] or [[newname,[oldname1,oldname2]]] or a dict: {'oldname':'newname','anotheroldname':'anothernewname'}




Sept 2008: if treeKeys is given, then the array of dicts will be shuffled to a single dict, with the dicts indexed by property (field) treeKeys.

Remember! You don't necessarily want to use this! It doesn't preserve order?!

July 2010: Added allowShortRows: so don't barf if some final elements are mising.

May 2012: Should I add the option of another row which gives descriptions of keys, ie variable labels? This doesn't fit into the Dict, but it could go into a secodn dict (or a cpbl codebook format).
Jan 2013: I will often be ussing pandas from now on, though this still is a useful.
    """
    assert not singletLeaves or treeKeys

    def verboseprint(ss,a='',b='',c='',d='',e='',f=''):
        return()
    #isNaN=['','no data/no data','no data','..','*','(dropped)']
    verboseprint ('tsvToDict: Reading ',filename,'...',)
    sys.stdout.flush()
    # Partly-friendly filename handling:
    if not filename.endswith('.tsv') and not filename.endswith('.csv') and not filename.endswith('.txt') and not os.path.splitext(filename)[1]:
        filename+='.tsv'

    # Check bloody file format
    testline=open(filename,'rt').readline()
    if '\r' in testline and '\n' not in testline:
        verboseprint( 'Assuming this is a MAC text format file (sigh...) and thus splitting by \\r, using lots of memory...')
        #if utf8:
        #    cells=[[c.strip('" \n\r').strip() for c in line.decode('utf-8').split(splitBy)] for line in re.split('\n',open(filename,'rt').read().replace('\r\n','\n').replace('\r','\n') )]
        #else:
        #    cells=[[c.strip('" \n\r').strip() for c in line.split(splitBy)] for line in re.split('\r|\n',open(filename,'rt').read())]
        xreadlines=re.split('\n',open(filename,'rt').read().replace('\r\n','\n').replace('\r','\n'))
        while not xreadlines[-1]:
            verboseprint ('Ignoring last line: "%s"'%xreadlines[-1])
            xreadlines.pop(len(xreadlines)-1)
    else:
        xreadlines=open(filename,'rt').xreadlines()
    # Ahh. don't strip the line before splitting in following! This removes final blank fields.
    if utf8:
        cells = [[c.strip('" \n').strip() for c in line.decode('utf-8').split(splitBy)] for line in xreadlines]
    else:
        cells = [[c.strip('" \n').strip() for c in line.split(splitBy)] for line in xreadlines]
    #Kluge to deal with DOS format files? ie with \r (ie old Macintosh format!!) rather than \n (unix) or \r\n (DOS):
    #print len(cells),len(cells[0])
    #print cells
    #   if len(cells)==1 and '\r' in cells[0]:
    #if len(cells)==1 and len(cells[0])==1 and '\r' in cells[0][0]:
    #    print 'Assuming this is a DOS text format file (sigh...) and thus splitting by \\r...([0][0] case!)'
    #    if utf8:
    #        cells=[[c.strip('" \n\r').strip() for c in line.decode('utf-8').split(splitBy)] for line in re.split('\r',cells[0][0])]
    #    else:
    #        cells=[[c.strip('" \n\r').strip() for c in line.split(splitBy)] for line in re.split('\r',cells[0][0])]


    # Make sure there are enough lines in thefile to provide data!
    if dataRow>=len(cells):
        print '   Warning! data file %s does not have any data'%filename
        return({})

    #fieldDescriptions=cells[1]
    #fieldNames=cells[2]
    if isinstance(keyRow,list):
        keys=keyRow
    elif isinstance(keyRow,int):
        keys=cells[keyRow]
    else:
        brokenTypeOfInputParam
    #print    cells[keyRow]
    #print    cells[dataRow]
    # Possibly replace some of the key names. This is crude, but: ignore spaces when replacing:

    if replaceHeaders:
        # Start off with the Identity map:
        #headerLookup={}

        # Nov 2009: this very strange line was here. Why lower()??? And why not just use ".get()" rather than have defaults?
        headerLookup=dict([(hni.lower().replace(' ',''),hni.lower().replace(' ','')) for hni in keys])
        headerLookup={} # Just use .get(), below.
        #print headerLookup
        assert isinstance(replaceHeaders,dict) or isinstance(replaceHeaders,list)
        if isinstance(replaceHeaders,dict):
            headerLookup.update(replaceHeaders)
        else:
            # Nov 2009: What the hell is the below? Why the lower()??? Above is if dict form passed; below is list
            for hn in replaceHeaders: # Build the look up table
                if not isinstance(hn[1],list):
                    hn[1]=[hn[1]]
                for hni in hn[1]:
                    headerLookup[hni.lower().replace(' ','')]=hn[0]
            #print headerLookup
            # Replace all keys with value from lookup table:
            #keys=[headerLookup[h.lower().replace(' ','')] or h for h in keys if headerLookup.has_key(h.lower().replace(' ','')) and h]

        # Nov 2009: this very strange line was here. Why lower()???
        #keys=[headerLookup[h.lower().replace(' ','')] or h for h in keys if h]
        keys=[headerLookup.get(h,headerLookup.get(h.lower(),headerLookup.get(h.lower().replace(' ',''),h))) or h for h in keys if h]

    # if formatRow not passed but headerFormats was, create formatRow
    if headerFormats and not formatRow:
        if isinstance(headerFormats,basestring):
            headerFormats=dict([[kk,headerFormats] for kk in keys])
        if isinstance(headerFormats,list):
            headerFormats=dict(zip(keys,headerFormats))
        dFormats=dict(headerFormats)
        formatRow=[dFormats.setdefault(k,'%s') for k in keys]
        #formatRow=[dict(headerFormats).setdefault(k,'%s') for k in keys]
    # Convert numeric values
    if formatRow != None:
        verboseprint( '      tsvToDict: Converting numbers...',)
        sys.stdout.flush()
        if isinstance(formatRow,int):
            fieldTypes=cells[formatRow]
        elif isinstance(formatRow,basestring):
            fieldTypes=formatRow.split('%')[1:]
        elif isinstance(formatRow,list):
            fieldTypes=formatRow
        #if len(fieldTypes)==1:
        #    fieldTypes=fieldTypes[0].split('%')[1:]
        for row in cells[dataRow:]:
            for ic in range(len(row)):
                if 'f' in fieldTypes[ic] or 'd' in fieldTypes[ic]:
                    if row[ic] in isNaN:
                        row[ic]=NaN
                    else:
                        if 'f' in fieldTypes[ic]:
                            row[ic]=float(row[ic])
                        elif 'd' in fieldTypes[ic]:
                            if '.' in row[ic]:
                                row[ic]=int(float(row[ic]))
                            else:
                                row[ic]=int(row[ic])
    # Sort the data (?!)
    if sort:
        cells[dataRow:]=sorted(cells[dataRow:])

    # Convert matrix format into vectors or list of dictionaries:
    # 2010 July (really? This hasn't been done before?!): dealing with possibility of short (incomplete) data rows.
    if vectors:
        verboseprint( '      tsvToDict: Returning dictionary of vectors...')
        tsv={}
        if allowShortRows:
            for ik in range(len(keys)):
                if keys[ik]:
                    def checkget(aa,ii,deff):
                        if len(aa)>ii:
                            return(aa[ii])
                        else:
                            return(deff)
                    # '' below should be facultative '' or NaN
                    tsv[keys[ik]]=[checkget(cells[dataRow+j],ik,{True:'',False:NaN}[fieldTypes[ic]=='s']) for j in range(len(cells[dataRow:]))]
        else:
            for ik in range(len(keys)):
                if keys[ik]:
                    tsv[keys[ik]]=[cells[dataRow+j][ik] for j in range(len(cells[dataRow:]))]
    else:
        verboseprint( '      tsvToDict: Returning list of dicts...')
        # Assume for now that keys are in English.. (str())
        # Oh dear, above is a bad idea: sometimes keys are numeric! So don't recast:
        # tsv=[ dict([(str(keys[i]),row[i]) for i in range(len(keys))]) for row in cells[dataRow:] ]
        lens=[len(row) for row in cells[dataRow:]]
        if min(lens)==max(lens) and len(keys)>min(lens):
            if any([kk not in [''] for kk in keys[min(lens):]]): # ie ignore it if they're just blanks / extra tabs.
                print('   Ignoring some keys due to excessive number of them (ie there are more headers than data columns?): %s'%filename, ' Extra headers are: ',keys[min(lens):])
            keys=keys[0:min(lens)]
        if min(lens)<len(keys):
            print '   Error / warning: input file has inconsistent number of fields: there are some bad rows!: %s\n'%filename
            #print lens
            #print cells[dataRow:]
            for irow in range(len(cells)-1,dataRow-1,-1): #Must loop backwards!
                if len(cells[irow])<len(keys):
                    if  allowShortRows:
                        cells[irow]+=['' for kk in range(len(keys)-len(cells[irow]))]
                    else:
                        print '  DELETING A ROW!: len=%d  key length = %d  min/max data length =%d/%d'%(len(cells[irow]),len(keys),min(lens),max(lens))
                        print cells.pop(irow) # Deletes that line!!
        tsv=[ dict([(keys[i],row[i]) for i in range(len(keys))]) for row in cells[dataRow:] ]


    if treeKeys: # Switch the array of dicts to a dict, with key treeKeys.
        import dictTrees
        tsv= dictTrees.dictTree(tsv,treeKeys)#dict([[x,[cii for cii in tsv if cii[treeKeys]==x][0]] for x in tsv if x[treeKeys]])
        if singletLeaves:
            tsv=tsv.singletLeavesAsDicts()
        #for kk in tsv:
        #    if len(tsv[kk])==1:
        #        tsv[kk]=tsv[kk][0]






    return(tsv)#,cells[0:dataRow]) # 21 Sept 2006: I took this second return argument out.



##############################################################################
##############################################################################
#
def popWordRE(sfrom,regexps,partword=False):
    ##########################################################################
    ##########################################################################
    """ Updated version of below...
    regexps is a *list* of strings.

    If the search-for string starts and ends with paren's, don't enforce word boundaries in the search, since the parens do not make word boundaries...
    """
    ##import unicodedata
    outlist=[]
    for ss in regexps:#[s.lower() for s in regexps]:
        """ Replace/Match word, possibly at end of string
         If search-for string has unicode, don't force it to be its own word! (re deficiency)
         Weirdness here: if the search-for string is short, let's make sure it's a word by itself:
         Without re.unicode \b doesn't think high unicode chars can be part of a word
        """
        # This line is a horrid kludge!!
        if partword or  ss.startswith(r'\.') or ss.endswith(r'\.') or (ss.startswith(r'\(') and ss.endswith(r'\)')) or (ss.startswith(r'\[') and ss.endswith(r'\]')):
            reg=ss
        else:
            reg=r'\b'+ss+r'\b'
        found=set(re.compile(reg,re.IGNORECASE|re.UNICODE).findall(sfrom))
        #if '(' in reg:# and len(reg)<10:
        #    print 'looking for ',reg, ' in ',sfrom
        if found:
            sfrom,ns=re.compile(reg,re.IGNORECASE|re.UNICODE).subn(' ',sfrom)
            outlist+=(list(found))
    return (sfrom.strip(),outlist)

def popStringIC(sfrom,ssearch):
    """ Move any instances of ssearch from sfrom to outlist. A tuple
    is returned: the possibly-modified sfrom and the outlist of removed
    ssearch's.  Case is ignored

    Note: ssearch must be a *list* of strings or regexps.
    """
    outlist=[]
    for ss in [s.lower() for s in ssearch]:
        # Replace/Match word, possibly at end of string
        # Weirdness here: if the search-for string is short, let's make sure it's a word by itself:
        if len(ss)<4:
            reg=r'\b'+ss+r'\b'
        else: # Otherwise, avoid \b because it doesn't think high unicode chars can be part of a word
            reg=ss
        sfrom,ns=re.compile(reg,re.IGNORECASE).subn(' ',sfrom)
        #if "astra coup" in sfrom.lower() and "coup" in ss:
        #    1/0

        #print '-->',sfrom,ns
        if ns:
            #print '-----------------======================='
            #print '  Sought "%s" in "%s" and found %d.  Left with %s'%(fs,sfrom,ns,sfrom)
            outlist.append(ss)
    return (sfrom.strip(),outlist)

##############################################################################
##############################################################################
#
def uniqueInOrder(alist,key=None,drop=None):    # Fastest order preserving
    # 2016 update: use     from  more_itertools import unique_everseen ?
    ##########################################################################
    ##########################################################################
    alist=list(alist) # Hmm. thi sis so that I can deal with mpl.array types? Aug2010
    if not alist:
        return(alist)

    if isinstance(alist[0],list): # Then this is NOT the fastest!!
        if all([not aa for aa in alist]):
            return(alist[0:1])
        def unique_items(L):
            found = set()
            for item in L:
                if item[0] not in found:
                    yield item
                    found.add(item[0])
        assert key==None
        assert drop==None
        return(list(unique_items(alist)))

    # So it's a list of hashable items, hopefully.
    if drop==None:
        drop=[]

    if key==None:
        setu = {}
        return [setu.setdefault(e,e) for e in alist if e not in setu and e not in drop]
    else: # Find dicts with unique values of key:
        NotDoneYet
        # set = {}
        #return [set.setdefault(e,e) for e in alist if e[key] not in [s[key] for s in set]]



##############################################################################
##############################################################################
#
def matchTableToTable(master,child,matchKeys=None,keepKeysMaster=None,keepKeysChild=None,masterKeyRow=0,childKeyRow=0,masterDataRow=1,childDataRow=1,primaryKey=None,reverse=False):
    ##########################################################################
    ##########################################################################
    """
    See concordanceFinder.py, if it still exists standalone, for examples of *usage*. But the algorithm is here.

    This has been a very useful tool for adding new fields to a concordance table, in particular for country-level data from different agencies which use different country names and sets.


    primaryKey is a key in the master table. It is the thing we would like all others to relate to. e.g. wp5 when doing gallup data analysis. [No, I think this is obselete, according to below. You can ignore it.]

    A "table" is a list of dicts. ie record order matters; column order does not.

You can pass either a dict or a filename for the first two arguments.

    matchKeys is a list of pairs, each like [masterKey,childKey]. Each pair (only one is needed) specifies keys to match on. For instance, if the first one is ['iso3','ISO3'] then the child table rows will first be assigned to master table rows where the child row's ISO3 matches the master's iso3. Leftover rows of the child can then be matched based on the next key pair.

    Lots still to deal with: what if multiple childs match to a master?
    Or vice versa? which should I allow?...


Sept 2010: child can be a filename,

OCt 2010: needs more docuemntaiton thorughout. should report un=matched items!
    """

    from copy import deepcopy

    if reverse: # untested
        child,master,matchKeys,keepKeysChild,keepKeysMaster,childKeyRow,masterKeyRow,childDataRow,masterDataRow,primaryKey=master,child,matchKeys,keepKeysMaster,keepKeysChild,masterKeyRow,childKeyRow,masterDataRow,childDataRow,primaryKey
        matchKeys=[mk[1:2]+mk[:1]+mk[2:] for mk in matchKeys]



    assert primaryKey==None # This is no longer used / implemented

    # Check inputs
    assert isinstance(matchKeys,list)
    assert isinstance(matchKeys[0],list)


    def table2tsv(filename,table):#,utf8=False):
        fout=open( filename,'wt')
        mk=sorted(table[0].keys())
        fout.write(('\t'.join([k for k in mk])+'\n'+\
     '\n'.join(['\t'.join([nm.get(k,'') for k in mk]) for nm in table])+'\n').encode('utf8'))
        fout.close()


    writeAutoFileMaster=False
    writeAutoFileChild=False
    if isinstance(master,basestring):
        writeAutoFileMaster=True
        masterFN=master
        master=tsvToDict(master,keyRow=masterKeyRow,dataRow=masterDataRow,vectors=False)
    if isinstance(master,dict): # hm, we need a list of dicts.   This came as a dicto f vectors.
        kk=master.keys()
        master=[dict([[kkk,master[kkk][ii]] for kkk in kk])        for ii in range(len(master[kk[0]]))]
    if isinstance(child,basestring):
        writeAutoFileChild=True
        childFN=child
        child=tsvToDict(child,keyRow=childKeyRow,dataRow=childDataRow,vectors=False) # This becomes a list of dicts
    if isinstance(child,dict): # hm, we need a list of dicts.   This came as a dicto f vectors.
        kk=child.keys()
        child=[dict([[kkk,child[kkk][ii]] for kkk in kk])        for ii in range(len(child[kk[0]]))]


    if not keepKeysMaster:
        keepKeysMaster=master[0].keys()
    if not keepKeysChild:
        keepKeysChild=child[0].keys()

    assert 'child' not in master
    assert primaryKey not in child

    def nopunclower(s):
        return(''.join([ss for ss in s if ss not in ['. ,']]).lower())

    master=deepcopy(master)
    keysCopyToChild=list(set([kk[0] for kk in matchKeys]))
    for c in child:
        for k in keysCopyToChild:
            c[k]=c.get(k,'')
    newchild=deepcopy(child)

    for mm in master: # Allow new data to be assigned onto master, duh.
        mm['child']={}#dict([[k,''] for k in keepKeysChild])

    for matchKeySet in matchKeys:
        childKey,masterKey=matchKeySet[1],matchKeySet[0]
        print 'Matching on %s (child) to  %s (master)'%(childKey,masterKey)
        print '    ',matchKeySet
        # Find keys with which to order incoming data:
        #childOrder=
        for icc in range(len(child))[::-1]:
            if child[icc][childKey]:
                imatches=[imm for imm in range(len(master)) if nopunclower(child[icc][childKey])==nopunclower(master[imm][masterKey]) or (master[imm][masterKey] and len(matchKeySet)>2 and matchKeySet[2]=='partial' and  ( nopunclower(child[icc][childKey]).find(nopunclower(master[imm][masterKey]))==0 or nopunclower(master[imm][masterKey]).find(nopunclower(child[icc][childKey]))==0 ) )]
                if imatches:
                    if len(imatches)>1:
                        print child[icc][childKey]
                        print master[imm][masterKey]

                    imm=imatches[0]
                    print '%s looks like %s'%(child[icc][childKey],master[imm][masterKey]),
                    for copykey in keysCopyToChild:
                        if not child[icc][copykey]: #Overwrite unless not ''
                            child[icc][copykey]=master[imm][copykey]
                    newchild[icc]=deepcopy(child[icc])##master[imm]['child'])

                    master[imm]['child']=child.pop(icc)

                    print ': popped %d of child, len(%d)'%(icc,len(child)+1)


    """
    from operator import itemgetter
    sorted(child, key=itemgetter('text'))
    """

    # Display matched:
    #keyOrderM=keepKeysMaster#[kk for kk in master[0].keys() if not kk=='child']
    #keyOrderC=keepKeysChild#child[0].keys()


    """ Generate a list with all master's primary key (and match keys), with a single (??) match of children, followed by (append) list of unmatched children.  ("newmaster"):    """
    newmaster=[]
    for mm in master:
        """ For new master, keep the primary key and any match keys: """
        mrow=dict([[k,mm[k]] for k in list(set(keepKeysMaster+[mk[0] for mk in matchKeys]))])#[primaryKey]+[mk[0] for mk in matchKeys]
        """ And also add the child keys: """
        for ck in list(set(keepKeysChild+[mk[1] for mk in matchKeys])):###list(set(keepKeysChild+[])):
            mrow[ck]=mm['child'].get(ck,'')
        newmaster+=[mrow]
    # And add the umatched children!!
    for cc in child:
        print ' FAILED to match this one with any key: '+' '.join(['%s="%s"'%(kk,str(cc[kk])) for kk in uniqueInOrder([mm[1] for mm in matchKeys])])
        mrow=dict([[k,cc.get(k,'')] for k in keepKeysMaster])##[primaryKey]+[mk[0] for mk in matchKeys]])
        for ck in list(set(keepKeysChild+[mk[1] for mk in matchKeys])):
            mrow[ck]=cc[ck]
        newmaster+=[mrow]
    if child in [[]]:
        print ' Match tables: managed to match every record!'

        #print '\t'.join([mm[k] for k in keyOrderM]+ [mm['child'][k] for k in keyOrderC])#+'\n'
    #for cc in child:  # Unmatched remaining ones
    #    print '\t'.join([mm[k] for k in keyOrderM]+ [mm['child'][k] for k in keyOrderC])#+'\n'


    """ Also generate a list of children simply with primary key added.  ("newchild"):  [done above]  """

    # I also want to add the unused master's to the newchild list:
    for mm in master:
        if not mm['child']:
            newchild+=[mm]


    if writeAutoFileChild:
        # Now make a tsv to finish matching by hand:
        table2tsv(childFN+'_automatch.tsv',newchild)
    if writeAutoFileMaster:
        # And make a tsv of child to check work:
        table2tsv(masterFN+'_automatch.tsv',newmaster)

    return(newmaster,newchild)


##############################################################################
##############################################################################
#
def importSpreadsheet(filename, masterKey=None):
    ##########################################################################
    ##########################################################################

    #return(tsvToDict(filename,masterKey=None,multiOutputs=True)

    """
    Grab a TSV spreadsheet and return both rows and columns.
    First row is column headers.
    If one column header is the master key, also offer a dict of rows based on that...
    rows,cols,colDict,keyDictDict,keyDictList=importSpreadsheet(filename)
    """
    rows=[LL.strip('\n').split('\t') for LL in open(filename,'rt').readlines()]
    # Transpose a square list of lists:
    cols=zip(*rows)
    colDict={}
    for col in cols:
        if col[0] not in colDict:
            colDict[col[0]]=col[1:]


    keyDictDict={}
    keyDictList={}

    if masterKey:
        for irow in range(len(colDict[masterKey])):
            if colDict[masterKey][irow]:
                keyDictDict[colDict[masterKey][irow]]=dict(zip(rows[0],rows[1+irow]))
                keyDictList[colDict[masterKey][irow]]=rows[1+irow]

    return(rows,cols,colDict,keyDictDict,keyDictList)




###########################################################################################
###
def orderListByRule(alist,orderRule,listKeys=None,dropIfKey=None):
    ###
    #######################################################################################
    """ Reorder alist according to the order specified in orderRule. The orderRule lists the order to be imposed on a set of keys. The keys are alist, if listkeys==None, or listkeys otherwise.  That is, the length of listkeys must be the same as of alist. That is, listkeys are the tags on alist which determine the ordering.  orderRule is a list of those same keys and maybe more which specifies the desired ordering.
    There is an optional dropIfKey which lists keys of items that should be dropped outright.
    """
    maxOR = len(orderRule)
    orDict = dict(zip(orderRule, range(maxOR)))
    alDict = dict(zip(range(maxOR, maxOR+len(alist)),
                      zip(alist if listKeys is None else listKeys, alist)))
    outpairs = sorted(  [[orDict.get(b[0],a),(b)] for a,b in alDict.items()]  )
    if dropIfKey is None: dropIfKey=[]
    outL = [b[1] for a,b in outpairs if b[0] not in dropIfKey]
    return outL

def test_orderListByRule():
    L1 = [1,2,3,3,5]
    L2 = [3,4,5,10]
    assert orderListByRule(L1, L2) == [3, 3, 5, 1, 2]
    assert orderListByRule(L1, L2, dropIfKey=[2,3]) == [5, 1,]
    Lv = [c for c in 'abcce']
    assert orderListByRule(Lv, L2, listKeys=L1) == ['c', 'c', 'e', 'a', 'b']
    assert orderListByRule(Lv, L2, listKeys=L1, dropIfKey=[2,3]) == ['e','a']



##########################################################################
##########################################################################
#
def transposedlist(lists):
    #
    ##########################################################################
    ##########################################################################
    if not lists: return []
    return map(lambda *row: list(row), *lists)


##########################################################################
##########################################################################
#
def readTSV(filepath, header=False, columnDict=False):
    #
    ##########################################################################
    ##########################################################################
    """
    hmmm!? See ImportSpreadsheet, above! and tsvDict ... How many times have I reinvnted the weheel??
    Depending on args, returns headers and data, or dict ,....
    """
    ff=[aline.strip('\n').split('\t') for aline in open(filepath,'rt').readlines()]

    if header or columnDict:
        hh=ff[0]
        ff=ff[1:]
    else:
        return(ff)
    if header and not columnDict:
        return(hh,ff)
    if columnDict:
        return(dict([[hh[ii],[fff[ii] for fff in ff]] for ii in range(len(hh))]))




if os.path.exists('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv'):
    masterCountryList=tsvToDict('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv',dataRow=4,keyRow=3)
    import pandas as pd
    dfCountryList=pd.DataFrame(masterCountryList)

##########################################################################
##########################################################################
#
def getCountryIDfromName(names):
    #
    ##########################################################################
    ##########################################################################
    """ Right now, returns universal country ID for a given name string.
    Still need to allow multiple guesses for the name of each country as param, somewhow

2010 Sept: HUH? why not country_bestName??
    """
    if not os.path.exists('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv'):
        NOT_AVAILABLE
        return()

    if isinstance(names,list):
        return([getCountryIDfromName(nn) for nn in names])
    assert isinstance(names,basestring)

    #if not masterCountryList:
    master=masterCountryList#tsvToDict('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv',dataRow=4,keyRow=3)
    favNames=[
    ['country_kauffman2'],
        ['country_UN'],
        ['country_WHO'],
        ['country_GWP3_wp5'],
        ['country_UN','partial'],
        ]
    mtab={}
    for kk in favNames:
        mtab=dict([[mm[kk[0]],mm['countryCode_CPBL']] for mm in master])
        if names in mtab:
            return(mtab[names])
        """ Should here check for partial!! """
    return('')

##########################################################################
##########################################################################
#
def getDictCountryIDtoISO3():
    #
    ##########################################################################
    ##########################################################################
    """ Returns a dict translating my country code to ISO3 name

N.B. This is NOT wp5 (Gallup). Nor is it returning the long name. For those, see recodeGallup.py [Sep 2010]
 Well, actually, this will now issue a warning if it conflicts with the latest wp5.

    """

    master=masterCountryList#tsvToDict('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv',dataRow=4,keyRow=3)

    master=tsvToDict('/home/cpbl/gallup/inputData/macro/countrycode_main.tsv',dataRow=4,keyRow=3)
    print [mm for mm in master if mm['countryCode_CPBL']=='167']

    for mmm in master:
        assert mmm['countryCode_CPBL']==mmm['countryCode_GWP3_wp5'] or not mmm['countryCode_GWP3_wp5'] # Ensure no lost wp5s
    response=dict([[mm['countryCode_CPBL'],mm.get('countryCode_ISO3','') ] for mm in master if mm['countryCode_CPBL']] )
    return(response)


def flattenList(listoflists,unique=False):#,noCopy=False):):
    """
what about these funny methods!?

[item for sublist in l for item in sublist]  # YES!! This is brilliant for one-level reduction. I've no idea how it works.

or sum(thelist,[]) !!!

CAUTION: The method below will FAIL on a list of lists of dicts. It will treat dicts like a list, and return its keys!
    """
    from pylab import flatten
    if unique:
        return(uniqueInOrder([xx for xx in flatten(listoflists)]))
    else:
        return([xx for xx in flatten(listoflists)])


def str2pathname(ss, includes_path = False, check=False):
    """ Remove some characters from a string, for safety (or simplicity) on POSIX systems.. 
    If passed with "check=True", it will simply check whether the string needs fixing.
    If passed with "includes_path", it will allow forward slashes but otherwise behave the same.
    """
    if check:
        return not ss == str2pathname(ss, includes_path = includes_path, check=False)
    subs=[
        ['_','-'],
        [r'$>$','gt'],
        ]
    todrop = u"""?"':.,()’ """+(not includes_path)*'/' #'
    for asub in subs:
        ss = ss.replace(asub[0],asub[1])
    ss=''.join([sss for sss in ss if sss not in todrop])
    return(ss)


def fileOlderThan(afile,bfiles,ageWarning=False):
    """ ie says whether the first fiel, afile, needs updating based on its parent, bfile.  ie iff afile does not exist or is older than bfile, this function returns true.

bfile can also be a list of files. in this case, the function returns true if any  of the bfiles is younger than the target file, afile; ie afile needs to be updated based on its antecedents, bfiles.

afile can also be a list of files... then the function returns true if any of the bfiles is younger than any of the afiles.

Rewritten, sept 2010, but features not yet complete. now afile,bfiles can be a filename, a list of filenames, or an mtime. And it's not vastly less inefficient than the first, recursive algorithm.
"""
    def oldestmtime(listoffiles): # Return oldest mtime of a list of filenames
        if any([not os.path.exists(afile) for afile in listoffiles]):
            return(-999999)
        return(min([os.path.getmtime(afile) for afile in listoffiles]))
    def newestmtime(listoffiles): # Return newst mtime of a list of filenames
        missingF=[afile  for afile in listoffiles if not os.path.exists(afile)]
        if missingF:
            print('File assumed to exist is missing!: ',missingF)
        assert not missingF #  assert all([os.path.exists(afile) for afile in listoffiles])
        return(max([os.path.getmtime(afile) for afile in listoffiles]))

    # Ensure are lists
    if isinstance(afile,basestring):
        afile=[afile]
    if isinstance(bfiles,basestring):
        bfiles=[bfiles]

    # Compare ages:
    aa=afile
    bb=bfiles
    if isinstance(afile,basestring) or isinstance(afile,list):
        aa=oldestmtime(afile)
    if isinstance(bfiles,basestring) or isinstance(bfiles,list):
        bb=newestmtime(bfiles)
    # So now I hope aa and bb contain the mtimes, or -999999
    isOlder= aa<bb or aa<-99999


    #if isinstance(afile,list):
    #    return(any([fileOlderThan(af,bfiles,ageWarning=ageWarning) for af in afile]))
    #if isinstance(bfiles,list):
    #    return(any([fileOlderThan(afile,bf,ageWarning=ageWarning) for bf in bfiles]))


    # Check for existence of bfiles:
    gotReq=False
    def cpblRequireDummy(afile):
        return
    try:
        from cpblMake import cpblRequire # I could skip this if it doesn't exist, or....

        gotReq=True
    except:
        cpblRequire=cpblRequireDummy
        #print('   Failed to import cpblMake...')
        #pass
    for bf in bfiles:
        if gotReq:
            cpblRequire(bf)



    # Check for exists but getting aged:
    if 1:#####isOlder: # Don't bother with warnings if we know it's older (worse than aged)
        for af in afile:
            if os.path.exists(af) and os.path.getmtime(af) < time.time()-30*24*3600 and ageWarning:
                print """     (N.B.: %s is older than a month. Consider recreating it?)"""%af

    return( isOlder)# not os.path.exists(afile) or os.path.getmtime(afile)<os.path.getmtime(bfiles) )

def fileOlderThanAMonth(af):
    import time
    return((not os.path.exists(af)) or (os.path.getmtime(af) < time.time()-30*24*3600))

def wassert(assertthat,msg):
    if not assertthat:
        cwarning(msg)
        return()
def cwarning(msg):
    print '\n\n'+msg+'\n\n'
    raw_input('Confirm to continue: ')
    print '\n\n'
    return()


def renameDictKey(dd,fromto):
    if isinstance(fromto[0],list) or isinstance(fromto[0],tuple):
        assert isinstance(fromto[0][0],basestring)
        return( sum([renameDictKey(dd,ft) for ft in fromto] ))

    howManyChanged=0
    if not fromto[0]==fromto[1] and fromto[0] in dd:

        dd[fromto[1]]= dd[fromto[0]]
        del dd[fromto[0]]
        howManyChanged+=1
    return(howManyChanged)


################################################################################################
################################################################################################
def dsetset(adict,keys,avalue):
    ############################################################################################
    ############################################################################################
    """
    July 2011: making the converse of dgetget... but I think rather more efficient.
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
    There are two calling formats. The 2011 format is 

       dgetget( mdict, [<list of keys>], default_value)

    while the older format is 

       dgetget( mdict, key1, key2, ...., keyN, default_value)

    The first is preferred, and the second/older version is simply translated into the first, which uses recursion to calculate the result.

    adict must exist and be a dict, of course.

    """
    # Backwards compatibility: Ancient dgetget(adict, key1, key2, key3=None,key4=None,key5=None,key6=None,keyn=None):
    if not hasattr(keys, '__iter__'): #isinstance(keys,list):
        keylist=[keys,defaultvalue]+list(args)
        keylist, defaultvalue= keylist[:-1] ,keylist[-1]
        return( dgetget(adict,keylist,defaultvalue))

    # New, recursive algorithm, which takes a list of keys as second argument:
    if keys[0] not in adict:
        return(defaultvalue)
    if len(keys)==1:
        return(adict[keys[0]])
    return(dgetget(adict[keys[0]],keys[1:],defaultvalue))

        
################################################################################################
################################################################################################
def shelfLoad(infilepath,default=False):
    ############################################################################################
    ############################################################################################
    """ God knows why there isn't already a one-liner for this: loading/saving a single object.
Dec 2011: adding a default option. ie if it doesn't exist, return {}. If default==False, then do not allow not existing.
"""
    import shelve
    if not infilepath.endswith('elf'):
        infilepath+='.pyshelf'
    if not os.path.exists(infilepath):
        if default not in [False]:
            return(default)
        else:
            File_not_found
    fid=shelve.open(infilepath)
    obj=fid['object'] # If this fails, maybe the file was made before I switched from manual shelve to shelfLoad
    fid.close()
    return(obj)
################################################################################################
################################################################################################
def shelfSave(outfilepath,anobj):
    ############################################################################################
    ############################################################################################
    import shelve
    if not outfilepath.endswith('elf'):
        outfilepath+='.pyshelf'
    if os.path.exists(outfilepath):
        os.remove(outfilepath) # It seems it's otherwise not necessarily updated?! 
    fid=shelve.open(outfilepath)
    fid['object']=anobj
    fid.close()
    return()

################################################################################################
""" It seems large pandas files cannot be saved (or re-loaded thereafter!) with pd.save (!!!) 
So let's use hd5 format. Also, let's save to a temporary file before renaming.
"""
################################################################################################
def saveLargePandas(fn,df,asboth=False,onlypickle=False):
    ############################################################################################
    ############################################################################################
    """ Should this check size of df first? """
    print(' Saving to ...'+fn)#d._path)
    import tempfile
    if asboth or onlypickle:
        tempfname=fn.replace('.pandas','.h5')+    os.path.split(tempfile.NamedTemporaryFile().name)[1]
        if 'apollo' in os.uname()[1]: 
            df.save(tempfname)
        else:
            df.write_pickle(tempfname)
        os.rename(tempfname,fn)
        if onlypickle: return
    tempfname=fn.replace('.pandas','.h5')+    os.path.split(tempfile.NamedTemporaryFile().name)[1]
    #d = pd.HDFStore(tempfname)
    #print('    Really saving to ...'+d._path)
    df.to_hdf(tempfname,'object',mode='w',table=True)
    #d['object']=df
    #d.close()
    os.rename(tempfname,fn.replace('.pandas','.h5'))
    print('   Overwrote '+fn.replace('.pandas','.h5'))
    return
def loadLargePandas(fn): # Unlike saveLargePandas, only uses hd5 so far
    print(' Loading from ...'+fn),
    if not os.path.exists(fn.replace('.pandas','.h5')):
        return(None) # Make sure not to open the file, since it would be created!
    d = pd.HDFStore(fn.replace('.pandas','.h5'))
    df=d['object']
    d.close()
    adf=df.copy() # Right now, I think it's keeping the file on disk, at least in part. .copy() gets it out of hdf/tables into RAM.
    print(' ... Done.')
    return(adf) 
def existsLargePandas(fn):
    return( os.path.exists(fn.replace('.pandas','.h5')))


try:   # amb added Jan 2014 - otherwise pylab import fails without X11
    from .mathgraph import *
except:
    print __file__+": can't import cpblUtilities.mathgraph"

if 0:
    try:  
        from .color import *
    except:
        print __file__+": can't import cpblUtilities.color"
    try: #This dependency should be excised....
        import cpblMake
        cpblRequire=cpblMake.cpblRequire
    except:
        print('   (Again?) Failed to import cpblMake (which should be liminated)')
        pass


def doSystemLatex(fname, texcode=None,
                  launchDaemon=False, # keep watching for the source file to change
                  display = True,
):
    #,latexPath=None,launch=None,launchViewer=None,tex=None,viewLatestSuccess=True,bgCompile=True,fgCompile=False):
    """
    Version 2 / from scratch. I'm eliminating arguments which can be taken from config file defaults. Focus on POSIX systems only.
    fname: either a fully-specified path to a .tex file, or a .tex filename located in the defaults['paths']['tex'] folder
    texcode: if fname is to be written, rather than read, this is the LaTeX source code to write there.

    In general, I find latexmk horribly buggy. -aux-directory does not work. -cd does not work until I upgrade to a newer version than the LTS release. It also claims it's failed, when it has apparently finished (according to the next run). However, it's passable, maybe.
As of July 2016, the tmpTEX folder isn't really used at all, except for the fdb_latexmk file.


Ah. Parse this and fix accordingly:


Date: Thu, 14 Jul 2016 14:39:54 -0400
From: John Collins <jcc8@psu.edu>
To: C P Barrington-Leigh <Chris.Barrington-Leigh@McGill.ca>
Subject: Re: latexmk

Hi Chris,

Almost certainly the cause of the problems you are finding is that the -aux-directory option is not supported by the distribution of TeX
you are using, which on linux systems is normally TeXLive.

Latexmk implements the -aux-directory option by passing the option to the underlying latex engine.  So the option doesn't work when it
isn't supported by by latex/pdflatex, as with TeXLive.   The only distribution that I know of that supports -aux-directory is MiKTeX.
The best that can done with TeXLive is to use the -out-directory option.

In fact, this issue is mentioned in latexmk's documentation, but obviously better diagnostics could be done.  There are interesting
possibilities for cross-operating-system uses, and non-standard TeX distributions, which is why I haven't implemented anything
previously.

But I now see that the combination of not finding the log file in the expected place and the use of the -aux-directory option (or the
equivalent setting of $aux_dir) is sufficient to indicate that -aux-directory is not supported by the latex program.  In this situation
a more informative error message for the user would be appropriate.  So I'll put this on the list of future improvements.

Of course, it would be rather nice to have an -aux-directory option, distinct from the -out-directory option.  But I haven't implemented
any fix ups that would let this be done.


> but when run again claims everything's up to date.

That's not quite a bug.  Latexmk sees that no source files have changed since the previous run, so that there is no point trying to run
pdflatex again.  In that sense, it is correct to say that everything is up to date.  But I would agree that since there was an error,
up-to-dateness doesn't seem exactly the right concept

Best regards and thanks for the communication,
John





    """
    # Choose .tex file location, either to read or to write
    ppa,fname=os.path.split(fname)
    latexPath = ppa+'/' if ppa else paths['tex']
    if fname.endswith('.tex'):
        fname=fname[0:-4]
    if texcode is not None: # Clobber freely
        fout=open(latexPath+fname+'.tex','wt')
        fout.write(texcode+'\n')
        fout.close()
    tmpLatexPath=latexPath+'tmpTEX/'
    if not os.path.exists(tmpLatexPath):
        os.mkdir(tmpLatexPath)
    assert os.path.exists(latexPath+fname+'.tex')
    if 1*"Use newest features of latexmk":
        # Use -cd option of latexmk, and assume that we want it compiled whereever the .tex file is, but in the local subfolder tmpTEX
        # -pvc should be optionalized here!
        # Insert " -commands " in the following for debuggin
        pdfout='{latexfolder}tmpTEX/{texfn}.pdf '.format(auxdir=latexPath+'tmpTEX',fullpath=latexPath+fname, latexfolder=latexPath, pvc=' -pvc '*launchDaemon, texfn=fname)
        cli="""
        cd {latexfolder}
        latexmk -silent  -cd -pdf {fullpath} --output-directory=tmpTEX {pvc}   && cp -a {pdfout} {latexfolder}{texfn}.pdf """.format(auxdir=latexPath+'tmpTEX',fullpath=latexPath+fname, latexfolder=latexPath, pvc=' -pvc '*launchDaemon, texfn=fname,pdfout=pdfout)
        print(cli)
        os.system(cli)
        # Only want to do following if success. How to tell?
        if display: # This seems like it should have a separate switch than launchDaemon:
            os.system('evince '+latexPath+fname+'.pdf&') # Don't use latexmk's -v because that will view the version in tmpTEX
    elif "Use own kludges":
        notnecyay
    return

###########################################################################################
###
def doSystemLatex_pre2016(fname,latexPath=None,launch=None,launchViewer=None,tex=None,viewLatestSuccess=True,bgCompile=True,fgCompile=False):
    ###
    #######################################################################################
    """
    Compiles latex. Give it a filename of the .tex file. Optionally, Supply tex code to overwrite that file.

    This is tailored very much to pystata's usage. Be careful before changing the default behaviour...
    cpbl, 2007-2011


    There needs to be a textmp folder inside the latexPath. (not safe yet)

    viewLatestSuccess=True means that a viewer that is launched should not show a broken PDF just because the latest compile didn't work. So turn this off if you don't want to be confused why your changes aren't having an effect (ie because the pop-up compilation process windows are failing to finish, so the -tmp.pdf file is not being copied to the viewing .pdf ...). Also, if you turn this off, you'll get lots of horrid error messages from your pdf viewer as the compilation is going on, since the PDF gets overridden... ACTUALLY, MAYBE THIS IS OBSELETE! SEE NEXT OPTION, bgCompile
    bgCompile=False will mean that it waits for the compile to complete successflly before launching a view with the result.  Note:!! This mode doesn't do what I want, yet.(?)

Then what is launch?
launch=True means ..... DEPRECATED?  THIS STILL NEEDS REVISING IN LIGHT OF NEW LAUNCHVIEWER AND BGCOMPILE AND VIEWLATESTSUCCESS.
launchViewer=None: ie decide based on remote or local terminal. [2011: I now have another python program for this; shoud use it...]
tex=None,
viewLatestSuccess=True,
bgCompile=True : ie compile in the background, either in a new window (if local X) or in the background if remote shell.

So there should be no way to run that doesn't result in compiling, one way or another...

Dec 2010: I don't think I know how to check for local display after all! make yet another argument. Or rely on config settings? or try/except

Oct 2016: Above sounds like a mess. This 
    """
    ppa,ppb=os.path.split(fname)
    assert latexPath is None or ppa ==''
    if latexPath is None:
        if ppa:
            latexPath=ppa
            fname=ppb
        else:
            latexPath=defaults['paths']['tex']


    if fname.endswith('.tex'):
        fname=fname[0:-4]
    if tex is not None:
        fout=open(latexPath+fname+'.tex','wt')
        fout.write(tex+'\n')
        fout.close()

    tmpLatexPath=latexPath+'tmpTEX/'
    if not os.path.exists(tmpLatexPath):
        os.mkdir(tmpLatexPath)

    #import shutil
    #shutil.copyfile(self.fpathname+'.partial.tex',self.fpathname+'.tex')#latexPath+'tables-allCR.tex')
    #os.rename(latexPath+'tablesPreview.tex',latexPath+'tables-allCR.tex')
    # On MS windows, since there's no reasonable interface, do the latex compilation automatically:
    texsh=tmpLatexPath+'TMPcompile_%s.bat'%fname
    shellfile=open(texsh,'wt')


    # For some reason, under Cygwin, the following executable
    # (pdflatex) is running under Windows native. So do not use
    # self.fpathname or specify a cygwin path
    uname=os.uname()
    islaptop=os.uname()[1] in ['cpbl-thinkpad']

    if uname[0].lower()=='linux' or uname[1].lower()=='linux': #defaults['os'] in ['unix']:
        if os.path.exists(latexPath+fname +'.tex'):
            # Crazy tex bug: if filename has "_tmp", this can fail! so use "-tmp".
            shellfile.write("""
            cp %(pp)s%(fp)s.tex %(tp)s%(fp)s-tmp.tex
            cd %(tp)s
            pdflatex %(fp)s-tmp |grep -i "\(Fatal error\)\|\(Output written\)\|\(Rerun \)"
            bibtex %(fp)s-tmp |grep "Error"
            echo             cp %(tp)s%(fp)s-tmp.pdf %(pp)s%(fp)s.pdf
            cp %(tp)s%(fp)s-tmp.pdf %(pp)s%(fp)s.pdf
            """%{'pp':latexPath,'tp':tmpLatexPath,'fp':fname}) #yap tables-allCR\npause
                      #

            if (launchViewer is None and islaptop) or launchViewer:#not bgCompile:
                # Do not include an ampersand following the evince call. Without, if no evince exists, one compile shell window will wait, but subsequent ones won't.
                shellfile.write("""
                evince """+latexPath+fname+'.pdf'+"""
                """)

            else:
                print 'Suppressed launching pdf viewer because not laptop or launchViewer==False'
        else:
            print '.tex file does not exist!!!'# Possible if it has been written onto windows instead..
            foiuweroiu123487698
    else:
        oiuoiuoiuoinnnnnnnnnnnnnnn
        # Used to use: defaults['native']['paths']['tex'] for pp, but defaults no longer available to this function
        shellfile.write('cd %(pp)s\npdflatex %(fp)s\n cp %(fp)s.pdf tmp%(fp)s.pdf\n'%{'pp':latexPath,'fp':fname}) #yap tables-allCR\npause\n

    #shellfile.write('cd %s\npdflatex tables-allCR.tex\npdflatex tables-allCR.tex\n'%latexPath) #yap tables-allCR\npause\n
    #if launch:
    #    pass
    shellfile.close()
    import subprocess
    shellDisplayV=subprocess.Popen(['echo $DISPLAY'], stdout=subprocess.PIPE,shell=True).communicate()[0]
    import socket
    whereami=socket.gethostbyaddr(socket.gethostname())
    os.system('bash %s &'%texsh)    

    # Now run a viewer for the resulting PDF: But this is not just a viewer. Is the line above a kludge?
    print 'shellDisplayV=',shellDisplayV
    if (':0.0' in shellDisplayV or ':0' in shellDisplayV) and bgCompile:#'cpbl-server' in whereami[0]:
        #print 'Compiling LaTeX: '+fname+'...'
        #print os.system('gnome-terminal -e "bash %s" &'%texsh)
        #os.system('bash %s &'%texsh)
        #print os.system('bash %stmpcompile.bat '%latexPath)
        ####if ':0.0' in shellDisplayV and launch and bgCompile:
        if     viewLatestSuccess and launchViewer:
            print os.system('sleep 2&evince '+latexPath+fname+'.pdf&')
        elif  launchViewer:
            print os.system('sleep 2&evince '+tmpLatexPath+fname+'-tmp.pdf&')
    elif fgCompile or ':0.0' in shellDisplayV  or ':0' in shellDisplayV:
        print 'Calling in foreground: '+texsh
        print os.system('bash %s '%texsh)
        print os.system('sleep 2&evince '+latexPath+fname+'.pdf&')

    else:
        print 'Suppressing launch of PDF viewer due to non local X terminal.'
        if launch:
            print '...  but making an attempt to compile in bg anyway. (Caution: stderr/out from LaTeX will follow, mixed in)..'
            print os.system('bash %s &'%texsh)
    return()


###########################################################################################
###
def latexFormatEstimateWithPvalue(x,pval=None,allowZeroSE=None,tstat=False,gray=False,convertStrings=True,threeSigDigs=None):
    ###
    #######################################################################################
    """
    This is supposed to encapsulate the colour/etc formatting for a single value and, optionally, its standard error or t-stat or etc. (take it out of formatpairedrow?)
    It's rather closely connected to cpbl's pystata package and latex_tables package, which produce statistical tables.

    It'll do the chooseSformat as well.

    May 2011.
    Still needs to learn about tratios and calculate own p...! if tstat==True
    """
    yesGrey=gray
    if isinstance(x,list):
        assert len(x)==2
        est,ses= x # Primary estimate, and secondary t-stat/se/p-value
        singlet=False
    else:
        singlet=True
        est=x###chooseSFormat(x,convertStrings=convertStrings,threeSigDigs=threeSigDigs)

    assert isinstance(pval,float) or pval in [] # For now, require p to be passed!

    if 0 and ses<1e-10 and not allowZeroSE:
        pair[0]=''
        pair[1]='' # Added Aug 2009... is not this right? It covers the (0,0) case.
    if pval not in [None,fNaN]: # This is if we specified p-values directly: then don't calculate it from t-stat, etc!
        significanceString=(['']+[tt[0] for tt in significanceTable if pval<= tt[2]*1.0/100.0])[-1]

    if significanceString and yesGrey:
            significanceString=r'\agg'+significanceString[1:]
    if not significanceString and yesGrey:
            significanceString=r'\aggc{'
    if yesGrey:
        greyString=r'\aggc'
    if singlet:
        return(significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs)+'}'*(not not significanceString))

    return([significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs)+'}'*(not not significanceString),
           significanceString+chooseSFormat(est,convertStrings=convertStrings,threeSigDigs=threeSigDigs,conditionalWrapper=[r'\coefp{','}'])])



def collapseByField(e,collapsefield,keepFields=None,agg=None):
    """
    This is a maybe-unneccesary wrapper for rec_groupby. I suppose it serves as an example more than anything. But it could be generalised to do analogous jobs for some other data structures...
    """
    import numpy as np
    assert isinstance(e,np.ndarray) # Structured array
    if agg is None:
        agg=np.mean
    if keepFields is None:
        newf=[(n,agg,n) for n in e.dtype.names if n not in (collapsefield)]

    #if 0:
    #from itertools import groupby
    #    alist= [(k,e[meanfield][list(g)].mean()) for k, g in groupby(np.argsort(e),e[collapsefield].__getitem__ )]

    import matplotlib as mpl
    return(mpl.mlab.rec_groupby(e,[collapsefield],newf))


def pandasReadTSV(tsvF,dtypeoverrides=None):
    """
    read_table is nice, but like the numpy version, I can't specify *some* of the dtypes
    """
    import pandas as pd
    fn=tsvF if tsvF.endswith('.tsv') or tsvF.endswith('.csv') else tsvF+'.tsv'
    dataDF=pd.read_table(fn)
    if dtypeoverrides:
        dt=dict(dataDF.dtypes)
        dtypeoverrides=dict([kk,np.dtype(vv)] for kk,vv in dtypeoverrides.items())
        dt.update(dtypeoverrides)
        dataDF=pd.read_table(fn,dtype=dt)
    return(dataDF)

def google_api_spreadsheet_to_pandas(credentials_json_file_or_dict, spreadsheetname, tabname):
    """
    Follow the directions here and linked here: https://github.com/burnash/gspread
    You'll need to share your doc with the strange email address in your credentials file.
    """
    import gspread # Installed with pip!  "pip install --user gspread "
    from oauth2client.service_account import ServiceAccountCredentials # Installed with pip!  "pip install --user oauth2client"
    scope = ['https://spreadsheets.google.com/feeds']
    if isinstance(credentials_json_file_or_dict,dict):
        credentials = ServiceAccountCredentials.from_json_keyfile_dict(credentials_json_file_or_dict, scope)
    else:
        credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_json_file_or_dict, scope)
        
    gc = gspread.authorize(credentials)
    wks = gc.open(spreadsheetname).worksheet(tabname)
    df=pd.DataFrame(wks.get_all_values())
    df.columns=df.iloc[0,:]
    df=df.iloc[1:]
    return(df)

def ods2pandas(infile,sheetname=None,tmpfile=None,skiprows=None, forceUpdate=False):#,header=None):
    """
    Pandas still cannot read ODF (grr)

# Or, the version I had already written here:
# http://stackoverflow.com/questions/17834995/read-opendocument-spreadsheets-to-pandas-dataframe
import pandas as pd
import os
if fileOlderThan('tmp.xlsx','myODSfile.ods'):
    os.system('unoconv -f xlsx -o tmp.xlsx myODSfile.ods ')
xl_file = pd.ExcelFile('tmp.xlsx')
dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
df=dfs['Sheet1']

    """
    if tmpfile is None:
        import tempfile
        outfile=  tempfile.NamedTemporaryFile().name+'.xlsx'
    else:
        outfile=tmpfile
    assert outfile.endswith('.xls') or outfile.endswith('.xlsx')
    if fileOlderThan(outfile,infile) or forceUpdate:
        os.system('ssconvert '+infile+' '+outfile)
    df=pd.read_excel(outfile,sheetname=sheetname,skiprows=skiprows)#,header=header)
    return(df)


def downloadOpenGoogleDoc(url, filename=None, format=None, pandas=False, update=True):
    """
    If you: Set a Google doc or Google drive document to be readable to anyone with the link:

    Then this function will return a version of it from online, if networked, or the latest download, if not.

    The filename (currently mandatory; should be fixed to be a hash of the url by default) should not include a path.

    This returns the full-path filename of the downloaded file, unless pandas is True for spreadsheets, in which case it returns a pandas version.

    If update False, and the file has already been downloaded, download will be skipped.
    """
    from  xlrd import XLRDError
    if not url.endswith('/'): url=url+'/'
    if format is None:
        format='xlsx'
    full_file_name        =paths['scratch']+filename
    if update or not os.path.exists(full_file_name):
        oss=' wget '+url+'export?format='+format+' -O '+full_file_name+'-dl'
        print(oss)
        result=os.system(oss)
        if result:
            print('  NO INTERNET CONNECTION, or other problem grabbing Google Docs file. Using old offline version ('+filename+')...')
        else:
            result=os.system(' mv '+ full_file_name+'-dl ' + full_file_name)
            assert not result
    else:
        print('   Using local (offline) version of '+filename)
    if pandas and format in ['xlsx']:
        return(   pd.ExcelFile(full_file_name)  )

    return(full_file_name)



def merge_dictionaries(default,update, verboseSource=False, allow_new_keys=True):
    """Given two dictionaries, this deep copies 'default' but recursively updates it with elements from
    'update'.  

    allow_new_keys = False ensures that only keys existing in the default are updated from the update, ie it ignores any data found in update which does not exist in the original.

    verboseSource: If not False, verboseSource must be a string, which denotes the updating source file description

    n.b. This function is duplicated in cpblUtilities_config.py, since I simply don't know how to import it otherwise.
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
    return result


def mergePDFs(listOfFns, outFn, allow_not_exist=True, delete_originals = True):
    """ merges all the PDFs in listOfFns, saves them to outFn and deletes orginals"""
    missing = [ff  for ff in listOfFns if not os.path.exists(ff)    ]
    if missing:
        print(' CAUTION: Following do not exist'+allow_not_exist*', so  being excluded from a merged PDF file'+':\n         '+'\n         '.join(missing) + '\n')
    if allow_not_exist:
        listOfFns = [ff for ff in  listOfFns if ff not in missing] # Preserve order
    if len(listOfFns)==0:
        print('   --- No existing PDFs to merge -- ')
        return
    try:  # marmite  # Sorry: There's a bug in line 69 of PyPDF2/utils.py, so I'm un-defaulting this.
        try:
            cmd = 'pdftk ' + ' '.join(listOfFns) + ' cat output ' + outFn
            if delete_originals:
                for fn in listOfFns: cmd+= ' && rm {} '.format(fn)
            os.system(cmd)
            print('(pdftk) Wrote '+outFn)
        except:
            print('Could not merge PDFs %s. pdftk not available. You might want to merge manually.' %   listOfFns)      
    except:  # apollo or cpblx230:
        print('Failing pdftk ... trying pypdf2')
        import PyPDF2
        merger = PyPDF2.PdfFileMerger()
        for fn in listOfFns: 
            merger.append(fn)
        merger.write(outFn)
        print('(pypdf) Wrote '+outFn)
        if delete_originals:
            for fn in listOfFns: os.remove(fn)
    return

def merge_pdfs_if_exist(infiles, outfile):
    osc = 'pdftk '+' '.join([ff for ff in infiles if ff not in missing])+ ' cat output {}'.format(outfile)
    os.system(osc)
    print('   SYSTEMCALL: '+osc+'\n')


# A couple of menu-ing / choice utilities: Make shorter one-liners from inquire
def inquirer_confirm(message, default=False):
    import inquirer
    ans = inquirer.prompt([inquirer.Confirm('singleQuestion',
                                            message= message,
                                            default=default)])
    return(ans['singleQuestion'])
def inquirer_list(choices, message=None, default=None):
    import inquirer
    message = 'Please choose' if message is None else message
    ans = inquirer.prompt([inquirer.List('singleQuestion',
                                         choices=choices,
                                            message= message,
                                            #default=default
    )])
    return(ans['singleQuestion'])
    
if 0:
    try: # Where is this needed? Should import it only where needed.        
        from parallel import *
    except ImportError:
        print(__file__ +":Unable to find CPBL's runFunctionsInParallel (cpblUtilities.parallel) module")

    try: # Where is this needed? Should import it only where needed.        
        from cpblUtilitiesUnicode import *
    except ImportError:
        print(__file__+":Unable to find CPBL's ragged unicode converstions module")


