#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import re
from copy import deepcopy

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

def y2k(ss):
    year=int(ss)
    if year < 1900:
        year=[[year+cc for cc in [1900,2000] if 1950<year+cc<2050][0]][0]
    else:
        print 'year ',year,' does not need y2k'
    return(year)

def _deprecated2013_cleanFields(LL):
    """ This simplifies whitespace and trims fields in a list of lists.. has this been done already? no.  Well, actually, it made no difference, so probably yes.
    """
    for i in range(len(LL)):
        LL[i]=list(LL[i]) # Convert a list of tuples into a list of lists
        for j in range(len(LL[i])):
            if type(LL[i][j])==type('a string'):
                LL[i][j]= re.sub('\s+',' ',LL[i][j]).strip()
    return(LL)



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

def _deprecated2013_removeLikeFields(a,b):
    """ Strip two dictionaries down to their non-identical elements .
    """
    # Careful! Python "=" assignment does not make copies, usually. The following only makes a "shallow" copy of the dictionaries. A simple assignment would not copy at all, just point.
    aa=a.copy()
    bb=b.copy()
    same={}
    for k in aa.keys():
        #if aa.has_key(k) and bb.has_key(k):
            #print 'Key: |%s|     values: |%s|   |%s|'%(k, aa[k],bb[k])
        if aa.has_key(k) and bb.has_key(k) and aa[k]==bb[k]:
            same[k]=a[k]
#            print 'Deleting',same[k]
            del aa[k]
            del bb[k]
#    print 'Removed %d=%d=%d fields from each of (%d,%d)'%(len(a)-len(aa),len(b)-len(bb),len(same),len(a),len(b))
#    print 'In common: ',same
    return(aa,bb,same)

###########################################################################################
###
def doSystem(acommand,verbose=False,bg=None):
    ###
    #######################################################################################
    """
    2011/2: Added some facility for background launching.
    """
    import os
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
    import sys
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


def processInjector(toRun,nSimultaneous,psGrepString):
    """
    toRun is a list of commands to run. This keeps n of them going at once
    """
    import commands
    import os
    import re
    import time

    #toRun=open(fileOfCommands,'r').readlines()
    nRunning=0
    while toRun:
        rval,ps=commands.getstatusoutput('ps -Afj |grep "%s" |grep -v "grep"|grep -v "injector"'%psGrepString)
        print ps
        nRunning=len(ps.split('\n'))
        if nRunning<nSimultaneous:
            ap=toRun.pop(0)+'&'
            os.system(ap)
            print 'Launched %s'%ap
            time.sleep(1)
        else:
            print '%d running; %d left'%(nRunning,len(toRun))
            time.sleep(10)


##############################################################################
##############################################################################
#
def popWordString(sfrom,sstrings):
    ##########################################################################
    ##########################################################################

    """Use function below, but ensure parens are literal.
    """
##  if 0:
##         outlist=[]
##         for ss in sstrings:
##             # Replace/Match word, possibly at end of string
##             reg=r'[\b\s]'+ss+r'[\b\s]'
##             sfrom,ns=re.compile(reg,re.IGNORECASE|re.UNICODE).subn(' ',sfrom)
##             if ns:
##                 outlist.append(ss)
##         return (sfrom.strip(),outlist)
    # Escape all regexp control characters: (this could be done with a re.sub !)
    regexps=[]
    for ss in sstrings:
        for ic in '*?+&$|().[]':
            ss=ss.replace(ic,'\\'+ic)
        regexps.append(ss)
    #regexps=[s.replace('(',r'\(').replace(')',r'\)') for s in sstrings]
    return(popWordRE(sfrom,regexps))

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
def unique_obselete(alist,key=None):    # Fastest order preserving
    """ NOTE!!! I have renamed this function as below, since numpy has a unique() which sorts.
    This copy with old name, "unique()", is for backwards compat.

    """
    ##########################################################################
    ##########################################################################
    assert ''=='REPLACE THIS CALL WITH UNIQUEINORDER'
    if key==None:
        set = {}
        return [set.setdefault(e,e) for e in alist if e not in set]
    else: # Find dicts with unique values of key:
        NotDoneYet
        # set = {}
        #return [set.setdefault(e,e) for e in alist if e[key] not in [s[key] for s in set]]
##############################################################################
##############################################################################
#
def uniqueInOrder(alist,key=None,drop=None):    # Fastest order preserving
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
def chooseSFormat(ff,conditionalWrapper=['',''],lowCutoff=None,lowCutoffOOM=True,convertStrings=False,highCutoff=1e8,noTeX=False,threeSigDigs=False,se=None, leadingZeros=False):
    ###
    #######################################################################################
    """ This chooses a reasonable number of significant figures for a numerical value in a results table...
    for LaTeX format.
    It takes "." to be a NaN, which it represents with a blank.
    It aims not to show more than three significant figures.
    It deals nicely with negatives.

     If conditionalWrapper is supplied, it is only given, enclosing the output, if the output is not empty.
If, instead of conditionalWrapper, "convertStrings" is supplied, then strings will be converted to floats or ints.
Error handling for conversions is not yet done.

If lowCutoff is supplied, smaller numbers than it will be shown as "zero".
If lowCutoffOOM in [True,'log'], we will use "<10^-x" rather than "zero" in above. It will, awkwardly, report $-<$10$^{-6}$ for -1e-7.
If lowCutoffOOM is some other string, then it will be used for small numbers. E.g, provide lowCutoffOOM='$<$10$^{-6}$' to show that for numbers smaller than the lowCutoff. Note that in this case the output is not sensitive to sign.

If highCutoff is supplied, larger numbers will be shown as "big". May 2011: reducing this from 1e6 to 1e8, since it was binding on sample sizes.

se is the standard error. you can just specify that for smarter choices about sig digs to show...

2014-03 adding leadingZeros: In regression tables, I don't want them. But in general I might.
"""
    if lowCutoff==None:
        lowCutoff==1.0e-99 # Sometimes "None" is explicitly passed to invoke default value.
    import numpy#from numpy import ndarray
    if isinstance(ff,list) or isinstance(ff,numpy.ndarray): # Deal with lists
        return([chooseSFormat(fff,conditionalWrapper=conditionalWrapper,lowCutoff=lowCutoff,convertStrings=convertStrings,threeSigDigs=threeSigDigs) for fff in ff])
    if ff=='': # Leave blanks unchanged
        return('')
    if ff=='.': # lone dots can mean NaN to Stata
        return('')
    if not isinstance(ff,int) and not isinstance(ff,float) and not convertStrings:
        return(conditionalWrapper[0]+str(ff)+conditionalWrapper[1])
    if isinstance(ff,basestring):
        #print "converting ",ff," to num:",
        if '.' in ff:
            ff=float(ff)
        else:
            ff=int(ff)
        #print '--> ',ff
    aa=abs(ff)
    if aa>highCutoff:
        return('big')
    if not aa>=0: # ie is "nan"
        return('')
    ss='%.1g'%ff
    if aa<lowCutoff:
        ss='0'
        if lowCutoffOOM in [True,'log']:
            negexp=int(np.ceil(np.log10(aa)))
            ss='-'*(ff<0)+ r'$<$10$^{%d}$'%negexp
        elif isinstance(lowCutoffOOM,basestring):
            ss=lowCutoffOOM
    if aa>=0.0001:
        ss=('%.4f'%ff)
    if aa>=0.001:
        ss=('%.3f'%ff)
    if aa>=0.01:
        ss='%.3f'%ff
    if aa>=0.1:
        ss='%.2f'%ff
    if threeSigDigs and aa>=0.1:
        ss='%.3f'%ff
    if aa>2.0:
        ss='%.1f'%ff
    if aa>10.0:
        ss='%.1f'%ff
    if aa>100.0:
        ss='%.0f'%ff
    if ss[0:2]=='0.' and not leadingZeros:
        ss=ss[1:]
    if ss[0:3]=='-0.' and not leadingZeros:
        ss='-'+ss[2:]
    if ss[0]=='-' and not noTeX:
        ss='$-$'+ss[1:]

    # Override all this for integers:
    if isinstance(ff,int):
        ss='$-$'*(ff<0)+str(abs(ff))


    return(conditionalWrapper[0]+ss+conditionalWrapper[1])



###########################################################################################
###
def orderListByRule(alist,orderRule,listKeys=None,dropIfKey=None):
    ###
    #######################################################################################

    """ Reorder alist according to the order specified in orderRule. The orderRule lists the order to be imposed on a set of keys. The keys are alist, if listkeys==None, or listkeys otherwise.  That is, the length of listkeys must be the same as of alist. That is, listkeys are the tags on alist which determine the ordering.  orderRule is a list of those same keys and maybe more which specifies the desired ordering.
    There is an optional dropIfKey which lists keys of items that should be dropped outright.
    """
    alist=list(alist)
    assert isinstance(alist,list) and isinstance(orderRule,list)
    alist=deepcopy(alist) #Let's not overwrite the passed array! to be safe and inefficient

    if listKeys==None:
        listKeys=deepcopy(alist)
    assert isinstance(listKeys,list) and len(listKeys)==len(alist)

    # First, drop outright any baddies. Need to do this in parallel for both listKeys and alist, which are of same length:
    if dropIfKey:
        for iopr in range(len(alist))[::-1]:
            if listKeys[iopr] in dropIfKey:
                alist.pop(iopr)
                listKeys.pop(iopr)

    newalist= [[] for _ in range(len(orderRule)+len(alist))] # Create list of empty lists
    debugprint( 'From order ',str(listKeys))
    # Order pairs that are specified in orderRule. When one is found in alist, place it in its ordered spot in newalist and delete it from (replace with []) alist:
    for iov in range(len(orderRule)):
        if orderRule[iov] in listKeys:
            iopr=[iii for iii in range(len(alist)) if listKeys[iii]==orderRule[iov]][0]
        #for iopr in range(len(alist)):
            #if alist[iopr] and listKeys[iopr]==orderRule[iov]:
            if 1:
                newalist[iov]=deepcopy(alist[iopr])
                alist[iopr]=[]
                #continue

    # Now add remaining (unspecified) pairs to end of newalist list and get rid of blanks:
    newalist=[pair for pair in newalist+alist if pair]
    #print( '    to order ',str([aa[0][0] for aa in newalist]))

    # SHould I also ensure uniqueness of list?


    return(newalist)



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
    WTF!? See ImportSpreadsheet, above! and tsvDict ... How many times have I reinvnted the weheel??
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
def flatten_deprectaed_BECAUSE_PYLAB_HAS_ONE(l, ltypes=(list, tuple),toList=False):
    """
    June 2010: option tolist gives a list rather than a generator!
    """
    if toList==True:
        return([LL for LL in flatten(l, ltypes=ltypes,toList=False)])

    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def oldFlattenList(listoflists,n=-1,unique=False,noCopy=False):
    debugprint( 'May 2010: I suspect that the "flatten" above, which returns a generator but does infinite flattening, is simply superior, at least for infinite depth..!!! THIS SHOULD PROBABLY BE PARTLY DEPRECATED. ')

    """Flatten a list by up to n (or infinity, when n==-1) dimensions
    June 2009: CPBL
    This doesn't work fulll yet, since the reduce() expects the list to be all lists. So this function does not reduce a list of which only some are lists.

    Not sure whether unique=True should be default, so always use the keyword explicitly for now.
    Dec 2009: of course unique should not be default!


    May 2010: noCopy is needed for, e.g. lists of graphics objects! You cannot sensibly deepcopy them. But they are often also tuples, which cannot use the method below...  AH!! Just use "flatten" with tolist=True for graphics objects nested list of lists.

    May 2010: trying to incorporate tuples. DOES NOT WORK YET!
    """
    import operator

    """ See above: must be all or none lists right now:
    """
    assert all([isinstance(ee,list) for ee in listoflists]) or all([not isinstance(ee,list) for ee in listoflists])

    def islist(oo):
        return(isinstance(oo,list) or isinstance(oo,tuple) ) # or isinstance plt.array??

    assert all([islist(ee) for ee in listoflists]) or all([not islist(ee) for ee in listoflists])

    if noCopy:
        outlist=listoflists
    else:
        outlist=deepcopy(listoflists)

    #altered=False
    if all([islist(ee) for ee in outlist]) and not n==0:
        #print outlist
        outlist=reduce(operator.add, outlist, [])
        #altered=True
        #print outlist


    if any([islist(ee) for ee in outlist]) and not n==0:
        outlist=flattenList(outlist,n=n-1,noCopy=noCopy) # Don't pass on unique... 'cause I just deal with that once at top level:
        #altered=True

    #print outlist
    if unique==True:
        outlist=uniqueInOrder(outlist)
    #print outlist
    return(outlist)


def str2latexOLD_see_unicode_file(ss):
    subs={
'$':r'\$',
'_':r'\_',
'%':r'\%',
'#':'\#',
'^':'\^',
'&':'\&',
'<=':r'$\leq$',
'>=':r'$\geq$',
'<':r'$<$',
'>':r'$>$',
#u'\xb4':"'", # apostrophe
#           u'\xe3':r'\"e', # e-umlaut ?
#           u'\xf3':r"\'o", # o acute
#           u'\xe3':r"\`O", # O grave. 227
#           u'\x92':"AE" # AE
           }
    subs.update(dict([
[ u"à", "\\`a" ], # Grave accent
[ u"è", "\\`e" ],
[ u"ì", "\\`\\i" ],
[ u"ò", "\\`o" ],
[ u"ù", "\\`u" ],
[ u"ỳ", "\\`y" ],
[ u"À", "\\`A" ],
[ u"È", "\\`E" ],
[ u"Ì", "\\`\\I" ],
[ u"Ò", "\\`O" ],
[ u"Ù", "\\`U" ],
[ u"Ỳ", "\\`Y" ],
[ u"á", "\\'a" ], # Acute accent
[ u"é", "\\'e" ],
[ u"í", "\\'\\i" ],
[ u"ó", "\\'o" ],
[ u"ú", "\\'u" ],
[ u"ý", "\\'y" ],
[ u"Á", "\\'A" ],
[ u"É", "\\'E" ],
[ u"Í", "\\'\\I" ],
[ u"Ó", "\\'O" ],
[ u"Ú", "\\'U" ],
[ u"Ý", "\\'Y" ],
[ u"â", "\\^a" ], # Circumflex
[ u"ê", "\\^e" ],
[ u"î", "\\^\\i" ],
[ u"ô", "\\^o" ],
[ u"û", "\\^u" ],
[ u"ŷ", "\\^y" ],
[ u"Â", "\\^A" ],
[ u"Ê", "\\^E" ],
[ u"Î", "\\^\\I" ],
[ u"Ô", "\\^O" ],
[ u"Û", "\\^U" ],
[ u"Ŷ", "\\^Y" ],
[ u"ä", "\\\"a" ],    # Umlaut or dieresis
[ u"ë", "\\\"e" ],
[ u"ï", "\\\"\\i" ],
[ u"ö", "\\\"o" ],
[ u"ü", "\\\"u" ],
[ u"ÿ", "\\\"y" ],
[ u"Ä", "\\\"A" ],
[ u"Ë", "\\\"E" ],
[ u"Ï", "\\\"\\I" ],
[ u"Ö", "\\\"O" ],
[ u"Ü", "\\\"U" ],
[ u"Ÿ", "\\\"Y" ],
[ u"ç", "\\c{c}" ],   # Cedilla
[ u"Ç", "\\c{C}" ],
[ u"œ", "{\\oe}" ],   # Ligatures
[ u"Œ", "{\\OE}" ],
[ u"æ", "{\\ae}" ],
[ u"Æ", "{\\AE}" ],
[ u"å", "{\\aa}" ],
[ u"Å", "{\\AA}" ],
[ u"–", "--" ],   # Dashes
[ u"—", "---" ],
[ u"ø", "{\\o}" ],    # Misc latin-1 letters
[ u"Ø", "{\\O}" ],
[ u"ß", "{\\ss}" ],
[ u"¡", "{!`}" ],
[ u"¿", "{?`}" ],
[ u"\\", "\\\\" ],    # Characters that should be quoted
[ u"~", "\\~" ],
[ u"&", "\\&" ],
[ u"$", "\\$" ],
[ u"{", "\\{" ],
[ u"}", "\\}" ],
[ u"%", "\\%" ],
[ u"#", "\\#" ],
[ u"_", "\\_" ],
[ u"≥", "$\\ge$" ],   # Math operators
[ u"≤", "$\\le$" ],
[ u"≠", "$\\neq$" ],
[ u"©", "\copyright" ], # Misc
[ u"ı", "{\\i}" ],
[ u"µ", "$\\mu$" ],
[ u"°", "$\\deg$" ],
[ u"‘", "`" ],    #Quotes
[ u"’", "'" ],
[ u"“", "``" ],
[ u"”", "''" ],
[ u"‚", "," ],
[ u"„", ",," ],
]))
    for asub in subs:
        ss=unicode(ss).replace(asub,subs[asub])   #.encode('utf-8')
        #ss=accentsToLaTeX(    ss.replace(asub[0],asub[1])   )
    return(ss)

def str2pathname(ss):
    subs=[
        ['_','-'],
        [r'$>$','gt'],
        ]
    todrop="""?"':.,/()’ """ #'
    for asub in subs:
        ss=ss.replace(asub[0],asub[1])
    ss=''.join([sss for sss in ss if sss not in todrop])
    return(ss)


def fileOlderThan(afile,bfiles,ageWarning=False):
    """ ie says whether the first fiel, afile, needs updating based on its parent, bfile.  ie iff afile does not exist or is older than bfile, this function returns true.

bfile can also be a list of files. in this case, the function returns true if any  of the bfiles is younger than the target file, afile; ie afile needs to be updated based on its antecedents, bfiles.

afile can also be a list of files... then the function returns true if any of the bfiles is younger than any of the afiles.

Rewritten, sept 2010, but features not yet complete. now afile,bfiles can be a filename, a list of filenames, or an mtime. And it's not vastly less inefficient than the first, recursive algorithm.
"""
    import os, time

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
def dgetget(adict,keys,defaultvalue,key3=None,key4=None,key5=None,key6=None,keyn=None):
    ############################################################################################
    ############################################################################################
    """
    July 2011: rewriting degetget, using recursion, and conforming only to the newer format in which a  list of the keys is passed.
    Much more efficient than the old version which took nargs!
    adict must exist and be a dict, of course.
    """
    # Backwards compatibility:
    if not isinstance(keys,list):
        return( dgetgetOLD(adict,keys,defaultvalue,key3=key3,key4=key4,key5=key5,key6=key6,keyn=keyn))

    # New, recursive algorithm, which takes a list of keys as second argument:
    if keys[0] not in adict:
        return(defaultvalue)
    if len(keys)==1:
        return(adict[keys[0]])
    return(dgetget(adict[keys[0]],keys[1:],defaultvalue))

################################################################################################
################################################################################################
def dgetgetOLD(adict,key1=None,key2=None,key3=None,key4=None,key5=None,key6=None,keyn=None):
    ############################################################################################
    ############################################################################################
    """
    DEPRECATED 2011. BUT STILL EXTANT IN MAY FUNCTIONS.

    Get a value from a dict of dicts, compact notation. This is just like dict.get but takes two keys deep.
    bykey(adict, topkey, subkey, defaultVal)
    Last parameter passed is the default return value. Unfortunately this is mandatory.

    May 2010: new optional format:
      dgetget(adict,listOfKeys,defaultVal)

    """
    assert not isinstance(key1,list) # Otherwise should be calling the newer version, dgetget

    if key3==None and isinstance(key1,list):
        # Assume calling format:       dgetget(adict,listOfKeys,defaultVal)
        if len(key1)==2:
            key1,key2,key3=key1[0],key1[1],key2
        elif len(key1)==3:
            key1,key2,key3,key4=key1[0],key1[1],key1[2],key2
        elif len(key1)==4:
            key1,key2,key3,key4,key5=key1[0],key1[1],key1[2],key1[3],key2
        elif len(key1)==5:
            return(adict.get(key1[0],{}).get(key1[1],{}).get(key1[2],{}).get(key1[3],{}).get(key1[4],key2)) # key2 is actually the default value
        elif len(key1)==6:
            return(adict.get(key1[0],{}).get(key1[1],{}).get(key1[2],{}).get(key1[3],{}).get(key1[4],{}).get(key1[5],key2)) # key2 is actually the default value
        else:
            1/0


    assert not key3==None # Why not just use dict.get if you're only asking for 1 deep?

    assert keyn==None # Not programmed yet for more than 2 keys deep

    if not key5==None: # Four deep, key5 is default
        return(adict.get(key1,{}).get(key2,{}).get(key3,{}).get(key4,key5)) # key4 is actually the default value
    if key5==None and not key4==None: # Three deep, key4 is default
        return(adict.get(key1,{}).get(key2,{}).get(key3,key4)) # key4 is actually the default value
    if key4==None: # Two deep, key3 is default
        return(adict.get(key1,{}).get(key2,key3)) # key3 is actually the default value
    # Three deep, key4 is default
    1/0
    if key3==None:
        return(adict.get(key1,key2))




try: # Where is this needed? Should import it only where needed.        
    from cpbl_tables import *
except ImportError:
    print(__name__+":Unable to find CPBL's tables TeX package")


        
################################################################################################
################################################################################################
def shelfLoad(infilepath,default=None):
    ############################################################################################
    ############################################################################################
    """ God knows why there isn't already a one-liner for this: loading/saving a single object.
Dec 2011: adding a default option. ie if it doesn't exist, return {}. If default==False, then do not allow not existing.
"""
    import shelve
    if not infilepath.endswith('elf'):
        infilepath+='.pyshelf'
    if not os.path.exists(infilepath):
        assert not default==False
        return(default)
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
    from mathgraph import *
except:
    print __name__+": can't import cpblUtilities.mathgraph"
try:  
    from color import *
except:
    print __name__+": can't import cpblUtilities.color"

if 0: #defaults['mode'] in ['gallup','rdc']:
    try: #This dependency should be excised....
        import cpblMake
        cpblRequire=cpblMake.cpblRequire
    except:
        print('   (Again?) Failed to import cpblMake (which should be liminated)')
        pass


# I meant to implement the following (somewhere!!) to use for the master codebook. not yet done. priorities! but below is finished...
###########################################################################################
###
def doSystemLatex(fname,latexPath=None,launch=None,launchViewer=None,tex=None,viewLatestSuccess=True,bgCompile=True,fgCompile=False):
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

Dec 2010: Damn. I don't think I know how to check for local display after all! make yet another argument.

    """
    import os
    ppa,ppb=os.path.split(fname)
    assert latexPath is None or ppa ==''
    if latexPath is None:
        if ppa:
            latexPath=ppa
            fname=ppb
        else:
            latexPath=defaults['native']['paths']['tex']


    if fname.endswith('.tex'):
        fname=fname[0:-4]
    if tex is not None:
        fout=open(latexPath+fname+'.tex','wt')
        fout.write(tex+'\n')
        fout.close()

    tmpLatexPath=latexPath+'tmpTEX/'
    if not os.path.exists(tmpLatexPath):
        os.mkdir(tmpLatexPath)


    import shutil
    #shutil.copyfile(self.fpathname+'.partial.tex',self.fpathname+'.tex')#latexPath+'tables-allCR.tex')
    #os.rename(latexPath+'tablesPreview.tex',latexPath+'tables-allCR.tex')
    # On MS windows, since there's no reasonable interface, do the latex compilation automatically:
    texsh=tmpLatexPath+'TMPcompile_%s.bat'%fname
    shellfile=open(texsh,'wt')


    # For some reason, under Cygwin, the following executable
    # (pdflatex) is running under Windows native. So do not use
    # self.fpathname or specify a cygwin path
    import os
    uname=os.uname()
    islaptop=os.uname()[1] in ['cpbl-thinkpad']

    if uname[0].lower()=='linux' or uname[1].lower()=='linux': #defaults['os'] in ['unix']:
        if os.path.exists(latexPath+fname +'.tex'):
            # Crazy tex bug: if filename has "_tmp", this can fail! so use "-tmp".
            shellfile.write("""
            cp %(pp)s%(fp)s.tex %(tp)s%(fp)s-tmp.tex
            cd %(tp)s
            pdflatex %(fp)s-tmp |grep "\(Fatal error\)\|\(Output\)\|\(Rerun\)"
            bibtex %(fp)s-tmp |grep "rror"
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
    import os
    import subprocess
    shellDisplayV=subprocess.Popen(['echo $DISPLAY'], stdout=subprocess.PIPE,shell=True).communicate()[0]
    import socket
    whereami=socket.gethostbyaddr(socket.gethostname())
    print 'shellDisplayV=',shellDisplayV
    if '0.0' in shellDisplayV and bgCompile:#'cpbl-server' in whereami[0]:
        print 'Compiling LaTeX: '+fname+'...'
        
        print os.system('gnome-terminal -e "bash %s" &'%texsh)
        #print os.system('bash %stmpcompile.bat '%latexPath)
        ####if ':0.0' in shellDisplayV and launch and bgCompile:
        if     viewLatestSuccess and launchViewer:
            print os.system('sleep 2&evince '+latexPath+fname+'.pdf&')
        elif  launchViewer:
            print os.system('sleep 2&evince '+tmpLatexPath+fname+'-tmp.pdf&')
    elif fgCompile or ':0.0' in shellDisplayV:
        print 'Calling in foreground: '+texsh
        print os.system('bash %s '%texsh)

    else:
        print 'Suppressing launch of PDF viewer due to non local X terminal.... right?'
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

###########################################################################################
###
def  manageParallelJobChunks(computationFunction,chunkIDs,nProc=30,shuffle=False):
    ###
    #######################################################################################
    """
    This takes the number or processors you want to keep running steadily, a function that does some calculation, and a list of values for an argument that the function takes. These values represent chunks of a job (e.g. counties in the US for some GIS calculation).
    Right now, there's no quality checking on the completion. Instead, when a job finishes (successfully or not), it will be assumed to have worked, and another will be launched.
    Right now, we don't deal with return values. So this is very simple. It just load balances...
    There's a pause of 1 second in between relaunches, so that if they're all failing instantly, we don't go too nuts.

Comment: isn't this entirely superceded by runFunctionsInParallel, which now has all the functionality?
    
    """
    jobs=[]
    njobs=0
    t=0
    from math import sqrt
    if shuffle:
        import random
        random.shuffle(chunkIDs)
    import time
    from multiprocessing import Process, Queue
    while chunkIDs:
        # Launch a new job
        if len([jj for jj in jobs if jj.is_alive()])<nProc: # Time to launch a new job!
            cname=chunkIDs.pop(0)
            jobs.append(Process(target=computationFunction,args=[cname],name=str(cname)))
            jobs[-1].start()

        # Display running jobs

        tableFormatString='%s\t%'+str(max([len(job.name) for job in jobs]))+'s:\t%9s\t%s\t%s\t%s'
        print('\n'+'-'*75+'\n'+ tableFormatString%('alive?','Job','exit code','Full','Empty','Func',)+ '\n'+'-'*75)
        print('\n'.join([tableFormatString%(job.is_alive()*'Yes:',job.name,job.exitcode,'','','(built-in function)' if not hasattr(computationFunction,'func_name') else computationFunction.func_name) for ii,job in enumerate(jobs) if job.is_alive()]))





        #queues[iii].full(),queues[iii].empty()
        print('-'*75+'\n')

        time.sleep(.2)

        # Pause before next display
        if len([jj for jj in jobs if jj.is_alive()])>=nProc: # Wait more than the minimum 1 second; and report status
            t+=1
            time.sleep(5+sqrt(t)) # Wait a while before next update. Slow down updates for really long runs.

    # We've launched them all!
    for job in jobs: job.join() # Wait for them all to finish... Hm, Is this needed to get at the Queues?




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
    import os
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


try: # Where is this needed? Should import it only where needed.        
    from parallel import *
except ImportError:
    print(__name__ +":Unable to find CPBL's runFunctionsInParallel (cpblUtilities.parallel) module")

try: # Where is this needed? Should import it only where needed.        
    from cpblUtilitiesUnicode import *
except ImportError:
    print(__name__+":Unable to find CPBL's ragged unicode converstions module")


