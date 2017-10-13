#!/usr/bin/python
#
# CPBL 2013-2014 September
# Incorporated old cpblTables.py, ie interface for prducing tables to be used by cpblTables.tex.
# Take a TSV/CSV file or (2014) a pandas DataFrame and generate a .tex include file that is used by my cpblTables.sty tools.
"""
2014 April: Provide tools for extracting a set of cells (location
 fixed) from an Excel sheet, too, doing some markup/formatting
 modifications, and outputting a CPBLtable for LaTeX.

Some possible work flows:

(1) XLS(x) to cpblTables, with some modification/ markup

#Grab a rectangle from a spreadsheet, and convert it to a Pandas DataFrame of TeX strings
df=spreadsheetToTexDF(my XLS filename, sheet=sheet name, topleft=(x,y), bottomright=(x2,y2))
#
df.

"""
import re
import os

import pandas as pd
import numpy as np
from cpblUtilities import doSystemLatex
from cpblUtilities.mathgraph import tonumeric

def open(fn,mm,encoding='utf-8'): # Replace default "open" to be unicode-safe
    import codecs
    return(codecs.open(fn,mm,encoding=encoding))

###########################################################################################
###
def chooseSFormat(ff,conditionalWrapper=['',''],lowCutoff=None,lowCutoffOOM=True,convertStrings=False,highCutoff=1e8,noTeX=False,sigdigs=4,threeSigDigs=False,se=None, leadingZeros=False):
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

To do: New parameter sigdigs only implemented for integers so far. And it needs to be reconciled with threeSigDigs (which should become deprecated).  threeSigDigs is really about precision; it corresponds closely/directly to decimal places.

   - N.B.: see new (2016Aug)   format_to_n_sigfigs and round_to_n_sigfigs in mathgraph.py, which should probably be used here.
   The latter  rounds correctly but does not do the displaying part.




"""
    if lowCutoff==None:
        lowCutoff==1.0e-99 # Sometimes "None" is explicitly passed to invoke default value.
    import numpy#from numpy import ndarray
    if isinstance(ff,list) or isinstance(ff,numpy.ndarray): # Deal with lists
        return([chooseSFormat(fff,conditionalWrapper=conditionalWrapper,lowCutoff=lowCutoff,convertStrings=convertStrings,sigdigs=sigdigs,threeSigDigs=threeSigDigs) for fff in ff])
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
        if lowCutoffOOM in [True,'log'] and not aa==0:
            negexp=int(np.ceil(np.log10(aa)))
            ss='-'*bool(ff<0)+ r'$<$10$^{%d}$'%negexp
        elif isinstance(lowCutoffOOM,basestring):
            ss=lowCutoffOOM
    else:
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
    if isinstance(ff,int) and not ff==0:
        round_to_n = lambda x, n: np.round(x, -int(np.floor(np.log10(abs(x)))) + (n - 1)) 
        ff=round_to_n(ff,sigdigs)
        #if ff>10**sigdigs:
        #    ff=int(np.round(ff % (10**sigdigs)))* (10**sigdigs)
        ss='$-$'*bool(ff<0)+str(abs(ff))

    return(conditionalWrapper[0]+ss+conditionalWrapper[1])


def extractExcelRange(spreadsheetfile, sheet=None ,cells=None,transpose=False):
    """
    Returns some of an excel spreadsheet values as a pandas DataFrame.
    sheet=None should just give first sheet. cells=None should give entire sheet.
    
    """

    if cells is not None:
        cols=''.join([cc for cc in cells if cc.isalpha() or cc in [':']]) # Extract "B:F"
        if ':' in cells:
            assert ':' in cells and len(cells.split(':'))==2
            rows=np.array(tonumeric(''.join([cc for cc in cells if cc.isdigit() or cc in [':']]).split(':'))) # Extract "24:56" and then to (24,56)
        else: # Maybe it's just one cell
            cols=cols+':'+cols
            rows=np.array(tonumeric(''.join([cc for cc in cells if cc.isdigit() or cc in [':']]))) # Extract "24"
            rows=[rows]*2 # to get [24,24]

    if spreadsheetfile.upper().endswith('.XLS') or spreadsheetfile.upper().endswith('.XLSX'):
        dfr = pd.ExcelFile(spreadsheetfile).parse(sheet,skiprows=rows[0]-1,parse_cols=cols).iloc[:(rows[1]-rows[0])]
        # Extract the inner matrix, and the column headers:
        dfm=np.concatenate([np.array([dfr.columns.tolist()]),dfr.as_matrix()])
        # Create the spreadsheet column and row numbers:
        index=[str(xx) for xx in range(rows[0],rows[1]+1)]
        columns= [chr(c) for c in xrange(ord(cols.split(':')[0]), ord(cols.split(':')[1])+1) ]
        # Put back into a DataFrame (easy), with column and row headers from spreadsheet (for cuteness):
        if transpose:
            dfm=dfm.T
            df=pd.DataFrame(dfm,columns=index,index=columns )
        else:
            df=pd.DataFrame(dfm,columns=columns,index=index)

    else:
        raise("I don't know that file type yet")
    return(df)


class latexTable(object):# pd.DataFrame):
    """
Define a class to manage conversion of a spreadsheet extract all the way to a cpblTables tex file


ToDo: create function isSpreadsheetCoordinate() to check for form BB78. Allow for specification of title and comments that way.

Allow for output as just the tabular component, rather than a full cpblTable, so that people can pull them into LyX and use LyX cross-references.

Pandas' excel import does not respect sig figs. There's anothre package that gives option to copy formatting???

Decide on and document combined decimals / sigdigs behaviour.

    """
    """
    Don't use pd.DataFrame as a subclass. This is not normal behaviour in python, and pd.DataFrame returns its own class from most functions, not the class it was given, so it's not useful/easy to do this. Instead, this class will just a dataframe internally, as self.df.

    """
    def __init__(self, spreadsheetfile=None, sheet=None, cells=None,topleft=None,bottomright=None,decimals=None,sigdigs=None, leadingZeros=True, title=None,caption=None,footercell=None,transpose=False):
        super(self.__class__, self).__init__() # To initialize the base class??!
        self._cformat=None
        self._tabletitle=None
        self._tablecaption=None
        self._tablefooter=None
        if caption is not None:
            self._tablecaption=caption
        if title is not None:
            self._tabletitle=title


        if spreadsheetfile.upper().endswith('.XLS') or spreadsheetfile.upper().endswith('.XLSX'):

            df=extractExcelRange(spreadsheetfile, sheet=sheet, cells=cells, transpose=transpose)

            # A horrid kludge: to get rid of what Pandas did with empty headers, originally: called them "Unnamed: 0", etc.
            df=df.applymap(lambda x: '{}' if x.__class__ in [str,unicode] and x.startswith('Unnamed: ') else x)
            
            # At this point, let's also deal with at least some LaTeX special chars:
            df=df.applymap(lambda x: x if x.__class__ not in [str,unicode] else  x.replace('$',r'\$'))

            if footercell is not None:
                fc=extractExcelRange(spreadsheetfile, sheet=sheet, cells=footercell)
                assert fc.shape in [(1,1)]
                self._tablefooter=fc.values[0][0]
        else:
            raise('unknown_spreadsheet_file_type')

        formatDFforLaTeX(df,row=None,sigdigs=sigdigs,colour=None,leadingZeros=leadingZeros)
        self.df=df
        return # init() just return None

    def emboldenRow(self,row):
        """
        One common task is to embolden a row.  It should be specified by a 0-based or by the row number, as a string, from Spreadsheet
 To do: row should be able to be a list. Or comma separated string. Or ":" range.
        """
        self.df.ix[row]='{'+r'\bfseries '+self.df.ix[row]+'}'
        if 0: # This does also the index. Not necessary, since my output is only the interior matrix, not headers.
            if row.__class__ is int:
                ii=self.df.index.tolist()
                ii[row]='{'+r'\bfseries '+ii[row]+'}'
                self.df.index=ii
        return(self) # Allows for easy chaining of command/modifications to table.

    def toTabular(self,outfile=None,footer=None,boldHeaders=False, boldFirstColumn=False,columnWidths=None,formatCodes='lc',hlines=False,vlines=None,centering=True):#   ncols=None,nrows=None,, alignment="c"): # tableName=None,landscape=None, masterLatexFile=None,
        """ Note that pandas already has a latex output function built in. If it deals with multiindices, etc, it may be better to edit it rather than to start over. However, my cpblTableC function expects it to be broken up into cells still.


    This function takes a dataframe whos entries have already been converted to LaTeX strings (where needed)!.
    So not much is left to do before passing to cpblTableStyc.
    If there is a _tablefooter defined, it will also be displayed in footnotesize font just below the tabular environment

    boldHeaders requires the array package. It inserts a command \rowstyle before the header. It requires definition of funny column format types, and it requires a special column format.

    boldFirstColumn just uses the built-in >{} column modifier. 

    columnWidths, if specified, can give LaTeX strings for the widths of some of the columns.

    formatCodes is really kludgy. This is a partial or complete list of single-letter codes for the column formats. These would be used as the base values, before adding emboldening, widths, etc.

    hlines: specifies whether to add horizontal lines in the middle rows of the table. More options (than True/False) can be implemented later. Default False

    vlines: grudgingly, this has a True/False option too, with False default. Note that if you specify the formatcodes in more detail, this will be ignored.
        
    centering: by default, the tabular environment is within a {\centering }. Set this to False to disable that.

Implementaiton comments:  pandas has a to_latex(), and it offers to bold the first column, etc, but the problem is that I have already got some TeX markup in some of the elements, so I'd have to undo the \textbackslashes that pd.DataFrame.to_latex() puts in. Instead, I'll stick to convertin to LaTeX by hand.
 2014 Apr: now column and row indices of internal df are not to be exported to LaTeX.
        """
        #cformat=list('l'+'c'*self.df.shape[1])
        # Pick the basic format codes to start with (ie ones which don't take arguments), e.g. lccccc....
        if formatCodes.__class__ in [str,unicode] and len(formatCodes)>=self.df.shape[1]:
            cformatl=list(formatCodes)
            if  vlines is not None:
                print(' Put the "|"s in your formatCodes, since you specified one.')
                failhere
        else:
            cformatl=list((formatCodes+1000*formatCodes[-1])[:(self.df.shape[1])])
        print("Format: ", cformatl)
        if columnWidths is not None:
            assert len(columnWidths)==self.df.shape[1]
            for ii,cw in enumerate(columnWidths):
                if cw is None: continue
                cformatl[ii]='m{'+cw+'}'
        if boldFirstColumn:
            cformatl[0]='>{\\bfseries}'+cformatl[0]
        #firstcol='>{\\bf}l' if boldFirstColumn else 'l'
        print("Format: ", cformatl)
        if boldHeaders:
            cformatl[0]='+'+cformatl[0]
            for ii,cw in enumerate(cformatl):
                if cw is None or ii==0: continue
                cformatl[ii]='^'+cformatl[ii]

        #cformat= '+'+firstcol+'|'+'|^c'*self.df.shape[1]+'|' if boldHeaders else None
        if vlines is None: 
            vlines =False
        if not len(formatCodes)>=self.df.shape[1]:
            
            cformat=vlines*'|'+(vlines*'|').join(cformatl)+vlines*'|'
        else: 
            cformat=''.join(formatCodes)
        if not boldHeaders and not boldFirstColumn and not columnWidths:
            cformat=None
        self._cformat=cformat

        # Caution! toprule, bottomrule require booktabs
        self._tabular_body=('\\\\ '+'\\hline '*hlines+'\n').join(['&'.join(RR)  for RR in self.df.as_matrix()[1:] ]) +'\\\\ \n\\bottomrule '  #cline{1-\\ctNtabCols}\n '
        self._tabular_header='\\toprule\n'+(boldHeaders*'\\rowstyle{\\bfseries}%\n') + ' & '.join(self.df.as_matrix()[0])+'\\\\ \n\\midrule '

        if outfile is not None:
            if not outfile.endswith('.tex'): outfile+='.tex'
            with open(outfile,'wt',encoding='utf-8') as f:
                f.write(r"""
                {"""+ centering*r"\centering"+r"""
                \begin{tabular}{"""+self._cformat+r"""}
                """+self._tabular_header+"""
                """+self._tabular_body+"""
                \end{tabular}  }
                """)
                if self._tablefooter is not None:

                    f.write(r"{\footnotesize "+self._tablefooter +'}\n')

#                print('Wrote '+outfile)
        
        return
        #         return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))




    def toCPBLtable(self,outfile,tableTitle=None,footer=None,boldHeaders=False, boldFirstColumn=False,columnWidths=None,formatCodes='lc',hlines=False,vlines=None,masterLatexFile=None,landscape=None,cformat=None,tableName=None):
        """A CPBLtable is a LaTeX format with ex-post customizable
        column specifiers, table type (e.g. longtable, table, etc), and some other formatting.

        toTabular() already calculates the body and header for the
        normal-orientation table, and stores it in self.  Note that in
        2014, there is not yet a facility here to make use of the
        transposed-table feature of the cpblTables.sty framework.

        Note that the toCPBLtable() deals with footers differently from the
        toTabular(). It will not stick the footer in the CPBLtable .tex file,
        but rather include it in the code which calls cpblTables.sty
        functions.

        masterLatexFile, if used,  depends on cpblStataLatex.py features

        """
        # Create the tabular core (body and header):
        self.toTabular(outfile=None,footer=None,boldHeaders=boldHeaders, boldFirstColumn=boldFirstColumn,columnWidths=columnWidths,formatCodes=formatCodes,hlines=hlines,vlines=vlines)
    
        if not outfile.endswith('.tex'): outfile+='.tex'
        includeTex,callerTex=cpblTableStyC(
            cpblTableElements(
                body=self._tabular_body,
                firstPageHeader=self._tabular_header,
                otherPageHeader=None,tableTitle=tableTitle,caption=self._tablecaption,footer=self._tablefooter,tableName=tableName,landscape=landscape,cformat=self._cformat),
            filepath=outfile,  masterLatexFile=masterLatexFile)

        #print(includeTex)
        with open(outfile,'wt',encoding='utf-8') as f:
            f.write(includeTex)
        if boldHeaders: print(r"""
    \usepackage{array}
    \newcolumntype{+}{>{\global\let\currentrowstyle\relax}}
    \newcolumntype{^}{>{\currentrowstyle}}
    \newcommand{\rowstyle}[1]{\gdef\currentrowstyle{#1}%
    #1\ignorespaces
    }
    """)
        #print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))
        return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))


def interleave_columns_as_rows(df):
    """ Assume every second column of the data frame is a standard error value for the column to its left. Move these values so they are below the point estimates.
    """
    alts= df.iloc[:,1::2].copy()
    alts.columns = df.columns[0::2]
    newdf=df.iloc[:,0::2].copy()
    newdf['__origOrd'] = range(len(newdf))
    newdf['_sorting']=1
    alts[ '_sorting']=2
    alts['__origOrd'] = range(len(newdf))
    anewdf = newdf.append(alts).sort_values(['__origOrd','_sorting'])
    anewdf=anewdf[[cc for cc in anewdf if cc not in ['__origOrd','_sorting']]]
    indd= anewdf.index.tolist()
    # Now let's get rid of the row labels on every second line, so they're not duplicated:
    for ii in range(1,len(indd),2):
        indd[ii]=''
    anewdf.index=indd
    return(anewdf)

def interleave_and_format_paired_columns_as_rows(odf, method='standard_errors'):
    """ Assume every second column of the data frame is a standard error value for the column to its left. Format these Move these values so they are below the point estimates.
    """
    assert method== 'standard_errors'
    df = odf.copy()
    cols = df.columns
    assert len(cols)%2 == 0
    import pystata
    for ii in range(0,len(cols),2):
        pystata.formatPairedRow_DataFrame(df, cols[ii], cols[ii+1], prefix='tmppref')
    df=df.iloc[:,len(cols):]
    df.columns = df.columns.map(lambda ss: ss[7:])
    if 0:print df
    return interleave_columns_as_rows(df)


    
class cpblTableElements():
    def __init__(self,
                 body=None, firstPageHeader=None, otherPageHeader=None, ncols=None, nrows=None, footer=None, landscape=None, cformat=None, tableTitle=None, caption=None, label=None, tableName=None,tableType=None):
        self.body=body
        self.firstPageHeader=firstPageHeader
        self.otherPageHeader=otherPageHeader
        self.ncols=ncols
        self.nrows=nrows
        self.footer=footer
        self.landscape=landscape
        self.format=cformat
        self.tableTitle=tableTitle
        self.caption=caption
        self.label=label
        self.tableType=tableType
        self.tableName=tableName


###########################################################################################
###
#def cpblTableStyC(format,filepath,firstPageHeader,otherPageHeader,body,tableTitle,caption,label,ncols,nrows,footer=None,tableName=None,landscape=None):
    ###
    #######################################################################################
###########################################################################################
###
# Original format:
def cpblTableStyC(tableElements=None,filepath=None,masterLatexFile=None,
                  tableElementsTrans=None, # Also allow definition of an alternative, transposed version of the same table to be coded in the same cpblTables file!
                  showTransposed=None):

                # I believe the title, caption, label, name should be in format, anyway,
    ###
    #######################################################################################
    """
    This is the most general way to create/add a table in LaTeX because the LaTeX code must already be generated (provide it in body, firstPageHeader,otherPageHeader, etc).  Other routines should be more specific in that they take some tabular data and generate LaTeX code for it, and then call this to package it together.


    Creates pieces of a table in a very modular way for my cpblTables.sty. Right now, automatically chooses the right format to use by default.
    Chooses landscape or not if landscape==None
    Output is text for the included .tex file and the calling controller latex code.

    June 2011: big innovation: version C of cpblTables can now contain two versions of the table in the same file, one a transposed alternative of the other. NOT IMPLEMENTED YET HERE!!!

    June 2011: retired original code in this function, created completely new calling format.:    folowing are now all contained in tableElements dict:
                body,firstPageHeader=None,otherPageHeader=None, ncols=None,nrows=None,footer=None,landscape=None,format=None,tableTitle=None,caption=None,label=None,tableName=None,
    What's left to be done on this is to set some of the transposed values to the non-transposed if they're undefined.

     ---

    body: This is a multiline string already in LaTeX format for the body of the table. Similarly, the header strings are LaTeX code.

    format: Leave this blank to make use of the "CA" style (automatic format) table... but does that work proprly yet??? aug 2009

    filepath: if specified, the tableStyC latex file will be written.  If it's not specified, 'PUT-TABLETEX-FILEPATH-HERE' will be placed in the callerTeX where the filepath should go.  [Why is it prefixing '/' to my file path? 2013sept]

    masterLatexFile: if provided,  this must be a (cpbl) latexRegressionFile class instance. If specified, the table will be included into this master file as well! Then the return arguments can both be ignored.

    showTransposed=None: when producing a .tex cpbltables file with both normal and transposed versions, which should be shown in the calling controller latex code.

    """

    """
%% Format for "C" types:
%% \cpblTableBWide{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

    """

    #if '\n' not in body: # May 2011: ... reformat?

    assert '\n' in tableElements.body    # Try to catch old-style calls to new calling sequence:    debugprint('cpblTableStyC',str((format,filepath,len(body),tableTitle,caption,label,ncols,nrows,footer,tableName,landscape)))

    tE=tableElements
    tEt=tableElementsTrans

    includeTex,callerTex='',''


    if showTransposed in ["both",True]:
        assert tEt.body
    if showTransposed is None:
        showTransposed=False

    if filepath==None:
        filepath='PUT-TABLETEX-FILEPATH-HERE'
    else:
        pp=os.path.split(filepath)[0]
        filepath=pp+(len(pp)>0)*'/'+os.path.split(filepath)[1].replace('_','-')

    for aTE in [xx for xx in [tE, tEt] if xx is not None]:
        if aTE.tableName==None: # Table name is for the TOC; table title is the begining of the caption
            aTE.tableName=aTE.tableTitle

        if aTE.ncols==None:
            aTE.ncols=len(aTE.body.split(r'\\')[1].split('&'))

        # At least for non-transposed, the function in cpblStata now defines the otherpageheader in detail, and then references it and adds one hline to make first page header. [nov 2010]
        if isinstance(aTE.firstPageHeader,list):
            assert isinstance(aTE.firstPageHeader[0],basestring)
            aTE.firstPageHeader='\n'.join(aTE.firstPageHeader)
        if aTE.otherPageHeader==None:
            aTE.otherPageHeader=r'\ctFirstHeader'

        # This section should, and doesn't yet, choose between wide, simple, and long. in landscape or not.
        aTE.squishCols=aTE.ncols>15
        #scaleTable=ncols>15
        aTE.tinyFont=aTE.ncols>15 or aTE.nrows>20
        if aTE.landscape==None:
            aTE.landscape=aTE.ncols>15


        if aTE.tableType is None: # Choose automatically
            aTE.tableType='CLongHere'+0*'Landscape'*aTE.landscape# Get rid of the landscape part: the entire thing is one big landscape environment now.
            # Dec2009: Following doesn't seem right. I still want to use "A" auto mode een if format is specified!
            aTE.tableType='C'+'A'*(format in [None,''])+'Long'
            aTE.tableType='CACLong' # Jan 2010: CAC means take caption from included .tex too!
            if aTE.nrows<20:
                aTE.tableType='CACSimple' # Jan 2010: CAC means take caption from included .tex too!
        assert aTE.tableType in ['CACSimple','CACLong'] # to be expanded!

        #%% \cpblTableBWide{1:format}{table (toc) name}{2:short caption}{3:caption}{4:label}{5:ncols for footer}{6:footer}{7:file}
        #%% Format for "C" types: (including CA..?)
        #%% \cpblTableCSimple{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

        aTE.tableCelements= [aTE.format,aTE.tableName,aTE.tableTitle,aTE.caption,aTE.label,aTE.footer,filepath]
        if not aTE.format:
            aTE.format='c'*aTE.ncols
        # The following does not affect the original variables, of course.
        for ii in range(len(aTE.tableCelements)):
            if not aTE.tableCelements[ii]:
                aTE.tableCelements[ii]=''
        # Jan 2010: Move caption and footer to CAC (automatic) caption! ie remove from explicit call! Put the automatic caption in blue, so it's clear it's not been moved/replaced (one can always simply remove the blue in the .tex file, when the .tex file is final.)
        aTE.CACcomments=r' {\color{blue} '+ ' '.join(aTE.tableCelements[3:6:2]) + '} '
        aTE.tableCelements[3],aTE.tableCelements[5]='',r'{\footnotesize\cpblColourLegend} '


    # Now, display the normal, non-transposed table
    if showTransposed in [False,'both']:
        callerTex=(r'\setlength\tabcolsep{1pt}'+'\n')*tE.squishCols+\
               r'{\usetinytablefont '*tE.tinyFont+\
               r'\cpblTable'+tE.tableType+'{'+'}{'.join(tE.tableCelements)+'}\n'+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*tE.squishCols +\
               r'}'*tE.tinyFont
    # Now, display the transposed table
    if showTransposed in [True,'both']:
        callerTex+=(r'\setlength\tabcolsep{1pt}'+'\n')*tEt.squishCols+\
               r'{\usetinytablefont '*tEt.tinyFont+\
               0*r'\cpblTableUseTransposedtrue '+\
               r'\cpblTable'+tEt.tableType+'Transposed{'+'}{'.join(tEt.tableCelements)+'}\n'+\
               0*r'\cpblTableUseTransposedfalse '+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*tEt.squishCols +\
               r'}'*tEt.tinyFont


    includeTex=r"""
    \renewcommand{\ctNtabCols}{%d}
    \renewcommand{\ctFirstHeader}{%s}
    \renewcommand{\ctSubsequentHeaders}{%s}
    \renewcommand{\ctBody}{%s}
    """%(tE.ncols,tE.firstPageHeader,tE.otherPageHeader,tE.body)  +r"""
    % Default caption:
    \renewcommand{\ctCaption}{"""+tE.CACcomments+r"""}
    % This .tex file is meant to be called by something from cpblTables.sty. 
    % If instead it is included directly, output something crude:
    \ifx\@ctUsingWrapper\@empty
    %Code to be executed if the macro is undefined
    \begin{table}
    \begin{tabular}{"""+tE.format+r"""}
    \ctFirstHeader
    \ctBody
    \end{tabular}
    \end{table}
    \else
    %Code to be executed if the macro IS defined
    \fi

    % For "CA" versions of cpblTables, the format need not be specified in the call:
    \renewcommand{\ctStartTabular}{\begin{tabular}{"""+tE.format+r"""}}
    \renewcommand{\ctStartLongtable}{\begin{longtable}[c]{"""+tE.format+r"""}}
    """
    if tEt is not None:
        includeTex+=r"""

% BEGIN TRANSPOSED VERSION
"""+r"""
    \renewcommand{\ctNtabColsTrans}{%d}
    \renewcommand{\ctFirstHeaderTrans}{%s}
    \renewcommand{\ctSubsequentHeadersTrans}{%s}
    \renewcommand{\ctBodyTrans}{%s}
    """%(tEt.ncols,tEt.firstPageHeader,tEt.otherPageHeader,tEt.body)  +r"""
    % Default caption:
    \renewcommand{\ctCaptionTrans}{"""+tEt.CACcomments+r"""}

    % Better yet, for version "CA" of cpblTables, define methods so that the format need not be specified in the call.
    \renewcommand{\ctStartTabularTrans}{\begin{tabular}{"""+tEt.format+r"""}}
    \renewcommand{\ctStartLongtableTrans}{\begin{longtable}[c]{"""+tEt.format+r"""}}
    """

    if not filepath in ['PUT-TABLETEX-FILEPATH-HERE']:
        import codecs
        with codecs.open(filepath,'wt',encoding='utf-8') as texfile:
            texfile.write(includeTex)
            texfile.close()

    if masterLatexFile:
        assert '/' in filepath # Can't finalise the callingTex if we don't know the filepath
        from cpblDefaults import defaults
        masterLatexFile.append(callerTex.replace(defaults['paths']['tex'],r'\texdocs '))

    return(includeTex,callerTex)



###########################################################################################
###
def legacy_cpblTableStyC(body,format=None,filepath=None,firstPageHeader=None,otherPageHeader=None,tableTitle=None,caption=None,label=None,ncols=None,nrows=None,footer=None,tableName=None,landscape=None,masterLatexFile=None,
                  # Also allow definition of an alternative, transposed version of the same table to be coded in the same cpblTables file!:
                  showTransposed=None,bodyTrans=None,formatTrans=None,firstPageHeaderTrans=None,otherPageHeaderTrans=None,ncolsTrans=None,nrowsTrans=None,landscapeTrans=None):
    # retired June 2011: new format! Not backwards compatible!!!!?!?
    ###
    #######################################################################################

    """
    This is the most general way to create/add a table in LaTeX because the LaTeX code must already be generated (provide it in body, firstPageHeader,otherPageHeader, etc).  Other routines should be more specific in that they take some tabular data and generate LaTeX code for it, and then call this to package it together.


    Creates pieces of a table in a very modular way for my cpblTables.sty. Right now, automatically chooses the right format to use by default.
    Chooses landscape or not if landscape==None
    Output is text for the included .tex file and the calling controller latex code.

    June 2011: big innovation: version C of cpblTables can now contain two versions of the table in the same file, one a transposed alternative of the other. NOT IMPLEMENTED YET HERE!!!

    body: This is a multiline string already in LaTeX format for the body of the table. Similarly, the header strings are LaTeX code.

    format: Leave this blank to make use of the "CA" style (automatic format) table... but does that work proprly yet??? aug 2009

    filepath: if specified, the tableStyC latex file will be written.  If it's not specified, 'PUT-TABLETEX-FILEPATH-HERE' will be placed in the callerTeX where the filepath should go.

    masterLatexFile: if provided,  this must be a (cpbl) latexRegressionFile class instance. If specified, the table will be included into this master file as well! Then the return arguments can both be ignored.

    showTransposed=None: when producing a .tex cpbltables file with both normal and transposed versions, which should be shown in the calling controller latex code.

    """

    """
%% Format for "C" types:
%% \cpblTableBWide{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

    """

    #if '\n' not in body: # May 2011: ... reformat?

    assert '\n' in body    # Try to catch old-style calls to new calling sequence:    debugprint('cpblTableStyC',str((format,filepath,len(body),tableTitle,caption,label,ncols,nrows,footer,tableName,landscape)))


    if tableName==None: # Table name is for the TOC; table title is the begining of the caption
        tableName=tableTitle

    includeTex,callerTex='',''


    if showTransposed in ["both",True]:
        assert bodyTrans
    if showTransposed is None:
        showTransposed=False

    if bodyTrans is not None:
        assert formatTrans
        assert firstPageHeaderTrans
        assert otherPageHeaderTrans
        assert ncolsTrans
        assert nrowsTrans
        assert landscapeTrans
        NotProgrammedYet_duplicateBelow

    if filepath==None:
        filepath='PUT-TABLETEX-FILEPATH-HERE'
    else:
        filepath=os.path.split(filepath)[0]+'/'+os.path.split(filepath)[1].replace('_','-')

    if ncols==None:
        ncols=len(body.split(r'\\')[1].split('&'))

    # At least for non-transposed, the function in cpblStata now defines the otherpageheader in detail, and then references it and adds one hline to make first page header. [nov 2010]
    if isinstance(firstPageHeader,list):
        assert isinstance(firstPageHeader[0],basestring)
        firstPageHeader='\n'.join(firstPageHeader)
    if otherPageHeader==None:
        otherPageHeader=r'\ctFirstHeader'

    # This section should, and doesn't yet, choose between wide, simple, and long. in landscape or not.
    squishCols=ncols>15
    #scaleTable=ncols>15
    tinyFont=ncols>15 or nrows>20
    if landscape==None:
        landscape=ncols>15

    tableType='CLongHere'+0*'Landscape'*landscape# Get rid of the landscape part: the entire thing is one big landscape environment now.
    # Dec2009: Following doesn't seem right. I still want to use "A" auto mode een if format is specified!
    tableType='C'+'A'*(format in [None,''])+'Long'
    tableType='CACLong' # Jan 2010: CAC means take caption from included .tex too!
    if nrows<20:
        tableType='CACSimple' # Jan 2010: CAC means take caption from included .tex too!

    #%% \cpblTableBWide{1:format}{table (toc) name}{2:short caption}{3:caption}{4:label}{5:ncols for footer}{6:footer}{7:file}
    #%% Format for "C" types: (including CA..?)
    #%% \cpblTableCSimple{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

    tableCelements= [format,tableName,tableTitle,caption,label,footer,filepath]
    if not format:
        format='c'*ncols
    # The following does not affect the original variables, of course.
    for ii in range(len(tableCelements)):
        if not tableCelements[ii]:
            tableCelements[ii]=''
    # Jan 2010: Move caption and footer to CAC (automatic) caption! ie remove from explicit call! Put the automatic caption in blue, so it's clear it's not been moved/replaced (one can always simply remove the blue in the .tex file, when the .tex file is final.)
    CACcomments=r' {\color{blue} '+ ' '.join(tableCelements[3:6:2]) + '} '
    tableCelements[3],tableCelements[5]='',r'{\footnotesize\cpblColourLegend} '

    callerTex=(r'\setlength\tabcolsep{1pt}'+'\n')*squishCols+\
               r'{\usetinytablefont '*tinyFont+\
               r'\cpblTable'+tableType+'{'+'}{'.join(tableCelements)+'}\n'+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*squishCols +\
               r'}'*tinyFont

    includeTex=r"""
    \renewcommand{\ctNtabCols}{%d}
    \renewcommand{\ctFirstHeader}{%s}
    \renewcommand{\ctSubsequentHeaders}{%s}
    \renewcommand{\ctBody}{%s}
    """%(ncols,firstPageHeader,otherPageHeader,body)  +r"""
    % Default caption:
    \renewcommand{\ctCaption}{"""+CACcomments+r"""}
    % This .tex file is meant to be called by something from
    % cpblTables.sty. If it is not, then output something crude:
    \ifx\@ctUsingWrapper\@empty
    %Code to be executed if the macro is undefined
    \begin{table}
    \begin{tabular}{"""+format+r"""}
    \ctFirstHeader
    \ctBody
    \end{tabular}
    \end{table}
    \else
    %Code to be executed if the macro IS defined
    \fi

    % Better yet, for version "CA" of cpblTables, define methods so that the format need not be specified in the call.
    \renewcommand{\ctStartTabular}{\begin{tabular}{"""+format+r"""}}
    \renewcommand{\ctStartLongtable}{\begin{longtable}[c]{"""+format+r"""}}
    """

    if not filepath in ['PUT-TABLETEX-FILEPATH-HERE']:
        texfile=open(filepath,'wt')
        texfile.write(includeTex)
        texfile.close()

    if masterLatexFile:
        assert '/' in filepath # Can't finalise the callingTex if we don't know the filepath
        from cpblDefaults import defaults
        masterLatexFile.append(callerTex.replace(defaults['paths']['tex'],r'\texdocs '))

    return(includeTex,callerTex)



################################################################################################
################################################################################################
def tsvToTable(infile,outfile=None):
    ############################################################################################
    ############################################################################################
    """ 2013 Sept.
     Transposed addition not done yet.
     Can pass filename or the tsv text contents of a file.

     2014March: this doesn't return the result yet!? Is it unused so far?
    """
    
    if '\n' in infile: # Oh. they've passed contents, not filename:
        ttt=[LL.strip().split('\t') for LL in infile.split('\n')]
        assert outfile is not None
    else:
        if outfile is None:
            assert infile.endswith('.tsv')
            outfile=infile[:-4]+'--.tex'
        ttt=[LL.strip().split('\t') for LL in open(infile,'rt').readlines()]
    body=ttt[1:]
    headers=[r"\sltheadername{"+tt+r"}" for tt in ttt[0]]
    assert headers
    assert body



    includeTex,callerTex=cpblTableStyC(cpblTableElements(body='\\\\ \hline \n'.join(['&'.join(LL) for LL in body])+'\\\\ \n\\cline{1-\\ctNtabCols}\n ',firstPageHeader='\\hline\\hline \n'+' & '.join(headers)+'\\\\ \n\\hline\\hline\n',otherPageHeader=None,tableTitle=None,caption=None,label=None,ncols=None,nrows=None,footer=None,tableName=None,landscape=None),
                                       filepath=None,#
                                       masterLatexFile=None)

    print(includeTex)
    with open(outfile,'wt',encoding='utf8') as f:
        f.write(includeTex)
    print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))
    
################################################################################################
################################################################################################
def tableToTSV(infilepath):
#def cpblTableToCSV(infilepath):
    ############################################################################################
    ############################################################################################
    """
    This is for the case when you want to recover data from a cpblTable tex file. 
    To do: it should offer pandas DF output.

    infilepath can be a filename or the .tex code of the file.

    Still need to allow option of substitution table of labels

    2013 Sept: n.b. The reverse function to this (TSV file to a cpblTable) is now a separate python executable.
 
    """
    """
2016: I'm deleting the following editLatexTable.py file from May 5 2008:

#!/usr/bin/python
import sys
print sys.argv

import re

# Oh DAMN this is no longer general. it's for cpblStata .tex files now.

fn=sys.argv[1]
ff=[re.sub(r'\\\\ *$','\t'+r'\\\\',
           re.sub(r'\\\\ }$','\t'+r'\\\\ }',
           ll.strip('\n \t').replace('\t',' ').replace('$$$$','').replace('&','\t&').replace(r'\showSEs{\\*}{\\}','\t'+r'\showSEs{\\*}{\\}'))) for ll in open(fn,'rt').readlines()]

print ff
# That's a good first approx, except that \\'s at the end of a line should not be left at the end of the line! ie they look 



fout=open(fn+'.csv','wt')
fout.write('\n'.join(ff))
fout.close()

import os
os.system('ooffice '+fn+'.csv&')

   """

    #
    if not '\n' in infilepath:
        tt=open(infilepath,'rt').read()
    else:
        tt=infilepath

    # Kludge June 2011 to deal with new, dual-format cpblTableC files:
    tt1=tt.split('% BEGIN TRANSPOSED VERSION')[0]
    # Find basic components:
    ff=re.findall(r'renewcommand{\\ctFirstHeader}{(.*?)}\s+\\renewcommand{\\ctSubsequentHeaders}{(.*?)\\renewcommand{\\ctBody}{(.*?)This .tex file is meant to be called',tt1,re.DOTALL)[0]

    #debugprint( 'Found %d out of 3 components...'%len(ff))

    # Header
    header=ff[0]+ ff[1] # Actual headers could be in first or second def'n; include both.
    headerDrop=[r'\begin{sideways}',r'\end{sideways}',r'\hline','\n']
    for hd in headerDrop:
        header=header.replace(hd,'')
    header=re.sub(r'\\sltc?r?headername{(.*?)}',r'\1',header)
    header=re.sub(r'\\\\','\n',header)


    # Body
    body=ff[2]
    signifReplacements=[
        [r'\\wrapSigOneThousandth{(.*?)}',r'\1****'],
        [r'\\wrapSigOnePercent{(.*?)}',r'\1***'],
        [r'\\wrapSigFivePercent{(.*?)}',r'\1**'],
        [r'\\wrapSigTenPercent{(.*?)}',r'\1*'],
        [r'\\YesMark','Yes'],
        [r'\\sltheadername{(.*?)}',r'\1'],
        [r'\\sltrheadername{(.*?)}',r'\1'],
        [r'\\sltcheadername{(.*?)}',r'\1'],
        [r'\\coefse{(.*?)}',r'\1'],
        [r'\\showSEs{',r''],
        [r'\\\\ }{}',''],
        [r'\$-\$','-'],
        [r'\\hline',''],
        [r'\\\\',''],
        [r'\\cline{1-\\ctNtabCols',''],
        ]
    for sr in signifReplacements:
        body=re.sub(sr[0],sr[1],body)
    body='\n'.join([LL for LL in  body.split('\n') if LL.strip() not in ['','}','%']])
    if not '\n' in infilepath:
        with open(infilepath+'____tmp.tsv','wt')  as fout:
            fout.write(  (header+body).replace('&','\t')  )
            os.system('libreoffice '+infilepath+'____tmp.tsv &')
    return(    (header+body).replace('&','\t')  )



# # Found on web, one of many. xls2csv exists but not ods2csv in Ubuntu!. Aug 2013. Adapted by CPBL to make tsv.
"""

csv has a unicodewriter too, which I should use below instead of this...
"""
import sys,zipfile,re,os,csv

def ods2tsv(filepath,outpath=None,replaceZerosWithBlanks=False):
    """
    I'm having to add various features. zeros to blanks: empties get turned into zeros when they are references..... hence replaceZerosWithBlanks
    """

    xml = zipfile.ZipFile(filepath).read('content.xml')

    def rep_repl(match):
        return '<table:table-cell>%s' %match.group(2) * int(match.group(1))
    def repl_empt(match):
        n = int(match.group(1))
        pat = '<table:table-cell/>'
        return pat*n if (n<100) else pat

    p_repl = re.compile(r'<table:table-cell [^>]*?repeated="(\d+)[^/>]*>(.+?table-cell>)')
    p_empt = re.compile(r'<table:table-cell [^>]*?repeated="(\d+)[^>]*>')
    xml = re.sub(p_repl, rep_repl, xml)
    xml = re.sub(p_empt, repl_empt, xml)

    from pyquery import PyQuery as pq

    d = pq(xml, parser='xml')

    from lxml.cssselect import CSSSelector

    ns={'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'}
    selr = CSSSelector('table|table-row', namespaces=ns)
    selc = CSSSelector('table|table-cell', namespaces=ns)
    rowxs = pq(selr(d[0]))
    data = []
    for ir,rowx in enumerate(rowxs):
        cells = pq(selc(rowx))
        if cells.text():
            data.append([cells.eq(ic).text().encode('utf-8') for ic in range(len(cells))])

    root,ext=os.path.splitext(filepath)
    #csv.register_dialect('unixpwd', delimiter='\t', quoting=csv.QUOTE_NONE)
    if outpath is None:
        outpath=''.join([root,'.tsv'])

    with open(outpath,'w',encoding='utf-8') as f:
        print(outpath+' should be writing now in utf8')
        for row in data:
            if replaceZerosWithBlanks:
                f.write('\t'.join([kk if not kk =='0' else '' for kk in row])+'\n')
            else:
                f.write('\t'.join(row)+'\n')

    """
    with open(outpath,'wb') as f:
        for row in data:
            dw = csv.writer(f,delimiter='\t', quoting=csv.QUOTE_NONE)#,dialect='unixpwd')    #quotechar=' ')#,QUOTE_NONE)#sep='\t')
            if replaceZerosWithBlanks:
                dw.writerow([kk if not kk =='0' else '' for kk in row])
            else:
                dw.writerow(row)
        """
#Test:
#ods2tsv(os.path.expanduser('~/tmp/schedule.ods')) #example


def single_to_multicolumn_fixer(onerow, fmt=None):
    """ Take a list of elements in one row of a LaTeX table, and find repeated values. Replace them with a multicolumn entry to span those cells.
(See also findAdjacentRepeats in pystata?)
    """
    if fmt is None: fmt='|c|'
    _lastval, _ncols = None,0
    outs=[]
    for ii,cc in enumerate(onerow+[None]):
        if cc == _lastval or _ncols==0:
            _ncols+=1
            _lastval = cc
        elif _ncols== 1:
            outs+=[_lastval]
            _lastval,_ncols = cc,1
        elif _ncols > 1:
            outs+=[r'\multicolumn{'+str(_ncols)+'}{'+fmt+'}{'+str(_lastval)+'}']
            _lastval, _ncols = cc, 1
        else:
            1/0
    return outs

def _test_single_to_multicolumn_fixer():
    assert single_to_multicolumn_fixer([1,2,3]) == [1,2,3]
    assert single_to_multicolumn_fixer([1,2,3,3,3,4,5,6]) == [1, 2, '\\multicolumn{4}{|c|}{3}', 4, 5, 6]


def dataframeWithLaTeXToTable(df,outfile,tableTitle=None,caption=None,label=None,footer=None,tableName=None,landscape=None, masterLatexFile=None,boldHeaders=False, boldFirstColumn=False,columnWidths=None,formatCodes='lc',hlines=False):#   ncols=None,nrows=None,, alignment="c"):
    """ Note that pandas already has a latex output function built in. If it deals with multiindices, etc, it may be better to edit it rather than to start over. [See issue #5 for columns]. However, my cpblTableC function expects it to be broken up into cells still.

This function takes a dataframe whos entries have already been converted to LaTeX strings (where needed)!.
So not much is left to do before passing to cpblTableStyc.

boldHeaders requires the array package. It inserts a command \rowstyle before the header. It requires definition of funny column format types, and it requires a special column format.

boldFirstColumn just uses the built-in >{} column modifier (or array package?}.

columnWidths, if specified, can give LaTeX strings for the widths of some of the columns.

formatCodes is really kludgy. This is a partial or complete list of single-letter codes for the column formats. These would be used as the base values, before adding emboldening, widths, etc.

hlines: if True, put horizontal lines on every line.

    """
    #cformat=list('l'+'c'*df.shape[1])
    # Pick the basic format codes to start with (ie ones which don't take arguments), e.g. lccccc....
    cformat=list((formatCodes+1000*formatCodes[-1])[:(df.shape[1])])
    if columnWidths is not None:
        assert len(columnWidths)==df.shape[1]
        for ii,cw in enumerate(columnWidths):
            if cw is None: continue
            cformat[ii]='m{'+cw+'}'
    if boldFirstColumn:
        cformat[0]='>{\\bfseries}'+cformat[0]
    #firstcol='>{\\bf}l' if boldFirstColumn else 'l'
    if boldHeaders:
        cformat[0]='+'+cformat[0]
        for ii,cw in enumerate(cformat):
            if cw is None or ii==0: continue
            cformat[ii]='^'+cformat[ii]

    #cformat= '+'+firstcol+'|'+'|^c'*df.shape[1]+'|' if boldHeaders else None
    cformat='|'+'|'.join(cformat)+'|'
    if not boldHeaders and not boldFirstColumn and not columnWidths:
        cformat=None

        
    if type(df.columns)==pd.MultiIndex:
        columnheaders=[]
        for icr in range(len(df.columns.values[0])):
            if 0: print [df.columns.values[ii][icr] for ii in range(len(df.columns.values))]
            onerow = single_to_multicolumn_fixer([df.columns.values[ii][icr] for ii in range(len(df.columns.values))],
                                                 fmt = cformat)
            columnheaders+=[ (boldHeaders*'\\rowstyle{\\bfseries}%\n') + ' & '.join(onerow )+'\\\\ \n'    ]
        if 0: print columnheaders
        firstPageHeader = '\\hline\n'+ '\n'.join( columnheaders) + ' \\hline\\hline\n '
    else:
        firstPageHeader = '\\hline\n'+(boldHeaders*'\\rowstyle{\\bfseries}%\n') + ' & '.join(df.columns.values)+'\\\\ \n\\hline\\hline\n'
    includeTex,callerTex=cpblTableStyC(cpblTableElements(
            body=('\\\\ '+hlines*'\\hline'+' \n').join(['&'.join(RR)  for RR in df.as_matrix() ]) +'\\\\ \n\\cline{1-\\ctNtabCols}\n ',firstPageHeader=firstPageHeader, otherPageHeader=None,tableTitle=tableTitle,caption=caption,label=label,footer=footer,tableName=tableName,landscape=landscape,cformat=cformat,),
                                       filepath=outfile,  masterLatexFile=masterLatexFile)
    #ncols=ncols,nrows=nrows,
    if 0: print(includeTex)
    with open(outfile,'wt',encoding='utf-8') as f:
        f.write(includeTex)
    #print(' Wrote '+includeTex)
    if boldHeaders: print(r"""
\usepackage{array}
\newcolumntype{+}{>{\global\let\currentrowstyle\relax}}
\newcolumntype{^}{>{\currentrowstyle}}
\newcommand{\rowstyle}[1]{\gdef\currentrowstyle{#1}%
#1\ignorespaces
}
""")
    #print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))
    return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))

def formatDFforLaTeX(df,row=None,sigdigs=None,colour=None,leadingZeros=False):
    """ Convert  dataframe entries from numeric to formatted strings.
    This is probably not for use by cpblLaTeX regression class. But useful in general for making LaTeX tables from pandas work.
    How to do this...?

    For colouring alternating rows, for instance, do not do it here! Just call rowcolors before tabular.
    Here's a nice colour: \definecolor{lightblue}{rgb}{0.93,0.95,1.0}
    """
    def sformat(aval,sigdigs=sigdigs):
        if isinstance(aval,basestring):
            return(aval)
        return(
        chooseSFormat(aval,conditionalWrapper=['',''],lowCutoff=None,lowCutoffOOM=False,convertStrings=False,highCutoff=1e90,noTeX=False,sigdigs=sigdigs,threeSigDigs=False,se=None,leadingZeros=leadingZeros)
)
    if row is None:
        for irow in range(len(df.index)):
            for cc in df.columns:
                df.iloc[irow]= [sformat(vv) for vv in df.iloc[irow]] # No chance of confusion with integer row labels
    else:
        for cc in df.columns:
            df.ix[row]= [sformat(vv) for vv in df.ix[row]] # .ix allows either index or label
        return
    
def colorizeDFcells(df,xyindices,colour,bold=False):
    """ Sure there's a better way to do this.... I want to map stuff back onto itself.
    Return a new df with a row or whatever colorized.
e.g.
df=colorizeDFcells(df,df.PR=='Canada','red')

To do a whole row, just reset_index() and ignore the index.
    """
    df.loc[xyindices]='{'+bold*r'\bfseries'+'\\color{'+colour+'}'+df.loc[xyindices]+'}'
    return
def wrapDFcells(df,xyindices,before='',after=''):
    """  """
    df.loc[xyindices]=before+df.loc[xyindices]+after
    return

def styleCtoStandalone(infile,tablestyle='simple'):
    """
    tablestyle: simple means LaTeX's tabular format

This seems like a good idea, but I've not done it yet. (nov 2015). Also, would be easier if I ended some definitions with }%bracketname etc rather than just }.
"""
    
    poss

# # # # # # # # # # # # # # # # # # 
if __name__ == '__main__':
    """ This should do a full demo(s):
    Write an excel file. Then create outputs from it.
    """
    # Demos:
    # (1) Create a LaTeX tabular file (use it with \input{} in a LaTeX document) from some Excel data
    from pandas import DataFrame
    l1,l2,l3 = [1,2,3,4], [6,7,1,2], [2,6,7,1]
    df = DataFrame({'Stimulus Time': l1, 'Reaction Time': l2,'foodle': l3})
    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    dff= latexTable("test.xlsx",sheet='sheet1',cells="B1:C3")
    dff.emboldenRow(0)
    dff.toTabular('test-cpblt',boldFirstColumn=True)
    # (2) Create a modular CPBL table (use it with cpblTables package in LaTeX) from some Excel data
    callerTeX=dff.toCPBLtable('test-cpbl',footer=None, boldFirstColumn=True,boldHeaders=True) #,columnWidths=None,formatCodes='lc',hlines=False,vlines=None,masterLatexFile=None,landscape=None,cformat=None)
    # (3) Use that cpblTables file and compile the result (this does not use anything from cpblTablesTex, actually, but is for completeness in the demo)
    with open('test-invoke-cpbl.tex','wt') as ff:
        ff.write(r"""\documentclass{article}
        \usepackage{index}
        \usepackage{longtable,lscape}
        \usepackage{rotating} % This is used for turning some column headers vertical
        \usepackage{hyperref}
        \usepackage{booktabs} % For toprule, midrule, bottomrule
        \usepackage{array} % for column styling?
        \usepackage{cpblTables}
        \usepackage{cpblRef}
        % Following is for row styling?
        \newcolumntype{+}{>{\global\let\currentrowstyle\relax}}
        \newcolumntype{^}{>{\currentrowstyle}}
        \newcommand{\rowstyle}[1]{\gdef\currentrowstyle{#1}%
        #1\ignorespaces
        }
        \begin{document}
        """+  callerTeX+r"""
        \end{document}
        """)
    # If cpblUtilities is around, compile the LaTeX, too:
    
    if not os.path.exists('./tmpTEX'):
        os.makedirs('./tmpTEX')
    os.rename('test-cpbl.tex','tmpTEX/test-cpbl.tex')
    doSystemLatex('test-invoke-cpbl.tex',latexPath='./',launch=True,tex=None,viewLatestSuccess=True,bgCompile=False)
    
    

