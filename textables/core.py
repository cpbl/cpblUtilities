#!/usr/bin/python
#
# CPBL 2013-2014 September
# Incorporated old cpblTables.py, ie interface for prducing tables to be used by cpblTables.tex.
# Take a TSV/CSV file or (2014) a pandas DataFrame and generate a .tex include file that is used by my cpblTables.sty tools.
"""
2014 April: Provide tools for extracting a set of cells (location
 fixed) from an Excel sheet, too, doing some markup/formatting
 modifications, and outputting a CPBLtable for LaTeX, or as a standalone PDF.
 
Because this was originally designed for the pystata package, the cpblTable format allows for normal and transposed versions of a table to coexist in an input .tex file, with the choice between these, and extensive formatting choices, available for adjustment in the input/calling line in LaTeX.

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
from cpblUtilities.textables.stats_formatting import chooseSFormat

def open(fn, mm,
         encoding='utf-8'):  # Replace default "open" to be unicode-safe
    import codecs
    return (codecs.open(fn, mm, encoding=encoding))

def texheader_without_CPBLtables():
    """ TO BE CLEANED UP STILL!!!!  NOT YET IMPLEMENTED"""
    return r"""
\usepackage[table]{xcolor}
\usepackage{relsize}
\usepackage{longtable} % Only needed for longtables
\usepackage{afterpage} % Only needed for longtables
\begin{document}
\newcommand{\ctNtabCols}{}
\newcommand{\ctFirstHeader}{}
\newcommand{\ctSubsequentHeaders}{}
\newcommand{\ctFirstHeaderTrans}{}
\newcommand{\ctSubsequentHeadersTrans}{}
\newcommand{\ctBody}{}
\newcommand{\ctBodyTrans}{}
\newcommand{\ctCaption}{}
\newcommand{\ctCaptionTrans}{}
\newcommand{\ctStartTabular}{}
\newcommand{\ctStartTabularTrans}{}
\newcommand{\ctStartLongtable}{}
\newcommand{\ctStartLongtableTrans}{}
\newcommand{\ctNtabColsTrans}{}
\newcommand{\sltrheadername}[1]{#1}
\newcommand{\sltcheadername}[1]{#1}
\newcommand{\cpblbottomrule}{}
\newcommand\cpbltoprule\hrule

\newcommand{\longTableFooter}{} % Only needed for longtables
\newcommand{\longtableContinueFooter}{}  % Only needed for longtables


\newcommand{\showSEs}[2]{#1} % To show SEs
    \newcommand{\coefse}[1]{{\smaller\smaller (#1)}}

    \definecolor{cSignifOne}{rgb}{.92,1,.92}
    \definecolor{cSignifTwo}{rgb}{.78,1,.78}
    \definecolor{cSignifThree}{rgb}{.5,1,.5}
    \definecolor{cSignifThousandth}{rgb}{.6,1,0} % .3 1 .3
    \newcommand{\colourswrapSigTenPercent}[1]{#1\cellcolor{cSignifOne}}
    \newcommand{\colourswrapSigFivePercent}[1]{#1\cellcolor{cSignifTwo}}
    \newcommand{\colourswrapSigOnePercent}[1]{#1\cellcolor{cSignifThree}}
    \newcommand{\colourswrapSigOneThousandth}[1]{#1\cellcolor{cSignifThousandth}}
    \newcommand{\wrapSigTenPercent}{\colourswrapSigTenPercent}
    \newcommand{\wrapSigFivePercent}{\colourswrapSigFivePercent}
    \newcommand{\wrapSigOnePercent}{\colourswrapSigOnePercent}
    \newcommand{\wrapSigOneThousandth}{\colourswrapSigOneThousandth}

    \newcommand{\YesMark}{{\sc y}}%\textcolor{blue}$\checkmark$}}
    \newcommand{\cpblColourLegend}{{\footnotesize \begin{tabular}{lcccc}
Significance:~ &
        \wrapSigOneThousandth{0.1\%}~~~~&
        \wrapSigOnePercent{1\%}~~~~&
        \wrapSigFivePercent{5\%}~~~~&
        \wrapSigTenPercent{10\%}
        \\ \end{tabular} }}

"""
    
def texheader_for_CPBLtables(margins=None, allow_underscore=True, standalone_table=False, no_CPBLtables=False):
    """ LaTeX code to load the packages (\usepackage) needed for use of cpblTables-format .tex files

no_CPBLtables= True excludes including cpblTables.sty! This is useful because without any non-standard packages, cpblTable .tex files can be included to generate a plain table by simply defining a few other macros.

A standalone_table means a PDF/image of a tabular environment, not in a PDF page nor with other material. (By contrast, a standalone document means one which does not require the cpblTables.sty )

Caution:  This must not be confused with pystata's texheader (It should be rewritten to use this.
    """
    if standalone_table:
        mmm = '' #margins='none'
    elif margins in ['None','none']:
        mmm=r'\usepackage[left=0cm,top=0cm,right=.5cm,nohead,nofoot]{geometry}'+'\n'
    elif margins in ['default',None]:
        mmm=''
    settings=r"""
%% This file created automatically by CPBL's latexRegressionFile class
\usepackage{amsfonts} % Some substitions use mathbb
\usepackage[utf8]{inputenc}
\usepackage{lscape}
\usepackage{rotating}
\usepackage{siunitx} % This allows use of special column controls for aligning decimals
\usepackage{relsize}
\usepackage{colortbl} %% handy for colored cells in tables, etc.
\usepackage{xcolor} %% For general, v powerful color handling: e.g simple coloured text.
"""+mmm+r"""
\usepackage[colorlinks]{hyperref}
"""+r'\usepackage{underscore}'*allow_underscore+r"""
\usepackage{multirow}
"""+ (not no_CPBLtables)*r'\usepackage{cpblTables} % http://github.com/cpbl/cpblUtilities'+r"""
"""+ no_CPBLtables*r"""
\newcommand\ctDraftComment
"""+r"""
\usepackage{xspace}
\ifdefined\draftComment
\else
\newcommand{\draftComment}[1]{{ \footnotesize\em\color{green} #1}\xspace}
\fi
\renewcommand{\ctDraftComment}[1]{{\sc\scriptsize ${\rm #1}$}} % Show draft-mode comments
\renewcommand{\ctDraftComment}[1]{{\sc\scriptsize #1}} % Show draft-mode comments
"""
    return({False:r' \documentclass{article} ',
            True: r'\documentclass{standalone} '}[standalone_table] + '\n'+settings)

    """ This is also for pystata . Should not be here... Need to rewrite it!

\usepackage{chngcntr} %This is for counterwithin command, below. amsmath and numberwithin would be an alternative.
        \begin{document}
        \title{Regression results preview}\author{CPBL}\maketitle
%TOC:\tableofcontents
%LOT:\listoftables
%LOF:\listoffigures
\counterwithin{table}{section}
\counterwithin{figure}{section}
% Following would require amsmath, instead of chngcntr:
%\numberwithin{figure}{section}
%\numberwithin{table}{section}
        \pagestyle{empty}%%\newpage
        \begin{landscape}
    """

# This stuff is for figures, not tables. should be in pystata
"""
% Below I will uglyly stick varioius things from cpblRef or etc so that if that package is not included, they will at least be defined:
\ifdefined\cpblFigureTC
\else
% The "TC" in this command means there are two caption titles: one for TOC, one for figure.
\newcommand{\cpblFigureTC}[6]{%{\begin{figure}\centerline{\includegraphics{#1.eps}}
% Arguments: 1=filepath, 2=height or width declaration, 3=caption for TOC, 4=caption title, 5=caption details, 6=figlabel
\begin{figure}
  \begin{center}
    \includegraphics[#2]{#1}\caption[#3]{{\bf #4.} #5\label{fig:#6}\draftComment{\\{\bf label:} #6 {\bf file:} #1}}
  \end{center}
\end{figure}
}
\fi
"""

def extractExcelRange(spreadsheetfile, sheet=None, cells=None,
                      transpose=False):
    """
    Returns some of an excel spreadsheet values as a pandas DataFrame.
    sheet=None should just give first sheet. cells=None should give entire sheet.
    
    """

    if cells is not None:
        cols = ''.join([cc for cc in cells
                        if cc.isalpha() or cc in [':']])  # Extract "B:F"
        if ':' in cells:
            assert ':' in cells and len(cells.split(':')) == 2
            rows = np.array(
                tonumeric(''.join([
                    cc for cc in cells if cc.isdigit() or cc in [':']
                ]).split(':')))  # Extract "24:56" and then to (24,56)
        else:  # Maybe it's just one cell
            cols = cols + ':' + cols
            rows = np.array(
                tonumeric(''.join([
                    cc for cc in cells if cc.isdigit() or cc in [':']
                ])))  # Extract "24"
            rows = [rows] * 2  # to get [24,24]

    if spreadsheetfile.upper().endswith(
            '.XLS') or spreadsheetfile.upper().endswith('.XLSX'):
        dfr = pd.ExcelFile(spreadsheetfile).parse(
            sheet, skiprows=rows[0] - 1,
            usecols=cols).iloc[:(rows[1] - rows[0])]
        # Extract the inner matrix, and the column headers:
        dfm = np.concatenate(
            [np.array([dfr.columns.tolist()]), dfr.as_matrix()])
        # Create the spreadsheet column and row numbers:
        index = [str(xx) for xx in range(rows[0], rows[1] + 1)]
        columns = [
            chr(c)
            for c in xrange(
                ord(cols.split(':')[0]), ord(cols.split(':')[1]) + 1)
        ]
        # Put back into a DataFrame (easy), with column and row headers from spreadsheet (for cuteness):
        if transpose:
            dfm = dfm.T
            df = pd.DataFrame(dfm, columns=index, index=columns)
        else:
            df = pd.DataFrame(dfm, columns=columns, index=index)

    else:
        raise ("I don't know that file type yet")
    return (df)


class latexTable(object):  # pd.DataFrame):
    """
Define a class to manage conversion of a DataFrame or spreadsheet extract all the way to a cpblTables tex file

ToDo: create function isSpreadsheetCoordinate() to check for form BB78. Allow for specification of title and comments that way.

Allow for output as just the tabular component, rather than a full cpblTable, so that people can pull them into LyX and use LyX cross-references.

Pandas' excel import does not respect sig figs. There's anothre package that gives option to copy formatting???

Decide on and document combined decimals / sigdigs behaviour.

    """
    """
    Don't make a subclass of pd.DataFrame (though I do in surveypandas...). This is not normal behaviour in python, and pd.DataFrame returns its own class from most functions, not the class it was given, so it's not useful/easy to do this. Instead, this class will just a dataframe internally, as self.df.

    """

    def __init__(self,
                 dataframe=None,
                 transposed_dataframe = None,
                 spreadsheetfile=None,
                 sheet=None,
                 cells=None,
                 topleft=None,
                 bottomright=None,
                 decimals=None,
                 sigdigs=None,
                 leadingZeros=True,
                 title=None,
                 caption=None,
                 footercell=None,
                 transpose=False):
        super(self.__class__,
              self).__init__()  # To initialize the base class??!
        self._cformat = None
        self._tabletitle = None
        self._tablecaption = None
        self._tablefooter = None
        if caption is not None:
            self._tablecaption = caption
        if title is not None:
            self._tabletitle = title

        if spreadsheetfile is not None:
            assert dataframe is None and transposed_dataframe is None
            self._init_from_spreadsheet(spreadsheetfile, sheet, cells)
            return
        
        if dataframe is not None:
            self.df = dataframe
            if transposed_dataframe is not None:
                self.tdf = transposed_dataframe
            return
        assert transposed_dataframe is None

            
    def _init_from_spreadsheet(self, spreadsheetfile, sheet, cells):
            if spreadsheetfile.upper().endswith(
                    '.XLS') or spreadsheetfile.upper().endswith('.XLSX'):

                df = extractExcelRange(
                    spreadsheetfile, sheet=sheet, cells=cells, transpose=transpose)

                # A horrid kludge: to get rid of what Pandas did with empty headers, originally: called them "Unnamed: 0", etc.
                df = df.applymap(
                    lambda x: '{}' if x.__class__ in [str, unicode] and x.startswith('Unnamed: ') else x
                )

                # At this point, let's also deal with at least some LaTeX special chars:
                df = df.applymap(
                    lambda x: x if x.__class__ not in [str, unicode] else x.replace('$', r'\$')
                )

                if footercell is not None:
                    fc = extractExcelRange(
                        spreadsheetfile, sheet=sheet, cells=footercell)
                    assert fc.shape in [(1, 1)]
                    self._tablefooter = fc.values[0][0]
            else:
                raise ('unknown_spreadsheet_file_type')

            formatDFforLaTeX(
                df,
                row=None,
                sigdigs=sigdigs,
                colour=None,
                leadingZeros=leadingZeros)
            self.df = df
            self.tdf = None # Transposed table



    def emboldenRow(self, row):
        """
        One common task is to embolden a row.  It should be specified by a 0-based or by the row number, as a string, from Spreadsheet
 To do: row should be able to be a list. Or comma separated string. Or ":" range.
        """
        self.df.ix[row] = '{' + r'\bfseries ' + self.df.ix[row] + '}'
        if 0:  # This does also the index. Not necessary, since my output is only the interior matrix, not headers.
            if row.__class__ is int:
                ii = self.df.index.tolist()
                ii[row] = '{' + r'\bfseries ' + ii[row] + '}'
                self.df.index = ii
        return (
            self
        )  # Allows for easy chaining of command/modifications to table.

    def toTabular(
            self,
            outfile=None,
            footer=None,
            boldHeaders=False,
            boldFirstColumn=False,
            columnWidths=None,
            formatCodes='lc',
            hlines=False,
            vlines=None,
            centering=True
    ):  #   ncols=None,nrows=None,, alignment="c"): # tableName=None,landscape=None, masterLatexFile=None,
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
        if formatCodes.__class__ in [str, unicode] and len(
                formatCodes) >= self.df.shape[1]:
            cformatl = list(formatCodes)
            if vlines is not None:
                print(
                    ' Put the "|"s in your formatCodes, since you specified one.'
                )
                failhere
        else:
            cformatl = list(
                (formatCodes + 1000 * formatCodes[-1])[:(self.df.shape[1])])
        print("Format: ", cformatl)
        if columnWidths is not None:
            assert len(columnWidths) == self.df.shape[1]
            for ii, cw in enumerate(columnWidths):
                if cw is None: continue
                cformatl[ii] = 'm{' + cw + '}'
        if boldFirstColumn:
            cformatl[0] = '>{\\bfseries}' + cformatl[0]
        #firstcol='>{\\bf}l' if boldFirstColumn else 'l'
        print("Format: ", cformatl)
        if boldHeaders:
            cformatl[0] = '+' + cformatl[0]
            for ii, cw in enumerate(cformatl):
                if cw is None or ii == 0: continue
                cformatl[ii] = '^' + cformatl[ii]

        #cformat= '+'+firstcol+'|'+'|^c'*self.df.shape[1]+'|' if boldHeaders else None
        if vlines is None:
            vlines = False
        if not len(formatCodes) >= self.df.shape[1]:

            cformat = vlines * '|' + (vlines * '|'
                                      ).join(cformatl) + vlines * '|'
        else:
            cformat = ''.join(formatCodes)
        if not boldHeaders and not boldFirstColumn and not columnWidths:
            cformat = None
        self._cformat = cformat
        huh
        # Caution! toprule, bottomrule require booktabs
        self._tabular_body = ('\\\\ ' + '\\hline ' * hlines + '\n').join(
            ['&'.join(RR) for RR in self.df.as_matrix()[1:]
             ]) + '\\\\ \n\\bottomrule '  #cline{1-\\ctNtabCols}\n '
        self._tabular_header = '\\toprule\n' + (
            boldHeaders * '\\rowstyle{\\bfseries}%\n'
        ) + ' & '.join(self.df.as_matrix()[0]) + '\\\\ \n\\midrule '

        if outfile is not None:
            if not outfile.endswith('.tex'): outfile += '.tex'
            with open(outfile, 'wt', encoding='utf-8') as f:
                f.write(r"""
                {""" + centering * r"\centering" + r"""
                \begin{tabular}{""" + self._cformat + r"""}
                """ + self._tabular_header + """
                """ + self._tabular_body + """
                \end{tabular}  }
                """)
                if self._tablefooter is not None:

                    f.write(r"{\footnotesize " + self._tablefooter + '}\n')

#                print('Wrote '+outfile)

        return
        #         return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))

    def toCPBLtable(self,
                    outfile,
                    tableTitle=None,
                    footer=None,
                    boldHeaders=False,
                    boldFirstColumn=False,
                    columnWidths=None,
                    formatCodes='lc',
                    hlines=False,
                    vlines=None,
                    masterLatexFile=None,
                    landscape=None,
                    cformat=None,
                    tableName=None):
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
        self.toTabular(
            outfile=None,
            footer=None,
            boldHeaders=boldHeaders,
            boldFirstColumn=boldFirstColumn,
            columnWidths=columnWidths,
            formatCodes=formatCodes,
            hlines=hlines,
            vlines=vlines)

        if not outfile.endswith('.tex'): outfile += '.tex'
        includeTex, callerTex = cpblTableStyC(
            cpblTableElements(
                body=self._tabular_body,
                firstPageHeader=self._tabular_header,
                otherPageHeader=None,
                tableTitle=tableTitle,
                caption=self._tablecaption,
                footer=self._tablefooter,
                tableName=tableName,
                landscape=landscape,
                cformat=self._cformat),
            filepath=outfile,
            masterLatexFile=masterLatexFile)

        #print(includeTex)
        with open(outfile, 'wt', encoding='utf-8') as f:
            f.write(includeTex)
        if boldHeaders:
            print(r"""
    \usepackage{array}
    \newcolumntype{+}{>{\global\let\currentrowstyle\relax}}
    \newcolumntype{^}{>{\currentrowstyle}}
    \newcommand{\rowstyle}[1]{\gdef\currentrowstyle{#1}%
    #1\ignorespaces
    }
    """)
        #print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))
        return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE', outfile))


    def toPDF(self,
              outfile,
              tableTitle=None,
              footer=None,
              boldHeaders=False,
              boldFirstColumn=False,
              columnWidths=None,
              formatCodes='lc',
              hlines=False,
              vlines=None,
              masterLatexFile=None,
              landscape=None,
              cformat=None,
              tableName=None):
        """ Create a standalone PDF using a cpblTable filename and its wrapper info.  This should be part of the class or static? Not sure where this should go. Pystata does not seem to use these things yet."""
        stophere

def underscores_to_hyphens_in_path(apath):
    """ For simplicity, remove underscores from filenames (but not folders)
    """
    a,b= os.path.split(apath)
    return a+os.sep+b.replace('_','-')
def test_underscores_to_hyphens_in_path():
    assert underscores_to_hyphens_in_path('/a_/b-/c_d._e')=='/a_/b-/c-d.-e' # Not OS-portable
        
def cpblTable_to_PDF(filename, aftertabulartex=None, caption=None, display=False, pdfcrop=False, transposed=False):
    """ caption not implemented yet
    aftertabulartex: footnotes, legend, etc to go below table

    transposed = True will use the transposed version instead of the first one, and it will append "-transposed" to the file name
    """
    if aftertabulartex is None: aftertabulartex=''
    pathstem = os.path.splitext(filename)[0]
    trans = 'Trans' if transposed else ''
    with open(pathstem+'-standalone'+(transposed*'-transposed')+'.tex','wt') as fout:
        fout.write(texheader_for_CPBLtables(margins = 'none',
                                        standalone_table = True,
                                        allow_underscore = True,)+r"""
\usepackage{threeparttable}
            \begin{document}
            \input{"""+ filename+r"""}
"""+ (aftertabulartex is  not None) * (r"""   \begin{threeparttable}""")+r"""
            \ctStartTabular"""+trans+r"""
            \ctFirstHeader"""+trans+r"""
            \ctBody"""+trans+r""" \hline
            \end{tabular}  """+ (aftertabulartex is  not None) * (r"""
            \begin{tablenotes}
            \item   """+aftertabulartex+r"""
            \end{tablenotes}
            \end{threeparttable}""")+ r"""
   \end{document}
            """)
    doSystemLatex(pathstem+'-standalone'+(transposed*'-transposed')+'.tex', display=display)
    if pdfcrop:
        os.system('pdfcrop {f} {f}'.format(f=pathstem+'-standalone.pdf'))


def interleave_pairs_of_rows(df, list_of_column_pairs):
    """"""
def test_interleave_se_columns_as_rows():
    """
    
    """
    from collections import OrderedDict
    df0= pd.DataFrame(OrderedDict([
        ['z', [10,20,30],],
        ['b', [1,2,3],],
         ['se', [4,5,6],],
        ['zz', [10,20,30],],
        ['b2', [7,8,9],],
        ['zaz', [10,20,30],],
         ['se2', [10,11,12],],
        ['b3', [13,14,15],],
        ['se3', [16,17,18]],
        ['foo', [5,5,5]],
    ]))
    mdf0 = df0.copy()

    mdf0.columns = pd.MultiIndex.from_tuples(zip(df0.columns.tolist(),
                                                 ['B'+cc for cc in df0.columns.tolist()]),
                                                 names=['foo','bar'])
    paircols = [cc for cc in df0.columns if cc[0] in ['b','s']]
    ipaircols = [ii for ii,cc in enumerate(df0.columns) if cc[0] in ['b','s']]
    mpaircols = mdf0.columns[ipaircols]

    df = df0[paircols]
    mdf = mdf0[paircols]
    
    df2 = interleave_se_columns_as_rows(df)
    assert  df2.columns.tolist()==['b', 'b2', 'b3']
    expected = np.array([[ 1,  7, 13],
       [ 4, 10, 16],
       [ 2,  8, 14],
       [ 5, 11, 17],
       [ 3,  9, 15],
       [ 6, 12, 18]])
    diff=np.abs(df2.as_matrix() - expected)
    assert not (df2.as_matrix()-expected).any().any()
    # Test LaTeX wrappers:
    df2w = interleave_se_columns_as_rows(df, wrap_se_for_LaTeX=True)
    assert df2w['b2'].tolist()[1] == r'\coefse{10}'

    
    # What about a multiindex?
    df3 = interleave_se_columns_as_rows(mdf)
    assert df3.columns.tolist() == [('b', 'Bb'), ('b2', 'Bb2'), ('b3', 'Bb3')]
    assert df3.to_csv(index=False, sep='\t') == 'b\tb2\tb3\nBb\tBb2\tBb3\n1\t7\t13\n4\t10\t16\n2\t8\t14\n5\t11\t17\n3\t9\t15\n6\t12\t18\n'


    # And when not all columns have se's?
    df4= interleave_se_columns_as_rows(df0, pairs_of_columns= paircols, wrap_se_for_LaTeX=False)
    assert  df4.to_csv(index=False, sep='\t') == 'z\tb\tzz\tb2\tzaz\tb3\tfoo\n10\t1\t10\t7\t10\t13\t5\n\t4\t\t10\t\t16\t\n20\t2\t20\t8\t20\t14\t5\n\t5\t\t11\t\t17\t\n30\t3\t30\t9\t30\t15\t5\n\t6\t\t12\t\t18\t\n'

    # Same, but with multiindex:
    df6 = interleave_se_columns_as_rows(mdf0, pairs_of_columns= mpaircols)
    assert df6.to_csv(index=False, sep='\t') == 'z\tb\tzz\tb2\tzaz\tb3\tfoo\nBz\tBb\tBzz\tBb2\tBzaz\tBb3\tBfoo\n10\t1\t10\t7\t10\t13\t5\n\t4\t\t10\t\t16\t\n20\t2\t20\t8\t20\t14\t5\n\t5\t\t11\t\t17\t\n30\t3\t30\t9\t30\t15\t5\n\t6\t\t12\t\t18\t\n'

    # And test the latex wrapping?
    df5= interleave_se_columns_as_rows(mdf0, pairs_of_columns= mpaircols, wrap_se_for_LaTeX=True)
    assert df5.to_csv(index=False, sep='\t') == 'z\tb\tzz\tb2\tzaz\tb3\tfoo\nBz\tBb\tBzz\tBb2\tBzaz\tBb3\tBfoo\n10\t1\t10\t7\t10\t13\t5\n\t\\coefse{4}\t\t\\coefse{10}\t\t\\coefse{16}\t\n20\t2\t20\t8\t20\t14\t5\n\t\\coefse{5}\t\t\\coefse{11}\t\t\\coefse{17}\t\n30\t3\t30\t9\t30\t15\t5\n\t\\coefse{6}\t\t\\coefse{12}\t\t\\coefse{18}\t\n'

    # What about if there is an index? In this case, we want to duplicate the values, unlike for all other columns which are not in the paircols list.
    df7_i = interleave_se_columns_as_rows(df0.set_index(['foo','z']), pairs_of_columns= paircols, wrap_se_for_LaTeX=True, duplicate_index=True)
    df7_noi = interleave_se_columns_as_rows(df0.set_index(['foo','z']), pairs_of_columns= paircols, wrap_se_for_LaTeX=True, duplicate_index=False)
    print('Caution, the duplicate_index=False option does not yet work for MultiIndex row index. TO DO.')

    
def interleave_se_columns_as_rows(odf, pairs_of_columns=None, wrap_se_for_LaTeX=False, duplicate_index=True):
    """ 
When pairs_of_columns=None, assume every second column of the data frame is a standard error value for the estimate (column) to its left. Move these values so they are below the point estimates.

    That is, new rows are inserted after each existing row in the df.

    If, instead, pairs_of_columns is specified, as a list of pairs of string or tuple (for MultiIndex columns) identifiers of columns, then only those are treated.

    Values in the DataFrame can be strings or numbers; they're not formatted/changed by this method, unless wrap_se_for_LaTeX = True.

    wrap_se_for_LaTeX=True:  Assuming values are already formatted (ie strings), this wraps the strings of the standard errors in '\coefse{','}'.  No change is made to the point estimate columns.

Normally, you should call this *after* formatting floating values for display, eg with

    """
    assert len(odf.columns) % 2 == 0 or (pairs_of_columns is not None and len(pairs_of_columns) % 2 == 0)
    if pairs_of_columns is None:
        pairs_of_columns = odf.columns
    if pairs_of_columns is not None:
        est_cols = pairs_of_columns[0::2]
        se_cols = pairs_of_columns[1::2]
        other_cols = [cc for cc in odf.columns if cc not in pairs_of_columns]
    else:
        est_cols = odf.columns[0::2]
        se_cols = odf.columns[1::2]
        other_cols= []
    df=odf.copy()
    alts = df.loc[:, se_cols].copy()
    alts.columns = est_cols
    if wrap_se_for_LaTeX:
        alts = alts.applymap(str).applymap(lambda ss:  r'\coefse{'+ss+'}' if ss else '')
    
    newdf = df[[cc for cc in df.columns if cc not in se_cols]].copy() # Template for new values.
    orig_columns = newdf.columns
    newdf['__origOrd'] = range(len(newdf))
    newdf['_sorting'] = 1
    alts['_sorting'] = 2
    alts['__origOrd'] = range(len(newdf))
    for oc in other_cols: # Doing this before append() avoids conversion of ints to NaNs
        alts.loc[:,oc] =''
    anewdf = newdf.append(alts).sort_values(['__origOrd', '_sorting']) # Pandas bug: append sorts columns


    # If desired, get rid of the row labels on every second line, so they're not duplicated (preserves MultiIndex row index)
    # (Otherwise, if you want not to have duplicated row labels, then use reset_index() before calling)
    if not duplicate_index:
        anewdf.index = [x if not ii%2 else '' for ii,x in enumerate(anewdf.index)]
    # Reimpose column order, to fix bug: https://github.com/pandas-dev/pandas/issues/4588
    # And also drop the sorting columns:
    anewdf = anewdf[orig_columns]

    return (anewdf)


def interleave_and_format_paired_columns_as_rows(odf,
                                                 method='standard_errors'):
    return DEPRECATED__USE_FUNC_ABOVE_interleave_se_columns_as_rows

    """ Assume every second column of the data frame is a standard error value for the column to its left. 
    
    This makes use of interleave_se_columns_as_rows, but in addition uses a formatter to format the estimates.  201804: I want to do that a bit differently, using new tools, so this function needs update. Just use the other (lower level) one, above, for now.

Format these Move these values so they are below the point estimates.
    """
    assert method == 'standard_errors'
    df = odf.copy()
    cols = df.columns
    assert len(cols) % 2 == 0
    import pystata
    for ii in range(0, len(cols), 2):
        pystata.formatPairedRow_DataFrame(
            df, cols[ii], cols[ii + 1], prefix='tmppref')
    df = df.iloc[:, len(cols):]
    df.columns = df.columns.map(lambda ss: ss[7:])
    if 0: print df
    return interleave_se_columns_as_rows(df)


class cpblTableElements():
    def __init__(self,
                 body=None,
                 firstPageHeader=None,
                 otherPageHeader=None,
                 ncols=None,
                 nrows=None,
                 footer=None,
                 landscape=None,
                 cformat=None,
                 tableTitle=None,
                 caption=None,
                 label=None,
                 tableName=None,
                 tableType=None):
        self.body = body
        self.firstPageHeader = firstPageHeader
        self.otherPageHeader = otherPageHeader
        self.ncols = ncols
        self.nrows = nrows
        self.footer = footer
        self.landscape = landscape
        self.format = cformat
        self.tableTitle = tableTitle
        self.caption = caption
        self.label = label
        self.tableType = tableType
        self.tableName = tableName


###########################################################################################
###
#def cpblTableStyC(format,filepath,firstPageHeader,otherPageHeader,body,tableTitle,caption,label,ncols,nrows,footer=None,tableName=None,landscape=None):
###
#######################################################################################
###########################################################################################
###
# Original format:
def cpblTableStyC(
        tableElements=None,
        filepath=None,
        masterLatexFile=None,
        tableElementsTrans=None,  # Also allow definition of an alternative, transposed version of the same table to be coded in the same cpblTables file!
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

    assert '\n' in tableElements.body  # Try to catch old-style calls to new calling sequence:    debugprint('cpblTableStyC',str((format,filepath,len(body),tableTitle,caption,label,ncols,nrows,footer,tableName,landscape)))

    tE = tableElements
    tEt = tableElementsTrans

    includeTex, callerTex = '', ''

    if showTransposed in ["both", True]:
        assert tEt.body
    if showTransposed is None:
        showTransposed = False

    if filepath == None:
        filepath = 'PUT-TABLETEX-FILEPATH-HERE'
    else:
        pp = os.path.split(filepath)[0]
        filepath = pp + (len(pp) > 0
                         ) * '/' + os.path.split(filepath)[1].replace('_', '-')

    for aTE in [xx for xx in [tE, tEt] if xx is not None]:
        if aTE.tableName == None:  # Table name is for the TOC; table title is the begining of the caption
            aTE.tableName = aTE.tableTitle

        if aTE.ncols == None:
            aTE.ncols = len(aTE.body.split(r'\\')[1].split('&'))

        # At least for non-transposed, the function in cpblStata now defines the otherpageheader in detail, and then references it and adds one hline to make first page header. [nov 2010]
        if isinstance(aTE.firstPageHeader, list):
            assert isinstance(aTE.firstPageHeader[0], basestring)
            aTE.firstPageHeader = '\n'.join(aTE.firstPageHeader)
        if aTE.otherPageHeader == None:
            aTE.otherPageHeader = r'\ctFirstHeader'

        # This section should, and doesn't yet, choose between wide, simple, and long. in landscape or not.
        aTE.squishCols = aTE.ncols > 15
        #scaleTable=ncols>15
        aTE.tinyFont = aTE.ncols > 15 or aTE.nrows > 20
        if aTE.landscape == None:
            aTE.landscape = aTE.ncols > 15

        if aTE.tableType is None:  # Choose automatically
            aTE.tableType = 'CLongHere' + 0 * 'Landscape' * aTE.landscape  # Get rid of the landscape part: the entire thing is one big landscape environment now.
            # Dec2009: Following doesn't seem right. I still want to use "A" auto mode een if format is specified!
            aTE.tableType = 'C' + 'A' * (format in [None, '']) + 'Long'
            aTE.tableType = 'CACLong'  # Jan 2010: CAC means take caption from included .tex too!
            if aTE.nrows < 20:
                aTE.tableType = 'CACSimple'  # Jan 2010: CAC means take caption from included .tex too!
        assert aTE.tableType in ['CACSimple', 'CACLong']  # to be expanded!

        #%% \cpblTableBWide{1:format}{table (toc) name}{2:short caption}{3:caption}{4:label}{5:ncols for footer}{6:footer}{7:file}
        #%% Format for "C" types: (including CA..?)
        #%% \cpblTableCSimple{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

        aTE.tableCelements = [
            aTE.format, aTE.tableName, aTE.tableTitle, aTE.caption, aTE.label,
            aTE.footer, filepath
        ]
        if not aTE.format:
            aTE.format = 'c' * aTE.ncols
        # The following does not affect the original variables, of course.
        for ii in range(len(aTE.tableCelements)):
            if not aTE.tableCelements[ii]:
                aTE.tableCelements[ii] = ''
        # Jan 2010: Move caption and footer to CAC (automatic) caption! ie remove from explicit call! Put the automatic caption in blue, so it's clear it's not been moved/replaced (one can always simply remove the blue in the .tex file, when the .tex file is final.)
        aTE.CACcomments = r' {\color{blue} ' + ' '.join(
            aTE.tableCelements[3:6:2]) + '} '
        aTE.tableCelements[3], aTE.tableCelements[
            5] = '', r'{\footnotesize\cpblColourLegend} '

    # Now, display the normal, non-transposed table
    if showTransposed in [False, 'both']:
        callerTex=(r'\setlength\tabcolsep{1pt}'+'\n')*tE.squishCols+\
               r'{\usetinytablefont '*tE.tinyFont+\
               r'\cpblTable'+tE.tableType+'{'+'}{'.join(tE.tableCelements)+'}\n'+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*tE.squishCols +\
               r'}'*tE.tinyFont
    # Now, display the transposed table
    if showTransposed in [True, 'both']:
        callerTex+=(r'\setlength\tabcolsep{1pt}'+'\n')*tEt.squishCols+\
               r'{\usetinytablefont '*tEt.tinyFont+\
               0*r'\cpblTableUseTransposedtrue '+\
               r'\cpblTable'+tEt.tableType+'Transposed{'+'}{'.join(tEt.tableCelements)+'}\n'+\
               0*r'\cpblTableUseTransposedfalse '+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*tEt.squishCols +\
               r'}'*tEt.tinyFont

    includeTex = r"""
    \renewcommand{\ctNtabCols}{%d}
    \renewcommand{\ctFirstHeader}{%s}
    \renewcommand{\ctSubsequentHeaders}{%s}
    \renewcommand{\ctBody}{%s}
    """ % (tE.ncols, tE.firstPageHeader, tE.otherPageHeader, tE.body) + r"""
    % Default caption:
    \renewcommand{\ctCaption}{""" + tE.CACcomments + r"""}
    % This .tex file is meant to be called by something from cpblTables.sty. 
    % If instead it is included directly, output something crude:
    \ifx\@ctUsingWrapper\@empty
    %Code to be executed if the macro is undefined
    \begin{table}
    \begin{tabular}{""" + tE.format + r"""}
    \ctFirstHeader
    \ctBody
    \end{tabular}
    \end{table}
    \else
    %Code to be executed if the macro IS defined
    \fi

    % For "CA" versions of cpblTables, the format need not be specified in the call:
    \renewcommand{\ctStartTabular}{\begin{tabular}{""" + tE.format + r"""}}
    \renewcommand{\ctStartLongtable}{\begin{longtable}[c]{""" + tE.format + r"""}}
    """
    if tEt is not None:
        includeTex += r"""

% BEGIN TRANSPOSED VERSION
""" + r"""
    \renewcommand{\ctNtabColsTrans}{%d}
    \renewcommand{\ctFirstHeaderTrans}{%s}
    \renewcommand{\ctSubsequentHeadersTrans}{%s}
    \renewcommand{\ctBodyTrans}{%s}
    """ % (tEt.ncols, tEt.firstPageHeader, tEt.otherPageHeader, tEt.body
           ) + r"""
    % Default caption:
    \renewcommand{\ctCaptionTrans}{""" + tEt.CACcomments + r"""}

    % Better yet, for version "CA" of cpblTables, define methods so that the format need not be specified in the call.
    \renewcommand{\ctStartTabularTrans}{\begin{tabular}{""" + tEt.format + r"""}}
    \renewcommand{\ctStartLongtableTrans}{\begin{longtable}[c]{""" + tEt.format + r"""}}
    """

    if not filepath in ['PUT-TABLETEX-FILEPATH-HERE']:
        import codecs
        with codecs.open(filepath, 'wt', encoding='utf-8') as texfile:
            texfile.write(includeTex)
            texfile.close()

    if masterLatexFile:
        assert '/' in filepath  # Can't finalise the callingTex if we don't know the filepath
        from cpblDefaults import defaults
        masterLatexFile.append(
            callerTex.replace(defaults['paths']['tex'], r'\texdocs '))

    return (includeTex, callerTex)


###########################################################################################
###
def legacy_cpblTableStyC(
        body,
        format=None,
        filepath=None,
        firstPageHeader=None,
        otherPageHeader=None,
        tableTitle=None,
        caption=None,
        label=None,
        ncols=None,
        nrows=None,
        footer=None,
        tableName=None,
        landscape=None,
        masterLatexFile=None,
        # Also allow definition of an alternative, transposed version of the same table to be coded in the same cpblTables file!:
        showTransposed=None,
        bodyTrans=None,
        formatTrans=None,
        firstPageHeaderTrans=None,
        otherPageHeaderTrans=None,
        ncolsTrans=None,
        nrowsTrans=None,
        landscapeTrans=None):
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

    assert '\n' in body  # Try to catch old-style calls to new calling sequence:    debugprint('cpblTableStyC',str((format,filepath,len(body),tableTitle,caption,label,ncols,nrows,footer,tableName,landscape)))

    if tableName == None:  # Table name is for the TOC; table title is the begining of the caption
        tableName = tableTitle

    includeTex, callerTex = '', ''

    if showTransposed in ["both", True]:
        assert bodyTrans
    if showTransposed is None:
        showTransposed = False

    if bodyTrans is not None:
        assert formatTrans
        assert firstPageHeaderTrans
        assert otherPageHeaderTrans
        assert ncolsTrans
        assert nrowsTrans
        assert landscapeTrans
        NotProgrammedYet_duplicateBelow

    if filepath == None:
        filepath = 'PUT-TABLETEX-FILEPATH-HERE'
    else:
        filepath = os.path.split(filepath)[0] + '/' + os.path.split(
            filepath)[1].replace('_', '-')

    if ncols == None:
        ncols = len(body.split(r'\\')[1].split('&'))

    # At least for non-transposed, the function in cpblStata now defines the otherpageheader in detail, and then references it and adds one hline to make first page header. [nov 2010]
    if isinstance(firstPageHeader, list):
        assert isinstance(firstPageHeader[0], basestring)
        firstPageHeader = '\n'.join(firstPageHeader)
    if otherPageHeader == None:
        otherPageHeader = r'\ctFirstHeader'

    # This section should, and doesn't yet, choose between wide, simple, and long. in landscape or not.
    squishCols = ncols > 15
    #scaleTable=ncols>15
    tinyFont = ncols > 15 or nrows > 20
    if landscape == None:
        landscape = ncols > 15

    tableType = 'CLongHere' + 0 * 'Landscape' * landscape  # Get rid of the landscape part: the entire thing is one big landscape environment now.
    # Dec2009: Following doesn't seem right. I still want to use "A" auto mode een if format is specified!
    tableType = 'C' + 'A' * (format in [None, '']) + 'Long'
    tableType = 'CACLong'  # Jan 2010: CAC means take caption from included .tex too!
    if nrows < 20:
        tableType = 'CACSimple'  # Jan 2010: CAC means take caption from included .tex too!

    #%% \cpblTableBWide{1:format}{table (toc) name}{2:short caption}{3:caption}{4:label}{5:ncols for footer}{6:footer}{7:file}
    #%% Format for "C" types: (including CA..?)
    #%% \cpblTableCSimple{1:format}{2:table (toc) name}{3:short caption title}{4:caption}{5:label}{6:footer}{7:file}

    tableCelements = [
        format, tableName, tableTitle, caption, label, footer, filepath
    ]
    if not format:
        format = 'c' * ncols
    # The following does not affect the original variables, of course.
    for ii in range(len(tableCelements)):
        if not tableCelements[ii]:
            tableCelements[ii] = ''
    # Jan 2010: Move caption and footer to CAC (automatic) caption! ie remove from explicit call! Put the automatic caption in blue, so it's clear it's not been moved/replaced (one can always simply remove the blue in the .tex file, when the .tex file is final.)
    CACcomments = r' {\color{blue} ' + ' '.join(tableCelements[3:6:2]) + '} '
    tableCelements[3], tableCelements[
        5] = '', r'{\footnotesize\cpblColourLegend} '

    callerTex=(r'\setlength\tabcolsep{1pt}'+'\n')*squishCols+\
               r'{\usetinytablefont '*tinyFont+\
               r'\cpblTable'+tableType+'{'+'}{'.join(tableCelements)+'}\n'+\
               (r'\setlength\tabcolsep{6pt} % 6pt is default value'+'\n')*squishCols +\
               r'}'*tinyFont

    includeTex = r"""
    \renewcommand{\ctNtabCols}{%d}
    \renewcommand{\ctFirstHeader}{%s}
    \renewcommand{\ctSubsequentHeaders}{%s}
    \renewcommand{\ctBody}{%s}
    """ % (ncols, firstPageHeader, otherPageHeader, body) + r"""
    % Default caption:
    \renewcommand{\ctCaption}{""" + CACcomments + r"""}
    % This .tex file is meant to be called by something from
    % cpblTables.sty. If it is not, then output something crude:
    \ifx\@ctUsingWrapper\@empty
    %Code to be executed if the macro is undefined
    \begin{table}
    \begin{tabular}{""" + format + r"""}
    \ctFirstHeader
    \ctBody
    \end{tabular}
    \end{table}
    \else
    %Code to be executed if the macro IS defined
    \fi

    % Better yet, for version "CA" of cpblTables, define methods so that the format need not be specified in the call.
    \renewcommand{\ctStartTabular}{\begin{tabular}{""" + format + r"""}}
    \renewcommand{\ctStartLongtable}{\begin{longtable}[c]{""" + format + r"""}}
    """

    if not filepath in ['PUT-TABLETEX-FILEPATH-HERE']:
        texfile = open(filepath, 'wt')
        texfile.write(includeTex)
        texfile.close()

    if masterLatexFile:
        assert '/' in filepath  # Can't finalise the callingTex if we don't know the filepath
        from cpblDefaults import defaults
        masterLatexFile.append(
            callerTex.replace(defaults['paths']['tex'], r'\texdocs '))

    return (includeTex, callerTex)


################################################################################################
################################################################################################
def tsvToTable(infile, outfile=None):
    ############################################################################################
    ############################################################################################
    """ 2013 Sept.
     Transposed addition not done yet.
     Can pass filename or the tsv text contents of a file.

     2014March: this doesn't return the result yet!? Is it unused so far?
    """

    if '\n' in infile:  # Oh. they've passed contents, not filename:
        ttt = [LL.strip().split('\t') for LL in infile.split('\n')]
        assert outfile is not None
    else:
        if outfile is None:
            assert infile.endswith('.tsv')
            outfile = infile[:-4] + '--.tex'
        ttt = [LL.strip().split('\t') for LL in open(infile, 'rt').readlines()]
    body = ttt[1:]
    headers = [r"\sltheadername{" + tt + r"}" for tt in ttt[0]]
    assert headers
    assert body

    includeTex, callerTex = cpblTableStyC(
        cpblTableElements(
            body='\\\\ \hline \n'.join(['&'.join(LL) for LL in body]) +
            '\\\\ \n\\cline{1-\\ctNtabCols}\n ',
            firstPageHeader='\\hline\\hline \n' + ' & '.join(headers) +
            '\\\\ \n\\hline\\hline\n',
            otherPageHeader=None,
            tableTitle=None,
            caption=None,
            label=None,
            ncols=None,
            nrows=None,
            footer=None,
            tableName=None,
            landscape=None),
        filepath=None,  #
        masterLatexFile=None)

    print(includeTex)
    with open(outfile, 'wt', encoding='utf8') as f:
        f.write(includeTex)
    print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE', outfile))


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
        tt = open(infilepath, 'rt').read()
    else:
        tt = infilepath

    # Kludge June 2011 to deal with new, dual-format cpblTableC files:
    tt1 = tt.split('% BEGIN TRANSPOSED VERSION')[0]
    # Find basic components:
    ff = re.findall(
        r'renewcommand{\\ctFirstHeader}{(.*?)}\s+\\renewcommand{\\ctSubsequentHeaders}{(.*?)\\renewcommand{\\ctBody}{(.*?)This .tex file is meant to be called',
        tt1, re.DOTALL)[0]

    #debugprint( 'Found %d out of 3 components...'%len(ff))

    # Header
    header = ff[0] + ff[
        1]  # Actual headers could be in first or second def'n; include both.
    headerDrop = [r'\begin{sideways}', r'\end{sideways}', r'\hline', '\n']
    for hd in headerDrop:
        header = header.replace(hd, '')
    header = re.sub(r'\\sltc?r?headername{(.*?)}', r'\1', header)
    header = re.sub(r'\\\\', '\n', header)

    # Body
    body = ff[2]
    signifReplacements = [
        [r'\\wrapSigOneThousandth{(.*?)}', r'\1****'],
        [r'\\wrapSigOnePercent{(.*?)}', r'\1***'],
        [r'\\wrapSigFivePercent{(.*?)}', r'\1**'],
        [r'\\wrapSigTenPercent{(.*?)}', r'\1*'],
        [r'\\YesMark', 'Yes'],
        [r'\\sltheadername{(.*?)}', r'\1'],
        [r'\\sltrheadername{(.*?)}', r'\1'],
        [r'\\sltcheadername{(.*?)}', r'\1'],
        [r'\\coefse{(.*?)}', r'\1'],
        [r'\\showSEs{', r''],
        [r'\\\\ }{}', ''],
        [r'\$-\$', '-'],
        [r'\\hline', ''],
        [r'\\\\', ''],
        [r'\\cline{1-\\ctNtabCols', ''],
    ]
    for sr in signifReplacements:
        body = re.sub(sr[0], sr[1], body)
    body = '\n'.join(
        [LL for LL in body.split('\n') if LL.strip() not in ['', '}', '%']])
    if not '\n' in infilepath:
        with open(infilepath + '____tmp.tsv', 'wt') as fout:
            fout.write((header + body).replace('&', '\t'))
            os.system('libreoffice ' + infilepath + '____tmp.tsv &')
    return ((header + body).replace('&', '\t'))


# # Found on web, one of many. xls2csv exists but not ods2csv in Ubuntu!. Aug 2013. Adapted by CPBL to make tsv.
"""

csv has a unicodewriter too, which I should use below instead of this...
"""
import sys, zipfile, re, os, csv


def ods2tsv(filepath, outpath=None, replaceZerosWithBlanks=False):
    """
    I'm having to add various features. zeros to blanks: empties get turned into zeros when they are references..... hence replaceZerosWithBlanks
    """

    xml = zipfile.ZipFile(filepath).read('content.xml')

    def rep_repl(match):
        return '<table:table-cell>%s' % match.group(2) * int(match.group(1))

    def repl_empt(match):
        n = int(match.group(1))
        pat = '<table:table-cell/>'
        return pat * n if (n < 100) else pat

    p_repl = re.compile(
        r'<table:table-cell [^>]*?repeated="(\d+)[^/>]*>(.+?table-cell>)')
    p_empt = re.compile(r'<table:table-cell [^>]*?repeated="(\d+)[^>]*>')
    xml = re.sub(p_repl, rep_repl, xml)
    xml = re.sub(p_empt, repl_empt, xml)

    from pyquery import PyQuery as pq

    d = pq(xml, parser='xml')

    from lxml.cssselect import CSSSelector

    ns = {'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'}
    selr = CSSSelector('table|table-row', namespaces=ns)
    selc = CSSSelector('table|table-cell', namespaces=ns)
    rowxs = pq(selr(d[0]))
    data = []
    for ir, rowx in enumerate(rowxs):
        cells = pq(selc(rowx))
        if cells.text():
            data.append([
                cells.eq(ic).text().encode('utf-8') for ic in range(
                    len(cells))
            ])

    root, ext = os.path.splitext(filepath)
    #csv.register_dialect('unixpwd', delimiter='\t', quoting=csv.QUOTE_NONE)
    if outpath is None:
        outpath = ''.join([root, '.tsv'])

    with open(outpath, 'w', encoding='utf-8') as f:
        print(outpath + ' should be writing now in utf8')
        for row in data:
            if replaceZerosWithBlanks:
                f.write('\t'.join([kk if not kk == '0' else ''
                                   for kk in row]) + '\n')
            else:
                f.write('\t'.join(row) + '\n')
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
    if fmt is None: fmt = '|c|'
    _lastval, _ncols = None, 0
    outs = []
    for ii, cc in enumerate(onerow + [None]):
        if cc == _lastval or _ncols == 0:
            _ncols += 1
            _lastval = cc
        elif _ncols == 1:
            outs += [_lastval]
            _lastval, _ncols = cc, 1
        elif _ncols > 1:
            outs += [
                r'\multicolumn{' + str(_ncols) + '}{' + fmt + '}{' + str(
                    _lastval) + '}'
            ]
            _lastval, _ncols = cc, 1
        else:
            1 / 0
    return outs


def _test_single_to_multicolumn_fixer():
    assert single_to_multicolumn_fixer([1, 2, 3]) == [1, 2, 3]
    assert single_to_multicolumn_fixer(
        [1, 2, 3, 3, 3, 4, 5, 6
         ]) == [1, 2, '\\multicolumn{4}{|c|}{3}', 4, 5, 6]


def dataframeWithLaTeXToTable(
        df,
        outfile,
        tableTitle=None,
        caption=None,
        label=None,
        footer=None,
        tableName=None,
        landscape=None,
        masterLatexFile=None,
        boldHeaders=False,
        boldFirstColumn=False,
        columnWidths=None,
        formatCodes=None, #'lc',
        formatString=None,
        hlines=False,
        pdfcrop=False):  #   ncols=None,nrows=None,, alignment="c"):
    """
TODO: Clarify relationship between this method and toCPBLtable()

 Note that pandas already has a latex output function built in. If it deals with multiindices, etc, it may be better to edit it rather than to start over. [See issue #5 for columns]. However, my cpblTableC function expects it to be broken up into cells still.

This function takes a dataframe whose entries have already been converted to LaTeX strings (where needed).
So not much is left to do before passing to cpblTableStyc.

outfile should not contain underscores. Any that do exist are replace by hyphens.

boldHeaders requires the array package. It inserts a command \rowstyle before the header. It requires definition of funny column format types, and it requires a special column format.

boldFirstColumn just uses the built-in >{} column modifier (or array package?}.

columnWidths, if specified, can give LaTeX strings for the widths of some of the columns.

formatCodes is really kludgy. This is a partial or complete list of single-letter codes for the column formats. These would be used as the base values, before adding emboldening, widths, etc.

formatString is an alternative. It specifies the final complete formatcode string, which means it overwrites 

hlines: if True, put horizontal lines on every line.

footer: this is footnotes, legend, etc to put right below the table (elsewhere called aftertabulartex)
    """
    #cformat=list('l'+'c'*df.shape[1])
    # Pick the basic format codes to start with (ie ones which don't take arguments), e.g. lccccc....
    outfile = underscores_to_hyphens_in_path(outfile)
    if formatString is not None: # This would overwrite cformat
        assert formatCodes is None and columnWidths is None
        assert not boldFirstColumn  # Alternative is not written yet
    if formatCodes is None:
        formatCodes = 'lc'
    cformat = list((formatCodes + 1000 * formatCodes[-1])[:(df.shape[1])])
    if columnWidths is not None:
        assert len(columnWidths) == df.shape[1]
        for ii, cw in enumerate(columnWidths):
            if cw is None: continue
            cformat[ii] = 'm{' + cw + '}'
    if boldFirstColumn:
        cformat[0] = '>{\\bfseries}' + cformat[0]
    #firstcol='>{\\bf}l' if boldFirstColumn else 'l'
    if boldHeaders:
        cformat[0] = '+' + cformat[0]
        for ii, cw in enumerate(cformat):
            if cw is None or ii == 0: continue
            cformat[ii] = '^' + cformat[ii]

    #cformat= '+'+firstcol+'|'+'|^c'*df.shape[1]+'|' if boldHeaders else None
    cformat = '|' + '|'.join(cformat) + '|'
    if not boldHeaders and not boldFirstColumn and not columnWidths:
        cformat = None
    if formatString is not None: cformat=formatString
    if type(df.columns) == pd.MultiIndex:
        columnheaders = []
        for icr in range(len(df.columns.values[0])):
            if 0:
                print [
                    df.columns.values[ii][icr]
                    for ii in range(len(df.columns.values))
                ]
            onerow = single_to_multicolumn_fixer(
                [
                    df.columns.values[ii][icr]
                    for ii in range(len(df.columns.values))
                ],
                fmt='|c|')#cformat)
            columnheaders += [(boldHeaders * '\\rowstyle{\\bfseries}%\n') +
                              ' & '.join(onerow) + '\\\\ \n']
        if 0: print columnheaders
        firstPageHeader = '\\hline\n' + '\n'.join(
            columnheaders) + ' \\hline\\hline\n '
    else:
        firstPageHeader = '\\hline\n' + (
            boldHeaders * '\\rowstyle{\\bfseries}%\n'
        ) + ' & '.join(df.columns.values) + '\\\\ \n\\hline\\hline\n'
    includeTex, callerTex = cpblTableStyC(
        cpblTableElements(
            body=('\\\\ ' + hlines * '\\hline' + ' \n').join(
                ['&'.join(RR) for RR in df.as_matrix()]) +
            '\\\\ \n\\cline{1-\\ctNtabCols}\n ',
            firstPageHeader=firstPageHeader,
            otherPageHeader=None,
            tableTitle=tableTitle,
            caption=caption,
            label=label,
            footer=footer,
            tableName=tableName,
            landscape=landscape,
            cformat=cformat, ),
        filepath=outfile,
        masterLatexFile=masterLatexFile)
    #ncols=ncols,nrows=nrows,
    if 0: print(includeTex)
    with open(outfile, 'wt', encoding='utf-8') as f:
        f.write(includeTex)
    #print(' Wrote '+includeTex)
    if boldHeaders:
        print(r"""
\usepackage{array}
\newcolumntype{+}{>{\global\let\currentrowstyle\relax}}
\newcolumntype{^}{>{\currentrowstyle}}
\newcommand{\rowstyle}[1]{\gdef\currentrowstyle{#1}%
#1\ignorespaces
}
""")
    # Also generate a PDF of the table?
    cpblTable_to_PDF(outfile, pdfcrop=pdfcrop, aftertabulartex=footer)
    #print(callerTex.replace('PUT-TABLETEX-FILEPATH-HERE',outfile))
    return (callerTex.replace('PUT-TABLETEX-FILEPATH-HERE', outfile))


def formatDFforLaTeX(df,
                     row=None,
                     sigdigs=None,
                     colour=None,
                     leadingZeros=False,
                     noTeX=False):
    """ Convert  dataframe entries from numeric to formatted strings.
    This is probably not for use by cpblLaTeX regression class. But useful in general for making LaTeX tables from pandas work.
    How to do this...?

    For colouring alternating rows, for instance, do not do it here! Just call rowcolors before tabular.
    Here's a nice colour: \definecolor{lightblue}{rgb}{0.93,0.95,1.0}


noTeX is passed on to chooseSFormat. If you want, for instance, to align your numbers using siunitx's S parameter in LaTeX, you for instance do not want to replace the minus signs with $-$. 
    """

    def sformat(aval, sigdigs=sigdigs):
        if isinstance(aval, basestring):
            return (aval)
        return (chooseSFormat(
            aval,
            conditionalWrapper=['', ''],
            lowCutoff=None,
            lowCutoffOOM=False,
            convertStrings=False,
            highCutoff=1e90,
            noTeX=noTeX,
            sigdigs=sigdigs,
            threeSigDigs=False,
            se=None,
            leadingZeros=leadingZeros))

    if row is None:
        for irow in range(len(df.index)):
            for cc in df.columns:
                df.iloc[irow] = [
                    sformat(vv) for vv in df.iloc[irow]
                ]  # No chance of confusion with integer row labels
    else:
        for cc in df.columns:
            df.ix[row] = [sformat(vv) for vv in df.ix[row]
                          ]  # .ix allows either index or label
        return


def colorizeDFcells(df, xyindices, colour, bold=False):
    """ Sure there's a better way to do this.... I want to map stuff back onto itself.
    Return a new df with a row or whatever colorized.
e.g.
df=colorizeDFcells(df,df.PR=='Canada','red')

To do a whole row, just reset_index() and ignore the index.
    """
    df.loc[
        xyindices] = '{' + bold * r'\bfseries' + '\\color{' + colour + '}' + df.loc[
            xyindices] + '}'
    return


def wrapDFcells(df, xyindices, before='', after=''):
    """  """
    df.loc[xyindices] = before + df.loc[xyindices] + after
    return




# # # # # # # # # # # # # # # # # # 
if __name__ == '__main__':
    
    test_interleave_se_columns_as_rows()
    test_underscores_to_hyphens_in_path()
    print('All tests succeeded.')
    astop    
    """ This should do a full demo(s):
    Write an excel file. Then create outputs from it.
    """
    # Demos:
    # (1) Create a LaTeX tabular file (use it with \input{} in a LaTeX document) from some Excel data
    from pandas import DataFrame
    l1, l2, l3 = [1, 2, 3, 4], [6, 7, 1, 2], [2, 6, 7, 1]
    df = DataFrame({'Stimulus Time': l1, 'Reaction Time': l2, 'foodle': l3})
    df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
    dff = latexTable("test.xlsx", sheet='sheet1', cells="B1:C3")
    dff.emboldenRow(0)
    dff.toTabular('test-cpblt', boldFirstColumn=True)
    # (2) Create a modular CPBL table (use it with cpblTables package in LaTeX) from some Excel data
    callerTeX = dff.toCPBLtable(
        'test-cpbl', footer=None, boldFirstColumn=True, boldHeaders=True
    )


    cpblTable_to_PDF('./test-cpbl')
    cpblTable_to_PDF('./test-cpbl', transposed = True)
    #,columnWidths=None,formatCodes='lc',hlines=False,vlines=None,masterLatexFile=None,landscape=None,cformat=None)
    # (3) Use that cpblTables file and compile the result (this does not use anything from cpblTablesTex, actually, but is for completeness in the demo)
    with open('test-invoke-cpbl.tex', 'wt') as ff:
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
        """ + callerTeX + r"""
        \end{document}
        """)
    # If cpblUtilities is around, compile the LaTeX, too:

    if not os.path.exists('./tmpTEX'):
        os.makedirs('./tmpTEX')
    os.rename('test-cpbl.tex', 'tmpTEX/test-cpbl.tex')
    doSystemLatex(
        'test-invoke-cpbl.tex',
        latexPath='./',
        launch=True,
        tex=None,
        viewLatestSuccess=True,
        bgCompile=False)
