#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Maybe many of my problems with unicodes being wrong can be solved by strictly using codecs.open rather than open.


See how-to : http://docs.python.org/release/3.0.1/howto/unicode.html

The rules for translating a Unicode string into a sequence of bytes are called an encoding

 For example, Python’s default encoding is the ‘ascii’ encoding. The rules for converting a Unicode string into the ASCII encoding are simple; for each code point:

If the code point is < 128, each byte is the same as the value of the code point.
If the code point is 128 or greater, the Unicode string can’t be represented in this encoding. (Python raises a UnicodeEncodeError exception in this case.)
Latin-1, also known as ISO-8859-1, is a similar encoding. Unicode code points 0-255 are identical to the Latin-1 values, so converting to this encoding simply requires converting code points to byte values; if a code point larger than 255 is encountered, the string can’t be encoded into Latin-1.

UTF-8 uses the following rules:

If the code point is <128, it’s represented by the corresponding byte value.
If the code point is between 128 and 0x7ff, it’s turned into two byte values between 128 and 255.
Code points >0x7ff are turned into three- or four-byte sequences, where each byte of the sequence is between 128 and 255.

-------------------

Since Python 3.0, the language features a str type that contain Unicode characters 

[ meaning u'' doesn't exist anymore?? Oh! I'm using Python 2.7 still.]

To insert a Unicode character that is not part ASCII, e.g., any letters with accents, one can use escape sequences in their string literals as such:

>>> "\N{GREEK CAPITAL LETTER DELTA}"  # Using the character name
'\u0394'
>>> "\u0394"                          # Using a 16-bit hex value
'\u0394'
>>> "\U00000394"                      # Using a 32-bit hex value
'\u0394'
In addition, one can create a string using the decode() method of bytes. This method takes an encoding, such as UTF-8, and, optionally, an errors argument.

One-character Unicode strings can also be created with the chr() built-in function, which takes integers and returns a Unicode string of length 1 that contains the corresponding code point. The reverse operation is the built-in ord() function that takes a one-character Unicode string and returns the code point value:

>>> chr(40960)
'\ua000'
>>> ord('\ua000')
40960

-------------------
So, byte.decode() goes from byte to str
str.encode() goes from str to bytes

s.decode(encoding)
u.encode(encoding)

-------------------


editing source:

Ideally, you’d want to be able to write literals in your language’s natural encoding. You could then edit Python source code with your favorite editor which would display the accented characters naturally, and have the right characters used at runtime.

UTF-8 is default, so no need to specify...



The most important tip is:

Software should only work with Unicode strings internally, converting to a particular encoding on output.


hm, okay, so why have I had any trouble?!  open() is unicode aware, and so I can give it an encoding='utf-8' param..... Uh, no. codecs.open() is. You need to replace open() with that.


Now there are two possible sources of UnicodeErrors. A) Python is trying to get unicode data from some sequence of bytes it got from somewhere outside and B) Python is trying to make a sequence of bytes from some unicode data because it will be sending these bytes to the outside world.
In both cases you will have to "help" python by explicitly stating what method it should use to make unicode from bytes (decoding) or bytes from unicode (encoding).


Dec 2011: upgraded to python3, in case that makes things easier...  It didn't.

2013 July: copied unicode_to_latex.py to my ~/bin.
"""


def ensure_unicode(
         obj, encoding='utf-8'):
     if isinstance(obj,list):
          return([ ensure_unicode(lobj, encoding=encoding) for lobj in obj])
     if isinstance(obj, basestring):
         if not isinstance(obj, unicode):
             obj = unicode(obj, encoding)
     return obj


########################################################################################################
########################################################################################################
########################################################################################################
# Yeah, but the following you need to know what encoding your text is in!
#  This strips accents, clearly, rather than rendering them in latex.
#  from stackexchange. Dec 2011
# Aha! Fine, in my case for the test code in the MAIN section of this file, that means utf-8, becaues this source file is in utf-8. IT works! Dec 2011 cpbl
########################################################################################################
import unicodedata

def not_combining(char):
        return unicodedata.category(char) != 'Mn'

def strip_accents(text, encoding):
        unicode_text= unicodedata.normalize('NFD', text.decode(encoding))
        return filter(not_combining, unicode_text).encode(encoding)





################################################################
# LaTeX accents replacement
# 2013: This seems obselete: a very complete one is in unicode_to_latex.py
str2texTranslation_table = dict([(ord(k), unicode(v)) for k, v in  [
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
]
])

################################################################
################################################################
def accentsToLaTeX(astr):
################################################################
################################################################
    """
2013 July: Obselete?? See new file unicode_to_latex.py, and its callers.
    
    Look for bad characters in strings. Make standard TeX-compatible substitutions here.

    or if TeX is false, do non-TeX version? [not done]

    # See http://www.petefreitag.com/cheatsheets/ascii-codes/ for instance, for a lookup!
    If you get a plotting error with 


N.B. Why do this?

Do you really need this? LaTeX has been supporting Unicode for a long time, it is sufficient to include \usepackage[utf8]{inputenc} in the preamble for UTF8 input

Hm, but  Python plotting parses TeX and I'm not sure I can put that header in... Maybe,

yes, it can. see the docs!

from matplotlib import rcParams
rcParams['text.usetex']=True
rcParams['text.latex.unicode']=True



Dec 2011: Ah!! Here's the answer. I should be doing this by using LaTeX's own database, e.g.  utf8enc.dfu  , and python's own package : import unicodedata

However, I'm proceeding with this horrid hack, thanks to the following work already done:

http://stackoverflow.com/questions/4578912/replace-all-accented-characters-by-their-latex-equivalent


Agh
 I thought this was working, but it now fails.  After 10 years I still don't understand high characteres!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

 sudo apt-get python-unac ... no. can't make sense out of that either.


 Dec 2011: str2latex(ss): I've just put a few in that function, and it's working for now. If python complains about 0x3fe, just print 0x3fe to find out the character number; if it's <256, it's in the html lookup above. So add it to my list in str2latex.
    """
    if 0:
        import unac  # Hmmm. some people don't like this?!~???!?


    import unicodedata
    def unfinished_strip_accents(s):
       return ''.join((c for c in unicodedata.normalize('NFD', unicode(s,errors='replace')) if unicodedata.category(c) != 'Mn'))
    return(strip_accents(astr,'utf-8'))

    import unicodedata

    def remove_accents(input_str):
        nkfd_form = unicodedata.normalize('NFKD', unicode(input_str))###'utf8',errors='ignore'))
        return u"".join([c for c in nkfd_form if not unicodedata.combining(c)])
    return(remove_accents(astr))

    subss={'\xb4':"'", # apostrophe
           '\xe3':r'\"e', # e-umlaut ?
           '\xf3':r"\'o", # o acute
           '\xe3':r"\`O", # O grave. 227
           }
    if 0:
        subss.update(dict([
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
    if 1:
        for ss in subss:
            if ss in unicode(astr):
                astr=astr.replace(ss,subss[ss])
        return(astr)
    # Nope! Ignore my efforts above!
    return(astr.translate(str2texTranslation_table))



def str2latex_punctuation(ss):
     subs=dict([
	     [ u"‘", "`" ],    #Quotes
	     [ u"’", "'" ],
	     [ u"“", "``" ],
	     [ u"”", "''" ],
	     [ u"‚", "," ],
	     [ u"„", ",," ],
	     ])
     for asub in subs.keys():
        if asub in ss:
             ss=ss.replace(asub.encode('utf-8'),subs[asub])   #
     return(ss)

def str2latex_specialAndMath(ss,safeMode=True):
     """
     This collects replacements that must not be done more than once!
     Also, the order matters, e.g. replacing '\\' after inserting it could be bad.
     So order matters, so don't use a dict!

     2013: safeMode now does nothing if it looks at all like it might have already been converted. 
     """

     if '\\' in ss and safeMode:
          return(ss)

     subs=[
	[ u"\\", "\\\\" ],    # Characters that should be quoted
	    ['$',r'\$'],
            ['_',r'\_'],
	    ['%',r'\%'],
	    ['#','\#'],
	    ['^','\^'],
	    ['&','\&'],
	    ['<=',r'$\leq$'],
	    ['>=',r'$\geq$'],
	    ['<',r'$<$'],
	    ['>',r'$>$'],
	[ u"~", "\\~" ],
	#[ u"&", "\\&" ],
	#[ u"$", "\\$" ],
	[ u"{", "\\{" ],
	[ u"}", "\\}" ],
	#[ u"%", "\\%" ],
	#[ u"#", "\\#" ],
         #[ u"_", "\\_" ],
	    ]
     # How will I check further for this being run more than once? March 2013 kludge:
     ss2=ss
     for asub in subs:
        ss2=ss2.replace(asub[1],'')
     try: # July 2015 workaround: Unicode failure on Apollo (RHEL).  Still need to figure out the problem and rewrite this sectoin.
          any(sst in ss2 for sst,ssr in subs)
     except UnicodeDecodeError:
          print('[u?]'),
          return(ss)
     if not any(sst in ss2 for sst,ssr in subs):
          return(ss) # If it looks like everything was already fixed, then return. If we have a partially fixed string, this wil fail still.

     for asub in subs:
        ss=ss.replace(asub[0],asub[1])
     return(ss)


def str2latex_accents(ss): # Should not need this!! Just use \usepackage[utf8]{inputenc} in LaTeX preamble
    subs=dict([[a,b] for a,b in [
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
	[ u"≥", "$\\ge$" ],   # Math operators
	[ u"≤", "$\\le$" ],
	[ u"≠", "$\\neq$" ],
	[ u"©", "\copyright" ], # Misc
	[ u"ı", "{\\i}" ],
	[ u"µ", "$\\mu$" ],
	[ u"°", "$\\deg$" ],
	]])
    for asub in subs:
        if asub in ss:
             ss=ss.replace(asub.encode('utf-8'),subs[asub])   #
        #ss=accentsToLaTeX(    ss.replace(asub[0],asub[1])   )
    return(ss)


if 0:
    {u'\xb4':"'", # apostrophe
           u'\xe3':r'\"e', # e-umlaut ?
           u'\xf3':r"\'o", # o acute
           u'\xe3':r"\`O", # O grave. 227
           u'\x92':"AE" # AE
           }
    ssubs={}
    #ssub.update(dict([a,b for a,b in [
    for asub in subs:
        ss=ss.replace(asub.encode('utf-8'),subs[asub])   #
        #ss=accentsToLaTeX(    ss.replace(asub[0],asub[1])   )
    return(ss)
"""
unicode(
,encoding='utf8')
"""

def str2latex(ss):
    """ Special and math must come first, otherwise i'll be replace replaced //'s
    """
    s1=str2latex_specialAndMath(ss)
    s2=str2latex_punctuation(s1)
    return(str2latex_accents(s2))



if __name__ == '__main__':

    # in a cp1252 environment
    #print strip_accents("déjà", "cp1252")
    print strip_accents("déjà", "utf-8")
    assert strip_accents("déjà", "utf-8")=="deja"
    # in a cp1253 environment
    #print strip_accents("καλημέρα", "cp1253")
    print strip_accents("καλημέρα", "utf-8")
    assert strip_accents("καλημέρα", "utf-8")=="καλημερα"

    print accentsToLaTeX("καλημέρα déjà")

    print str2latex_punctuation('It´s coo')

    print str2latex('It´s coo'), "'" in str2latex('It´s coo') # This string is already encoded. As utf-8. That's why it is displayed as an apostrophe in Emacs.  So, you should pass *already encoded* strings to str2latex. Putting a "u" in front of this tring gives an error because ...
    # Oops. the above is not working. the apostrophe is not replaced.

    if 1: # I need help dealing with these. It ought to be prefixed with the escape character. Maybe just do that by hand? no...
	    whatisthis=u'Danny\xb4s song' # without the u prefix, the following fails. ie unicode(whatisthis) doesn't recover...
	    uversion=unicode(whatisthis).encode('utf-8')  # Huh? why does this work?
	    print uversion
	    uversion=whatisthis.encode('utf-8')  # Huh? why does this work?
	    print uversion
	    print str2latex(uversion), "'" in str2latex(uversion) # oh-oh. str2latex doesn't catch this.
    #str2texTranslation_table = dict([(ord(k), unicode(v)) for k, v in  [

    from codecs import open # Must always do this when writing files (!) to get unicode-aware file managment?!
    fout=open('tmptmptmptest.tex','wt',encoding='utf-8')
    tex=r"""
    \documentclass{article}
 \usepackage[utf8]{inputenc} 
\begin{document}
Hi. Here's some très bien stuff.
\end{document}
"""
    fout.write(tex).encode('utf-8') # Fails on Apollo (RHEL) and I don't know why. Fine under Ubuntu
    fout.close()
    import os
    os.system('pdflatex tmptmptmptest')
    os.system('evince  tmptmptmptest.pdf')
    
