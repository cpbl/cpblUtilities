cpblUtilities
=============

Various tools: plotting, color mapping /colormaps, file storage, etc, etc

utilities.py

color.py
 Smart generation / use of colormaps, including scaling them to data in order to maximise contrast
 Colorbars for non-images
 etc
 
configtools.py


dictTrees.py
 Defines a class for recursive trees which contain, at the bottom levels, lists of some kind.

mapping.py
 Fill in blank SVG maps (tagged geographic outlines) with data
 
mathgraph.py
  Assorted plotting (matplotlib) and numerical routines

optimization.py
  
  
parallel.py
  Defines runFunctionsInParallel, a general purpose tool for parallelizing tasks, managing load, etc.
  
stats.py


textables
Defines a flexible LaTeX format used especially for regression output (data), which allows later on-the-fly layout choices/changes to table formatting. Also, associated tools, including extraction of tabular data from Stata, from Excel, from LibreOffice. 
