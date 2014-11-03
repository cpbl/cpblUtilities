#!/usr/bin/python



# Define some CIFAR colour scheme colours:
CIFARgreen=[168,180,0]
CIFARcyan=[0,179,190]
CIFARpink=[130,0,81] 
CIFARgrey=[103,92,83] 

from pylab import array

cifarC={
'green':[.6588,.70588,0],#{168,180,0}
'cyan':[0,0.70196,0.7451],#{0,179,190}
'pink':[0.5098,0,0.31765],#{130,0,81} 
'grey':[0.40392,0.36078,0.32549],#{103,92,83}
'darkgreen':array([91,143,34])/255.,
'lightblue':array([161,222,233])/255.,
'yellow':array([236,227,84])/255.,
'palekhaki':array([181,163,0])/255.,
'black':[0,0,0],
}

def defcolours():
    return(cifarC)
# Following retired April 2010
    adict={'green':CIFARgreen,'cyan':CIFARcyan,'pink':CIFARpink,'grey':CIFARgrey,'black':[0,0,0]}
    for cc in adict:
        adict[cc]=[vv/255. for vv in adict[cc]]
    return(adict)


colours=defcolours()
colors=colours


if __name__=='__main__':
    kk=colours.keys()
    from cpblUtilities import categoryBarPlot
    categoryBarPlot(kk,[1]*len(colours),barColour=[colours[kkk] for kkk in kk],horiz=True)#,labelLoc='opposite bar or inside')
    print colours
    import pylab as plt
    plt.show()
#return(
