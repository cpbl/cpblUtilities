import matplotlib as mpl
import matplotlib.pyplot as plt


def set_foregroundcolor(ax, color):
     '''For the specified axes, sets the color of the frame, major ticks,                                                             
         tick labels, axis labels, title and legend                                                                                   
     '''
     for tl in ax.get_xticklines() + ax.get_yticklines():
         tl.set_color(color)
     for spine in ax.spines:
         ax.spines[spine].set_edgecolor(color)
     for tick in ax.xaxis.get_major_ticks():
         tick.label1.set_color(color)
     for tick in ax.yaxis.get_major_ticks():
         tick.label1.set_color(color)
     ax.axes.xaxis.label.set_color(color)
     ax.axes.yaxis.label.set_color(color)
     ax.axes.xaxis.get_offset_text().set_color(color)
     ax.axes.yaxis.get_offset_text().set_color(color)
     ax.axes.title.set_color(color)
     lh = ax.get_legend()
     if lh != None:
         lh.get_title().set_color(color)
         #lh.legendPatch.set_edgecolor('none')
         labels = lh.get_texts()
         for lab in labels:
             lab.set_color(color)
     for tl in ax.get_xticklabels():
         tl.set_color(color)
     for tl in ax.get_yticklabels():
         tl.set_color(color)


def set_backgroundcolor(ax, color):
     '''Sets the background color of the current axes (and legend).                                                                   
         Use 'None' (with quotes) for transparent. To get transparent                                                                 
         background on saved figures, use:                                                                                            
         pp.savefig("fig1.svg", transparent=True)                                                                                     
     '''
     ax.patch.set_facecolor(color)
     lh = ax.get_legend()
     if lh != None:
         lh.legendPatch.set_facecolor(color)

         
##############################################################################
##############################################################################
#
def figureToInverseVideo(fig=None, debug=False):
    ##########################################################################
    ##########################################################################
    """
2013 June: backgrounds of boxed text missing. (bbox). Cannot figure out how. 
Added: setp(fff,'facecolor','k') No! Actually, use facecolor='k' option in savefig!
Replaced a bunch of loops with a findobj!
    
    May 2012, ... I've finally made some progress on this. Set figure axis background color, foreground color, and save figure as transparent. Plus, check that black text, black lines .... and then also, harder! other black stuff like patches are dealt with.

sep2012:    Agh. what about stuff drawn in bg colour, eg to over? I want to swtich that to the new bg colour! Not done. Wasneeded for PQ/liberal colour band in rdc/regressoinsQuebec.

debug = True will wait for confirmation at each step, in order to check what's happening
    """
    if fig is None:
        fig=plt.gcf()
    k2w = {'k':'w', (0,0,0):(1,1,1), (0,0,0,0):(1,1,1,0), (0,0,0,1):(1,1,1,1), (0.0,0.0,0.0,1):(1,1,1,1)} # Lookup for blacks to whites
    def gray2gray(colour):
        """ Invert the gray level for gray colours """
        if hasattr(colour, 'shape'): # Is mpl.array!
            assert len(colour) in [3,4]
            colour=list(colour)
            assert colour[0]<=1
            if colour[0]==colour[1]==colour[2]:
                colour[0]= 1-colour[0]
                colour[1]= colour[0]
                colour[2]= colour[0]
                return colour
        return None
    
    def cpause(ss='Press enter'):
        if not debug: return
        plt.show(), plt.draw()
        raw_input(ss)
    cpause('About to invert colours')

    def check_and_set_color_using_function(oo, set_function, colour):
        """  The set_function may be the object's set_color or its set_facecolor, for instance. Thus, all three parameters must be set.
        """
        origc=colour
        if hasattr(colour, 'shape') or isinstance(colour,tuple): # Is mpl.array or tuple
            assert len(colour) in [3,4]
            colour=list(colour)
        """ Invert the gray level for gray colours """
        if len(colour) in [3,4]  and colour[0]==colour[1]==colour[2]:
            assert colour[0]<=1
            colour[0]= 1-colour[0]
            colour[1]= colour[0]
            colour[2]= colour[0]
            set_function(colour)
            if debug: print(' Found gray: {} --> {} in {}'.format(origc,colour, oo.__class__.__name__))
            return
        # Deal here with hex grays (NOT DONE YET
        #
        if tuple(colour) in k2w: # "black" to "white"
            if debug: print(' Found "black": {} --> {} in {}'.format(origc,k2w[tuple(colour)], oo.__class__.__name__))
            set_function(k2w[tuple(colour)])
            return
        if debug: print(' Not changing color {} of {}'.format(colour, oo.__class__.__name__))

    def check_and_set_color(oo):
        if hasattr(oo,'get_color'):
            check_and_set_color_using_function(oo, oo.set_color, oo.get_color())
            return
        if hasattr(oo,'get_edgecolor'):
            check_and_set_color_using_function(oo, oo.set_edgecolor, oo.get_edgecolor())
        if hasattr(oo,'get_facecolor'):
            check_and_set_color_using_function(oo, oo.set_facecolor, oo.get_facecolor())
        
    def OBSELETE_checkColour(oo):
        colour=oo.get_color()
        origc=colour
        if hasattr(colour, 'shape'): # Is mpl.array!
            assert len(colour) in [3,4]
            colour=list(colour)
        """ Invert the gray level for gray colours """
        if len(colour) in [3,4]  and colour[0]==colour[2]==colour[3]:
            assert colour[0]<=1
            colour[0]= 1-colour[0]
            colour[1]= colour[0]
            colour[2]= colour[0]
            oo.set_color(colour)
            if debug: print(' Found gray: {} --> {}'.format(origc,colour))
            return
        # Deal here with hex grays (NOT DONE YET
        #
        if colour in k2w: # "black" to "white"
            if debug: print(' Found "black": {} --> {}'.format(origc,k2w[colour]))
            oo.set_color(k2w[colour])
            return
        if debug: print(' Not changing color {} of {}'.format(colour, oo.__class__.__name__))

    def OBSELET_checkFaceEdgeColour(oo):
        if hasattr(oo,'get_edgecolor'):
            colour=oo.get_edgecolor()
            if hasattr(colour, 'shape'): # Is mpl.array!
                if colour.shape in [(1,4)]:
                    if sum(colour[0][0:3])==0:
                        # Leave alpha as is; set black to white:
                        colour[0][0:3]=[1,1,1]
            elif len(colour) in [3,4]:
                colour=list(colour)
                if colour in ['k',(0,0,0),(0,0,0,0),]:
                    o.set_edgecolor('w')
            else:
                ffffooijoweiruiuiuuoiuiu
        colour=oo.get_facecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_facecolor('w')
            if colour in ['w']:#,(0,0,0),(0,0,0,0),]:
                foiofofooffo
                
        else:
            ffffooijoweiruiuiuuoiuiu

            
    for aaa in plt.findobj(fig,mpl.axes.Axes):
        set_backgroundcolor(aaa,'black') # Not None?
        cpause('Axes bgs')
        set_foregroundcolor(aaa,'white')
        cpause('Axes fgs')

    for o in fig.findobj(lambda ooo: hasattr(ooo, 'set_facecolor') ):
        check_and_set_color(o)
        ###checkFaceEdgeColour(o)
    cpause('All faces')
            
    for o in fig.findobj(mpl.lines.Line2D):
        check_and_set_color(o)
    cpause('lines')        
    for o in fig.findobj(mpl.text.Text):
        check_and_set_color(o)
    cpause('text')        



    if 0: # Wow!! All these are obselete, given the above (new 2013 June)!?
        for o in fig.findobj(mpl.patches.Patch):
            checkFaceEdgeColour(o)
        for o in fig.findobj(mpl.collections.PolyCollection):
            checkFaceEdgeColour(o)

        for o in fig.findobj(mpl.patches.Rectangle):
            checkFaceEdgeColour(o)

    # what about figure itself? See also  facecolor of savefig...
    if fig.get_facecolor()[0] in [.75, 1.0]: # Not sure why not just always set it to 'k'.? (201710cpbl)
        fig.set_facecolor('k')
    elif fig.get_facecolor() not in [(0.0, 0.0, 0.0, 1.0)]:
        print('deal with this')
        print fig.get_facecolor()
        deal_with_this


        
    return()

##############################################################################
##############################################################################
#
def figureToGrayscale(fig=None, debug=True):
    ##########################################################################
    ##########################################################################
    """
    May 2012. Rewriting this from scratch, copied from figureToInverseVideo, after having some luck with the latter.
    This is only just started, though.
    """
    if fig is None:
        fig=plt.gcf()
    def cpause(ss='Press enter'):
        if not debug: return
        plt.show(), plt.draw()
        raw_input(ss)

    if 1: # Note reall
        for aaa in plt.findobj(fig,mpl.axes.Axes):
            set_backgroundcolor(aaa,None) # Not None?
            set_foregroundcolor(aaa,'black')
            cpause('anax')
    def meanColour(oo):
        colour=oo.get_color()
        if hasattr(colour, 'shape'): # Is mpl.array!
            #iopoio
            assert len(colour) in [3,4]
            colour=list(colour)
        if colour not in  ['w','white','k','black',(0,0,0),(0,0,0,0),(1,1,1),(1,1,1,1),]:
            if isinstance(colour,str):
                o.set_color('k')
            #Not finished!!!!
        #Not finished!!!!
    def meanFaceEdgeColour(oo):
        colour=oo.get_edgecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_edgecolor('w')
        else:
            ffffooijoweiruiuiuuoiuiu
        colour=oo.get_facecolor()
        if hasattr(colour, 'shape'): # Is mpl.array!
            if colour.shape in [(1,4)]:
                if sum(colour[0][0:3])==0:
                    # Leave alpha as is; set black to white:
                    colour[0][0:3]=[1,1,1]
        elif len(colour) in [3,4]:
            colour=list(colour)
            if colour in ['k',(0,0,0),(0,0,0,0),]:
                o.set_facecolor('w')
        else:
            ffffooijoweiruiuiuuoiuiu

    for o in fig.findobj(mpl.lines.Line2D):
        meanColour(o)

    cpause('lines')        
    for o in fig.findobj(mpl.text.Text):
        meanColour(o)
    cpause('text')
    """
    for o in fig.findobj(mpl.patches.Patch):
        meanFaceEdgeColour(o)
    for o in fig.findobj(mpl.collections.PolyCollection):
        meanFaceEdgeColour(o)


    for o in fig.findobj(mpl.patches.Rectangle):
        meanFaceEdgeColour(o)
        """
    return()

##############################################################################
##############################################################################
#
def figureToGrayscaleOld(): # This is a disaster for envelopes, no!? it sets all colors t ogrey!!! needs fixing!  May 2012: delete this when y
    ##########################################################################
    ##########################################################################
    fig=plt.gcf()

    from pylab import mean
    for o in fig.findobj(mpl.patches.Rectangle):
        rgba=o.get_facecolor()
        if not isinstance(rgba,str) and len(rgba)==4:
            gg=mean(rgba[:-1])
            o.set_facecolor([gg,gg,gg,rgba[-1]])

    for o in fig.findobj(mpl.lines.Line2D):
        o.set_color('k')

    # Set all text to black:
    for o in fig.findobj(mpl.text.Text):
        o.set_color('k')


    # Set all envelopes / patch polys to black (this seems to make them appropriately gray if alpha<1 !?)
    setp(fig.findobj(mpl.collections.PolyCollection),'color','k')

    return()


if __name__ == '__main__':
    """ Demo functionality """
    plt.close('all')
    fig = plt.figure(117)
    plt.plot((1,10),(1,10), 'r', label='red'), plt.plot((2,10),(1,10), 'b', label='blue'), plt.plot((4,10),(1,10), color = [.2,.2,.2], label='dark grey'), plt.plot((5,10),(1,10), color = [.8,.8,.8], label='light grey'), plt.plot((3,10),(1,10), 'k',label='main'),     plt.title('My title'),     plt.xlabel('foo'), plt.legend()
    plt.show()
    figureToInverseVideo(fig, debug= True)
    
