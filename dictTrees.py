
"""
Aug 2006: cpbl

A dictTree incorporates a heirarchical ordering to keys in a set of dictionaries. 

This module takes a list of dictionaries and makes a tree out of a subset of the keys, with the assumption that the sets of possible values for those keys is not too large (ie so a tree is sensible).

It deals with regular and non-regular trees. (?(

it can actually makes sure the tree is regular -- ie every branch has every value, even if most are empty. I will call this a "regular tree"
Alternatively, I could write methods that made comparisons and searches safe (ie not running up against a key not existing in some branch....


STILL NEEDED: Merge trees, getBranches, ...


Definitions:

tree: a nested set of dicts with a list of some kind of list at the bottom
level of each dict. Typically, the terminal lists are lists of dicts
which have discrete-valued properties used to separate them in the
tree heirarchy. For instance, a set of dicts describing properties of
fruit could be turned into a tree based on the sequence of properties:
'colour','shape','sweetness'. The resulting tree might have top-level
keys 'red','green','yellow'.

subtree: a tree

address: a list of keys that specifies the heirarchy of dict entries leading to a particular subtree or a leaf.

branch: an address to a subtree or to a leaf

keynames: the sequence of properties (of a set of dicts distributed within lists at leaves) used to define the tree structure.

leaf: a list or other object at a bottom-level of a tree.

regular tree: a tree with the property that at a particular level (depth), dicts are similar, ie have the same set of keys...... By convention, we ensure that non-regular trees have the property of no empty leaves.

"""
        
##     def print_name(self): 
##         print "I'm", self.name
##     def call_parent(self): 
##         c_parent.print_name(self)

################################################################################
# Try again, 13 Aug 2006:
################################################################################
def dictToTree(listOfDicts,keys):
    """
    Use a recursive algorithm to add each of the keys in keys as a layer in a tree.
    This is NOT a regular tree. This (not regular) tree has the property of no empty leaves.
    
    A tree is a sequence of key names and a nested set of dictionaries with, at each terminal node,a list of dictionaries.  The nested dictionaries have keys which correspond to values of the sequence of key names. [huh? rewrite this]

    This function creates a dictionary of lists. The dictionary keys are the various possible values of the first key in parameter "keys" taken by the elements of listOfDicts. The values in the dictionary are the elements of listOfDicts sorted by the property which is the first key in parameter "keys".

Old version:


    treeList={}
    key=keys[0]
    for dd in listOfDicts:
        if treeList.has_key(dd[key]):
            treeList[dd[key]].append(dd)
        else:
            treeList[dd[key]]=[dd]
    if len(keys)>1:
        for tk in treeList.keys():
            treeList[tk]=dictToTree(treeList[tk],keys[1:])
    return(treeList)


Upgraded to deal with values of a key that are lists rather than scalars. Default behaviour will be to *mlutiply* the leaves -- ie split each record amongst the multiple branches it fits under.

"""

    treeList={}
    key=keys[0]
    if not listOfDicts:
	return(dictTree())
    assert isinstance(listOfDicts,list)
    for dd in listOfDicts:
        if isinstance(dd[key],list):# and not isinstance(dd[key][0],dict):
            # This is an .... feature. Unlikely:
            print 'splitting up (duplicating) multiplet key value: %s'%key
            toadd=[]
            for branchA in dd[key]:
                toadd=dd.copy()
                toadd[key]=branchA
                if treeList.has_key(toadd[key]):
                    treeList[toadd[key]].append(toadd)
                else:
                    treeList[toadd[key]]=[toadd]
        elif treeList.has_key(dd[key]):
            treeList[dd[key]].append(dd)
        else:
            treeList[dd[key]]=[dd]
    if len(keys)>1:
        for tk in treeList.keys():
            treeList[tk]=dictToTree(treeList[tk],keys[1:])
    return(treeList)
    
#def flatten(atree):
#    return(dictList(atree))
def flatten(atree): # Make a single-level list of all the elements in all the leaves # Rename this collapse? No renamed flatten  in May 2010
    leaves=getLeaves(atree)
    dl=[]
    for i in leaves:
        dl+=i
    return(dl)

def getLeaves(atree):
    """ Get a list of the leaves. A leaf is a terminal (end-node) lists (of dicts). This essentially "undoes" the dictToTree
    No, in May 2010 it didn't undo the dictToTree because it returned a list of lists of dicts.
    The function flatten, above, does the real collapse.
    May 2012: indeed, this is surely wrong/useless. 
    """
    leaves=[]
    if isinstance(atree,dict ):
        for k in atree.keys():
            #print 'Leaves now had length ',len(leaves)
            leaves+=getLeaves(atree[k])
            #print '   Leaves now has length ',len(leaves)
    elif isinstance(atree,list ):
        #print '   Adding end leaves of length ', len(atree),' to eaves now has length ',len(leaves)
        leaves+=[atree]
    else:
        1/0
    return(leaves)

def getBranches(atree,branchAddress=[],depth=99999):
    """ Get a list of the addresses of all leaves. A leaf is a
    terminal (end-node) lists (of dicts). An address is a list of
    keys.

    Alogorithm: given a pair (subTree, branchAddress), either return
    the branchAddress (if the subTree is really a list, ie leaf) or
    return a list of the branch addresses of all the subTrees of the
    given node.

    19 Sep 2006: upgrade: If depth is specified, only looks to given
    depth.
    """
    branchAddresses=[]
    branch=branchAddress
    #print 'Incoming: ba=',branchAddress
    if isinstance(atree,dict ) and depth>0:
        for k in atree.keys():
            branchAddresses+=getBranches(atree[k],branchAddress=branch+[k],depth=depth-1)
    #elif isinstance(atree,list ):
    else: # We have either reached a leaf (non-dict) or the requested depth.
        #print 'Done a leaf: ba=',branch
        branchAddresses=[branch]
    return(branchAddresses)

def subTree(listofDicts,keys):
    """ This should work: ie subTree and getLeaf are interchangeable. They return a atree or a leaf as appropriate. Keys is an address (list of keys).
    """
    return(getLeaf(listofDicts,keys))

def getLeaf(atree,keys,defaultValue=[]):
    """ Return the leaf at the address (list of keys) given by keys, or defaultValue if there is no such leaf.

Hm. April 2010: This may return a subtree rather than a leaf, right?
    """
    if keys[0] not in atree:
        return(defaultValue)
    if len(keys)>1:
        return(getLeaf(atree[keys[0]],keys[1:],defaultValue=defaultValue))
    else:
        return(atree[keys[0]])

def leafExists(atree,keys):
    if len(keys)>1:
        if atree.has_key(keys[0]):
            return(leafExists(atree[keys[0]],keys[1:]))
        else:
            return(0)
    else:
        return(atree.has_key(keys[0]))
        
def setLeaf(atree,keys,newLeaf):
    if len(keys)>1:
        if keys[0] not in atree:
            atree[keys[0]]={}
        setLeaf(atree[keys[0]],keys[1:],newLeaf)
    else:
        atree[keys[0]]=newLeaf

def addtoLeaf(atree,keys,newLeaf):
    setLeaf(atree,keys,getLeaf(atree,keys)+newLeaf)

def dictToRegularTree(listOfDicts,keys):
    pass

def commonBranches(tree1,tree2,depth=999999): # Previously called "compare()"
    """
    ####################################################################################
    # Compare two trees: find common branches
    ####################################################################################
    It gives lists of all the common branch names (and: not anymore: all the
    unmatched branch names) at any level of depth for two dictTrees

    Algorithm: if depth==0, or if If passed trees are leaves (ie not
    dicts) return null.  Otherwise, get a list of commons for each
    subtree, and append that subtree's key to it.

    The [None] trick and the interior if statement is a bit awkward,
    but was the only way I could...

    """
    if depth==0 or not isinstance(tree1,dict) or not isinstance(tree2,dict):
        return([None])
    else: # Recurse to give more detail on common branches:
        k1=set(tree1.keys())
        k2=set(tree2.keys())
        #noMatchK=sorted(list((k1-k2) | (k2-k1))) # Set unions over set differences # Not used
        branches=[]
        for key in sorted(list(k1&k2)):
            for branch in commonBranches(tree1[key],tree2[key],depth=depth-1):
                if branch !=None:
                    branches+=[[key]+branch]
                else:
                    branches+=[[key]]
            #c=commonBranches(getSubTree(tree1,[branch[-1]]),getSubTree(tree2,[branch[-1]]),depth=depth-1)
        return(branches)

def diffBranches(tree1,tree2,depth=999999): 
    """
    ####################################################################################
    # Compare two trees: find non-common branches. This is the complement to commonBranches()
    ####################################################################################

    It gives lists of all the not-in-common branch names at any level
    of depth for two dictTrees.

    Algorithm: if depth==0, or if If passed trees are leaves (ie not
    dicts) return null.  Otherwise, get a list of commons for each
    subtree, and append that subtree's key to it.

    The [None] trick and the interior if statement is a bit awkward,
    but was the only way I could...

    """
    b1=getBranches(tree1,depth=depth)
    b2=getBranches(tree2,depth=depth)
    diff=[]
    diff+=[b for b in b1 if b not in b2]
    diff+=[b for b in b2 if b not in b1]
    return(diff)


def old_compare(tree1,tree2,level=0): # Garbage.
    """ This works brilliantly. It gives lists of all the common branch names and all the unmatched branch names at any level of depth for two dictTrees.
    Explain the level of depth...

    I don't understand this function, so I have written a new one which returns branches, but cannot do the difference at the same time as comparison...

    No.. It returns dictTrees of the common and non-common subsets.
    No... it's a list or a dict...
    uhh. need to reassess what it does and what it should be called.
    For level==0, it gives a list of common keys.
    For level>0, it gives a dictTree of depth 1 .... and isn't doing anything sensible.

    """
    
    if level==0:
        k1=set(tree1.keys())
        k2=set(tree2.keys())
        noMatchK=(k1-k2) | (k2-k1) # Set unions over set differences
        return(sorted(list(k1&k2)),sorted(list(noMatchK))) # Return intersection, difference
    else:
        commonKeys,difk = compare(tree1,tree2) #Get list of common keys
        comDict,difDict={},{}
        for k in commonKeys: # For each key
            c,d=compare(tree1[k],tree2[k],level=level-1)
            if isinstance(c,list):
                comDict[k]=c
                difDict[k]=d
            else:
                comDict.update(c)
                difDict.update(d)
    return(comDict,difDict)
    

        
####################################################################################
# Following was from gmSalesData, but seems relevant here
####################################################################################
def dictSearch(dictList,keys,values):
    """ This does not use a tree. It just brute-force searches through a list of dicts. This is what I first used in cars.py and gmSalesData before making use of trees.

     :: I am not yet fluent with writing a class. What I want to do is simple: find all elements of a list of, say, dictionaries, which have a given set of values for a given set of keys. So write something to do this more generally, using only built-ins: [I had been using dictTree, but it's slightly more specialised]

    Two calling forms:
    
    """
    if isinstance(values,dict):
        values=[values[k] for k in keys]
    # Successively slice down the list; keep track of set of indices to elements we're after
    iDict=range(len(dictList))
    for ik in range(len(keys)):
        iDict=[i for i in iDict if dictList[i][keys[ik]]==values[ik] ]
    #if len(iDict)>1 and sum([dictList[iDict[0]][k]<>dictList[iDict[1]][k] for k in keys])>0:
    #    print 'Trouble!!!'
    #    stop
    #if len(iDict)>1:
    #    print 'Multiples',iDict,' look like:'
    #    for i in iDict:
    #        print dictList[i]['line']
    return(iDict)



def class_dictToTree(dictList,keys):
    """
    # Maybe should make a class, so that comparisons are possible, etc.
    # Turn a list of dictionaries into a recursive tree, ie sorted (branched) by each possible value of each key
    """
    tree={}
    for dd in dictList:
        branch=tree
        for k in keys[0:-2]:
            if not branch.has_key(dd[k]):
                branch[dd[k]]={}
            branch=branch[dd[k]]
        k=keys[-1]
        if not branch.has_key(dd[k]):
            branch[dd[k]]=[]
        else:
            print 'Duplicate',dd['line'],branch[dd[k]][0]['line']
        branch[dd[k]].append(dd)
    return(tree)


################################################################################
# Following was from processWhatCar, but seems relevant here
################################################################################
def displayTree(tree,keys,level=0):
    """ Recursive display of a tree made by my segmentListOfDict. "key" is the property to display in the bottom level of the tree
    """
    if isinstance(tree,list):
         print  ''.join(['   '*level+'%s\n'%'\t'.join([str(rr[k]) for k in keys]) for rr in tree])
    elif isinstance(tree,dict):
        for rr in tree.keys():
            print '   '*level + rr+ ':'
            displayTree(tree[rr],keys,level+1)
    return()




################################################################################
# MAIN: test functions
################################################################################



#import gmSalesData
#reload(gmSalesData)
#gms=gmSalesData.gms

#gmTree=dictToTree(gms,['make','model'])


# 20 Sep 2006: PRoblem. I'm now adding the list of categories that
# generate the address keys as a member of a dictTree. (This is useful
# for regenerating the tree, which I am doing for laziness in one form
# of getBranch). But then I need to be careful with the __init__ form
# which does not create a true dictTree...  Okay. This would take more
# work: implement __keynames__ in all functions..

class dictTree (dict):#treeDict: ##(dict): # Base class type is a dict
    """
    dictTree() -> new empty nested dictionary tree.
    dictTree(list of dicts, list of branch keynames) -> new dictList initialized from a list of dictionaries with some keys in common
    dictTree(dict) -> cast a homemade dict as a dictTree

    Some members:
    
    """
    def __init__(self,*args): 
        """ Allow instantiation from a list of Dicts or from a dict.
        April 2010: I'm a little confused: why is what is returned a dict, not a dictTree?
        """
        if len(args)==0: # dictTree()
            pass
        if len(args)==2: # dictTree(listOfDicts,keys)
            listOfDicts,keys=args[0],args[1]
            self.update(dictToTree(listOfDicts,keys))
            self.__keynames__=keys
        if len(args)==1: # dictTree(adict) # Needs more error checking for format...
            self.clear()
            self.update(dict(args[0]))
    def commonBranches(self,tree2,depth=999999):
        return(commonBranches(self,tree2,depth=depth))
    def subTree(self,branch):
        """subTree(branch): get a single branch of the tree
        subTree(branches): form a new tree with only the members of the specified set of branches
        """
        if branch and isinstance(branch[0],list): # Join multiple branches
            branches=branch
            dicts=[self.getLeaf(b) for b in branches]
            NotDone
            return(dictTree())
        else:
            st=getLeaf(self,branch)
            if isinstance(st,dict):# ought to update __keynames__ here...
                return(dictTree(st)) 
            else: #SHould i even allow this?
                print 'subTree(): Caution! This is actually a leaf, not a subtree...'
                return(st)
    def getLeaf(self,branch,defaultValue=[]): # A leaf should always be an end-node: a list, not a dict
        return(getLeaf(self,branch,defaultValue=defaultValue))
    def delBranch(self,branch):
        dd, """Does not work! Use a recursive method?"""
        del self.getLeaf(branch[:-1])[branch[-1]]
        return(self)
    def sample(self):
        self.update(dictToTree([{'a':1,'b':2,'c':3},{'a':1,'b':3,'c':3},{'a':1,'b':2,'c':4},{'a':2,'b':3,'c':4}],['a','b']))
        return(self)
    def getBranches(self,branchAddress=[],depth=99999):
        return(getBranches(self,branchAddress=[],depth=depth))
    def getLeaves(self):
        return(getLeaves(self))
    def isLeaf(self,keys):
	return(self.leafExists(keys))
    def leafExists(self,keys):
        return(leafExists(self,keys))
    def setLeaf(self,keys,newLeaf):
        return(setLeaf(self,keys,newLeaf)) # There is no return value...
    def flatten(self):
        return(flatten(self))
    def commonBranches(self,tree2,depth=999999):
        return(commonBranches(self,tree2,depth=depth))
    def uniqueBranches(self,tree2,depth=999999):
        """Returns all branches in self that are not in tree2, to given depth.
        So maybe I want to erase diffBranches function"""
        b1=getBranches(self,depth=depth)
        b2=getBranches(tree2,depth=depth)
        diff=[]
        diff=[b for b in b1 if b not in b2]
        return(diff)
    def merge(self, other):
	return(self.mergeTree(other))
    def mergeTree(self, other):
        'Combine trees: add all leaves of other tree to leaves of first. Duplicate elements in the leaves will be listed twice in the result...'
        ' This could also be done as a __add__ operator... But how to not change the object? Copy it all??'
        for branch in other.getBranches():
            addtoLeaf(self,branch,other.getLeaf(branch))
            #if self.leafExists(branch):
            #    self.setLeaf(branch,self.getLeaf(branch)+other.getLeaf(branch))
            #else:                    
            #    self.setLeaf(branch,other.getLeaf(branch))
        return

    def leavesAreSinglets(self):
        """
        Check to see wether the next function is "safe"
        May 2011
        """
        return(all([1==len(LL) for LL in self.getLeaves()]))
    def singletLeavesAsDicts(self):
        """ April 2010: . First addition in years. This horribly named function returns a non-treeDict recursive dict which has a dict at each leaf rather than a list of dicts. You had better have a tree for which all leaves are len(1) for this.
 Note. This does not change the object; it simply returns a dict.
        """
        assert all([len(LL)==1 for LL in self.getLeaves()])
        from copy import deepcopy
        outD=deepcopy(self)
        for AA in outD.getBranches():
            outD.setLeaf(AA,outD.getLeaf(AA)[0])
        # The above object no longer fulfills the requirements of a treeDict..
        return(outD)

    """ Old code, probably for regular trees only:
    def emptyTree(self,values):
        if len(values)==1: # We're at the lowest level: make lists rather than more levels
            branch=dict([[v,[]] for v in values[0]])
        else: # Recurse to next level
            branch=dict([[v, emptyTree(self,values[1:]) ] for v in values[0]])
        return(branch)        

    def addleaf(self,branches,leaf): # Add object leaf to the list at end of a branch
        # Here it helps that dicts are all pointers??
        if len(branches)==1: # We're at the lowest level: just insert the leaf here
            branch=dict([[v,[]] for v in values[0]])
        else: # Recurse to next level
            branch=dict([[v, emptyTree(self,values[1:]) ] for v in values[0]])
        return(branch)        
        
    def __init__(self,dictList,keys):
        # Create a tree from a list of dictionaries.
        # Check for numbers of possible values in each key:
        values=[list(set([g[k] for g in gm])) for k in keys]
        nv= [len(list(set([g[k] for g in gm]))) for k in keys]
        print 'Numbers of values in your tree (total size is product):',nv
        print values
        # First, create the empty, complete tree with a recursive function:
        self.t=emptyTree
        ## Now fill it with the data "dictLlist":
        #for dd in dictList:
        #    branch=tree
        #for k in keys[0:-2]:
        #    if not branch.has_key(dd[k]):
        #        branch[dd[k]]={}
        #    branch=branch[dd[k]]
        #k=keys[-1]
        #if not branch.has_key(dd[k]):
        #    branch[dd[k]]=[]
        #else:
        #    print 'Duplicate',dd['line'],branch[dd[k]][0]['line']
        #branch[dd[k]].append(dd)
    """
        

#class regularDictTree (dictTree):


