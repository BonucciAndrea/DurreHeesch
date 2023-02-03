import copy
import time

import concurrent.futures
import random
from random import shuffle
from multiprocessing import Process

f = open("U_2822.conf")

Info = []
Current = []
CurrentAd = []
counter = 0

# We split our file of configurations up in to readable parts:
for x in f:
    if counter == 0:
        Current.append(int(x.split()[0]))
        counter = counter + 1
        continue
    if counter == 1:
        y = x.split()
        Current.append(int(y[0]))
        Current.append(int(y[1]))
        Current.append(int(y[2]))
        counter = counter + 1
        continue
    if counter == 2:
        counter = counter + 1
        continue
    if counter == 3:
        y = x.split()
        for z in range(len(y)):
            y[z] = int(y[z])
        if y[0] == 1024:
            counter = counter + 1
            continue
        else:
            CurrentAd.append(y)
            continue
    if counter == 4:
        if x == "\n":
            counter = 0
            Current.append(CurrentAd)
            Info.append(Current)
            Current = []
            CurrentAd = []
            continue
        else:
            continue

# Turning a coloring into an essential coloring (a representation of its equivalence class up to permutation):
def ColorPermuter(l):
    order = []
    newl = []
    for x in l:                             # We determine the order of the colors
        if x in order:
            continue
        else:
            order.append(x)
    for x in l:
        for y in range(len(order)):         # Assign a color its new color with respect to correct ordering
            if x == order[y]:
                newl.append(y)              # We add the correct colors to our new coloring
    return newl

# Change an arrow sequence (ex: [1,1,0,0,1,...]) into a sequence of block indeces (ex: [0,1,2,1,0,3,...])
def ArrowToBlock(l):
    n = 0
    l1 = [0]
    for x in range(len(l) - 1):             # We go through the list of arrows:
        if l[x] == 1:                       # If up arrow
            while n in l1:                  #       look at which numbers we've used
                n = n +1
            l1.append(n)                    #       add the next available number to our list of block indeces
            n = 0
        else:
            while l1[n] != l1[-1]:          # Take current block index and find first instance of this in our list
                n = n + 1
            l1.append(l1[n-1])              # Add the previous entry
            n = 0
    return l1

def ArrowGen(x, u, b):                      # Generates a list of all possible (valid) lists of arrows recursively
    count = 0                               # x is length of list
    l = []                                  # u is a upper bound on allowed number of up arrows (or 1's) (always start at #blocks/2 rounded up)
    current = []                            # b is a lower bound on allowed (#(up arrows) - #(down arrows)) (alwayst start at 0)
    if x == 0:
        return []
    for y in range(2):
        if y == 0:
            if count == b:                  # if b == 0 then we can't add any down arrows (see 2.8.2.ii in report)
                continue
            current.append(y)
            current1 = list(current)
            z = ArrowGen(x-1, u, b + 1)     # We have added a down arrow, so b increases by 1
            if z == list([]):               # If ArrowGen returns an empty list we have reached x == 0
                l.append(current1)
            for h in z:
                for j in h:
                    current1.append(j)
                l.append(current1)
                current1 = list(current)
            del current[-1]
        if y == 1:
            if u == 0:
                continue
            current.append(y)
            current1 = list(current)
            z = ArrowGen(x-1, u-1, b - 1)   # We have added an up arrow, so both u and b decrease by 1
            if z == list([]):
                l.append(current1)
            for h in z:
                for j in h:
                    current1.append(j)
                l.append(current1)
                current1 = list(current)
    return l

# Generating all possible block decompositions for a given coloring c and a partition p
#       Returns a list of block decompositions
#       Block decomposition: a list of blocks
#       Block: a list of (Kempe) sectors
#       Sector: a list (Kempe chain) of vertices
def BlockGen(c, p):
    CurrSec = [0]                       # The sector we're currently generating. Start by placing vertex 0 in the first sector
    KempSec = []                        # Set of Kempe sectors
    if p == 0:                          # Partition {{0,1},{2,3}}
        part = [2,3]                    #           {{0,p+1},{the other,numbers}}
    if p == 1:
        part = [1,3]                    # Partition {{0,2},{1,3}}
    if p == 2:
        part = [1,2]                    # Partition {{0,3},{1,2}}
    if p == 0 and 2 not in c and 3 not in c:                    # Special case of the coloring (0,1,0,1,0,1,...) with partition {{0,1},{2,3}}
        return [[[list(range(len(c)))]]]
    for x in range(1, len(c)):          # Compare current color to previous color and check if they are in the same partition:
        if (c[x] in part and c[x-1] in part) or (c[x] not in part and c[x-1] not in part):
            CurrSec.append(x)           # Append to the current sector if they are
        else:
            KempSec.append(CurrSec)     # If the colors are in a different partition, add the current sector as a full sector to the set of Kempe sectors
            CurrSec = [x]
    KempSec.append(CurrSec)             # If we've run through all colors (thus all vertices), append the current sector to the list of Kempe sectors

    if c[-1] not in part:               # Special case where a Kempe sector contains both the first and last vertices of the ring
        for x in KempSec[-1]:
            KempSec[0].append(x)
        del KempSec[-1]
    B = []
    CurrBlock = []
    SecNum = len(KempSec)               # Number of blocks is equal to 1 or 1 + (#Kempe Sectors)/2 (see source[1] in report, coll 6.4.6)
    Arrows = ArrowGen(SecNum, int(SecNum / 2), 0)       # Number of blocks is equal to the number of up arrows (see report)
    SimBlocks = []
    for x in Arrows:
        SimBlocks.append(ArrowToBlock(x))               # Simblocks is a list of kempe sectors represented by the index of the block they belong to (see ArrowToBlock)
    for x in SimBlocks:
        for y in range(len(x)):
            if len(CurrBlock) > x[y] + 1:
                CurrBlock[x[y]].append(KempSec[y])      # x[y] is the index of the block the kempe sector KempSec[y] belongs to
            else:
                CurrBlock.append([KempSec[y]])
        B.append(CurrBlock)
        CurrBlock = []
    return B

# Create a list of all possible combinations of 0's and 1's
# We use this in KempeInt below
def OnesAndZeros(n):
    N = []
    CurrN = []
    if n == 0:
        return list([[]])
    for x in range(2):
        if x == 0:
            CurrN.append(x)
            CurrN1 = list(CurrN)
            y = OnesAndZeros(n-1)
            for z in y:
                for q in z:
                    CurrN1.append(q)
                N.append(CurrN1)
                CurrN1 = list(CurrN)
            del CurrN[-1]
        else:
            CurrN.append(x)
            CurrN1 = list(CurrN)
            y = OnesAndZeros(n - 1)
            for z in y:
                for q in z:
                    CurrN1.append(q)
                N.append(CurrN1)
                CurrN1 = list(CurrN)
    return N

# Perform all Kempe Interchanges given a coloring, partition and block decomposition
def KempeInt(B,c,p):
    C = list([])
    if p == 0:
        part = [2,3]
    if p == 1:
        part = [1,3]
    if p == 2:
        part = [1,2]
    CurrC = list(c)
    length = len(B)
    if length < 3:      # If there are only <3 blocks then performing an interchange doesn't result in a new essential coloring
        return []
# Switching the last block is the same as switching every block but the last one.
# This also extends to switching every block of the same type but the last one.
#  Hence we only need to consider OnesAndZeros(length-2) instead of OnesAndZeros(length)
    x = OnesAndZeros(length-2)
    for y in x:
        for z in range(2, length): # Range starts at 2 converting a coloring to its equivalent essential coloring makes switching the first two blocks obsolete
            for q in B[z]:
                for w in q:
                    if y[z-2] == 0:     # If 0 don't perform an interchange
                        CurrC[w] = c[w]
                    else:
                        if c[w] == 0:   # Else, perform an interchange
                            CurrC[w] = p+1
                        if c[w] == p+1:
                            CurrC[w] = 0
                        if c[w] == part[0]:
                            CurrC[w] = part[1]
                        if c[w] == part[1]:
                            CurrC[w] = part[0]
        CurrC = ColorPermuter(CurrC)
        if CurrC not in C:
            C.append(CurrC)
        CurrC = list(c)
    return C

#We use this is EssColorGenerator. This generates all not nessesarily essential colorings of size n whose first color is not f and last color is not 0.
def FourComp(n, f):
    N = []
    CurrN = []
    L = list(range(4))
    L.remove(f)
    if n == 1 and 0 in L:
        L.remove(0)
    if n == 0:
        return [[]]
    for x in range(len(L)):
        if x < len(L) - 1:
            CurrN.append(L[x])
            CurrN1 = list(CurrN)
            y = FourComp(n-1, L[x])
            for z in y:
                for q in z:
                    CurrN1.append(q)
                N.append(CurrN1)
                CurrN1 = list(CurrN)
            del CurrN[-1]
        else:
            CurrN.append(L[x])
            CurrN1 = list(CurrN)
            y = FourComp(n - 1, L[x])
            for z in y:
                for q in z:
                    CurrN1.append(q)
                N.append(CurrN1)
                CurrN1 = list(CurrN)
    return N

# Generatese a list of all possible essential colorings of length n
def EssColorGenerator(n):
    C = []
    if n % 2 == 0:
        ra = range(n-1)
    else:
        ra = range(n-2)
    for x in ra:
        CurrC = []
        for y in range(x + 2):
            if y % 2 == 0:
                CurrC.append(0)
            else:
                CurrC.append(1)
        if n == x + 2:
            C.append(CurrC)
            break
        else:
            CurrC.append(2)
            if n == x + 3:
                C.append(CurrC)
                continue
        for q in FourComp(n - x - 3, 2):
            CurrC1 = list(CurrC)
            for k in q:
                CurrC1.append(k)
            C.append(CurrC1)
    return C

# Check if a coloring is extendable to a coloring of the whole configuration.
# Takes in an adjacency matrix M and a list of possible colors for every vertex, where the colors of the ring are predetermined
def ExtendCheck(M,P):
    checker = 1
    while checker == 1:
        checker = 0
        for x in range(len(P)):
            for y in M[x]:
                if len(P[y-1]) == 1 and P[y-1][0] in P[x]:
                    checker = 1
                    P[x].remove(P[y-1][0])
    if list([]) in P:
        return False
    for x in range(len(P)):
        P1 = list(P)
        if len(P[x]) > 1:
            for y in P[x]:
                P1[x] = [y]
                if ExtendCheck(M, copy.deepcopy(P1)):
                    return True
            return False
    return True

#Initializes the list into which the 11 lists of essential colorings will be placed
Colorings = [0]*11

#The Node objects are the individual nodes in the tree in which the good colorings will be stored. A node has a color, and no children initially, but one can add childrn to it.
class Node:
    def __init__(self, color):
        self.color = color
        self.children = list([])
    def addChild(self, childcolor):
        self.children.append(Node(childcolor))

#Initializes a Root node (tho this will be done later as well)
Root = Node(0)

#Adds a coloring c to a tree with root node r.
def addToTree(c, r):
    d = list(c)
    if len(d) == 1:
        return
    for y in r.children:
        if d[1] == y.color:
            del d[0]
            addToTree(d, y)
            return
    r.addChild(d[1])
    del d[0]
    addToTree(d, r.children[-1])

#Checks if a coloring is in a tree.
def isInTree(c, r):
    d = list(c)
    if len(d) == 1:
        return True
    for y in r.children:
        if d[1] == y.color:
            del d[0]
            if isInTree(d, y):
                return True
            else:
                return False
    return False

#Converts a tree to a list (debugging purpuses)
def treeToList(r):
    l = list([])
    CurrC = list([r.color])
    if len(r.children) == 0:
        return list([[r.color]])
    for x in r.children:
        for y in treeToList(x):
            for z in y:
                CurrC.append(z)
            l.append(copy.deepcopy(CurrC))
            CurrC = list([r.color])
    return l

#The nodes of the tree in which the blocks will be stored. It can have a list of blcok decompositions, an "In" child and an "Out" child.
class BNode:
    def __init__(self):
        self.In = 0
        self.Out = 0
        self.BlockDecomp = list([])
    def AddIn(self):
        self.In = BNode()
    def AddOut(self):
        self.Out = BNode()

#Adds a list of Block Decompositions to the block-tree. c is a coloring, p is a partition, l is a "level" for recursion purpuses (starts at 0), r is the root node.
def AddBlockDecomp(c, p, l, r):
    if len(c) == l:
        if r.BlockDecomp == list([]):
            r.BlockDecomp = BlockGen(c, p)
        return
    if c[l] == 0 or c[l] == p+1:
        if r.Out == 0:
            r.AddOut()
        x = r.Out
        AddBlockDecomp(c, p, l+1, x)
        return
    else:
        if r.In == 0:
            r.AddIn()
        x = r.In
        AddBlockDecomp(c, p, l+1, x)
        return

#Accesses a list of block decompositions from a coloring & partition. l is level (starts at 0), r is root node.
def GetBlockDecomp(c, p, l, r):
    if len(c) == l:
        return r.BlockDecomp
    if c[l] == 0 or c[l] == p+1:
        if r.Out == 0:
            return False
        x = r.Out
        return GetBlockDecomp(c, p, l+1, x)
    else:
        if r.In == 0:
            return False
        x = r.In
        return GetBlockDecomp(c, p, l+1, x)

#Initializes block tree.
BRoot = BNode()

#Checks if a kempe interchange for c via block decomp B and partition p leads to a coloring in the tree with root node r. This does not generate a list of kempe interchanges, rather uses deduction to eliminate possibilities (from the list Poss) continuously
#until it reaches a contradiction, or something in the tree. The elements within "Poss" stand for 0 = "don't change this color", 1 = "change this color" and 2 = "don't know". The code would work perfectly even if Poss only contained 2s in the beginning, but because
#we know not to change the 1st 2 blocks, we will account for this in our main algorythm. Also, as we do not know what the kempe interchanges are from this function, we cannot check if they are essential. This can lead to it outputting false negatives. We account for
#this in the main algorythm.
def KempeInTree(c, B, p, Poss, l, r):
    if len(c)-1 == l:       #If the "level" gets to the last coloring, we are done.
        return True
    for x in B:             #Rewrites the Possibility list according to the rules.
        PossNum = 2
        for y in x:
            for z in y:
                if Poss[z] != 2:
                    PossNum = Poss[z]
                Poss[z] = PossNum
    if Poss[l+1] == 2:      #If we don't know wether to change or not to change the next color, we try both.
        if c[l+1] == 0:
            for x in r.children:
                if x.color == 0:
                    Poss[l+1] = 0
                    if KempeInTree(c, B, p, list(Poss), l+1, x):
                        return True
                if x.color == p+1:
                    Poss[l+1] = 1
                    if KempeInTree(c, B, p, list(Poss), l+1, x):
                        return True
            return False
        if c[l+1] == p+1:
            for x in r.children:
                if x.color == 0:
                    Poss[l+1] = 1
                    if KempeInTree(c, B, p, list(Poss), l+1, x):
                        return True
                if x.color == p+1:
                    Poss[l+1] = 0
                    if KempeInTree(c, B, p, list(Poss), l+1, x):
                        return True
            return False
        for x in r.children:
            if x.color != 0 and x.color != p+1:
                if c[l+1] == x.color:
                    Poss[l + 1] = 0
                    if KempeInTree(c, B, p, list(Poss), l + 1, x):
                        return True
                else:
                    Poss[l + 1] = 1
                    if KempeInTree(c, B, p, list(Poss), l + 1, x):
                        return True
        return False
    if Poss[l+1] == 0:      #If the next color is not to be changed, we check if that is possible.
        for x in r.children:
            if x.color == c[l+1]:
                return KempeInTree(c,B,p,Poss,l+1,x)
        return False
    else:                   #If the next color is to be changed, we check if that is possible.
        if c[l+1] == 0:
            for x in r.children:
                if x.color == p+1:
                    return KempeInTree(c,B,p,Poss,l+1,x)
            return False
        if c[l+1] == p+1:
            for x in r.children:
                if x.color == 0:
                    return KempeInTree(c,B,p,Poss,l+1,x)
            return False
        for x in r.children:
            if (x.color != 0 and x.color != p+1):
                if c[l+1] != x.color:
                    return KempeInTree(c,B,p,Poss,l+1,x)
        return False

# The main algorithm:
def algorythm6(x):
    Root = Node(0)
    t = time.time()
    Ind = x[0]          # Index of configuration
    VerNum = x[1]       # Number of vertices of configuration
    RinNum = x[2]       # Length of ring
    ExColNum = x[3]     # Number of directly extendable colorings
    DegNum = []         # List of degrees of all vertices
    AdMat = []          # Adjacency matrix of configuration
    for y in x[4]:
        DegNum.append(y[1])
    for y in x[4]:
        AdVec = []
        for z in range(2, len(y)):
            AdVec.append(y[z])
        AdMat.append(AdVec)
    AllColorings = Colorings[RinNum - 6]    # Select list of essential colorings that apply to current ring size
    GoodNo = 0                              # The number of good colorings (see report for definition of good)
    if RinNum % 2 == 1:                     # We determine the amount of essential colorings like this to cut computation time. See page 3 in the report.
        TotalCol = int((3**(RinNum-1) - 1) / 8)
    else:
        TotalCol = int((3 ** (RinNum - 1) + 5) / 8)
    ColsToIgnore = [0]*TotalCol             # This is going to keep track of the colors we have already found and thus don't need to check again
    for y in range(TotalCol):
        P = list(range(VerNum))             # Here we create the possibility coloring-matrix used by ExtendCheck
        for z in range(RinNum):             # We assign the possible colors to all the vertices of the ring
            P[z] = [int(AllColorings[y][z])] # The coloring of the ring is given
        for z in range(RinNum,VerNum):      # We assign the possible colors to all the vertices of smaller graph inside the ring
            P[z] = [0,1,2,3]                # This can be any coloring
        if ExtendCheck(AdMat, P):
            addToTree(list(AllColorings[y]), Root) # If a coloring is extendable, add it to the data tree
            ColsToIgnore[y] = 1
            GoodNo = GoodNo + 1             # We also increase the number of good colorings
        if GoodNo == ExColNum:              # Once we reach all of the directly extendable colorings, we break the for loop
            break
    while GoodNo != TotalCol:               # It's time to start the Kempe Chain Game
        CurrentGoodNo = GoodNo
        for z in range(TotalCol):
            if ColsToIgnore[z] == 1:        # Skip over any colorings that have been checked and determined good before
                continue
            y = AllColorings[z]
            checker = 0
            for q in range(3):              # q represents our color partition
                B = GetBlockDecomp(y,q,0,BRoot)     # Select the relevant block decompositions given our coloring and partition
                checker1 = 1
                for b in B:
                    Poss = [2]*RinNum               #We start a possibility list where we set 2 for each color (i.e. "don't know if I should change it") Then we set the first color in the first 2 blocks to "don't chenge".
                    Poss[0] = 0
                    if q == 1 or q == 2:
                        Poss[1] = 0
                    else:
                        for h in range(RinNum):
                            if y[h] == 2:
                                Poss[h] = 0
                                break
                    if KempeInTree(y, b, q, Poss, 0, Root): # If we find a Kempe Interchange which gives a good coloring, our color y is considered good.
                        continue                            # We move to the next block decomposition.
                    checker1 = 0
                    break                   # If there is no such Kempe Interchange we know our coloring is not yet good. Thus we break the for loop
                if checker1 == 1:           # If checker1 == 1 we have found a partition for which our coloring is good
                    checker = 1
                    break                   # Hence we break the partiton for-loop
            if checker == 1:                # If checker == 1 we have found that our coloring is now good
                addToTree(AllColorings[z], Root) # We add it to the data tree
                ColsToIgnore[z] = 1         # Ignore this coloring in the future
                GoodNo = GoodNo + 1         # Increase the amount of good colorings
#
        if CurrentGoodNo == GoodNo:         # If this equivalence holds, we didn't find any new good colorings when going through all possible colorings
        # This means we could have found a counter example, but we also could have found a false negative because the algorythm tried to Kempe Interchange to a non-essential coloring
        # This is our backup:
            for z in range(TotalCol):
                if ColsToIgnore[z] == 1:
                    continue
                y = AllColorings[z]
                checker = 0
                for q in range(3):
                    B = GetBlockDecomp(y, q, 0, BRoot)
                    checker1 = 1
                    for b in B:
                        checker2 = 0
                        for w in KempeInt(b, y, q): # Instead of trying for a valid Kempe Interchange we now generate all of the possible interchanges
                            if isInTree(w, Root):   # We check if any of the Kempe Interchanges result in a good coloring
                                checker2 = 1
                                break
                        if checker2 == 0:           # If checker2 == 0 we have found no such Kempe Interchange
                            checker1 = 0
                            break
                    if checker1 == 1:
                        checker = 1
                        break
                if checker == 1:
                    addToTree(AllColorings[z], Root)
                    ColsToIgnore[z] = 1
                    GoodNo = GoodNo + 1
            if CurrentGoodNo == GoodNo:             # If we now find no new good coloring after going through every possible coloring,
                return False                        # we have found a counterexample to the Four Color Theorem
    t = time.time() - t
    print(t)
    return True                             # If we exit all loops we can conlude that this configuration can always be colored using just 4 colors

# First we generate a list all essential colorings
for x in range(6,17):
    Colorings[x-6] = EssColorGenerator(x)
    print(x)

# Then we generate a list of all possible block decompositions
for x in range(6,17):
    for y in Colorings[x-6]:
        for z in range(3):
            AddBlockDecomp(y,z,0,BRoot)
    print(x)

# Then we start the algorithm
for x in range(2822):
    print(x+1, algorythm6(Info[x]))

# We also ran the code on Lex's miner. The following code ensures python uses the miner's multiple cores.

# cores = 3
# random.seed(0)
# nums = sorted([Info[x] for x in range(2822)], key=lambda k: random.random())
# batches = [[m for m in nums[n:min(n+cores,10)]] for n in range(0,10,cores)]
# with concurrent.futures.ProcessPoolExecutor() as executor:
#                 res = executor.map(algorythm6, batches)
#                 for r in res:
#                     if r == False:
#                         print("Neen")
#                     else:
#                         print("J")

# The code finished with J. 