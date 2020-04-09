# STUDENT NAMES: (TODO: Samih Sghier, Yefet, Thomas)

## This program performs various search algorithms to transform one English
## word into another, with each intermediate word also being a valid English word.
##
## Original author: Dr. Forrest Stonedahl
## Purpose: Educational purposes for CSC 320: Artificial Intelligence at Augustana College
## Date: December 06, 2017

# Note: for this problem, the *world states* will simply be represented by python Strings,
#   whereas the *search states* 
#   (which include the path by which the agent could arrive at a particular world state)
#    will be represented by instances of the SearchNode class.

from queue import Queue, LifoQueue, PriorityQueue  # See: https://docs.python.org/3/library/queue.html
from functools import total_ordering
import warnings
warnings.filterwarnings("ignore")
# For our set of legal English words, we'll use the official scrabble tournament dictionary from 1998...

# TODO: For quicker debugging, you may first try using the smaller dictionaries 
#        ("super_small_dict.txt", "TWL98_atmost4.txt", etc)
# TODO: But for timing measurements & analysis, go back to using the full "TWL98.TXT" later
LEGAL_WORD_SET = frozenset(open("TWL98.txt", "r").read().split())
ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # just the letters A-Z (no punctuation, etc)


########### Begin Action CLASS #############
# NOTE: You should not need to change this class at all, but you should understand it.
class Action:
    """Each action consists of the ACTION string, the action cost, 
      the starting state, and the state that would result if this action were taken."""

    def __init__(self, actionName, actionCost, startState, resultingState):
        self.actionName = actionName
        self.actionCost = actionCost
        self.startState = startState
        self.resultingState = resultingState

    def getActionName(self):
        return self.actionName

    def getActionCost(self):
        return self.actionCost

    def getStartState(self):
        return self.startState

    def getResultingState(self):
        return self.resultingState

    def __str__(self):  # like Java's toString() method...
        return "Action(%s $%s %s->%s)" % (self.actionName, self.actionCost, self.startState, self.resultingState);

    def __repr__(self):  # also like Java's toString() method, but used instead of __str__ in some contexts
        return str(self)


########### End Action CLASS #############

########### Begin SearchNode CLASS #############

# NOTE: You should not need to change this class at all, but try to understand it.
@total_ordering
class SearchNode:
    """Each search node contains the current state (i.e. word), 
       the complete list of Actions taken to reach this search state,
       the cumulative action cost ("gone" cost) to reach this state,
       a heuristic cost (estimate of distance to the goal)
       and a priority function that computes a priority value that can be used 
       for choosing which fringe node to expand."""

    def __init__(self, currentState, actionHistory, goneCost, heuristicCost, priorityFunction):
        self.currentState = currentState
        self.actionHistory = actionHistory
        self.goneCost = goneCost
        self.heuristicCost = heuristicCost
        self.priorityFunction = priorityFunction

    def getCurrentState(self):
        return self.currentState

    def getActionHistory(self):
        """returns a list of all the actions taken to reach this state"""
        return self.actionHistory

    def getGoneCost(self):
        """ returns the cumulative sum of all the action costs taken to reach this state"""
        return self.goneCost

    def getPriority(self):
        return self.priorityFunction(self.goneCost, self.heuristicCost)

    def getStatesAlongSearchPath(self):
        """returns just a list of the world states that the search process went through,
           without the description of the actions taken."""
        path = []
        for action in self.actionHistory:
            path.append(action.getStartState())
        path.append(self.currentState)
        return path

    def __eq__(self, other):  # overload the == operator
        """ Compare search nodes based on their priorities..."""
        return isinstance(other, SearchNode) and self.getPriority() == other.getPriority()

    def __lt__(self, other):  # overload the < operator
        """ Just compare search nodes in terms of their total action costs..."""
        return self.getPriority() < other.getPriority()

    def __str__(self):  # like Java's toString() method...
        return "SearchNode(%s, g=%s, h=%s, %s)" % (
        self.currentState, self.goneCost, self.heuristicCost, self.actionHistory)

    def __repr__(self):  # also like Java's toString() method...
        return str(self)


########### End SearchNode CLASS #############

########### Begin search priority functions #############

def zeroPriority(goneCost, heuristicCost):
    """ all fringe nodes have identical 0 priority -- for use with non-priority queue/stack. """
    return 0


def ucsPriority(goneCost, heuristicCost):
    """ priority function to use for UCS"""
    return goneCost


def greedyPriority(goneCost, heuristicCost):
    """ priority function to use for greedy best-first search"""
    return heuristicCost


def aStarPriority(goneCost, heuristicCost):
    """ priority function to use for A* search"""
    return goneCost + heuristicCost


########### End search priority functions #############

########### Begin heuristic functions #############

def zeroHeuristic(fromWord, goalWord):
    return 0


def wrongLocationsHeuristic(fromWord, goalWord):
    """simply returns the number of letter slots in fromWord and goalWord that don't match."""
    wrongCount = 0
    wrongCount = 0
    for i in range(max(len(fromWord), len(goalWord))):
        try:
            if fromWord[i] != goalWord[i]:
                wrongCount += 1
        except IndexError as ex:
            wrongCount += 1
    return wrongCount


def betterHeuristic(fromWord, goalWord):
    #The goal of our invented betterHeuristic method is to return an
    #estimated cost to reach the goalWord from the current state of the word if we use DELETE, SWAP, INSERT OR REPLACE actions.
    tempWord = fromWord
    totalCost = 0
    # uncoment for debug
    # print("initial state: ", tempWord)
    swapCount = 0
    # adds up cost for swaping 
    for i in range(max(len(fromWord), len(goalWord))):
        try: 
            if tempWord[i] != goalWord[i] and (tempWord[i] in goalWord[i:]):
                tempWord = swap(tempWord, i, goalWord.index(fromWord[i]))
                swapCount += 1
        except IndexError as ex:
            swapCount += 1
    # uncoment for debug
    # if swapCount > 0:
    #     print("after swapping: ", tempWord)
    # adds up cost for replacing letter
    replaceCount = 0
    for i in range(max(len(fromWord), len(goalWord))):
        try: 
            if tempWord[i] != goalWord[i]:
                tempWord = changeLetter(tempWord, i, goalWord[i])
                replaceCount += 1
        except IndexError as ex:
            replaceCount += 1
    # uncoment for debug
    # if replaceCount > 0:
    #     print("after replacing: ", tempWord)
    # adds up cost for adding
    if(len(tempWord) < len(goalWord)):
        totalCost = abs(len(tempWord)-len(goalWord)) * 100
    # adds up cost for deleting 
    if(len(tempWord) > len(goalWord)):
        totalCost += abs(len(tempWord)-len(goalWord)) * 100
    return swapCount + totalCost + (replaceCount * 10)


########### End heuristic functions #############

########### Begin successor function and helper functions for it  #############

def swap(word, index1, index2):
    """ returns a new word where the letters at the two indices have been swapped.
    
      NOTE: to work properly, index1 must be less than index2"
      e.g. swap("ABCDE",2,4)  will return "ABEDC"    """
    if (index1 > index2):
        raise Exception("index1 must be less than index2 in swap method")
    elif (index1 == index2):
        return word
    else:
        return word[:index1] + word[index2] + word[index1 + 1:index2] + word[index1] + word[index2 + 1:]


def changeLetter(word, index, newLetter):
    """returns a new word which is the result of replacing the letter in word at the specified index with newLetter
    
    e.g. changeLetter("YIK", 1, 'A')  will return "YAK"  """
    return word[:index] + newLetter + word[index + 1:]


def insertLetter(word, index, newLetter):
    """returns a new word which is the result of inserting the new letter in word BEFORE the specified index

    index should be between 0 and len(word) inclusive.
    e.g. insertLetter("ABC", 2, 'K')  will return "ABKC"  """
    return word[:index] + newLetter + word[index:]


def deleteLetter(word, index):
    """returns a new word that results from deleting the letter at  the specified index
    
    e.g. deleteLetter("ABC", 1)  will return "AC"  """
    word = word[0:index] + "" + word[index + 1:len(word)]
    return word


def successorActions(word):
    """ returns a list of legal actions that can be taken from this word """
    sucList = []
    # consider swapping each pair of letters in the word (action cost 1)
    for i in range(len(word) - 1):
        for j in range(i + 1, len(word)):
            resultWord = swap(word, i, j)
            if resultWord != word and resultWord in LEGAL_WORD_SET:
                # Note: the following line creates a new Action object (Python doesn't use the "new" keyword).
                action = Action('SWAP %d,%d' % (i, j), 1, word, resultWord)
                sucList.append(action)
    # consider changing each letter to another letter of the alphabet (action cost 10)
    for i in range(len(word)):
        for newLetter in ALPHABET:
            resultWord = changeLetter(word, i, newLetter)
            if resultWord != word and resultWord in LEGAL_WORD_SET:
                sucList.append(Action('CHG %d %s->%s' % (i, word[i], newLetter), 10, word, resultWord))
    # consider inserting a new letter at each position (action cost 100)
    for i in range(len(word) + 1):
        for newLetter in ALPHABET:
            resultWord = insertLetter(word, i, newLetter)
            if resultWord in LEGAL_WORD_SET:
                sucList.append(Action('INS %d %s' % (i, newLetter), 100, word, resultWord))
    # consider removing each letter in the word (action cost 100)
    for i in range(len(word)):
        resultWord = deleteLetter(word, i)
        if resultWord in LEGAL_WORD_SET:
            sucList.append(Action('DEL %d' % (i), 100, word, resultWord))
    return sucList


########### End successor function #############

# NOTE: You should not need to change this function at all, but it is the HEART
#        of all the search algorithms, so you SHOULD understand it.
def genericGraphSearch(startWord, goalWord, fringe, priorityFunction, heuristicFunction=zeroHeuristic, searchDepthLimit=100):
    """ Runs a general state-space GRAPH search algorithm. 
        Returns a tuple (solution, numNodesCreated, maxMemoryUsedEstimate)
           If no solution was found,  the returned solution will be None
    
     Function parameters:
        startWord - start state for the search
        goalWord - goal state for the search
        fringe - empty fringe data structure, 
                (FIFO queue, LIFO queue, or priority queue, depending on what search algorithm you want.)
        priorityFunction - a function that takes in the backward cost (g) from the start 
                             and the forward heuristic cost (h) 
                             and computes a priority value (only gets used if passed in a priorityQueue!)
        heuristicFunction - a function that takes in a currentState and a goalState and estimates the cost to reach the goal.
        searchDepthLimit - give up on plans that require more than this number of actions
       """
    closedSet = set([]) # keep track of all previously expanded states, so we don't try them again.
    heuristicCostStart = heuristicFunction(startWord, goalWord)
    searchRoot = SearchNode(startWord,[],0, heuristicCostStart, priorityFunction) #create the first node in the search tree
    fringe.put(searchRoot)
    
    # the next 3 lines are not part of the search process, just used to help measure algorithm efficiency.
    numNodesCreated = 1 # as a way to estimate for computational effort / time-efficiency
    roughMemoryCounter = 1 # as a way to (roughly!) estimate the current amount of memory used
    maxRoughMemoryCounter = 0 # tracks the max amount of (roughly estimated) memory used at one time
    
    while (not fringe.empty()):
        curNode = fringe.get()
        roughMemoryCounter -= len(curNode.getActionHistory())
        curWord = curNode.getCurrentState()
        if (curWord == goalWord):
            return curNode, numNodesCreated, maxRoughMemoryCounter
        elif (curWord not in closedSet) and (searchDepthLimit == 0 or len(curNode.getActionHistory()) < searchDepthLimit):        
            closedSet.add(curWord)
            actionList = successorActions(curWord)
            for action in actionList:
                newWord = action.getResultingState()
                if (newWord not in closedSet):  # we can save some memory by not adding things to the fringe if they're already in the closed set
                    newActionHistory = curNode.getActionHistory() + [action] #create new list with action added on the end
                    newGoneCost = curNode.getGoneCost() + action.getActionCost()
                    newHeuristicCost = heuristicFunction(newWord,goalWord)
                    childNode = SearchNode(newWord, newActionHistory, newGoneCost, newHeuristicCost, priorityFunction)
                    roughMemoryCounter += len(newActionHistory)
                    numNodesCreated += 1
                    fringe.put(childNode)
            maxRoughMemoryCounter = max(roughMemoryCounter,maxRoughMemoryCounter) #update max memory statistic if needed
    return (None, numNodesCreated, maxRoughMemoryCounter)



def dfs(startWord, goalWord, searchDepthLimit=100):
    return genericGraphSearch(startWord, goalWord, LifoQueue(), zeroPriority, searchDepthLimit=searchDepthLimit)


def bfs(startWord, goalWord):
    return genericGraphSearch(startWord, goalWord, Queue(), zeroPriority)# TODO: Fix this line to call genericGraphSearch with the right parameters (use a Queue object for the fringe!)


def ucs(startWord, goalWord):
    return genericGraphSearch(startWord, goalWord, Queue(), ucsPriority)  # TODO: Fix this line to call genericGraphSearch with the right parameters


def greedy(startWord, goalWord, ):
    # TODO: Note, if you want to test greedy with your own heuristic function,
    # you'll need to change the value of the heuristicFunction keyword parameter in the call below!
    return genericGraphSearch(startWord, goalWord, PriorityQueue(), greedyPriority,
                              heuristicFunction=wrongLocationsHeuristic)


def aStar(startWord, goalWord):
    return genericGraphSearch(startWord, goalWord, PriorityQueue(), aStarPriority,
                              heuristicFunction=betterHeuristic)

def iterativeDeepening(startWord, goalWord, maxDepthLimit=100):
    totalSteps = 0
    overallMaxMemory = 0
    ## TODO: Use a FOR loop to repeatedly call depth-limited DFS until a solution is found, or we hit maxDepthLimit
    ##   totalSteps calculate the SUM of all the steps each DFS search took in every round
    ##   overallMaxMemory should end up as the MAX of all the memory that each DFS search took
    ##   as soon as a solution is found, return it (along with the steps & memory efficiency data).
    for i in range(maxDepthLimit):
        node, numNodesCreated, maxMemory  = dfs(startWord, goalWord, maxDepthLimit)
        totalSteps += numNodesCreated
        overallMaxMemory += maxMemory
        if (node != None):
            return node, totalSteps, overallMaxMemory
    # if the iteration down to maxDepthLimit failed, we return None (along with the efficiency info).
    return (None, totalSteps, overallMaxMemory)


def runSearch(start, goal, searchAlgFunction):
    """ runs the specified search algorithm from start to goal
       and reports the path it found, the overall cost,
       and some information/estimates about time & memory efficiency."""
    import time
    startTime = time.clock()
    solution, numSteps, maxMemory = searchAlgFunction(start, goal)
    endTime = time.clock()

    print("\nSearch method " + searchAlgFunction.__name__)
    # print(solution)  # useful for debugging -- maybe too much information though?
    if (solution != None):
        print(solution.getStatesAlongSearchPath())
        print("Total cost: ", solution.getGoneCost())
    else:
        print(None)
    print("# of nodes searched: ", numSteps)
    print("Elapsed time: %.3f sec" % (endTime - startTime))
    print("Rough max memory: ", maxMemory, " Action objects")


def main():
    # a few temporary commands just for debugging/testing
    # print(deleteLetter("ABCDE", 2))
    # should print ABDE

    #print(successorActions("CART"))
    # if you're using "super_small_dict.txt", this should print: 
    # [Action(CHG 3 T->D $10 CART->CARD), Action(DEL 2 $100 CART->CAT), Action(DEL 3 $100 CART->CAR)]

    # Note: since Python allows "functional programming", we can pass
    #       a function (like dfs) as a parameter to another function!
    #  (essentially, functions ARE just another type of object in Python!)
    #runSearch("karim", "CART", dfs)
    # node, overallSteps, maxMemory = iterativeDeepening("cat", "dog")
    # print('NODE', node)
    # print('OVERALLSTEPS', overallSteps)
    # print ('MAXMEMORY' ,  maxMemory)

    # We can also loop through a LIST of FUNCTION objects --
    # which is a convenient way to test them all![dfs, bfs, ucs, greedy, aStar, iterativeDeepening]:
    # betterHeuristic("s", "sami")
    for searchAlg in [dfs, bfs, ucs, greedy, aStar, iterativeDeepening]:
        runSearch("SNAKE", "BIRDS", searchAlg)
    #     # runSearch("HUMAN", "ROBOT", searchAlg)
    #     # runSearch("STONEDAHL", "ROBOT", searchAlg)
    #     # runSearch("ROBOT", "STONEDAHL", searchAlg)
    #     print('--------------------------------------------------------------------------------------------')


if __name__ == '__main__':
    main()
