
# coding: utf-8

# In[1]:

from collections import defaultdict
import math
import random
import sys
import bisect
from queue import PriorityQueue

infinity = float('inf')

def memoize(fn, slot=None):
    """Memoize fn: make it remember the computed value for any argument list.
    If slot is specified, store result in that slot of first argument.
    If slot is false, store results in a dictionary."""
    if slot:
        def memoized_fn(obj, *args):
            if hasattr(obj, slot):
                return getattr(obj, slot)
            else:
                val = fn(obj, *args)
                setattr(obj, slot, val)
                return val
    else:
        def memoized_fn(*args):
            if args not in memoized_fn.cache:
                memoized_fn.cache[args] = fn(*args)
            return memoized_fn.cache[args]

        memoized_fn.cache = {}

    return memoized_fn

class Queue:

    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(order, f): Queue in sorted order (default min-first).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
        item in q       -- does q contain item?
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        raise NotImplementedError

    def extend(self, items):
        for item in items:
            self.append(item)

class PriorityQueue(Queue):

    """A queue in which the minimum (or maximum) element (as determined by f and
    order) is returned first. If order is min, the item with minimum f(x) is
    returned first; if order is max, then it is the item with maximum f(x).
    Also supports dict-like lookup."""

    def __init__(self, order=min, f=lambda x: x):
        self.A = []
        self.order = order
        self.f = f

    def append(self, item):
        bisect.insort(self.A, (self.f(item), item))

    def __len__(self):
        return len(self.A)

    def pop(self):
        if self.order == min:
            return self.A.pop(0)[1]
        else:
            return self.A.pop()[1]

    def __contains__(self, item):
        return any(item == pair[1] for pair in self.A)

    def __getitem__(self, key):
        for _, item in self.A:
            if item == key:
                return item

    def __delitem__(self, key):
        for i, (value, item) in enumerate(self.A):
            if item == key:
                self.A.pop(i)

class Problem(object):

    def __init__(self, initial, goal=None):
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        move = []
        idx = state.index(0)
        if idx in [0,1,3,4,6,7]:
            move.append("R")
        if idx in [1,2,4,5,7,8]:
            move.append("L")
        if idx in [3,4,5,6,7,8]:
            move.append("U")
        if idx in [0,1,2,3,4,5]:
            move.append("D")
        return move

    def result(self, state, action): 
        if action == "R":
            copy = list(state)
            idx = copy.index(0)
            copy[idx],copy[idx+1] = copy[idx+1],copy[idx]
        elif action == "L":
            copy = list(state)
            idx = copy.index(0)
            copy[idx],copy[idx-1] = copy[idx-1],copy[idx]
        elif action == "U":
            copy = list(state)
            idx = copy.index(0)
            copy[idx],copy[idx-3] = copy[idx-3],copy[idx]
        elif action == "D":
            copy = list(state)
            idx = copy.index(0)
            copy[idx],copy[idx+3] = copy[idx+3],copy[idx]
        return tuple(copy)

    def goal_test(self, state):

        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1
    
    def h(self,node):
        h1 = 0
        state = node.state
        board,tmp = [],[]
        for k in range(9):
            tmp.append(state[k])
            if (k+1)%3 == 0 and k != 0:
                board.append(tmp)
                tmp = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] != 0: h1 += abs(int(board[i][j]/3 - i)) + abs(int(board[i][j]%3 - j))
        return h1

    def value(self, state):
        raise NotImplementedError
        
class Node:

    def __init__(self, state, parent=None, action=None, path_cost=0):
        "Create a search tree Node, derived from a parent by an action."
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node %s>" % (self.state,)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        "List the nodes reachable in one step from this node."
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        next = problem.result(self.state, action)
        return Node(next, self, action,
                    problem.path_cost(self.path_cost, self.state,
                                      action, next))

    def path(self):
        "Return a list of nodes forming the path from the root to this node."
        node, path_back = self, []
        while node:
            path_back.append(node.state)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)

def best_first_graph_search(problem, f):
    f = memoize(f, 'f')
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = PriorityQueue(min, f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def astar_search(problem, h=None):
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n))

def recursive_best_first_search(problem, h=None):
    h = memoize(h or problem.h, 'h')

    def RBFS(problem, node, flimit):
        if problem.goal_test(node.state):
            return node, 0   # (The second value is immaterial)
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, infinity
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            best = successors[0]
            if best.f > flimit:
                return None, best.f
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = infinity
            result, best.f = RBFS(problem, best, min(flimit, alternative))
            if result is not None:
                return result, best.f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf = RBFS(problem, node, infinity)
    return result

def judge_solve(state):
    board,tmp = [],[]
    for k in range(9):
        tmp.append(state[k])
        if (k+1)%3 == 0 and k != 0:
            board.append(tmp)
            tmp = []
    invr = 0
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] != 0: tmp.append(board[i][j])
    for i in range(len(tmp)):
        for j in range(i+1,len(tmp)):
            if tmp[j] < tmp[i]: invr += 1
    if invr%2 == 1: return False
    else: return True

board = tuple()
in_ = open("Input_Pro3.22_8Puzzle.txt",'r')
for i in in_:
    tmp = i.split(',')
    board = tuple(map(int,tmp))
in_.close()

if judge_solve(board):
    out = open("Output_Pro3.22_8Puzzle.txt",'w')
    problem = Problem(board,(0,1,2,3,4,5,6,7,8))
    out.write("Path for A Star:" + '\n')
    res = astar_search(problem).path()
    for i in res:
        #print (str(i))
        out.write(str(i))
        out.write('\n')
    out.write("Path for RBFS:" + '\n')
    res2 = recursive_best_first_search(problem).path()
    for j in res2:
        #print (str(j))
        out.write(str(j))
        out.write('\n')
    #print(str(recursive_best_first_search(problem).path()))
    out.close()
else:
    out = open("Output_Pro3.22_8Puzzle.txt",'w')
    out.write("Unsolvable")
    out.close()

