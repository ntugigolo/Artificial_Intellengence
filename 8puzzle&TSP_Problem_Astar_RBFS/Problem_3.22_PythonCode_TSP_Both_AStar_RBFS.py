
# coding: utf-8

# In[2]:

import numpy as np
import operator
import random
from scipy.spatial.distance import pdist, squareform
def minimum_spanning_tree(X, copy_X=True):
    """X are edge weights of fully connected graph"""
    if copy_X:
        X = X.copy()
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    # initialize with node 0:
    visited_vertices = [0]
    num_visited = 1
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None)
        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices)
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
        # add edge to tree
        spanning_edges.append(new_edge)
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf
        num_visited += 1
    return np.vstack(spanning_edges)

def test_mst(opt,num_city):
    P = np.random.uniform(size=(num_city, 2))
    X = squareform(pdist(P))
    edge_list = minimum_spanning_tree(X)
    if opt == 1: return edge_list
    else: return X

class Node:
    def __init__(self, state, f=0, g=0 ,h=0):
        self.state = state
        self.f = f
        self.g = g
        self.h = h
        self.parent = None

def RBFS(startState, actionsF, takeActionF, goalTestF, hF):
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    closeset = list()
    closeset.append(startNode)
    return RBFSHelper(startNode, actionsF, takeActionF, goalTestF, hF, float('inf'),closeset)

def RBFSHelper(parentNode, actionsF, takeActionF, goalTestF, hF, fmax,closeset):
    if goalTestF(closeset):
        path = []
        GG = 0
        for i in closeset:
            path.append(i.state)
            GG += i.g
        #GG += hF(path[len(path)-1])
        return (path,GG)
        #return ([parentNode.state], parentNode.g)
    ## Construct list of children nodes with f, g, and h values
    actions = actionsF(parentNode,closeset,2)
    children = []
    if not actions:
        return ("failure", float('inf'))
    for action in actions:
        stepCost = actions[action]
        h = hF(action)
        g = parentNode.g + stepCost
        f = max(h+g, parentNode.f)
        childNode = Node(state=action, f=f, g=g, h=h)
        children.append(childNode)
    """
    for action in actions:
        (childState,stepCost) = takeActionF(parentNode.state, action)"""
    while True:
        # find best child
        children.sort(key = lambda n: n.f) # sort by f value
        bestChild = children[0]
        if bestChild.f > fmax:
            return ("failure",bestChild.f)
        # next lowest f value
        flag = 0
        for i in closeset:
            if bestChild.state == i.state: flag = 1
        if flag == 0: closeset.append(bestChild)
        alternativef = children[1].f if len(children) > 1 else float('inf')
        # expand best child, reassign its f value to be returned value
        result,bestChild.f = RBFSHelper(bestChild, actionsF, takeActionF, goalTestF, hF, min(fmax,alternativef),closeset,hy)
        if result is not "failure":
            #result.insert(0,parentNode.state)
            return (result, bestChild.f)

def AStar(startState, actionsF, takeActionF, goalTestF, hF):
    h = hF(startState)
    startNode = Node(state=startState, f=0+h, g=0, h=h)
    return aStar(startNode, actionsF, takeActionF, goalTestF, hF)

def aStar(startNode, actionsF, takeActionF, goalTestF, hF):
    openset,closeset = set(),set()
    current = startNode
    openset.add(current)
    while openset:
        current = min(openset,key=lambda o:o.g + o.h)
        if goalTestF(closeset):
            path,cost = [],[]
            while current.parent:
                path.append(current)
                current = current.parent
                cost.append(current.g+current.h)
            path.append(current)
            return (path[::-1],cost[0])
        openset.remove(current)
        closeset.add(current)

        actions = actionsF(current,closeset,1)
        #for action in actions:
        (childState,stepCost) = takeActionF(current.state, actions)
        node = Node(state=childState, f=current.g+stepCost+hF(childState), g=current.g+stepCost, h=hF(childState))
        if node in closeset:
            node.parent = current
            continue
        if node in openset:
            new_g = current.g + stepCost
            if node.g > new_g:
                node.g = new_g
                node.parent = current
        else:
            node.parent = current
            openset.add(node)

if __name__ == "__main__":
    f = open("Input_Pro3.22_TSP.txt",'r')
    for i in f:
        tmp = i.split(',')
    num_city,start = int(tmp[0]),int(tmp[1])
    f.close()
    #tree = test_mst(1,num_city)
    entire = test_mst(2,num_city)
    def eveluate_MST(edge_list):
        sum_ = 0
        for i in edge_list:
            sum_ += entire[i[0]][i[1]]
        return sum_

    def actionsF(s,closeset,opt):
        try:
            tmp = []
            dic = {}
            if len(closeset) == num_city:
                return (start,entire[s.state][start])
            for i in closeset:
                tmp.append(i.state)
            for i in range(num_city):
                if i not in tmp:
                    copy = tmp[:]
                    copy.append(i)
                    copy.remove(start)
                    sub_entire = np.delete(entire,copy,0)
                    sub_entire = np.delete(sub_entire,copy,1)
                    if len(sub_entire) > 1:
                        edge_list = minimum_spanning_tree(sub_entire)
                        mst_l = eveluate_MST(edge_list)
                    else: mst_l = 0
                    dic[i] = entire[s.state][i] + mst_l
            sorted_dic = sorted(dic.items(), key=operator.itemgetter(1))
            if opt == 1: return (sorted_dic[0][0],sorted_dic[0][1])
            else: return dic
            #return [(succ,float(entire[s][succ])) for succ in successors[s]]
        except KeyError:
            return []
    def takeActionF(s,a):
        return a
    def goalTestF(s):
        return len(s) == num_city
    def h1(s):
        return float(entire[s][start])
    f = open("Output_Pro3.22_TSP.txt",'w')
    #f.write("Distance between each city:"+'\n'+str(entire)+'\n')
    #f.write("Tree relation:"+'\n'+str(successors)+'\n')
    res = AStar(start,actionsF,takeActionF,goalTestF,h1)
    path = []
    for i in res[0]:
        path.append(i.state)
    f.write("Path for Astar from start to end is " + str(path) + " for a cost of "+ str(res[1]))
    f.write('\n')
    f.write("Path for RBFS from start to end is")
    result = RBFS(start,actionsF,takeActionF,goalTestF,h1)
    f.write(str(result[0]))
    f.write('\n'+"\n")
    #f.write(str(path))
    f.close()
    #print ("Path from start to end is", result[0], " for a cost of ", result[1])
