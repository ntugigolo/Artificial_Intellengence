
# coding: utf-8

# In[8]:

import numpy as np
import operator
from scipy.spatial.distance import pdist, squareform
import sys
import time
import random
import argparse
from math import sqrt
import urllib.request
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
    P = np.random.uniform(low = 0.0,high = 10,size=(num_city, 2))
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

def init_random_tour(tour_length):
    tour = []
    f = open("Input_Pro4.3.txt",'r')
    for i in f:
        tmp = i.split(',')
    start = int(tmp[1])
    f.close()
    for i in range(0,tour_length):
        if i == start:
            continue
        else:
            tour.append(i)
    random.shuffle(tour)
    return [start]+tour +[start]

def tour_length(entire,tour):
    total = 0
    num_locations = len(tour)
    for i in range(num_locations-1):
        total += entire[tour[i],tour[i+1]]
    return total
def all_pairs(size,shuffle=random.shuffle):
    '''generates all i,j pairs for i,j from 0-size uses shuffle to randomise (if provided)'''
    r1,r2 = [],[]
    for i in range(1,size):
        r1.append(i)
        r2.append(i)
    if shuffle:
        shuffle(r1)
        shuffle(r2)
    for i in r1:
        for j in r2:
            yield (i,j)
def reversed_sections(tour):
    for i,j in all_pairs(len(tour)-1):
        if i != j:
            copy=tour[:]
            if i < j:
                copy[i:j+1]=reversed(tour[i:j+1])
            else:
                copy[j:i+1]=reversed(tour[j:i+1])
                #copy[i+1:]=reversed(tour[:j])
                #copy[:j]=reversed(tour[i+1:])
            if copy != tour: # no point returning the same tour
                yield copy


def hillclimb(init_function,move_operator,objective_function,max_evaluations):
    '''
    hillclimb until either max_evaluations is reached or we are at a local optima
    '''
    best=init_function()
    best_score=objective_function(best)
    
    num_evaluations=1
    
    while num_evaluations < max_evaluations:
        # examine moves around our current position
        move_made=False
        for next in move_operator(best):
            if num_evaluations >= max_evaluations:
                break
            # see if this move is better than the current
            next_score=objective_function(next)
            num_evaluations+=1
            if next_score > best_score:
                best=next
                best_score=next_score
                move_made=True
                break # depth first search   
        if not move_made:
            break # we couldn't find a better move (must be at a local maximum)
    
    return (num_evaluations,best_score,best)

def hillclimb_and_restart(init_function,move_operator,objective_function,max_evaluations):
    '''
    repeatedly hillclimb until max_evaluations is reached
    '''
    best=None
    best_score=0
    
    num_evaluations=0

    while num_evaluations < max_evaluations:
        remaining_evaluations=max_evaluations-num_evaluations
        
        evaluated,score,found=hillclimb(init_function,move_operator,objective_function,remaining_evaluations)
        num_evaluations+=evaluated
        if score > best_score or best is None:
            best_score=score
            best=found
    return (num_evaluations,best_score,best)
        
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
    f = open("Input_Pro4.3.txt",'r')
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
                    sub_entire = np.delete(entire,copy,0)
                    sub_entire = np.delete(sub_entire,copy,1)
                    if len(sub_entire) > 1:
                        edge_list = minimum_spanning_tree(sub_entire)
                        mst_l = eveluate_MST(edge_list)
                    else: mst_l = 0
                    dic[i] = entire[s.state][i]
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

    init_function=lambda: init_random_tour(num_city)
    move_operator = reversed_sections
    max_iterations = 200000
    objective_function=lambda tour: -tour_length(entire, tour)
    # Execute hill_climb
    iterations,score,best=hillclimb_and_restart(init_function,move_operator,objective_function,max_iterations)
    path = []
    for loc in best:
        path.append(loc)
    #path.append(0)
    path[:] = path[::-1]
    f = open("Output_Pro4.3(a).txt",'w')
    #f.write("Distance between each city:"+'\n'+str(entire)+'\n')
    f.write("Path for Hillclimbing from start to end is " + str(path))
    f.write("\n")
    res = AStar(start,actionsF,takeActionF,goalTestF,h1)
    path = []
    for i in res[0]:
        path.append(i.state)
    f.write("Path for Astar from start to end is " + str(path) + " for a cost of "+ str(res[1]))
    #f.write(str(path))
    f.close()
    #print ("Path from start to end is", result[0], " for a cost of ", result[1])

