
# coding: utf-8

# In[2]:

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

def crossover(parent1,parent2):
    new = []
    r1,r2 = random.randint(1,len(parent1)-2),random.randint(1,len(parent1)-2)
    while r1 == r2: r2 = random.randint(1,len(parent1)-2)
    if r1 > r2: r1,r2 = r2,r1
    for i in range(len(parent1)):
        new.append(-1)
    new[0] = parent1[0]
    new[len(new)-1] = parent1[len(parent1)-1]
    new[r1:r2+1] = parent1[r1:r2+1]
    res = parent1[1:r1] + parent1[r2+1:-1]
    order = []
    for j in range(r2+1,len(parent2)-1):
        if parent2[j] in res: order.append(parent2[j])
    for j in range(1,r2+1):
        if parent2[j] in res: order.append(parent2[j])
    for j in range(r2+1,len(parent2)-1):
        new[j] = order[0]
        order.remove(order[0])
    for j in range(1,r1):
        new[j] = order[0]
        order.remove(order[0])
    return new    
        
def mutate(obj):
    r1,r2 = random.randint(1,len(obj)-2),random.randint(1,len(obj)-2)
    while r1 == r2: r2 = random.randint(1,len(obj)-2)
    obj[r1],obj[r2] = obj[r2],obj[r1]
    return obj

def Genetic(population,elitism,init_function,objective_function,crossover,mutate,max_iterations):
    store = []
    for i in range(population):
        best=init_function()
        store.append(best)
    for i in range(max_iterations):
        store.sort(key = lambda x: objective_function(x))
        store1,store2 = [],[]
        for idx in range(int(population*elitism)):
            store1.append(store[idx])
        
        num_crosser = int((population - int(population*elitism))/2)
        for idy in range(num_crosser):
            x1,x2 = random.choice(store),random.choice(store)
            while x1 == x2: x2 = random.choice(store)
            x3,x4 = crossover(x1,x2),crossover(x2,x1)
            while x3 == x4: x4 = crossover(x2,x1)
            store2.append(x3)
            store2.append(x4)
        
        for idx in range(num_crosser):
            ran = random.randint(0,len(store2)-1)
            store2[ran] = mutate(store2[ran])
        store = store1 + store2
    return store[0]
        
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
    max_iterations = 4000
    ##calculate score
    objective_function=lambda tour: tour_length(entire, tour)
    elitism = 0.2
    population = 10
    # Execute hill_climb
    result = Genetic(population,elitism,init_function,objective_function,crossover,mutate,max_iterations)

    f = open("Output_Pro4.3(b).txt",'w')
    #f.write("Distance between each city:"+'\n'+str(entire)+'\n')
    f.write("Path for Genetic Algorithm from start to end is " + str(result))
    f.write("\n")
    res = AStar(start,actionsF,takeActionF,goalTestF,h1)
    path = []
    for i in res[0]:
        path.append(i.state)
    f.write("Path for Astar from start to end is " + str(path) + " for a cost of "+ str(res[1]))
    #f.write(str(path))
    f.close()
    #print ("Path from start to end is", result[0], " for a cost of ", result[1])

