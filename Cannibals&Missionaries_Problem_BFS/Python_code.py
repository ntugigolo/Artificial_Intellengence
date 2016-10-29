
# coding: utf-8

# In[34]:

import math
class State():
    def __init__(self,cannibalLeft, missionaryLeft, boat, cannibalRight, missionaryRight):
        self.canL = cannibalLeft
        self.mL = missionaryLeft
        self.boat = boat
        self.canR = cannibalRight
        self.mR = missionaryRight
        self.parent = None
    def isvalid(self):
        if self.canL >= 0 and self.canR >= 0         and self.mL >= 0 and self.mR >= 0        and (self.mL==0 or self.mL >= self.canL)        and (self.mR==0 or self.mR >= self.canR):
            return True
        else: return False
        
    def isGoal(self):
        if self.mL == 0 and self.canL == 0: return True
        else: return False
        
def successor(cur_state):
    children = []
    def helper(new_state,children,cur_state):
        if new_state.isvalid():
            new_state.parent = cur_state
            children.append(new_state)
        
    if cur_state.boat == "left":
        new_state = State(cur_state.canL,cur_state.mL-2,'right',cur_state.canR,cur_state.mR+2)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL-2,cur_state.mL,'right',cur_state.canR+2,cur_state.mR)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL-1,cur_state.mL-1,'right',cur_state.canR+1,cur_state.mR+1)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL,cur_state.mL-1,'right',cur_state.canR,cur_state.mR+1)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL-1,cur_state.mL,'right',cur_state.canR+1,cur_state.mR)
        helper(new_state,children,cur_state)
    else:
        new_state = State(cur_state.canL,cur_state.mL+2,'left',cur_state.canR,cur_state.mR-2)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL+2,cur_state.mL,'left',cur_state.canR-2,cur_state.mR)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL+1,cur_state.mL+1,'left',cur_state.canR-1,cur_state.mR-1)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL,cur_state.mL+1,'left',cur_state.canR,cur_state.mR-1)
        helper(new_state,children,cur_state)
        new_state = State(cur_state.canL+1,cur_state.mL,'left',cur_state.canR-1,cur_state.mR)
        helper(new_state,children,cur_state)
    return children
def storeprocedure(state):
    path = []
    path.append(state)
    parent = state.parent
    while parent:
        path.append(parent)
        parent = parent.parent
    f = open('Output.txt','w')
    for i in range(len(path)):
        tmp = path[len(path)-i-1]
        f.write("("+"cannibalLeft:"+ str(tmp.canL) +', '+'missionaryLeft:' + str(tmp.mL) + ', '+'boat:'+tmp.boat+', '    + "cannibalRightt:"+ str(tmp.canR) +', '+'missionaryRight:' + str(tmp.mR) + ')')
    f.close()
        
def breadth_first_search():
    try:
        f = open("Input.txt",'r')
        j = list()
        for i in f:
            j = i.split(',')
        initial = State(int(j[0]),int(j[1]),j[2],int(j[3]),int(j[4]))  
        f.close()
    except IOError:
        initial = State(3,3,'left',0,0)
    if initial.isGoal(): 
        return initial
    frontier = list()
    explored = set()
    frontier.append(initial)
    while frontier:
        state = frontier.pop(0)
        if state.isGoal():
            return state
        explored.add(state)
        children = successor(state)
        for child in children:
            if (child not in explored) or (child not in frontier):
                frontier.append(child)
    return None
if __name__ == "__main__":
    solution = breadth_first_search()
    storeprocedure(solution)

