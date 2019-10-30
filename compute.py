#    Author   #
#  Jay Bansal #
#  193050004  #

import argparse
from itertools import cycle
import csv
from pulp import *
import numpy as np
import random

def evaluate_value_fn(S,A,R,T,G,P):
    a = []
    b = [None]*S
    for i in range(S):
        c = [None]*S
        for j in range(S):
            c[j] = G*T[i][P[i]][j]
        c[i] = c[i]-1
        a.append(c)
        b[i] = -1*sum([T[i][P[i]][k]*R[i][P[i]][k] for k in range(S)])
    a = np.array(a)
    b = np.array(b)
    V = np.linalg.solve(a, b).tolist()
    return V

def evaluate_action_value_fn(S,A,R,T,G,V):
    Q = []
    for i in range(S):
        a = [None]*A
        for j in range(A):
            a[j] = sum([T[i][j][k]*(R[i][j][k]+G*V[k]) for k in range(S)])
        Q.append(a)
    return Q

def lp_policy_fn(S,A,R,T,G,t):
    allowed_error = 1e-05
    V = []
    var = [None]*S
    prob = LpProblem("Finding optimal value function",LpMinimize)
    for i in range(S):
        var[i] = LpVariable("V("+str(i)+")",None,None)
    prob += lpSum([var[i] for i in range(S)]), "Sum of all values is minimzed"
    for i in range(S):
        for j in range(A):
            prob += var[i] >= lpSum([T[i][j][k]*(R[i][j][k]+G*var[k]) for k in range(S)])
    # prob.writeLP("ValueFunctionLP.lp")
    prob.solve()
    for v in prob.variables():
        V.append(v.varValue)
    P = [None]*S
    for j in range(S):
        for i in range(A):
            if abs(V[j] - sum([T[j][i][k]*(R[j][i][k]+G*V[k]) for k in range(S)]))<=allowed_error:
                P[j] = i
                break
    V = evaluate_value_fn(S,A,R,T,G,P) #updating V to more precise values
    return V,P

def hpi_policy_fn(S,A,R,T,G,t):
    V = [None]*S
    P = [None]*S
    # start with arbitrary policy
    for i in range(S):
        P[i] = random.randrange(A)
    newP = P.copy()
    P = [None]*S
    # optimality condition (No improvable states)
    while P != newP:
        #calculate new Policy (switch to some improving action for all improving actions)
        P = newP.copy()
        newP = [None]*S
        Q = evaluate_action_value_fn(S,A,R,T,G,evaluate_value_fn(S,A,R,T,G,P))
        for i in range(S):
            imp_actions=[]
            for j in range(A):
                if Q[i][j] > Q[i][P[i]]:
                    imp_actions.append(j)
            if imp_actions:
                newP[i] = random.choice(imp_actions)
            else:
                newP[i] = P[i]
    return evaluate_value_fn(S,A,R,T,G,P),P
    
def read_mdp(file):
    R=[]
    T=[]
    new=[]
    with open(file,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            if("" in row):
                row.remove("")
            new.append(row)
        new = cycle(new)
        S = int(next(new)[0])
        A = int(next(new)[0])
        for i in range(S):
            ll=[]
            for j in range(A):
                ll.append([float(i) for i in next(new)])
            R.append(ll)
        for i in range(S):
            ll=[]
            for j in range(A):
                ll.append([float(i) for i in next(new)])
            T.append(ll)
        G = float(next(new)[0])
        t = next(new)[0]
        return S,A,R,T,G,t
    
dictionary_of_fn = {
    'lp':(lp_policy_fn),
    'hpi':(hpi_policy_fn),
}

def main():
    S,A,R,T,G,t = read_mdp(args.mdp)
    # S,A,R,T,G,t = read_mdp('/home/jay/Desktop/workspace/data/continuing/test.txt')
    policy_evaluation = dictionary_of_fn[args.algorithm]
    # policy_evaluation = dictionary_of_fn['hpi']
    V,P = policy_evaluation(S,A,R,T,G,t)
    for i in range(S):
        print(str(V[i])+'\t'+str(P[i]))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', default='/home/jay/Desktop/workspace/data/continuing/MDP2.txt', type=str, help='location of the MDP file')
    parser.add_argument('--algorithm', default='lp', choices=['lp','hpi'], help='Algorithm used')
    args = parser.parse_args()
    main()
