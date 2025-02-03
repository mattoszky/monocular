"""
Write a fitness function (use real code or pseudocode) for a genetic algorithm that optimizes the following problem 
(traveling salesman problem): Given N+1 cities and their coordinates in an (x,y) reference system represented 
by the Nx2 matrix X=[x0 y0; x1 y1; …; xN yN], starting from City 0, find the sequence of cities that minimizes 
the length of the path that one needs to travel to visit all cities (City 1, City 2, …, City N) only once and come back to City 0. 
Suppose the input individual is represented as an N-dimensional vector S = (s1, s2, … sN) containing N different integers 
in the range [1,N], i.e., a permutation of length N
"""

from math import sqrt

def fitness(ind, cities):
    percorso_totale = 0
    for i in range(len(ind) - 1):
        percorso_totale += sqrt((cities[ind[i]][0]-cities[ind[i+1]][0])**2 + (cities[ind[i]][1]-cities[ind[i+1]][1])**2)
        
    #calcolo distanza da città 0 a prima città
    percorso_totale += sqrt((cities[0][0]-cities[ind[0]][0])**2 + (cities[0][1]-cities[ind[0]][1])**2)
    #calcolo distanza da ultima città a città 0
    percorso_totale += sqrt((cities[ind[-1]][0]-cities[0][0])**2 + (cities[ind[-1]][1]-cities[0][1])**2)
    
    return percorso_totale


cities = [(1,0), (2,0), (3,0), (4,0), (5,0), (6,0)]
ind = [1,2,3,5,4]
#calcolo della fitness:
print(fitness(ind, cities))
