# Name: Hasanat Jahan


#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import math
import sys
from collections import deque

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
comm_size = comm_world.Get_size()
global_best_tour = 1000000
global_best_path = []
best_results = {}


# Problem: Visit each city once and return to hometown with a minimum cost
# In searching for solutions, we build a tree. The leaves of the tree correspond to tours and the nodes represent partial tours 

# graph representation as an adjacency matrix 
graph = [[0, 1, 5, 1, 1, 1, 1, 1, 1, 1], 
        [2, 0, 2, 1, 1, 1, 1, 1, 1, 1],
        [1, 4, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 4, 5, 0, 1, 1, 1, 1, 1, 1],
        [1, 4, 5, 1, 0, 2, 1, 1, 1, 1],
        [1, 4, 5, 1, 1, 0, 1, 1, 1, 1],
        [2, 4, 5, 1, 1, 1, 0, 1, 1, 1],
        [1, 4, 5, 1, 1, 1, 1, 0, 2, 4],
        [1, 4, 1, 1, 1, 1, 4, 1, 0, 1],
        [1, 4, 5, 1, 1, 1, 1, 1, 1, 0]]


# Now the variables used for the MPI process 
n = 10
my_stack = []
free_tour_dict = {}


def isEmpty(my_stack):
    return len(my_stack) == 0


# 4. Push_copy makes the function create a copy of the tour before actually pushing it onto the stack 
def push_copy(stack, tour):
    free_tour_dict[str(tour)] = 'visited'
    stack.append(tour)

# 1. City_count examines the partial tour to see if there are n cities in the partial tour 
def city_count(curr_tour):
    # assuming that the current tour is an array 
    return len(curr_tour)



# We build a parallel algorithm based on the second iterative solution 
def get_partial_tour():
    partial_tours = []
    for i in range(1, n):
        partial_tours.append(float(i))

    return np.array(partial_tours)


def best_tour(tour):
    global global_best_tour
    cost = 0
    # first we calculate the cost of the tour 
    
    for i in range(len(tour) - 1):
        cost += sent_adjacency[tour[i]][tour[i+1]]
    
    cost += sent_adjacency[tour[len(tour) -1]][0] 
    # is it the smallest tour that it has found so far 
    if cost < global_best_tour:
        global_best_tour = cost 
        return True


# update the best tour
def update_best_tour(tour):
    global global_best_tour
    global global_best_path
    global best_results

    tour_cost = (tour, global_best_tour)
    snd_buf = np.asarray(tour_cost, dtype=object)

    for dest in range(0, comm_size):
        if dest != my_rank:
            # use a synchronous send or a non-blocking send 
            comm_world.isend(snd_buf, dest, tag=11)
    
    comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)
    while comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11):
        received_payload = comm_world.recv(source=MPI.ANY_SOURCE, tag=11)
        received_tour = received_payload[0]
        received_cost = received_payload[1]     
        if received_cost <= global_best_tour:
            global_best_tour = received_cost
            global_best_path = received_tour
        comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)

    rcv_buf = np.empty((), dtype=np.intc)
    new_snd_buf = np.array(global_best_tour, dtype=np.intc)

    best_results.update({global_best_tour : global_best_path})


# is the next tour feasible 
# it checks to see if the city or vertex has already been visited 
# if not, whether it can possibly lead to a least-cost tour 
def feasible(curr_tour, city, visited):
    # feasible should also check if 
    for i in range(len(curr_tour)-1):
        if sent_adjacency[curr_tour[i]][curr_tour[i+1]] == 0:
            return False        
    
    if sent_adjacency[curr_tour[len(curr_tour) -1]][0] == 0:
        return False 

    # if city not in visited:
    if city not in curr_tour:
        return True

# take the current tour and append the city at the end
def add_city(curr_tour, city):
    curr_tour.append(city)

def remove_last_city(curr_tour):
    curr_tour = curr_tour[:-1]

# we can mitigate push copy costs by saving our free tours in a data structure 
def free_tour(curr_tour):
    free_tour_dict[str(curr_tour)] = "visited"


#########################################################
# Partition the Tree 

nprocs = comm_world.Get_size()

if my_rank == 0:
    sendbuf = get_partial_tour()

    # count: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)

    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(nprocs)]
    displ = np.array(displ)
else:
    sendbuf = None
    # initialize count on worker processes
    count = np.zeros(nprocs, dtype=int)
    displ = None

# broadcast count
comm_world.Bcast(count, root=0)

# initialize recvbuf on all processes
recvbuf = np.zeros(count[my_rank])

comm_world.Scatterv([sendbuf, count, displ, MPI.DOUBLE], recvbuf, root=0)

########################################################
# tree partition ends here 

# main loop 
snd_buf_graph = np.array(graph, dtype=int)
sent_adjacency = comm_world.bcast(snd_buf_graph, root = 0)

# Now to create the partial tours and push it onto my_stack of my_rank
visited = set() 

for i in range(len(recvbuf)):
    my_stack.append([0, int(recvbuf[i])])


while not isEmpty(my_stack):
    curr_tour = my_stack.pop()
    if city_count(curr_tour) == n:
        if best_tour(curr_tour):
            update_best_tour(curr_tour)
    else:
        # visited = set() 
        for city in range(n-1, 0, -1):
            if feasible(curr_tour, city, visited):
                add_city(curr_tour, city)
                push_copy(my_stack, curr_tour)
                remove_last_city(curr_tour)

                visited.add(city)

    free_tour(curr_tour)


if my_rank == 0:
    # iterate through the keys and find the best result
    min_cost = 20000
    for cost in best_results:
        if cost < min_cost:
            min_cost = cost 

    print(f"The best tour is {best_results[cost]} with the cost of {cost}")




