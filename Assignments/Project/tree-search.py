# Name: Hasanat Jahan
# NOTE: NOW TO WORK ON THE IPROBE TO GET THE BEST TOUR 
# NOTE: ONCE THAT IS DONE AND WE CAN PRINT THE BEST TOUR SUCCESFULLY THEN WORK ON THE TERMINATION FUNCTION


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

# Problem: Visit each city once and return to hometown with a minimum cost
# In searching for solutions, we build a tree. The leaves of the tree correspond to tours and the nodes represent partial tours 
# Each node of the tree has an associated cost, ie, the cost of the partial tour. We use this to eliminate parts of the tree 
# if we find a partial tour or node of the tree that couldn't lead to a less expensive final tour, we don't bother searching there 


# graph representation as an adjacency matrix 
graph = [[0, 4, 5, 1, 1, 1, 1, 1, 1, 1], 
        [1, 0, 5, 1, 1, 1, 1, 1, 1, 1],
        [1, 4, 0, 1, 1, 1, 1, 1, 1, 1],
        [1, 4, 5, 0, 1, 1, 1, 1, 1, 1],
        [1, 4, 5, 1, 0, 1, 1, 1, 1, 1],
        [1, 4, 5, 1, 1, 0, 1, 1, 1, 1],
        [1, 4, 5, 1, 1, 1, 0, 1, 1, 1],
        [1, 4, 5, 1, 1, 1, 1, 0, 1, 1],
        [1, 4, 5, 1, 1, 1, 1, 1, 0, 1],
        [1, 4, 5, 1, 1, 1, 1, 1, 1, 0]]

cities = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]



# Now the variables used for the MPI process 
n = 10
my_stack = []
# init to be higher than any possible value 
global_best_tour = 1000000
# starting from the initial vertiex 
hometown = graph[0]
initial_tour = [hometown]
free_tour_dict = {}


def isEmpty(my_stack):
    return len(my_stack) == 0


# we represent partial tours as stack records 

# 4. Push_copy makes the function create a copy of the tour before actually pushing it onto the stack 
def push_copy(stack, tour):
    free_tour_dict[str(tour)] = 'visited'
    stack.append(tour)

# 1. City_count examines the partial tour to see if there are n cities in the partial tour 
def city_count(curr_tour):
    # assuming that the current tour is an array 
    return len(curr_tour)



# we would represent each tour with an array so we could have something like 0 -> 2 -> 3, 13 
# NOTE: 
# 1. City_count examines the partial tour to see if there are n cities in the partial tour 
# 2. Best_tour: we can check if the current complete tour has a lower cost than the current "best tour" 
# 3. Update_best_tour: we can replace the current best tour by calling this function 
# 4. Push_copy makes the function create a copy of the tour before actually pushing it onto the stack 


# We would like all the processes to have the copy of the adjacency matrix but since there is a shared graph this does not need to be distributed 
# We build a parallel algorithm based on the second iterative solution 
def get_partial_tour():
    partial_tours = []
    for i in range(1, 10):
        partial_tours.append(float(i))

    return np.array(partial_tours)


# global best tour is a local variable 
def best_tour(tour):
    cost = 0
    # first we calculate the cost of the tour 
    # NOTE: it was not checked if cost function is working or not 
    print(f"this is sent adjacency ceroero {sent_adjacency[0]}")
    
    for i in range(len(tour) - 1):
        cost += sent_adjacency[0][tour[i]][tour[i+1]]
    
    cost += sent_adjacency[tour[len(tour) -1]][0] 
    # is it the smallest tour that it has found so far 
    if cost < global_best_tour:
        global_best_tour = cost 
        return True


# update the best tour
def update_best_tour(tour):
    snd_buf = np.array(global_best_tour, dtype=np.intc)

    for dest in range(0, comm_size):
     if dest != rank:
          # use a synchronous send or a non-blocking send 
          comm_world.isend(snd_buf, dest, tag=11)
    
    comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)
    while comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11):
     received_cost = comm_world.recv(source=MPI.ANY_SOURCE, tag=11)
     if received_cost < global_best_tour:
          global_best_tour = received_cost
     comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)

    rcv_buf = np.empty((), dtype=np.intc)
    new_snd_buf = np.array(global_best_tour, dtype=np.intc)
    comm_world.Allreduce(new_snd_buf, rcv_buf, op=MPI.MIN)

    print(f"My rank {rank} and then the min {rcv_buf}")


    return

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
# HERE WE PARTITION THE TREE 
# def partition_tree(my_rank, my_stack, comm_size):
    # Process 0 will generate a list of comm_size partial tours .
    # Memory wont be shared, it will send the initial partial tours to the ap
    # So it will send many tours using scatter to each process  
    # process 0 will need to send the initial tours to the appropriate process 
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

# print('After Scatterv, process {} has data:'.format(my_rank), recvbuf)
########################################################
# tree partition ends here 

# main loop 
snd_buf_graph = np.array(graph, dtype=int)
sent_adjacency = comm_world.bcast(snd_buf_graph, root = 0)

# Now to create the partial tours and push it onto my_stack of my_rank
visited = set() 

for i in range(len(recvbuf)):
    my_stack.append([0, int(recvbuf[i])])
    # visited.add(int(recvbuf[i]))


while not isEmpty(my_stack):
    curr_tour = my_stack.pop()
    print(f"current tour {curr_tour}")
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

# at the end we reduce the value of the best tour and print it from process 0 
