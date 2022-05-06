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

# Problem: Visit each city once and return to hometown with a minimum cost
# In searching for solutions, we build a tree. The leaves of the tree correspond to tours and the nodes represent partial tours 
# Each node of the tree has an associated cost, ie, the cost of the partial tour. We use this to eliminate parts of the tree 
# if we find a partial tour or node of teh tree that couldn't lead to a less expensive final tour, we don't bother searching there 


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

# NOTE: Have Process 0 broadcas the adjacency matrix to all the processes 
if my_rank == 0:
    snd_buf = np.array(graph, dtype=np.intc)
    sent_adjacency = comm_world.bcast(snd_buf)


# Now the variables used for the MPI process 
n = 10
my_stack = []
# init to be higher than any possible value 
best_tour = 1000000
# starting from the initial vertiex 
hometown = graph[0]
initial_tour = [hometown]


# functions for terminated 
def my_avail_tour_count():
    return

def fullfill_request(my_stack):
    return


def send_rejects():
    return

def out_of_work():
    return 

def clear_msg():
    return

def no_work_left():
    return

def isEmpty(my_stack):
    return len(my_stack) == 0

# terminated function for dynamically partitioned solver with MPI 
def terminated(my_stack):
    if my_avail_tour_count(my_stack) >= 2:
        fullfill_request(my_stack)
        return False # still more work to do
    # at most one available tour
    else: 
        send_rejects() #tell everyone that requested that I have none 
        # there is still more work to do
        if not isEmpty(my_stack):
            return False
        else:
            if size == 1:
                return True

            out_of_work()
            work_request_sent = False 

            while(True):
                clear_msg()
                
                # no more work left - quit 
                if(no_work_left()):
                    return True


# we represent partial tours as stack records 

# 4. Push_copy makes the function create a copy of the tour before actually pushing it onto the stack 
def push_copy(stack, tour):
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
def partition_tree(my_rank, my_stack, comm_size):
    # Process 0 will generate a list of comm_size partial tours .
    # Memory wont be shared, it will send the initial partial tours to the ap
    # So it will send many tours using scatter to each process  
    # process 0 will need to send the initial tours to the appropriate process 
    partial_tours = []
    split_sizes = []
    split =[]
    if my_rank == 0:
        # first to create the different partial tours 
        for i in range(1, n):
            partial_tours.append([0, i])

        snd_buf = np.array(partial_tours)

        # now to equally divide the number of processes 
        # using reference: https://www.kth.se/blogs/pdc/2019/11/parallel-programming-in-python-mpi4py-part-2/

        # ave, res = divmod(snd_buf.size/(n-1), comm_size)
        # print(f"ave {ave} res {res}")
        # count = [ave + 1 if p < res else ave for p in range(comm_size)]
        # count = np.array(count)
        split = np.array_split(partial_tours, comm_size, axis = 0)
        print(f"This is split {split}")
        
        split_sizes = []

        for i in range(0,len(split),1):
            split_sizes = np.append(split_sizes, len(split[i]))

        # displacement: the starting index of each sub-task
        # displ = [sum(count[:p]) for p in range(comm_size)]
        # displ = np.array(displ)
        split_sizes_input = split_sizes * (n-1)
        displ = np.insert(np.cumsum(split_sizes_input), 0, 0)[0:-1]

        split_sizes_output = split_sizes*512
        displacements_output = np.insert(np.cumsum(split_sizes_output),0,0)[0:-1]
        
    else:
        snd_buf = None
        # initialize count on worker processes
        count = np.zeros(comm_size, dtype=np.intc)
        displ = None

        split_sizes_input = None
        displacements_input = None
        split_sizes_output = None
        displacements_output = None

    # on all processes we have the initial rcv_buf 
    # recv_buf = np.zeros(count[my_rank])
    recv_buf = np.empty(len(split_sizes))

    comm_world.Scatterv([snd_buf, len(split), displ, MPI.INT], recv_buf, root = 0)
    

    print(f"After scatter this is the partial tour {recv_buf} with process {my_rank}")


def best_tour(tour):
    return

# update the best tour
def update_best_tour(tour):
    return

# is the next tour feasible 
def feasible(curr_tour, city):
    return True

def add_city(curr_tour, city):
    return

def remove_last_city(curr_tour):
    curr_tour = curr_tour[:-1]

def free_tour(curr_tour):
    return





# main loop 
partition_tree(my_rank, my_stack, comm_size)

while not isEmpty(my_stack):
    curr_tour = my_stack.pop()
    
    if city_count(curr_tour) == n:
        if best_tour(curr_tour):
            update_best_tour(curr_tour)
    else:
        for city in range(n-1, 1, -1):
            add_city(curr_tour, city)
            push_copy(my_stack, curr_tour)
            remove_last_city(curr_tour)

    free_tour(curr_tour)


