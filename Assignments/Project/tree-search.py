# Name: Hasanat Jahan

#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np
import math
import sys

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()


# Create a graph representation 
# Directed graph representation with 10 vertices to represent 10 cities 
# copied from: https://www.bogotobogo.com/python/python_graph_data_structures.php
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()  

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

# Now to create the graph with 10 vertices 
g = Graph()

g.add_vertex('a')
g.add_vertex('b')
g.add_vertex('c')
g.add_vertex('d')
g.add_vertex('e')
g.add_vertex('f')
g.add_vertex('g')
g.add_vertex('h')
g.add_vertex('i')
g.add_vertex('j')

# Replace the weights with random numbers for the final run 
g.add_edge('a', 'b', 7)  
g.add_edge('a', 'c', 9)
g.add_edge('a', 'f', 14)
g.add_edge('b', 'c', 10)
g.add_edge('b', 'd', 15)
g.add_edge('c', 'd', 11)
g.add_edge('c', 'f', 2)
g.add_edge('d', 'e', 6)
g.add_edge('e', 'f', 9)

for v in g:
    for w in v.get_connections():
        vid = v.get_id()
        wid = w.get_id()
        print( vid, wid, v.get_weight(w))

for v in g:
    print(v.get_id(), g.vert_dict[v.get_id()])


# Now the variables used for the MPI process 
n = 10
my_stack = []
# init to be higher than any possible value 
best_tour = 1000000
# starting from the initial vertiex 
hometown = g.vert_dict["a"]
initial_tour = [hometown]
print(initial_tour)


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

# terminated function for dynamically partitioned solver with MPI 
def terminated(my_stack):
    if my_avail_tour_count(my_stack) >= 2:
        fullfill_request(my_stack)
        return False # still more work to do
    # at most one available tour
    else: 
        send_rejects() #tell everyone that requested that I have none 
        # there is still more work to do
        if not my_stack.empty():
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

# NOTE: Tehere should be a main while loop in the search function 

# 1. If the my_rank==0 then we figure out how many processes there should be - not sure about this part 
# It is very similar to the serial implementation of the tree search 


# We would like all the processes to have the copy of the adjacency matrix but since there is a shared graph this does not need to be distributed 
# We build a parallel algorithm based on the second iterative solution 
def partition_tree(my_rank, my_stack):
    return 

# checks if it is the best tour 
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
partition_tree(my_rank, my_stack)

while not my_stack.empty():
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


