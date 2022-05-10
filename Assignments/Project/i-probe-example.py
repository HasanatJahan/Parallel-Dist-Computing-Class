#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import random
import time

comm_world = MPI.COMM_WORLD
rank = comm_world.Get_rank()
buffer = 0.0
comm_size = comm_world.Get_size()


# Dada's Example Code 
#################################################
# if rank == 0:
#    data = {'a': 7, 'b': 3.14}
#    time.sleep(3)
#    comm.send(data, dest=1, tag=11)
# elif rank == 1:
#    while not comm.Iprobe(source=0, tag=11):
#         print('rank 1 Doing some work...')
#         time.sleep(1)
#    rdata = comm.recv(source=0, tag=11)
#    print('rank 1: got ', rdata)
##################################################

global_best_tour = 10000
snd_buf = np.array(global_best_tour, dtype=np.intc)


global_best_tour = random.randrange(100)

for dest in range(0, comm_size):
     if dest != rank:
          # use a synchronous send or a non-blocking send 
          comm_world.isend(snd_buf, dest, tag=11)


comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)
while comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11):
     received_cost = comm_world.recv(source=MPI.ANY_SOURCE, tag=11)
     if received_cost < global_best_tour:
          global_best_tour = received_cost
     print(f"Global best tour {global_best_tour}")
     comm_world.Iprobe(MPI.ANY_SOURCE, tag = 11)



