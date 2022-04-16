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


def calculate_trap_area(left_endpoint, right_endpoint, trap_count, base_len):
    estimate = 0
    x = 0
    i = 0
    
    func = lambda a : a * a

    estimate = (func(left_endpoint) +  func(right_endpoint)) / 2.0
    for i in range(1, int(trap_count)):
        x = left_endpoint + i * base_len
        estimate += func(x)
    
    estimate = estimate * base_len
    return estimate

# starting and ending range of the whole function 
a = 0
b = 1

n = int(sys.argv[1])

h = (b-a) / n
local_n = n / size

local_a = a + my_rank * local_n * h 
local_b = local_a + local_n * h 
local_sum = calculate_trap_area(local_a, local_b, local_n, h)

# init the buffer for send and receive
snd_buf = np.array(local_sum, dtype=np.float_)
rcv_buf = np.empty((), dtype=np.float_) # uninitialized 0 dimensional float array

if my_rank != 0:
    comm_world.Send((snd_buf, 1, MPI.DOUBLE), 0, tag=0)

else:
    total_sum = local_sum
    for i in range(1, size):
        request = comm_world.Irecv((rcv_buf, 1, MPI.DOUBLE), source=i, tag=0)
        status = MPI.Status()
        request.Wait(status)
        np.copyto(snd_buf, rcv_buf)
        total_sum += rcv_buf


if my_rank == 0:
    print(f"With n = {n} trapezoids, out estimate of the integral of x^2 from 0 to 1 is {total_sum}")

				



