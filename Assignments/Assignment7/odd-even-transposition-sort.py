#!/usr/bin/env python3
from mpi4py import MPI
import numpy as np

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

# we have to take user input for a 16 values
input_arr = [15, 11, 9, 16, 3, 14, 8, 7, 4, 6, 12, 10, 5, 2, 13, 1]

def compute_partner(phase, my_rank):
    partner = 0
    # even phase 
    if phase % 2 == 0: 
        if my_rank % 2 != 0:
            partner = my_rank - 1
        else:
            partner = my_rank + 1
    else:
        if my_rank % 2 != 0:
            partner = my_rank + 1
        else:
            partner = my_rank - 1        
    
    # NOTE: MPI.PROC_NULL DOING SOMETHING WONKY AND THIS FOR LOOP DOES NOT ALL CAPTURE NEGATIVES 
    if partner < 0 or partner >= size:
        partner = MPI.PROC_NULL

    return partner

# first we sort local keys 
local_n = len(input_arr) / size
starting_index = int(my_rank * local_n)
ending_index = int(starting_index + local_n) # taking into account the slicing 
local_arr = np.array(input_arr[starting_index:ending_index])
# sort the local keys 
local_arr.sort()

rcv_arr = []

snd_buf = np.array(local_arr, dtype=np.intc)
rcv_buf = np.empty(len(snd_buf), dtype=np.intc)

for phase in range(0, size):
    partner = compute_partner(phase, my_rank)
    # print("phase, partner, my_rank", phase, partner, my_rank)
    # if the current process is not idle 
    if partner >= 0 and partner < size and ((phase % 2 != 0 and (my_rank != 0 or my_rank != 3) and (partner != 0 or partner != 3)) or phase % 2== 0):
        # print("inside --> phase, partner, my_rank", phase, partner, my_rank)

        # send keys to partner 
        comm_world.Sendrecv(snd_buf, partner, 0, rcv_buf, partner, 0)

        # print("rcv_buf", rcv_buf)
        # print("phase", phase, "my_rank", my_rank, "partner", partner, "local arr" , local_arr)
        # print("phase", phase,"rcv_buf before", rcv_buf)
        np.copyto(snd_buf, rcv_buf)

        # merge them together and sort 
        joined_list = np.concatenate((local_arr, rcv_buf))
        joined_list = sorted(joined_list)
        # print("joined_list", joined_list)
        half_len = int(len(joined_list)/2)

        if my_rank < partner:
            # keep the smaller keys 
            local_arr = joined_list[0:half_len]
            # print("local_arr after join", local_arr)
        else:
            local_arr = joined_list[half_len:len(joined_list)]
            # print("local_arr after join", local_arr)


        np.copyto(snd_buf, local_arr)

        print("phase", phase, "my_rank", my_rank, "partner", partner, "local arr" , local_arr)


sorted_arr = comm_world.gather(local_arr, 0)
flat_sorted_arr = []
if(sorted_arr!= None):
    flat_sorted_arr = [ item for elem in sorted_arr for item in elem]
    print(f"Final sorted list: {flat_sorted_arr}")