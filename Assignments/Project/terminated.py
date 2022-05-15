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


my_stack = []


# # functions for terminated 
def my_avail_tour_count(my_stack):
    return len(my_stack)

# fullfill request checks to see if the process has received a request for work 
# if it has then it splits its stack and sends work to the requesting process 
def fullfill_request(my_stack):
	if comm_world.Iprobe(MPI.ANY_SOURCE, tag = 30):
		# split the stack 
		size_of_stack = len(my_stack)
		split_stack = my_stack[0:size_of_stack/2]
     	snd_buf = np.array(split_stack, dtype=np.intc)
     	# how to send to the requesting process? how do i know which process/
     	comm_world.isend(snd_buf, dest, tag=11)

    else: 
    	return


# send_rejects checks for any work requests from other processes 
# and sends a "no work"
# reply to each requesting process 
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

