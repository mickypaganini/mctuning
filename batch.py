from __future__ import print_function
import threading
from joblib import Parallel, delayed
import time
import Queue
import os
import sys

# Modify
#########################

# Define all the command line scripts to run here.
TASKS = [
#    "python train.py test.yaml Baseline test.yaml fsr{fsr} --ntrain 100000 --ntest 100000 --nval 100000 --model rnn --patience 10 --nepochs 1000 --lr 0.001 &> w_rnn_fsr{fsr}.log".format(fsr=option) for option in ['05', '07', '09', '11', '13', '15', '17', '19', '20']
] + [
    "python train.py test.yaml Baseline fsr12.yaml Baseline --ntrain 100000 --ntest 100000 --nval 100000 --model rnn --patience 10 --nepochs 1000 --lr 0.001 --checkpoint checkpoint_{t}  &> uw_rnn_fsr12_{t}.log".format(t=option) for option in range(12)
]
# Define which GPUs to consider
DEVICE_NUMBERS = [0, 1, 2, 3]

# Define the number of processes that can run on a gpu concurrently
NUM_PROC_MAX = 3

# Don't Modify
#########################

print('Found {} tasks.'.format(len(TASKS)))

# Don't have print go crazy inside threads
_print = print
RLOCK = threading.RLock()


def print(*args, **kwargs):
    with RLOCK:
        _print(*args, **kwargs)
#        sys.stdout.flush()
                            
NUM_GPU = len(DEVICE_NUMBERS)
print('found {} devices'.format(NUM_GPU))

# Put GPU IDs in queue
print('queueing devices...')
device_queue = Queue.Queue(maxsize=NUM_GPU * NUM_PROC_MAX)
for device_id in DEVICE_NUMBERS:
    for p in range(NUM_PROC_MAX):
        print('enqueueing device:{}(replica:{})'.format(device_id, p))
        device_queue.put(device_id)

print('found {} devices'.format(NUM_GPU))


def run_task(task_id):
    print('task_id:{}, recieved new task'.format(task_id))

    # Dequeue a GPU ID. This will block if it's not done.
    gpu = device_queue.get()

    # dequeue a task to run
    cmd = TASKS[task_id]

    print('task_id:{}, running on GPU:{}'.format(task_id, gpu))

    job = "CUDA_VISIBLE_DEVICES={} {}".format(gpu, cmd)
    print('task_id:{}, will run: gpu{}://{}'.format(task_id, gpu, cmd))
    os.system(job)

    # return gpu id to queue for the next task to be able to use
    device_queue.put(gpu)

# Change loop
start_time = time.time()
Parallel(n_jobs=NUM_GPU * NUM_PROC_MAX, backend='threading')(
    delayed(run_task)(i) for i in range(len(TASKS))
)
print('{} tasks finished in {} minutes'
      .format(len(TASKS), (time.time() - start_time) / 60))
