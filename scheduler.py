import sched, time
import os

s = sched.scheduler(time.time, time.sleep)

def print_time(a='default'):
    print("From print_time", time.time(), a)
    os.system('python train.py configs/EDA_BiFPN_1layer_64channels.py')


def print_some_times():
    print(time.time())
    s.enter(9000, 1, print_time, kwargs={'a': 'keyword'})
    s.run()
    print(time.time())


print_some_times()