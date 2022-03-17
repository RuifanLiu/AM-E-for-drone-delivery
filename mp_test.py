# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 21:21:45 2021

@author: s313488
"""

from __future__ import print_function

import time
import multiprocessing

data = range(30)

def lol():
    for i in data:
        start_time = time.time()
        time.sleep(2)
        t = time.time()
        print('t: ', t, end='\t')
        print(t - start_time, "lol seconds")


def worker(n):
    start_time = time.time()
    time.sleep(2)
    t = time.time()
    print('t: ', t, end='\t')
    print(t - start_time, "multiprocesor seconds")


def mp_handler():
    p = multiprocessing.Pool(30)
    p.map(worker, data)

if __name__ == '__main__':
    lol()
    mp_handler()