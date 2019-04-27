import numpy as np
from time import sleep
import cv2

def roll(A, *coord):
    if (len(A.shape) != len(coord)):
        raise Exception('Number of dimensions of input array ['+str(len(A.shape))+\
                        '] and coordinate for rolling ['+str(len(coord))+'] don\'t match')
    for i in range(len(coord)):
        if coord[i] != 0:
            A = np.roll(A, coord[i], axis=i)
    return A

def lap(A):
    result = -A +.2*(np.roll(A, 1, axis = 0)+np.roll(A, 1, axis=1)+np.roll(A, -1, axis=0)+np.roll(A,-1, axis=1))+.05*(roll(A, 1, 1)+roll(A, 1, -1)+roll(A, -1, 1)+roll(A, -1, -1))
    return result

PPS = 3
size = 400

A = np.ones((size,size))
B = np.zeros((size,size))

B+=np.random.randn(size,size)/5.5

dt = 1
k = .055
f = .015
DA = 1
DB = .5



for i in range(10000):
    if i%2 == 0:
        im = np.kron(A, np.ones((PPS, PPS)))
        cv2.imshow('Test',im)
        cv2.waitKey(1)

    A = A + (DA*lap(A)-A*B*B+f*(1-A))*dt
    B = B + (DB*lap(B)+A*B*B-(k+f)*B)*dt