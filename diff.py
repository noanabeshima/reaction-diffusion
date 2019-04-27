import numpy as np
import time
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

PPS = 4
X, Y = 650, 400

A = np.ones((Y,X))
B = np.zeros((Y,X))
B+=np.random.randn(Y,X)/5.55

dt = 1
k = .048
f = .012
DA = 1
DB = .5

im = np.kron(1-A, np.ones((PPS, PPS)))
cv2.imshow('Gray-Scott',im)
cv2.waitKey(1)
cv2.setWindowProperty('Gray-Scott', 0, 1.)
cv2.setWindowProperty('Gray-Scott', 1, 1.)


for i in range(100000):    
    im = np.kron(A, np.ones((PPS, PPS)))
    cv2.imshow('Gray-Scott',im)
    if cv2.waitKey(1)==27:
        break
    cv2.setWindowProperty('Gray-Scott', 0, 1.)
    
    A = A + (DA*lap(A)-A*B*B+f*(1-A))*dt
    B = B + (DB*lap(B)+A*B*B-(k+f)*B)*dt