import numpy as np
import cv2


def roll(A, *coord):
    # Returns a shifted version of the array
    # See np.roll for more information
    if (len(A.shape) != len(coord)):
        raise Exception('Number of dimensions of input array [' + str(len(A.shape)) +
                        '] and coordinate for rolling [' + str(len(coord)) + '] don\'t match')
    for i in range(len(coord)):
        if coord[i] != 0:
            A = np.roll(A, coord[i], axis=i)
    return A


def laplacian(A):
    result = -A + .2 * (np.roll(A, 1, axis=0) + np.roll(A, 1, axis=1) + np.roll(A, -1, axis=0) +
                        np.roll(A, -1, axis=1)) + .05 * (roll(A, 1, 1) + roll(A, 1, -1) +
                                                         roll(A, -1, 1) + roll(A, -1, -1))
    return result


pixels_per_square = 3
X, Y = 650, 400

A = np.ones((Y, X))
B = np.zeros((Y, X))

B += np.random.randn(Y, X) / 5.5

# "physical" constants described in Sim's page
dt = 1
k = .055
f = .015
DA = 1
DB = .5

while True:
    im = np.kron(A, np.ones((pixels_per_square, pixels_per_square)))
    cv2.imshow('Test', im)

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

    # update arrays
    A = A + (DA * laplacian(A) - A * B * B + f * (1 - A)) * dt
    B = B + (DB * laplacian(B) + A * B * B - (k + f) * B) * dt
