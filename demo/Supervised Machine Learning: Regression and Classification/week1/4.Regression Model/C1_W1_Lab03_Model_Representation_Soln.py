import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    AT = A.T
    print("A: ", A)
    print("A.T: ", AT)
    W = np.array([[1, 2, 3], [4, 5, 6]])
    print("W: ", W)
    Z = np.dot(A, W)
    print("Z: ", Z)
    Z1 = np.matmul(A, W)
    print("Z1: ", Z1)
    print("=====================================")
    Z2 = np.matmul(np.array([[200, 17]]), np.array([[1, -3, 5], [-2, 4, -6]])) + np.array([[-1, 1, 2]])
    print()
    print("=====================================")
    print(sigmoid(Z2))


if __name__ == '__main__':
    main()
