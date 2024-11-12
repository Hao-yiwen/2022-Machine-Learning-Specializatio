import numpy as np
import time

a = np.zeros(4)
print(a)
b = np.zeros((4,))
print(b)
c = np.random.random_sample((4,))
print(c)
d = np.arange(3, 7, 1)
print(d)
e = np.random.rand(5)
print(e)

print(e[2].shape)

print('---------------------------------')

a = np.arange(10)
print(a)

f = a[2:7:1]
print(f)

f = a[2:7]
print(f)
f = a[2:]
print(f)
f = a[:7]
print(f)
f = a[:]
print(f)
f = a[2:7:2]
print(f)
print('---------------------------------')

a = np.array([1, 2, 3, 4])

if __name__ == "__main__":
    pass
