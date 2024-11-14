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
b = -a
print(b)
b = sum(a)
print(b)
b = np.mean(a)
print(b)
b = a ** 2
print(b)
print('---------------------------------')
a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(a + b)
print('---------------------------------')
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print(e)
print('---------------------------------')
a = np.array([1, 2, 3, 4])
b = 5 * a
print(b)

print('----------')


def my_dot(a, b):
    x = 0
    for i in range(a.shape[0]):
        x += a[i] * b[i]
    return x


a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(my_dot(a, b))

print(np.dot(a,b), np.dot(a,b).shape)

print('---------------------------------')

print(np.random.seed(1))
a = np.random.rand(1000000)
print(a)
b = np.random.rand(1000000)

tic = time.time()
print(tic)
c=np.dot(a,b)
tioc = time.time()
print(1000*(tioc-tic))

tic = time.time()  # capture start time
c = my_dot(a,b)
toc = time.time()  # capture end time

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

print('---------------')

X = np.array([[1],[2],[3],[4]])
w=np.array([2])
c=np.dot(X[1],w)
print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

print('-----------')
a = np.zeros((1,5))
print(a)

a=np.zeros((2,1))
print(a)

a=np.random.random_sample((1,1))
print(a)

a= np.array([[5],[4],[3]])
print(f'a shape is {a.shape}, np.array: a = {a}')

print('-----------')

# reshape用于重新调整数组的形状
# np.arange(6)生成[0,1,2,3,4,5]
# reshape(-1,2)将数组重组为n行2列的形式，其中-1表示自动计算行数
# 这里会得到一个3行2列的数组:
# [[0,1],
#  [2,3], 
#  [4,5]]
a = np.arange(6).reshape(3, 2)  
print(a)

print('-----------')

a= np.arange(20).reshape(-1,10)
print(a)

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# 访问第2行的所有元素
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

if __name__ == "__main__":
    pass
