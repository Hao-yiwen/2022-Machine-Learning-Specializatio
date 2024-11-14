def example(*args, **t):
    print(args)
    print(type(args))
    print(t)
    print(type(t))

example([1, 2, 3, 4, 5])
example(1,2,name=3)
if __name__ == '__main__':
    pass