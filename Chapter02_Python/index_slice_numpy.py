import numpy as np

l0 = [i for i in range(30)]
l0 = l0[1:30:3]
print(l0)

l1 = list(range(20))
print(l1)

# Start, Stop, Step
l2 = l1[0:20:2]
print(l2)

# 0:Stop:1
l3 = l1[:10]
print(l3)

my_array = np.zeros(shape=(2, 2), dtype=np.int32)
print(my_array)

my_reshaped_array = np.reshape(my_array, newshape=(4,))
print(my_reshaped_array)

my_reshaped_array2 = np.reshape(my_array, newshape=(4,))
print(my_reshaped_array2)

# range [0, 10]
my_random_array = np.random.randint(low=0, high=11, size=20)
print("my_random_array: ", my_random_array)

my_random_array2 = np.random.uniform(low=0.0, high=10.0, size=4)
print("my_random_array2: ", my_random_array2)
