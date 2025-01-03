import matplotlib.pyplot as plt
import numpy as np


x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print(x)

myArray = np.array([10, 20, 30, 40, 50], dtype=np.float32)  # noqa: N816
print(np.max(myArray))

y = np.array([-2, 1, 2, -10, 22], dtype=np.float32)
print(y)

print(np.max(x))
print(np.min(x))
print(np.mean(x))
print(np.median(x))

# Scatter Plot
plt.scatter(x, myArray, color="green")
plt.legend(["f(x)"])
plt.title("This is a title")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()

# Plot
plt.scatter(x, y, color="red")
plt.plot(x, y, color="blue")
plt.legend(["f(x)"])
plt.title("This is a title")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()
