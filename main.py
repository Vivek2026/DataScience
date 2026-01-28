n = 5

print("Lower Triangular:")
for i in range(1, n + 1):
    print("* " * i)

print("\nUpper Triangular:")
for i in range(n, 0, -1):
    print("  " * (n - i) + "* " * i)

print("\nPyramid:")
for i in range(1, n + 1):
    print(" " * (n - i) + "* " * i)
