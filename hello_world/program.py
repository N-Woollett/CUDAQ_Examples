import cudaq

kernel = cudaq.make_kernel()
qubit = kernel.qalloc()
kernel.x(qubit)
kernel.mz(qubit)

result = cudaq.sample(kernel)
print(result)