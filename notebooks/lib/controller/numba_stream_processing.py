#TODO: utilize multiple stream processors for high throughput.
# Stream Management
# Streams allow concurrency of execution on a single device within a given context. Queued work items in the same stream execute sequentially, but work items in different streams may execute concurrently. Most operations involving a CUDA device can be performed asynchronously using streams, including data transfers and kernel execution. For further details on streams, see the CUDA C Programming Guide Streams section.

# Streams are instances of numba.cuda.cudadrv.driver.Stream:

# nota from numba at https://numba.pydata.org/numba-doc/dev/cuda-reference/host.html

numba.cuda.cudadrv.driver.Device.reset()
"""
	deletes the context for the device. This will destroy all memory allocations, events, and streams created within the context.
"""


classnumba.cuda.cudadrv.driver.Stream(context, handle, finalizer, external=False)
https://numba.pydata.org/numba-doc/dev/cuda-reference/host.html#numba.cuda.cudadrv.driver.Stream

work items in different streams may execute concurrently

conda install numba cudatoolkit pyculib

