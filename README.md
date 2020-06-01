# pashmap

pashmap (**pa**rallelized **sh**ared memory **map**) provides tools to process data in memory in a parallelized way with an emphasis on shared memory and zero copy. Thus, it is currently limited to a single node using threads or processes. It uses the map pattern similar to Python's builtin map() function, where a Callable is applied to potentially many elements in a collection. To avoid the high cost of IPC or other communication schemes, the results are meant to be written directly to memory shared between all workers as well as the calling site.

## Quick guide

To use it, simply import it, define your kernel function of choice and map away!
```python
import numpy as np
import pashmap as pm

# Get some random input data
inp = np.random.rand(100)

# Allocate output array via pashmap.
outp = pm.array(100)

# Define a kernel function multiplying each value with 3.
def triple_it(worker_id, index, value):
    outp[index] = 3 * value
    
# Map the kernel function.
pm.map(triple_it, inp)

# Check the result
np.testing.assert_allclose(outp, inp*3)
```
The runtime environment is controlled via a so called map context. The default context object is `LocalContext`, which does not actually parallelize anything and runs in the same process. You may either create an explicit context object and use it directly or change the default context, e.g.

```python
pm.set_default_context('processes', num_workers=4)
```
Now each map call will spawn a process pool and hand off the work. This only works on \*nix systems supporting the fork() system call, as it expects any input data to be shared. The output array returned by array() will also reside in shared memory with this context.

The input array passed to map() is called the map target and is automatically wrapped in a suitable MapTarget object, here SequenceTarget. This works for a number of common array and collection types, but you may also implement your own MapTarget object to wrap anything else. For example, there is built-in support for DataCollection objects from the EXtra-data toolkit accessing run files from the European XFEL facilty:
```python
def analysis_kernel(worker_id, index, train_id, data):
    # Do something with the data and save it to shared memory.

run = extra_data.open_run(proposal=700000, run=1)
pm.map(analysis_kernel, run)
```