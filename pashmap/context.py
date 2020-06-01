
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

import mmap
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from os import cpu_count
from queue import Queue
from threading import local

import numpy as np

from .target import MapTarget


class MapContext:
    """Context to execute map operations.

    A map operation applies a single callable to each item in a
    collection. The context define the runtime conditions for this
    operation, which may be in a process pool, for example.

    As some of these environments may require special memory semantics,
    the context also provides a method of allocating ndarrays. The only
    abstract method required by an implementation is map.
    """

    def __init__(self, num_workers):
        """Initialize this map context.

        In order to use the default allocation methods in this type,
        in particular reduce_array(), the number of workers in this
        context need to be passed to its initializer or stored in a
        `num_workers` property.

        Args:
            num_workers (int): Number of workers used in this context.
        """

        self.num_workers = num_workers

    def array(self, shape, dtype=np.float64):
        """Allocate an array shared with all workers.

        The implementation may decide how to back this memory, but it
        is required that all workers of this context may read AND write
        to this memory. The default implementation allocates directly
        on the heap.

        Args:
            shape (int or tuple of ints): Shape of the array.
            dtype (data-type): Desired data-type for the array.

        Returns:
            (numpy.ndarray) Created array object.
        """

        return np.zeros(shape, dtype=dtype)

    def reduce_array(self, shape, dtype=np.float64):
        """Allocate a shared array for each worker.

        The returned array will contain an additional prepended axis
        with its shape corresponding to the number of workers in this
        context, i.e. with one dimension more than specified by the
        shape parameter. These are useful for parallelized reduction
        operations, where each worker may work with its own accumulator.

        Args:
            Same as array()

        Returns:
            (numpy.ndarray) Created array object.
        """

        if isinstance(shape, int):
            return self.array((self.num_workers, shape), dtype)
        else:
            return self.array((self.num_workers,) + tuple(shape), dtype)

    def map(self, kernel, target):
        """Apply kernel to each element in the target.

        This method performs the actual map operation. The target may
        either be an explicit MapTarget object or any other supported
        type, which can be wrapped into a default target object.

        Args:
            kernel (Callable): Kernel function to apply.
            target (MapTarget or Any): Target object to apply to or
                any supported type to be wrapped automatically.

        Returns:
            None
        """

        raise NotImplementedError('map')

    @staticmethod
    def run_worker(kernel, target, share, worker_id):
        """Main worker loop.

        This staticmethod contains the actual inner loop for a worker,
        i.e. iterating over the target and calling the kernel function.

        Subtypes may call this method after sorting out the required
        parameters through their specific machinery.

        Args:
            kernel (Callable): Kernel function to apply.
            target (MapTarget): Target object to apply to.
            share (Any): Share of the target for this worker.
            worker_id (int): Identification of this worker.

        Returns:
            None
        """

        for entry in target.iterate(share):
            kernel(worker_id, *entry)


class LocalContext(MapContext):
    """Local map context.

    Runs the map operation directly in the same process and thread.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(num_workers=1)

    def map(self, kernel, target):
        if not isinstance(target, MapTarget):
            target = MapTarget.get_default_target(target)

        self.run_worker(kernel, target, next(iter(target.split(1))), 0)


class PoolContext(MapContext):
    """Abstract map context for multiprocessing.Pool interface.

    This class contains the common machinery required for a map context
    based on the Pool interface. A subtype is still required to
    implemenent the map() method with its actual call signature and then
    call this type's map method in turn.
    """

    def __init__(self, num_workers=None):
        if num_workers is None:
            num_workers = min(cpu_count() // 2, 10)

        super().__init__(num_workers=num_workers)

    def map(self, kernel, target, pool_cls):
        """Apply kernel to each element in the target.

        Incomplete map method to be called by a subtype.

        Args:
            kernel (Callable): Kernel function to apply.
            target (MapTarget): Target object to apply to.
            pool_cls (type): Pool implementation to use.

        Returns:
            None
        """

        self.kernel = kernel

        for worker_id in range(self.num_workers):
            self.id_queue.put(worker_id)

        if not isinstance(target, MapTarget):
            target = MapTarget.get_default_target(target)

        with pool_cls(self.num_workers, self.init_worker, (target,)) as p:
            p.map(self.run_worker, target.split(self.num_workers))


class ThreadContext(PoolContext):
    """Map context using a thread pool.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.worker_storage = local()
        self.id_queue = Queue()

    def map(self, kernel, target):
        super().map(kernel, target, ThreadPool)

    def init_worker(self, target):
        self.worker_storage.worker_id = self.id_queue.get()
        self.worker_storage.target = target

    def run_worker(self, share):
        super().run_worker(self.kernel, self.worker_storage.target, share,
                           self.worker_storage.worker_id)


class ProcessContext(PoolContext):
    """Map context using a process pool.

    The memory allocated by this context is backed by anonymous mappings
    via `mmap` and thus shared for both reads and writes with all worker
    processes created after the allocation. This requires the start
    method to be `fork`, which is only supported on Unix systems.
    """

    _instance = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        try:
            self.mp_ctx = get_context('fork')
        except ValueError:
            raise ValueError('fork context required')

        self.id_queue = self.mp_ctx.Queue()

    def array(self, shape, dtype=np.float64):
        if isinstance(shape, int):
            n_elements = shape
        else:
            n_elements = 1
            for _s in shape:
                n_elements *= _s

        n_bytes = n_elements * np.dtype(dtype).itemsize
        n_pages = n_bytes // mmap.PAGESIZE + 1

        buf = mmap.mmap(-1, n_pages * mmap.PAGESIZE,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)

        return np.frombuffer(memoryview(buf)[:n_bytes],
                             dtype=dtype).reshape(shape)

    def map(self, kernel, target):
        super().map(kernel, target, self.mp_ctx.Pool)

    def init_worker(self, target):
        # Save reference in process-local copy
        self.__class__._instance = self

        self.worker_id = self.id_queue.get()
        self.target = target

    @classmethod
    def run_worker(cls, share):
        # map is a classmethod here and fetches its process-local
        # instance, as the instance in the parent process is not
        # actually part of the execution

        self = cls._instance
        super(cls, self).run_worker(self.kernel, self.target, share,
                                    self.worker_id)
