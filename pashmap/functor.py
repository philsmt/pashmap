
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from collections.abc import Sequence

import numpy as np


class Functor:
    """Mappable object.

    In functional programming, a functor is basically something that can
    be mapped over. This interface specifically provides the machinery
    to distribute a share of the value to each worker. The simplest
    functor is SequenceFunctor, which assigns a set of indices to worker
    and then iterates over the assigned indices in its body.

    If a custom type extends this class and implements the wrap
    classmethod, then it take advantage of automatic wrapping of values
    passed to a map call.
    """

    _functor_types = []

    @classmethod
    def __init_subclass__(cls):
        cls._functor_types.append(cls)

    @classmethod
    def try_wrap(cls, value):
        """Attempt to wrap a value in a functor.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (Functor) Functor wrapping the given value or the same value
                if already a subtype of Functor.

        Raises:
            ValueError: If no or more than one default functor types
                could be applied.
        """

        if isinstance(value, cls):
            return value

        functor = None

        for functor_type in cls._functor_types:
            cur_functor = functor_type.wrap(value)

            if cur_functor is not None:
                if functor is not None:
                    raise ValueError('default functor is ambiguous')

                functor = cur_functor

        if functor is None:
            raise ValueError(f'no default functor for {type(value)}')

        return functor

    @classmethod
    def wrap(self, value):
        """Wrap value in this functor type, if possible.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (Functor) Functor if wrapping is possible or None.
        """

        return

    def split(self, num_workers):
        """Split this functor into work units.

        The values contained in the returned Iterable are passed to this
        functor's iterate method later on. It may consist of any value
        suitable to describe each work unit, e.g. an iterable of indices
        of a sequence.

        Args:
            num_workers (int): Number of workers processing this
                functor.

        Returns:
            (Iterable) Iterable of elements for each work unit.
        """

        raise NotImplementedError('split')

    def iterate(self, share):
        """Iterate over a share of this functor.

        Args:
            share (Any): Element of the Iterable returned by to iterate
                over.

        Returns:
            None
        """

        raise NotImplementedError('iterate')


class SequenceFunctor(Functor):
    """Functor wrapping a sequence.

    This functor can wrap any indexable collection, e.g. list, tuples,
    or any other type implementing __getitem__. It automatically wraps
    any value implementing the collections.abc.Sequence type. The kernel
    is passed the current index and sequence value at that index.
    """

    def __init__(self, sequence):
        """Initialize a sequence functor.

        Args:
            sequence (Sequence): Sequence to process.
        """

        self.sequence = sequence

    @classmethod
    def wrap(cls, value):
        if isinstance(value, Sequence):
            return cls(value)

    def split(self, num_workers):
        return np.array_split(np.arange(len(self.sequence)), num_workers)

    def iterate(self, indices):
        for index in indices:
            yield index, self.sequence[index]


class NdarrayFunctor(SequenceFunctor):
    """Functor wrapping an numpy.ndarray.

    This functor extends SequenceFunctor to use additional functionality
    provided by numpy ndarrays, e.g. iterating over specific axes and
    more efficient indexing and should works for any array_like object
    that supports numpy-style slicing. However, specifying an explicit
    axis may cause conversion to an ndarray or break unexpectedly.
    """

    def __init__(self, array, axis=None):
        """Initialize an ndarray functor.

        Args:
            array (numpy.ndarray): Array to map over.
            axis (int, optional): Axis to map over, first axis by
                default or if None.
        """

        self.sequence = np.swapaxes(array, 0, axis) \
            if axis is not None else array

    @classmethod
    def wrap(cls, value):
        if isinstance(value, np.ndarray):
            return cls(value)

    def iterate(self, indices):
        yield from zip(indices, self.sequence[indices])


# Ideas for wrapping functors: xarray.DataArray/Dataset, pandas


class ExtraDataFunctor(Functor):
    """Functor for EXtra-data DataCollection.

    This functor wraps an EXtra-data DataCollection and performs the map
    operation over its trains. The kernel is passed the current train's
    index in the collection, the train ID and the data mapping.
    """

    def __init__(self, dc):
        self.dc = dc

        import extra_data as xd
        ExtraDataFunctor.xd = xd

    @classmethod
    def wrap(cls, value):
        if value.__class__.__name__ != 'DataCollection':
            # Avoid importing EXtra-data if not even the name matches.
            return

        try:
            import extra_data as xd
        except ImportError:
            return

        if isinstance(value, xd.DataCollection):
            return cls(value)

    def split(self, num_workers):
        return np.array_split(np.arange(len(self.dc.train_ids)), num_workers)

    def iterate(self, indices):
        subdc = self.dc.select_trains(ExtraDataFunctor.xd.by_index[indices])

        # Close all file handles inherited from the parent collection
        # to force re-opening them in each worker process.
        for f in subdc.files:
            f.close()

        for index, (train_id, data) in zip(indices, subdc.trains()):
            yield index, train_id, data
