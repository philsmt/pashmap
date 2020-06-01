
# Distributed under the terms of the BSD 3-Clause License.
# The full license is in the file LICENSE, distributed with this software.
#
# Author: Philipp Schmidt <philipp.schmidt@xfel.eu>
# Copyright (c) 2020, European X-Ray Free-Electron Laser Facility GmbH.
# All rights reserved.

from collections.abc import Sequence

import numpy as np


class MapTarget:
    """Target for map operation.

    This object wraps the value processed in a map operation to control
    its distribution into individual work units and then iterating over
    the entries in each work unit. The simplest target type is
    SequenceTarget, which distributes a set of indices to each worker
    and then iterates over the assigned indices in its body.

    If a custom type extends this class and implements the wrap
    classmethod, then it take advantage of automatic wrapping of values
    passed to a map call.
    """

    _target_types = []

    @classmethod
    def __init_subclass__(cls):
        cls._target_types.append(cls)

    @classmethod
    def get_default_target(cls, value):
        """Get default map target.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (MapTarget) Target object wrapping the given value.

        Raises:
            ValueError: If no or more than one default target types can
                be found.
        """

        if isinstance(value, cls):
            return value

        target = None

        for target_type in cls._target_types:
            cur_target = target_type.wrap(value)

            if cur_target is not None:
                if target is not None:
                    raise ValueError('ambiguous default target requires an '
                                     'explicit map target')
                target = cur_target

        if target is None:
            raise ValueError(f'no default target for {type(value)}')

        return target

    @classmethod
    def wrap(self, value):
        """Wrap value in this target type, if possible.

        Args:
            value (Any): Value to wrap for map operation.

        Returns:
            (MapTarget) Target object if wrapping is possible or None.
        """

        return

    def split(self, num_workers):
        """Split this target into work units.

        The values contained in the returned Iterable are passed to this
        target's iterate method later on. It may consist of any value
        suitable to describe each work unit, e.g. an iterable of indices
        of a sequence.

        Args:
            num_workers (int): Number of workers processing this target.

        Returns:
            (Iterable) Iterable of elements for each work unit.
        """

        raise NotImplementedError('split')

    def iterate(self, share):
        """Iterate over a share of this target.

        Args:
            share (Any): Element of the Iterable returned by
                :method:split to iterate over.

        Returns:
            None
        """

        raise NotImplementedError('iterate')


class SequenceTarget(MapTarget):
    """Map target for a sequence.

    This target wraps any indexable collection, e.g. list, tuples, numpy
    ndarrays or any other type implementing __getitem__. The kernel is
    passed the current index and sequence value at that index.

    Note that only ndarray and types implementing the Sequence interface
    are currently automatically detected to use this target type. Other
    types, e.g. xarray's DataArray, need to be wrapped manually.
    """

    def __init__(self, sequence):
        """Initialize this sequence target.

        Args:
            sequence (Sequence): Sequence to process.
        """

        self.sequence = sequence

    @classmethod
    def wrap(cls, value):
        if isinstance(value, (Sequence, np.ndarray)):
            # Note that ndarray does NOT implement Sequence itself!
            return cls(value)

    def split(self, num_workers):
        return np.array_split(np.arange(len(self.sequence)), num_workers)

    def iterate(self, indices):
        for index in indices:
            yield index, self.sequence[index]


# Ideas for targets: xarray.DataArray/Dataset, pandas


class ExtraDataTarget(MapTarget):
    """Map target for EXtra-data DataCollection.

    This target wraps an EXtra-data DataCollection and performs the map
    operation over its trains. The kernel is passed the current train's
    index in the collection, the train ID and the data mapping.
    """

    def __init__(self, dc):
        self.dc = dc

        import extra_data as xd
        ExtraDataTarget.xd = xd

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
        subdc = self.dc.select_trains(ExtraDataTarget.xd.by_index[indices])

        # Close all file handles inherited from the parent collection
        # to force re-opening them in each worker process.
        for f in subdc.files:
            f.close()

        for index, (train_id, data) in zip(indices, subdc.trains()):
            yield index, train_id, data
