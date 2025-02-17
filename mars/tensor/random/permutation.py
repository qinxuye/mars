# Copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from numbers import Integral

import numpy as np

from ... import opcodes as OperandDef
from ...core.operand import OperandStage
from ...serialization.serializables import KeyField, Int32Field
from ..operands import TensorOperandMixin, TensorShuffleProxy
from ..utils import gen_random_seeds, validate_axis
from ..datasource import tensor as astensor
from ..array_utils import as_same_device, device
from .core import TensorRandomMapReduceOperand


def _permutation_on_axis(ar, axis, rs, xp):
    try:
        return rs.permutation(ar, axis=axis)
    except TypeError:
        # numpy starts to support axis from 1.18
        if axis == 0:
            return rs.permutation(ar)
        indices = xp.arange(ar.shape[axis])
        rs.shuffle(indices)
        slc = (slice(None),) * axis + (indices,)
        return ar[slc]


class TensorPermutation(TensorRandomMapReduceOperand, TensorOperandMixin):
    _op_type_ = OperandDef.PERMUTATION

    input = KeyField("input")
    axis = Int32Field("axis")

    reduce_size = Int32Field("reduce_size")

    def _set_inputs(self, inputs):
        super()._set_inputs(inputs)
        self.input = self._inputs[0]

    def __call__(self, x):
        return self.new_tensor([x], x.shape, order=x.order)

    @classmethod
    def tile(cls, op):
        in_tensor = op.inputs[0]
        out_tensor = op.outputs[0]

        state = np.random.RandomState(op.seed)
        if len(op.input.chunks) == 1:
            chunk_op = op.copy().reset_key()
            chunk_op._state = None
            chunk_op.seed = gen_random_seeds(1, state)[0]
            c = op.input.chunks[0]
            chunk = chunk_op.new_chunk([c], shape=c.shape, index=c.index, order=c.order)
            new_op = op.copy()
            return new_op.new_tensors(
                op.inputs,
                shape=out_tensor.shape,
                order=out_tensor.order,
                nsplits=op.input.nsplits,
                chunks=[chunk],
            )

        chunk_size = in_tensor.chunk_shape[op.axis]
        map_seeds = gen_random_seeds(chunk_size, state)
        reduce_seeds = gen_random_seeds(chunk_size, state)
        reduce_chunks = []
        if in_tensor.ndim > 1:
            cs = in_tensor.chunk_shape
            left_chunk_shape = cs[: op.axis] + cs[op.axis + 1 :]
            idx_iter = itertools.product(*[range(s) for s in left_chunk_shape])
        else:
            idx_iter = [()]
        for idx in idx_iter:
            map_chunks = []
            for j in range(chunk_size):
                in_idx = list(idx)
                in_idx.insert(op.axis, j)
                c = in_tensor.cix[tuple(in_idx)]
                chunk_op = TensorPermutation(
                    stage=OperandStage.map,
                    seed=map_seeds[c.index[op.axis]],
                    axis=op.axis,
                    reduce_size=chunk_size,
                    dtype=c.dtype,
                    gpu=c.op.gpu,
                )
                map_chunk = chunk_op.new_chunk(
                    [c], shape=c.shape, index=c.index, order=out_tensor.order
                )
                map_chunks.append(map_chunk)

            proxy_chunk = TensorShuffleProxy(
                dtype=out_tensor.dtype, _tensor_keys=[in_tensor.key]
            ).new_chunk(map_chunks, shape=())

            for ordinal, c in enumerate(map_chunks):
                chunk_op = TensorPermutation(
                    stage=OperandStage.reduce,
                    n_reducers=len(map_chunks),
                    reducer_ordinal=ordinal,
                    seed=reduce_seeds[c.index[op.axis]],
                    axis=op.axis,
                )
                chunk_shape = list(c.shape)
                chunk_shape[op.axis] = np.nan
                reduce_chunk = chunk_op.new_chunk(
                    [proxy_chunk],
                    shape=tuple(chunk_shape),
                    order=out_tensor.order,
                    index=c.index,
                    dtype=out_tensor.dtype,
                )
                reduce_chunks.append(reduce_chunk)

        new_op = op.copy()
        nsplits = list(in_tensor.nsplits)
        nsplits[op.axis] = [np.nan] * len(nsplits[op.axis])
        return new_op.new_tensors(
            op.inputs,
            out_tensor.shape,
            order=out_tensor.order,
            chunks=reduce_chunks,
            nsplits=nsplits,
        )

    @classmethod
    def _execute_map(cls, ctx, op):
        (x,), device_id, xp = as_same_device(
            [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True
        )

        out_chunk = op.outputs[0]
        reduce_size = op.reduce_size
        with device(device_id):
            rs = xp.random.RandomState(op.seed)
            to_reduce_idxes = rs.randint(reduce_size, size=x.shape[op.axis])
            for to_reduce_idx in range(reduce_size):
                reduce_idx = (
                    out_chunk.index[: op.axis]
                    + (to_reduce_idx,)
                    + out_chunk.index[op.axis + 1 :]
                )
                slc = (slice(None),) * op.axis + (to_reduce_idxes == to_reduce_idx,)
                ctx[out_chunk.key, reduce_idx] = x[slc]

    @classmethod
    def _execute_reduce(cls, ctx, op: "TensorPermutation"):
        inputs = list(op.iter_mapper_data(ctx))
        inputs, device_id, xp = as_same_device(inputs, device=op.device, ret_extra=True)

        with device(device_id):
            rs = xp.random.RandomState(op.seed)
            data = xp.concatenate(inputs, axis=op.axis)
            if op.axis == 0:
                rs.shuffle(data)
            else:
                data[...] = _permutation_on_axis(data, op.axis, rs, xp)
            ctx[op.outputs[0].key] = data

    @classmethod
    def execute(cls, ctx, op):
        if op.stage == OperandStage.map:
            cls._execute_map(ctx, op)
        elif op.stage == OperandStage.reduce:
            cls._execute_reduce(ctx, op)
        else:
            (x,), device_id, xp = as_same_device(
                [ctx[c.key] for c in op.inputs], device=op.device, ret_extra=True
            )

            with device(device_id):
                rs = xp.random.RandomState(op.seed)
                ctx[op.outputs[0].key] = _permutation_on_axis(x, op.axis, rs, xp)


def permutation(random_state, x, axis=0, chunk_size=None):
    r"""
    Randomly permute a sequence, or return a permuted range.

    Parameters
    ----------
    x : int or array_like
        If `x` is an integer, randomly permute ``mt.arange(x)``.
        If `x` is an array, make a copy and shuffle the elements
        randomly.
    axis : int, optional
        The axis which `x` is shuffled along. Default is 0.
    chunk_size : : int or tuple of int or tuple of ints, optional
        Desired chunk size on each dimension
    Returns
    -------
    out : Tensor
        Permuted sequence or tensor range.
    Examples
    --------
    >>> import mars.tensor as mt
    >>> rng = mt.random.RandomState()
    >>> rng.permutation(10).execute()
    array([1, 2, 3, 7, 9, 8, 0, 6, 4, 5]) # random
    >>> rng.permutation([1, 4, 9, 12, 15]).execute()
    array([ 9,  4, 12,  1, 15]) # random
    >>> arr = mt.arange(9).reshape((3, 3))
    >>> rng.permutation(arr).execute()
    array([[3, 4, 5], # random
           [6, 7, 8],
           [0, 1, 2]])
    >>> rng.permutation("abc")
    Traceback (most recent call last):
        ...
    numpy.AxisError: x must be an integer or at least 1-dimensional
    """
    if isinstance(x, (Integral, np.integer)):
        from ..datasource import arange

        x = arange(x, chunk_size=chunk_size)
    else:
        x = astensor(x, chunk_size=chunk_size)
        if x.ndim < 1:
            raise np.AxisError("x must be an integer or at least 1-dimensional")

    axis = validate_axis(x.ndim, axis)
    seed = gen_random_seeds(1, random_state.to_numpy())[0]
    op = TensorPermutation(seed=seed, axis=axis, dtype=x.dtype, gpu=x.op.gpu)
    return op(x)
