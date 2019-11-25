#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 1999-2020 Alibaba Group Holding Ltd.
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
from collections import deque
import sys
import weakref

from .graph import DAG
from .compat import six
from .utils import kernel_mode, enter_build_mode


class Tileable(object):
    __slots__ = ()

    @property
    def op(self):
        raise NotImplementedError

    @property
    def inputs(self):
        raise NotImplementedError

    @property
    def chunks(self):
        raise NotImplementedError

    def is_coarse(self):
        raise NotImplementedError

    def copy_from(self, other):
        raise NotImplementedError

    def tiles(self):
        return handler.tiles(self)

    def single_tiles(self):
        return handler.single_tiles(self)

    @kernel_mode
    def build_graph(self, graph=None, cls=DAG, tiled=False, compose=True,
                    **build_chunk_graph_kwargs):
        tileable_graph = graph if not tiled else None
        tileable_graph_builder = TileableGraphBuilder(graph=tileable_graph, graph_cls=cls)
        tileable_graph = tileable_graph_builder.build([self])
        if not tiled:
            return tileable_graph
        chunk_graph_builder = ChunkGraphBuilder(
            graph=graph, graph_cls=cls, compose=compose,
            **build_chunk_graph_kwargs)
        return chunk_graph_builder.build([self], tileable_graph=tileable_graph)

    @staticmethod
    def _to_coarse_graph(graph):
        new_graph = type(graph)()
        visited = dict()
        for n in graph:
            if n not in visited:
                new_node = n.to_coarse()
                visited[n] = new_node
                new_graph.add_node(new_node)
            for succ in graph.successors(n):
                if succ not in visited:
                    new_node = succ.to_coarse()
                    visited[succ] = new_node
                    new_graph.add_node(new_node)
                new_graph.add_edge(visited[n], visited[succ])
        return new_graph

    def visualize(self, graph_attrs=None, node_attrs=None, **kw):
        from graphviz import Source

        g = self.build_graph(**kw)
        dot = g.to_dot(graph_attrs=graph_attrs, node_attrs=node_attrs,
                       result_chunk_keys={c.key for c in self.chunks})

        return Source(dot)


class TilesError(Exception):
    pass


class DataNotReady(TilesError):
    pass


class TilesFail(TilesError):
    pass


class NotSupportTile(Exception):
    pass


class OperandTilesHandler(object):
    _handlers = {}

    @classmethod
    def _get_op_cls(cls, op):
        if isinstance(op, type):
            return op
        return type(op)

    @classmethod
    def register(cls, op, tile_handler):
        cls._handlers[cls._get_op_cls(op)] = tile_handler

    @classmethod
    def _assign_to(cls, tile_after_tensor_datas, tile_before_tensor_datas):
        assert len(tile_after_tensor_datas) == len(tile_before_tensor_datas)

        for tile_after_tensor_data, tile_before_tensor_data in \
                zip(tile_after_tensor_datas, tile_before_tensor_datas):
            if tile_before_tensor_data is None:
                # garbage collected
                continue
            tile_after_tensor_data.copy_to(tile_before_tensor_data)
            tile_before_tensor_data.op.outputs = tile_before_tensor_datas

    @kernel_mode
    def _dispatch(self, op):
        op_cls = self._get_op_cls(op)
        try:
            handler = self._handlers[op_cls]
            return handler(op)
        except KeyError as e:
            if hasattr(op_cls, 'tile'):
                # has tile implementation
                return op_cls.tile(op)
            for op_clz in self._handlers.keys():
                if issubclass(op_cls, op_clz):
                    self._handlers[op_cls] = self._handlers[op_clz]
                    return self._handlers[op_cls](op)

            raise e

    def dispatch(self, to_tiles):
        return self._dispatch(to_tiles.op)

    def single_tiles(self, to_tiles):
        if to_tiles.is_coarse() and to_tiles.op:
            dispatched = self._dispatch(to_tiles.op)
            self._assign_to([d.data for d in dispatched], to_tiles.op.outputs)

        return to_tiles

    def tiles(self, tiles_obj):
        graph = DAG()
        visited = {id(tiles_obj)}
        q = deque([tiles_obj])

        while q:
            to_tiles = q.popleft()
            if to_tiles not in graph:
                graph.add_node(to_tiles)
            objs = to_tiles.inputs or []
            for o in objs:
                if not isinstance(o, Tileable):
                    continue
                if o not in graph:
                    graph.add_node(o)
                graph.add_edge(o, to_tiles)
                if id(o) in visited:
                    continue
                visited.add(id(o))
                q.append(o)

        for node in graph.topological_iter():
            if node.is_coarse() and node.op:
                tiled = self._dispatch(node.op)
                self._assign_to([t.data for t in tiled], node.op.outputs)
                for tile_before_data, tile_after in zip(node.op.outputs, tiled):
                    _tileable_data_to_tiled[tile_before_data] = tile_after.data

        return tiles_obj


handler = OperandTilesHandler()
register = OperandTilesHandler.register

_tileable_data_to_tiled = weakref.WeakKeyDictionary()


@enter_build_mode
def get_tiled(tileable):
    tileable_data = tileable.data if hasattr(tileable, 'data') else tileable
    return _tileable_data_to_tiled[tileable_data]


class GraphBuilder(object):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None):
        self._graph_cls = graph_cls
        if graph is not None:
            self._graph = graph
        else:
            self._graph = graph_cls()
        self._node_processor = node_processor

    def _get_inputs(self, node):
        return node.inputs or []

    def _add_nodes(self, nodes, visited):
        graph = self._graph

        while len(nodes) > 0:
            node = nodes.pop()
            if self._node_processor:
                # if node processor registered, process the node first
                node = self._node_processor(node)

            visited.add(node)
            if not graph.contains(node):
                graph.add_node(node)
            children = self._get_inputs(node)
            for c in children:
                if self._node_processor:
                    c = self._node_processor(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
                for n in c.op.outputs:
                    if n not in visited:
                        nodes.append(n)

    def build(self, tileables, tileable_graph=None):
        raise NotImplementedError


class TileableGraphBuilder(GraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None,
                 trace_inputs=True):
        super(TileableGraphBuilder, self).__init__(graph=graph, graph_cls=graph_cls,
                                                   node_processor=node_processor)
        self._trace_inputs=trace_inputs
        self._tileable_set = None

    def _get_inputs(self, node):
        inputs = super(TileableGraphBuilder, self)._get_inputs(node)
        if not self._trace_inputs:
            return [inp for inp in inputs if inp in self._tileable_set]
        else:
            return inputs

    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        if tileable_graph is not None:  # pragma: no cover
            return tileable_graph

        visited = set()
        nodes = list(itertools.chain(
            *(tileable.op.outputs for tileable in tileables)))
        self._tileable_set = set(nodes)
        self._add_nodes(nodes, visited)
        return self._graph


class ChunkGraphBuilder(GraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None, compose=True,
                 on_tile=None, on_tile_success=None, on_tile_failure=None):
        super(ChunkGraphBuilder, self).__init__(graph=graph, graph_cls=graph_cls,
                                                node_processor=node_processor)
        self._compose = compose
        self._on_tile = on_tile
        self._on_tile_success = on_tile_success
        self._on_tile_failure = on_tile_failure

    def _tile(self, tileable_data):
        cache = _tileable_data_to_tiled
        on_tile = self._on_tile

        if tileable_data in cache:
            return [cache[o] for o in tileable_data.op.outputs]

        if on_tile is None:
            # copy tileable
            op = tileable_data.op.copy()
            tds = op.new_tileables([cache[inp] for inp in tileable_data.inputs],
                                   kws=[o.params for o in tileable_data.op.outputs],
                                   output_limit=len(tileable_data.op.outputs),
                                   **tileable_data.extra_params)
            tds[0].single_tiles()
        else:
            tds = on_tile(tileable_data)
            if not isinstance(tds, (list, tuple)):
                tds = [tds]
            assert len(tileable_data.op.outputs) == len(tds)
        for t, td in zip(tileable_data.op.outputs, tds):
            cache[t] = td.data
        return tds

    def _get_tileable_data_graph(self, tileables, tileable_graph):
        if tileable_graph is None:
            # if tileable_data graph not provided
            # create a new one via GraphBuilder
            tileable_graph_builder = TileableGraphBuilder(
                graph_cls=type(self._graph),
                node_processor=self._node_processor)
            tileable_graph = tileable_graph_builder.build(tileables)
        return tileable_graph

    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        tileable_graph = self._get_tileable_data_graph(tileables, tileable_graph)

        # do tiles and add nodes or edges to chunk graph
        tileables_set = set(tileables)
        keys = []
        visited = set()
        tiled_op = set()
        for tileable_data in tileable_graph.topological_iter():
            nodes = []
            # do tiling
            if tileable_data.op in tiled_op:
                continue
            try:
                tiled = self._tile(tileable_data)
                tiled_op.add(tileable_data.op)
                for t, td in zip(tileable_data.op.outputs, tiled):
                    if self._on_tile_success is not None:
                        td = self._on_tile_success(t, td)
                    nodes.extend(c.data for c in td.chunks)
                    if t in tileables_set:
                        keys.extend(c.key for c in td.chunks)
                    self._add_nodes(nodes, visited)
            except:  # noqa: E722
                exc_info = sys.exc_info()
                if self._on_tile_failure:
                    self._on_tile_failure(tileable_data.op, exc_info)
                else:
                    raise
        if self._compose:
            self._graph.compose(keys=keys)
        return self._graph


class IterativeChunkGraphBuilder(ChunkGraphBuilder):
    def __init__(self, graph=None, graph_cls=DAG, node_processor=None, compose=True,
                 on_tile=None, on_tile_success=None, on_tile_failure=None):
        self._failed_ops = set()
        self._iterative_tileable_graphs = []
        self._iterative_chunk_graphs = []
        self._done = False
        super(IterativeChunkGraphBuilder, self).__init__(
            graph=graph, graph_cls=graph_cls, node_processor=node_processor,
            compose=compose, on_tile=on_tile,
            on_tile_success=self._wrap_on_tile_success(on_tile_success),
            on_tile_failure=self._wrap_on_tile_failure(on_tile_failure))
        if self._graph_cls is None:
            self._graph_cls = type(self._graph)

    def _wrap_on_tile_failure(self, on_tile_failure):
        def inner(op, exc_info):
            if isinstance(exc_info[1], TilesFail):
                self._failed_ops.add(op)
            else:
                if on_tile_failure is not None:
                    on_tile_failure(op, exc_info)
                else:
                    six.reraise(*exc_info)
        return inner

    def _wrap_on_tile_success(self, on_tile_success):
        def inner(tile_before, tile_after):
            # if tile succeed, add the node before tiling
            # to current iterative tileable graph
            if on_tile_success is not None:
                tile_after = on_tile_success(tile_before, tile_after)
            iterative_tileable_graph = self._iterative_tileable_graphs[-1]
            iterative_tileable_graph.add_node(tile_before)
            for inp in tile_before.inputs:
                if inp in iterative_tileable_graph:
                    iterative_tileable_graph.add_edge(inp, tile_before)
            return tile_after
        return inner

    @property
    def failed_ops(self):
        return self._failed_ops

    @property
    def iterative_tileable_graphs(self):
        return self._iterative_tileable_graphs

    @property
    def iterative_chunk_graphs(self):
        return self._iterative_chunk_graphs

    @property
    def done(self):
        return self._done

    def _tile(self, tileable_data):
        if any(inp.op in self._failed_ops for inp in tileable_data.inputs):
            raise TilesFail('Tile fail due to failure of inputs')
        return super(IterativeChunkGraphBuilder, self)._tile(tileable_data)

    @enter_build_mode
    def build(self, tileables, tileable_graph=None):
        tileable_graph = self._get_tileable_data_graph(tileables, tileable_graph)
        self._graph = self._graph_cls()
        self._failed_ops.clear()
        self._iterative_tileable_graphs.append(type(tileable_graph)())

        chunk_graph = super(IterativeChunkGraphBuilder, self).build(
            tileables, tileable_graph=tileable_graph)
        self._iterative_chunk_graphs.append(chunk_graph)
        if len(self._failed_ops) == 0:
            self._done = True
        return chunk_graph
