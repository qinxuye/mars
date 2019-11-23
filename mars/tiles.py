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
        graph_builder = GraphBuilder(graph=graph, graph_cls=cls)
        return graph_builder.build([self], tiled=tiled, compose=compose,
                                   **build_chunk_graph_kwargs)
        # from .utils import build_fetch
        #
        # executed_keys = set(executed_keys or [])
        # if tiled and self.is_coarse():
        #     self.tiles()
        #
        # graph = graph if graph is not None else cls()
        # keys = None
        #
        # if tiled:
        #     nodes = list(c.data for c in self.chunks)
        #     keys = list(c.key for c in self.chunks)
        # else:
        #     nodes = list(self.op.outputs)
        #
        # node_to_fetch = dict()
        #
        # def _generate_fetch_node(n):
        #     if n in node_to_fetch:
        #         return node_to_fetch[n]
        #     fn = build_fetch(n).data
        #     node_to_fetch[n] = fn
        #     return fn
        #
        # visited = set()
        # while len(nodes) > 0:
        #     node = nodes.pop()
        #
        #     # replace executed tensor/chunk by tensor/chunk with fetch op
        #     if node.key in executed_keys:
        #         node = _generate_fetch_node(node)
        #
        #     visited.add(node)
        #     if not graph.contains(node):
        #         graph.add_node(node)
        #     children = node.inputs or []
        #     for c in children:
        #         if c.key in executed_keys:
        #             visited.add(c)
        #             c = _generate_fetch_node(c)
        #         if not graph.contains(c):
        #             graph.add_node(c)
        #         if not graph.has_successor(c, node):
        #             graph.add_edge(c, node)
        #     nodes.extend([c for c in itertools.chain(*[inp.op.outputs for inp in node.inputs or []])
        #                   if c not in visited])
        # if tiled and compose:
        #     graph.compose(keys=keys)
        #
        # if not tiled and any(not n.is_coarse() for n in graph):
        #     return self._to_coarse_graph(graph)
        #
        # return graph

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
        if graph is not None:
            self._graph = graph
        else:
            self._graph = graph_cls()
        self._node_processor = node_processor

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
            children = node.inputs or []
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

    @enter_build_mode
    def build_tileable_data_graph(self, tileables):
        visited = set()
        nodes = list(itertools.chain(
            *(tileable.op.outputs for tileable in tileables)))
        self._add_nodes(nodes, visited)
        return self._graph

    def _tile(self, tileable_data, on_tile):
        cache = _tileable_data_to_tiled

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
        for t, td in zip(tileable_data.op.outputs, tds):
            cache[t] = td.data
        return tds

    @enter_build_mode
    def build_chunk_graph(self, tileables, compose=False, tileable_graph=None,
                          on_tile=None, on_tile_success=None, on_tile_failure=None):
        if tileable_graph is None:
            # if tileable_data graph not provided
            # create a new one via GraphBuilder
            tileable_graph_builder = GraphBuilder(
                graph_cls=type(self._graph),
                node_processor=self._node_processor)
            tileable_graph = tileable_graph_builder.build_tileable_data_graph(tileables)

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
                tiled = self._tile(tileable_data, on_tile)
                tiled_op.add(tileable_data.op)
                for t, td in zip(tileable_data.op.outputs, tiled):
                    if on_tile_success is not None:
                        td = on_tile_success(t, td)
                    nodes.extend(c.data for c in td.chunks)
                    if t in tileables_set:
                        keys.extend(c.key for c in td.chunks)
                    self._add_nodes(nodes, visited)
            except:  # noqa: E722
                exc_info = sys.exc_info()
                if on_tile_failure:
                    on_tile_failure(exc_info)
                else:
                    raise
        if compose:
            self._graph.compose(keys=keys)
        return self._graph

    def build(self, tileables, tiled=False, compose=True, **build_chunk_graph_kwargs):
        if not tiled:
            return self.build_tileable_data_graph(tileables)
        else:
            return self.build_chunk_graph(tileables, compose=compose,
                                          **build_chunk_graph_kwargs)
