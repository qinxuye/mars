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

from .graph import DAG
from .utils import kernel_mode


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
    def build_graph(self, graph=None, cls=DAG, tiled=False, compose=True, executed_keys=None):
        from .utils import build_fetch

        executed_keys = set(executed_keys or [])
        if tiled and self.is_coarse():
            self.tiles()

        graph = graph if graph is not None else cls()
        keys = None

        if tiled:
            nodes = list(c.data for c in self.chunks)
            keys = list(c.key for c in self.chunks)
        else:
            nodes = list(self.op.outputs)

        node_to_fetch = dict()

        def _generate_fetch_node(n):
            if n in node_to_fetch:
                return node_to_fetch[n]
            fn = build_fetch(n).data
            node_to_fetch[n] = fn
            return fn

        visited = set()
        while len(nodes) > 0:
            node = nodes.pop()

            # replace executed tensor/chunk by tensor/chunk with fetch op
            if node.key in executed_keys:
                node = _generate_fetch_node(node)

            visited.add(node)
            if not graph.contains(node):
                graph.add_node(node)
            children = node.inputs or []
            for c in children:
                if c.key in executed_keys:
                    visited.add(c)
                    c = _generate_fetch_node(c)
                if not graph.contains(c):
                    graph.add_node(c)
                if not graph.has_successor(c, node):
                    graph.add_edge(c, node)
            nodes.extend([c for c in itertools.chain(*[inp.op.outputs for inp in node.inputs or []])
                          if c not in visited])
        if tiled and compose:
            graph.compose(keys=keys)

        if not tiled and any(not n.is_coarse() for n in graph):
            return self._to_coarse_graph(graph)

        return graph

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

        return tiles_obj


handler = OperandTilesHandler()
register = OperandTilesHandler.register


# class GraphBuilder(object):
#     def __init__(self, graph=None, graph_cls=DAG):
#         if graph is not None:
#             self._graph = graph
#         else:
#             self._graph = graph_cls()
#
#     def build_tileable_graph(self):
#         pass
#
#     def build_chunk_graph(self):
#         pass
