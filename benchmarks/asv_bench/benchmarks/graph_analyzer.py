# Copyright 1999-2022 Alibaba Group Holding Ltd.
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

import mars.tensor as mt
import mars.dataframe as md
from mars.config import options
from mars.core.graph.builder.utils import build_graph, ChunkGraph
from mars.resource import Resource
from mars.services.task import Task
from mars.services.task.analyzer import GraphAnalyzer


class ChunkGraphAnalyzerSuite:
    """
    Benchmark that times performance of chunk graph assigner
    """

    def setup(self):
        self.df = md.DataFrame(
            mt.random.rand(1000, 10, chunk_size=(1, 10)), columns=list("abcdefghij")
        )
        self.band_resource = {
            (f"worker-{i}", "numa-0"): Resource(num_cpus=16) for i in range(50)
        }
        self._gen_info()

    def _gen_info(self):
        df2 = self.df.copy()
        df2["k"] = df2["c"]
        df2["l"] = df2["a"] * (1 - df2["d"])
        df2["m"] = df2["e"] * (1 + df2["d"]) * (1 - df2["h"])
        self.df2 = df2
        self.graph = ChunkGraph([])
        g = build_graph([df2], tile=True)
        g.copyto(self.graph)
        self.graph.results.extend(g.iter_indep(True))

    def time_analyze(self):
        analyzer = GraphAnalyzer(
            self.graph, self.band_resource, Task(), options, dict()
        )
        subtask_graph = analyzer.gen_subtask_graph()
        assert len(subtask_graph) > 1
