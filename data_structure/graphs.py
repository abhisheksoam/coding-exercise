#  BFS and DFS for disconnected graph
"""
Graphs:
Vertices, Edges

Edge - Adjacency matrix

Functions:
bfs, dfs
has path
get path
is connected
return all connected components
weighted and directed graphs

MST (Minimum spanning tree)

Algorithms:
Detect Cycle: Check if the path exist for two node twice.
    Union find algorithm or DSU: Maintain parent matrix
Krushkal: Pick the edge with minimum weight, which doesn't form a cycle in the MST.
Prim
Djikstra
Bellman Ford
Floyd Warshall
"""


class Vertices:
    def __init__(self, id, data=None):
        self.id = id
        self.data = data


class Edge:

    def __init__(self, v1, v2, weight=None):
        self.v1 = v1
        self.v2 = v2
        self.weight = weight


class Graph:

    def __init__(self, vertices_size, edges):
        self.vertices_size = vertices_size
        self.edge_size = edges
        self.last_vertex_id = -1
        self.vertices = {}
        self._build_adjacency_matrix()

    def _build_adjacency_matrix(self):
        self.adjacency_matrix = self.vertices_size * [[0] * self.vertices_size]

    def has_path(self, v1, v2):
        pass

    def add_vertex(self, data=None):
        id = self.last_vertex_id + 1
        _vertex = Vertices(id=id, data=data)
        self.last_vertex_id += 1
        self.vertices[id] = _vertex

    def get_vertex_id(self, vertex):
        pass

    def add_edge(self, v1, v2):
        self.adjacency_matrix[][]

    def get_path(self):
        pass

    def is_connected(self):
        pass

    def get_connected_component(self):
        pass


g = Graph(vertices_size=6)
