class Node:
    def __init__(self, value):
        self.value = value


class Edge:
    def __init__(self, node_a, node_b, distance=None, bi_directional=False):
        self.node_a = node_a
        self.node_b = node_b
        self.distance = distance
        self.bidirectional = bi_directional


class Graph:
    def __init__(self):
        self.nodes = {}
        self.root = None

    def create_root(self, value):
        node = Node(value)
        self.nodes[node] = []
        self.root = node

    def get_root(self):
        return self.root

    def add_node(self, value, neighbour, distance=None):
        node = Node(value)
        edge = Edge(node, neighbour, distance, bi_directional=True)
        self.nodes[node] = [edge]
        self.nodes[neighbour].append(edge)

    def get_nodes(self):
        return self.nodes.keys()

    def get_edges(self):
        output = {}
        for node, edges in self.nodes.items():
            for edge in edges:
                if edge not in output:
                    output[edges] = True

        return list(output.keys())

    def dfs_traversal(self):
        visited, stack = set(), [self.root]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                for edge in self.nodes[vertex]:
                    stack.append(edge.node_b)

        return visited

    def bfs_traversal(self):
        visited, stack = set(), [self.root, None]
        while stack:
            vertex = stack.pop(0)
            if vertex in visited:
                continue

            if vertex is None:
                if not stack:
                    break
                else:
                    stack.append(None)
            else:
                visited.add(vertex)
                for edge in self.nodes[vertex]:
                    stack.append(edge.node_b)

        return visited

    def print_graph(self):
        for node in self.nodes:
            print("Node: ", node.value, end="\n")
            print("Edges:")
            for edge in self.nodes[node]:
                print(edge.node_a.value, "->", edge.node_b.value, end="\n")

            print("\n")


g = Graph()
g.create_root(5)
print(g.get_nodes())
g.add_node(4, g.get_root())
g.print_graph()

print(g.dfs_traversal())
print(g.bfs_traversal())
