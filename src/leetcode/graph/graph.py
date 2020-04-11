import collections


class Node:
    def __init__(self, val=0, neighbors=[]):
        self.val = val
        self.neighbors = neighbors


class Solution:
    """
    https://leetcode.com/problems/clone-graph/
    """

    # TODO:
    def cloneGraph(self, node: "Node") -> "Node":
        if not node:
            return None

        queue = [node, None]
        root = None
        while queue:
            current_node = queue.pop(0)
            if current_node is None:
                if queue:
                    queue.append(None)
                else:
                    break
            else:
                new_node = Node(current_node.val)
                if root is None:
                    root = new_node

                for _ in current_node.neighbors:
                    queue.append(_)
                    neighbours_new_node = Node(_.val)
                    new_node.neighbors.append(neighbours_new_node)

        return root
