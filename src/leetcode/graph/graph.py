import collections
from typing import List


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

    """
    https://leetcode.com/problems/find-the-town-judge/
    :solution_seen True
    :time_complexity O(n)
    :space_complexity O(n)
    :category easy
    :example_case 
    print(s.findJudge(2, [[1, 2]])) 
    """

    def findJudge(self, N: int, trust: List[List[int]]) -> int:
        trusted = [0] * (N + 1)
        for a, b in trust:
            trusted[b] += 1
            trusted[a] -= 1

        check_value = N - 1
        for i in range(1, N + 1):
            if trusted[i] == check_value:
                return i

        return -1


s = Solution()
