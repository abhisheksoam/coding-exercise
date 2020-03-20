"""
https://leetcode.com/problems/copy-list-with-random-pointer/
"""


class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random


# TODO:
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        def deep_copy(node, prev):
            if not node:
                return None

            if prev == node:
                return None

            n1 = Node(node.val)
            n1.next = deep_copy(node.next, node)
            n1.random = deep_copy(node.random, node)
            return n1

        new_head = deep_copy(head, None)

        return new_head



s = Solution()
head = Node(1)
head.next = Node(2)
head.random = head.next
head.next.random = head.next
s.copyRandomList(head)
