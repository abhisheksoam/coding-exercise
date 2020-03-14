# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Node:
    def __init__(self, x: int, next: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next
        self.random = random


class Solution:
    """
    https://leetcode.com/problems/odd-even-linked-list/
    """

    def oddEvenList(self, head: ListNode) -> ListNode:
        pass

    """
    # Definition for a Node.
    class Node:
        def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
            self.val = int(x)
            self.next = next
            self.random = random
    """

    def copyRandomList(self, head: "Node") -> "Node":
        pass

    """
    https://leetcode.com/problems/palindrome-linked-list/
    """
    # TODO:

    """
    https://leetcode.com/problems/reverse-linked-list/
    """

    # TODO:
    def reverseList(self, head: ListNode) -> ListNode:
        def helper(node):
            if not node:
                return None

            current = node
            next_node = helper(node.next)
            next_node.next = current
            return next_node

        return helper(head)

s = Solution()
node = ListNode(4)
node.next = ListNode(5)
node.next.next = ListNode(1)
node.next.next.next = ListNode(9)

s.deleteNode(node=ListNode(5))
