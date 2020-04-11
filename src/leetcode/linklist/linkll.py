# Definition for singly-linked list.
class ListNode:
    def __init__(self, x, input_data=[]):
        self.val = x
        self.next = None
        self.input_data = input_data

    def create(self):
        head = None
        current = head
        for value in self.input_data:
            if head is None:
                head = ListNode(value)
                current = head

            current.next = ListNode(value)
            current = current.next

        return head


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

    """
    https://leetcode.com/problems/delete-node-in-a-linked-list/
    """

    # TODO:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
