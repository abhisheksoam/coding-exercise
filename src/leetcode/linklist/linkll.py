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

    def reverseList(self, head: ListNode) -> ListNode:
        def helper(node):
            if node.next is None:
                return node

            node.next = helper(node.next)
            return head

        return helper(head)

    """
    Print linklist
    """

    def print(self, root):
        while root is not None:
            print(root.val, end="->")
            root = root.next
        print("\n")

    """
    https://leetcode.com/problems/delete-node-in-a-linked-list/
    """

    # TODO:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """

    def middleNode(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        while fast.next is not None:
            for i in range(0, 2):
                if fast.next is not None:
                    fast = fast.next

            if slow.next is not None:
                slow = slow.next

        return slow

    """
    https://leetcode.com/problems/intersection-of-two-linked-lists/
    :solution_seen False
    :time_complexity O(N)
    :space_complexity O(N)
    
    
    """

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        node = headA
        map = {}
        while node is not None:
            map[id(node)] = node
            node = node.next

        node = headB
        while node is not None:
            node_id = id(node)
            if map.get(node_id):
                return map.get(node_id)

            node = node.next

        return None

    """
    https://leetcode.com/problems/swap-nodes-in-pairs/        
    :category medium
    :space_complexity O(1)
    :time_complexity O(n)
    
    Example
    s = Solution()
    s.print(head)
    print("\n")
    output = s.swapPairs(head)
    s.print(output)
    """

    def swapPairs(self, head: ListNode) -> ListNode:
        def helper(head):
            if not head:
                return None

            first = head
            second = head.next
            if second is None:
                return first

            first.next = second.next
            second.next = first

            first.next = helper(first.next)

            return second

        return helper(head)


head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)
# head.next.next.next.next.next = ListNode(6)


# s = Solution()
# s.print(head)
# head = s.reverseList(head)
# s.print(head)
