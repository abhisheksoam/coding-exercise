# Definition for singly-linked list.
from typing import List


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # Reverse the LL
        previous = head
        current = head.next
        while current.next is not None:
            previous = current

        # Delete simply

    def print_ll(self, head: ListNode):
        while head is not None:
            print(head.val, '->', end=' ')
            head = head.next

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = None
        last = None
        while l1 is not None and l2 is not None:
            if l1.val >= l2.val:
                new_node = ListNode(l2.val)
                l2 = l2.next
            else:
                new_node = ListNode(l1.val)
                l1 = l1.next

            if head:
                last.next = new_node
                last = new_node
            else:
                head = new_node
                last = head

        while l1 is not None:
            new_node = ListNode(l1.val)
            l1 = l1.next
            if head:
                last.next = new_node
                last = new_node
            else:
                head = new_node
                last = head

        while l2 is not None:
            new_node = ListNode(l2.val)
            l2 = l2.next
            if head:
                last.next = new_node
                last = new_node
            else:
                head = new_node
                last = head

        return head

    def swapPairs(self, head: ListNode) -> ListNode:
        if not head:
            return head

        node1 = head
        if node1.next is None:
            return head
        else:
            node2 = head.next

    def is_candidate_absent(self, output, solution):
        for solution_no, candidate in output.items():
            valA, valB, valC = solution[0], solution[1], solution[2]
            if candidate.get(valA) and candidate.get(valB) and candidate.get(valC):
                return False

        return True

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        from collections import defaultdict
        process = {}
        output = defaultdict(dict)
        index = 0
        size = len(nums)
        for value in nums:
            process[value] = process.get(value, 0) + 1

        for i in range(0, size):
            for j in range(i + 1, size):
                valA, valB = nums[i], nums[j]
                valC = - (valA + valB)
                if process.get(valC):
                    if self.is_candidate_absent(output, [valA, valB, valC]):
                        output[index] = {
                            valA: True,
                            valB: True,
                            valC: True
                        }
                        index = index + 1
        l = []
        for key, value in output.items():
            p = []
            for key in value.keys():
                p.append(key)
            l.append(p)
        return l

    # def myAtoi(self, str: str) -> int:
    #     size = len(str)
    #     str.lstrip()
    #     start = 0
    #     digits_found = None
    #     optional_sign = None
    #     digits_sequence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #     while start < size:
    #         current_char = str[start]
    #         if not optional_sign and :
    #         # if not digits_found

    def convert(self, s: str, numRows: int) -> str:
        size = len(str)
        processing_array = [[''] * size] * numRows



s = Solution()
nums = [-1, 0, 1, 2, -1, -4]
print(s.threeSum(nums))

# node = ListNode(1)
# node.next = ListNode(2)
# node.next.next = ListNode(3)
# node.next.next.next = ListNode(4)
# node.next.next.next.next = ListNode(5)

# s.print_ll(node)
# print('\n')
# s.print_ll(s.removeNthFromEnd(node, 1))
