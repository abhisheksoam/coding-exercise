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
            print(head.val, "->", end=" ")
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
                valC = -(valA + valB)
                if process.get(valC):
                    if self.is_candidate_absent(output, [valA, valB, valC]):
                        output[index] = {valA: True, valB: True, valC: True}
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
        processing_array = [[""] * size] * numRows

    """
    https://leetcode.com/problems/contains-duplicate/
    """

    def containsDuplicate(self, nums: List[int]) -> bool:
        d = {}
        for value in nums:
            count = d.get(value, 0)
            if count == 0:
                d[value] = 1
            else:
                return True

        del d
        return False

    """
    https://leetcode.com/problems/contains-duplicate-ii/
    """

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        for index, value in nums:
            try:
                if nums[index + 1 + k] == value:
                    return True
            except:
                pass

        return False

    def hammingWeight(self, n: int) -> int:
        count = 0
        for digit in str(bin(n)):
            if digit == "1":
                count = count + 1
        return count

    """
    https://leetcode.com/problems/pascals-triangle/
    """

    def generate(self, numRows: int) -> List[List[int]]:
        res = []
        previous = []
        for i in range(1, numRows + 1):
            current = [0] * i
            current[0] = 1
            current[-1] = 1
            if not previous:
                res.append(current)
                previous = current
            else:
                sum = []
                for i in range(0, len(previous)):
                    try:
                        sum.append(previous[i] + previous[i + 1])
                    except:
                        pass
                index = 0
                for i in range(1, len(current) - 1):
                    current[i] = sum[index]
                    index += 1

                res.append(current)
                previous = current
        return res

    """
    https://leetcode.com/problems/pascals-triangle-ii/
    """

    def getRow(self, rowIndex: int) -> List[int]:
        previous = []
        for i in range(0, rowIndex + 2):
            if not previous:
                previous = [1]
            else:
                current = [0] * i
                current[0] = 1
                current[-1] = 1

                sum = []
                for i in range(0, len(previous)):
                    try:
                        sum.append(previous[i] + previous[i + 1])
                    except:
                        pass
                index = 0
                for i in range(1, len(current) - 1):
                    current[i] = sum[index]
                    index += 1

                previous = current

        return previous

    """
    https://leetcode.com/problems/bulb-switcher/
    """

    def bulbSwitch(self, n: int) -> int:
        import math

        return math.sqrt(n)

    """
    https://leetcode.com/problems/reverse-bits/
    """

    # TODO:
    def reverseBits(self, n: int) -> int:
        pass


"""
https://leetcode.com/problems/find-median-from-data-stream/
"""
# TODO:
import heapq


class MedianFinder:
    def __init__(self):
        """
        initialize your data structure here.
        """
        # self.list = list()
        self.heap = []

    # def merge_list(self, input):
    #     output = []
    #     i, j = 0, 0
    #     size = len(self.list)
    #     size1 = len(input)
    #     while i < size and j < size1:
    #
    #         if self.list[i] < input[j]:
    #             output.append(self.list[i])
    #             i += 1
    #         else:
    #             output.append(input[j])
    #             j += 1
    #
    #     while i < size:
    #         output.append(self.list[i])
    #         i += 1
    #
    #     while j < size1:
    #         output.append(input[j])
    #         j += 1
    #
    #     self.list = output

    def addNum(self, num: int) -> None:
        # Add a number to array and merge it with the list
        heapq.heappush(self.heap, num)

    def findMedian(self) -> float:
        arr = heapq.nsmallest(len(self.heap), self.heap)
        size = len(arr)
        if size % 2 == 0:
            median_index = size // 2
            return (arr[median_index] + arr[median_index - 1]) / 2

        else:
            return arr[(size // 2)]


"""
https://leetcode.com/problems/lru-cache/
"""


# TODO:
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity

    def get(self, key: int) -> int:
        pass

    def put(self, key: int, value: int) -> None:
        pass
