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

    def is_candidate_absent(self, output, solution):
        for solution_no, candidate in output.items():
            valA, valB, valC = solution[0], solution[1], solution[2]
            if candidate.get(valA) and candidate.get(valB) and candidate.get(valC):
                return False

        return True

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
    https://leetcode.com/problems/complement-of-base-10-integer/
    """

    def findComplement(self, num: int) -> int:
        return num >> 1


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
        self.lru = []
        self.hashmap = {}
        self.operations = 0

    def get(self, key: int) -> int:
        index = self.hashmap.get(key)
        if self.operations - (index + 1) < self.capacity:
            value = self.lru[index][1]
            self.put(key, value)
            return value
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        self.lru.append((key, value))
        self.hashmap[key] = self.operations
        self.operations += 1


cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
print(cache.get(1))
cache.put(3, 3)
print(cache.get(2))
cache.put(4, 4)
print((cache.get(1)))
cache.get(3)
cache.get(4)

"""
https://leetcode.com/problems/add-and-search-word-data-structure-design/
"""


# TODO:
class TrieNode:
    def __init__(self, character, word=False):
        self.character = character
        self.children = {}
        self.word = word


class WordDictionary:
    def __init__(self):
        """
        Initialize your data structure here.
       """
        self.root = TrieNode("0", word=False)

    def insert(self, word, root):
        for index, char in enumerate(word):
            children = root.children
            if children.get(char):
                root = root.children.get(char)
            else:
                node = TrieNode(char)
                root.children[char] = node
                root = node

        root.word = True

    def addWord(self, word: str) -> None:
        """
        Adds a word into the data structure.
        """
        self.insert(word, self.root)

    def _search(self, word, root):
        if not word:
            if root.word:
                return True
            return False

        char = word[0]
        if char == ".":
            for node in root.children.values():
                return self._search(word[1:], node)
        else:
            node = root.children.get(char)
            if not node:
                return False
            return self._search(word[1:], node)

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        """
        return self._search(word, self.root)


# obj = WordDictionary()
# obj.addWord("at")
# obj.addWord("and")
# obj.addWord("an")
# obj.addWord("add")
# obj.addWord("bat")
# print(obj.search("a.d"))

"""
https://leetcode.com/problems/range-sum-query-2d-immutable/
"""


class NumMatrix:
    def __init__(self, matrix: List[List[int]]):
        self.matrix = matrix

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        sum = 0
        for i in range(row1, row2 + 1):
            for j in range(col1, col2 + 1):
                sum = sum + self.matrix[i][j]

        return sum


# Your NumMatrix object will be instantiated and called as such:
matrix = [
    [3, 0, 1, 4, 2],
    [5, 6, 3, 2, 1],
    [1, 2, 0, 1, 5],
    [4, 1, 0, 1, 7],
    [1, 0, 3, 0, 5],
]

obj = NumMatrix(matrix)
param_1 = obj.sumRegion(2, 1, 4, 3)
print(param_1)
