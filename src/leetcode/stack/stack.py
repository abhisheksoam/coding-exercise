class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for char in s:
            stack.append(char)

        previous = None
        while stack:
            current = stack.pop()
            if previous:
                if previous != ")" and current != "(":
                    return False
                elif previous != "}" and current != "{":
                    return False
                elif previous != "]" and current != "[":
                    return False

            previous = current

        return True


"""
https://leetcode.com/problems/min-stack/
"""


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.list = []

    def push(self, x: int) -> None:
        self.list.insert(0, x)
        if x < self.min:
            self.min = x

    def pop(self) -> None:
        self.list.pop(0)

    def top(self) -> int:
        return self.list[0]

    def getMin(self) -> int:
        min = float("inf")
        for a in self.list:
            if a < min:
                min = a

        return min


"""
https://leetcode.com/problems/lru-cache/
"""

class LRUCache:

    def __init__(self, capacity: int):

    def get(self, key: int) -> int:

    def put(self, key: int, value: int) -> None:


