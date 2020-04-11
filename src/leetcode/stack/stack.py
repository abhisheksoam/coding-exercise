class Solution:
    """
    https://leetcode.com/problems/valid-parentheses/
    """

    def isValid(self, s: str) -> bool:
        stack = []

        for char in s:
            try:
                top = stack[-1]
            except Exception as error:
                stack.append(char)
                continue

            if char == ")" and top == "(" or char == "(" and char == ")":
                stack.pop()

            elif char == "}" and top == "{" or char == "{" and char == "}":
                stack.pop()
            elif char == "]" and top == "[" or char == "[" and char == "]":
                stack.pop()

            else:
                stack.append(char)

        if stack:
            return False

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
        self.smallest = None

    def push(self, x: int) -> None:
        self.list.insert(0, x)
        if self.smallest is None:
            self.smallest = x

        if x < self.smallest:
            self.smallest = x

    def pop(self) -> None:
        value = self.list.pop(0)
        if value == self.smallest:
            self._set_min()

    def _set_min(self):
        if self.list:
            self.smallest = min(self.list)
        else:
            self.smallest = None

    def top(self) -> int:
        return self.list[0]

    def getMin(self) -> int:
        return self.smallest


"""
https://leetcode.com/problems/lru-cache/
"""


# TODO
class LRUCache:
    def __init__(self, capacity: int):
        pass

    def get(self, key: int) -> int:
        pass

    def put(self, key: int, value: int) -> None:
        pass
