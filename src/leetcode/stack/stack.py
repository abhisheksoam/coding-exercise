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
