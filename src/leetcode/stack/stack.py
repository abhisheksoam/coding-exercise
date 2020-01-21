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
