from typing import List


class Solution:
    def firstUniqChar(self, s: str) -> int:
        process = {}
        for char in s:
            process[char] = process.get(char, 0) + 1

        for index, char in enumerate(s):
            if process.get(char) == 1:
                return index

        return -1

    def frequencySort(self, s: str) -> str:
        pass

    """
    
    """

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 0:
            return ""

        indexing_position = 0
        char_at_indexing_position = None
        looping = True
        common_prefix = ""
        while looping:
            for word in strs:
                try:
                    if char_at_indexing_position is None:
                        char_at_indexing_position = word[indexing_position]
                        continue
                    else:
                        if word[indexing_position] != char_at_indexing_position:
                            looping = False

                except IndexError:
                    looping = False

            if looping:
                indexing_position = indexing_position + 1
                common_prefix = common_prefix + char_at_indexing_position
                char_at_indexing_position = None

        return common_prefix

    """
    https://leetcode.com/problems/zigzag-conversion/
    """

    def convert(self, s: str, numRows: int) -> str:
        size = len(s)
        process = size * [[None] * size]

        zigzag = False
        index = 0
        diagonal_limit = numRows - 2
        diagonal_count, rows = 0, 0
        for i in range(0, size):
            for j in range(0, size):
                if rows == numRows:
                    zigzag = True

                if diagonal_limit == diagonal_count:
                    zigzag = False

                try:
                    if zigzag:
                        process[j - 1][i - 1] = s[index]
                        diagonal_count = diagonal_count + 1
                    else:
                        process[j][i] = s[index]
                        rows = rows + 1
                except Exception as e:
                    break

                index = index + 1
        res = ""
        for i in range(0, size):
            for j in range(0, size):
                char = process[i][j]
                if process[i][j] is not None:
                    res = res + char

        print(res)
        return res

    def multiply(self, num1: str, num2: str) -> str:
        phase = 0
        for lower in reversed(num2):
            carry_forward = 0
            sum = 0
            for upper in reversed(num2):
                product = int(lower) * int(upper)
                if carry_forward:
                    product = carry_forward + product

                if product >= 10:
                    pass

    """
    https://leetcode.com/problems/basic-calculator-ii/
    """

    def calculate(self, s: str) -> int:
        operations = ["+", "-", "*", "/"]
        stack = s
        operator = None
        res = ""

s = Solution()
s.convert("PAYPALISHIRING", 3)
