from typing import List
from collections import OrderedDict


class Solution:
    ## Jump game 2
    def jump(self, nums: List[int]) -> int:
        size = len(nums)
        dp, i = [-1] * size, size - 2
        dp[size - 1] = 0
        while i >= 0:
            current_value = nums[i]
            minimum_value = float("inf")
            j = i + 1
            while j <= i + current_value and j < size:
                if dp[j] < minimum_value:
                    minimum_value = dp[j]

                j = j + 1

            dp[i] = minimum_value + 1
            i = i - 1
        return dp[0]

    ## Jump game 1
    def canJump(self, nums: List[int]) -> bool:
        size = len(nums)
        dp, i = [-1] * size, size - 2
        dp[size - 1] = 0
        while i >= 0:
            current_value = nums[i]
            minimum_value = float("inf")
            j = i + 1
            while j <= i + current_value and j < size:
                if dp[j] < minimum_value:
                    minimum_value = dp[j]

                j = j + 1

            dp[i] = minimum_value + 1
            i = i - 1

        if dp[0] == -1 or dp[0] == float("inf"):
            return False
        return True

    def roman(self, number):
        symbol_map = OrderedDict()
        symbol_map[1000] = 'M'
        symbol_map[900] = 'CM'
        symbol_map[500] = 'D'
        symbol_map[400] = 'CD'
        symbol_map[100] = 'C'
        symbol_map[90] = 'XC'
        symbol_map[50] = 'L'
        symbol_map[40] = 'XL'
        symbol_map[10] = 'X'
        symbol_map[9] = 'IX'
        symbol_map[5] = 'V'
        symbol_map[4] = 'IV'
        symbol_map[1] = 'I'

        roman = ""
        while number > 0:
            for value, symbol in symbol_map:
                while number >= value:
                    roman = roman + symbol
                    number = number - value
        return roman

    # """Write a function to generate all possible n pairs of balanced parentheses. """
    def generate_paranthesis(self, n):
        dp = {}

        def recursion(n):
            if n == 1:
                return ["{}"]

            output = recursion(n - 1)
            result = []
            for word in output:
                for i in range(0, len(word)):
                    temp = word[0:i] + "{}" + word[i:]
                    if temp not in dp:
                        result.append(temp)
                        dp[temp] = True
            return result

        print(recursion(4))
        # print(list(dp.keys()))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    s = Solution()
    s.generate_paranthesis(3)
    # print(s.roman(720))
    # print(s.jump([3, 2, 1, 0, 4]))
    # print((s.canJump([3, 2, 1, 0, 4])))
