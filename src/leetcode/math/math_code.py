class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        max = 2147483648 - 1
        min = -2147483648

        negative = False
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
            negative = True

        def helper(dividend, divisor, quotient):
            if dividend < divisor:
                return quotient
            return helper(dividend - divisor, divisor, quotient + 1)

        import math

        if divisor == 1 or divisor == -1:
            result = int(math.fabs(dividend))
        else:
            result = helper(math.fabs(dividend), math.fabs(divisor), 0)

        if negative:
            result = -int(result)

        if result > max:
            result = max

        if result < min:
            result = min

        return result

    """
    https://leetcode.com/problems/powx-n/
    """

    def myPow(self, x: float, n: int) -> float:
        def divide_conq(input):
            if not input:
                return []

            size = len(input)
            l = 0

        if n == 0:
            return 1

        bottoms_up = n < 0
        negative = x < 0 and n % 2 != 0

        x = abs(x)
        n = abs(n)

        input = [x] * n

        # if bottoms_up:
        #     result = 1 / result
        #
        # if negative:
        #     result = - result
        #
        # return result

    """
    https://leetcode.com/problems/excel-sheet-column-number/
    """

    def titleToNumber(self, s: str) -> int:
        pass

    """
    https://leetcode.com/problems/sqrtx/
    """

    # TODO:
    """
    DP
    https://leetcode.com/problems/coin-change/
    """


s = Solution()
print(s.divide(-2147483648, -1))
