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

    """
    https://leetcode.com/problems/power-of-three/
    """

    def isPowerOfThree(self, n: int) -> bool:
        def helper(n):
            if n == 1:
                return True
            elif n < 1:
                return False

            return helper(n / 3)

        return helper(n)

    """
    https://leetcode.com/problems/count-primes/
    Approach: 
    """

    def countPrimes(self, n: int) -> int:
        output = set()

        for i in range(2, int(n ** 0.5) + 1):
            integer = i
            counter = 2
            value = counter * integer
            while value <= n:
                output.add(value)
                counter = counter + 1
                value = counter * integer

        count = 0

        for i in range(2, n):
            if i not in output:
                count = count + 1

        return count

    """
    https://leetcode.com/problems/factorial-trailing-zeroes/
    """

    def trailingZeroes(self, n: int) -> int:
        zero_count = 0
        while n / 5 >= 5:
            zero_count = zero_count + n // 5
            n = n // 5

        return zero_count


s = Solution()
# print(s.divide(-2147483648, -1))
print(s.countPrimes(10))
