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

    def myPow(self, x: float, n: int) -> float:
        bottoms_up = n < 0
        negative = x < 0 and n % 2 != 0

        result = x
        while n > 1:
            result = result * x
            n = n - 1

        if bottoms_up:
            result = 1 / result

        if negative:
            result = - result

        return result


s = Solution()
print(s.divide(-2147483648, -1))
