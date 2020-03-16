"""
Problem Statement:
Let us say that you are given a number N,
you've to find the number of different ways to write it as the sum of 1, 3 and 4.
# Recurrence Relation
F(n) = F(n-1) + F(n-3) + F(n-4)
"""


class CountWaysSummingUp:
    def recursion(self, n, count=0):
        if n == 0:
            return count + 1
        elif n < 0:
            return count
        else:
            return (
                self.count_ways_summing_up(n - 1, count)
                + self.count_ways_summing_up(n - 3, count)
                + self.count_ways_summing_up(n - 4, count)
            )

    def dp(self, n):
        pass
