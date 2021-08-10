from typing import List


class Solution:
    """
    https://leetcode.com/problems/rotting-oranges/
    """

    def orangesRotting(self, grid: List[List[int]]) -> int:
        res = -1
        if not grid:
            return res


i = [[2, 1, 1], [1, 1, 0], [0, 1, 1]]
s = Solution()
s.orangesRotting(i)
