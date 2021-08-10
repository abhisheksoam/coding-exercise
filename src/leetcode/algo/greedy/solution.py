from typing import List


class Solution:
    """
    https://leetcode.com/problems/gas-station/
    """

    # TODO:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        gas_len = len(gas)
        result = -1
        remaining = 0
        solution = 0
        for i in range(0, gas_len):
            gas_i = gas[i]
            cost_i = cost[i]

            if gas_i + remaining > cost_i:
                solution = i
            else:
                remaining += cost_i

        return result


s = Solution()
s.canCompleteCircuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])
