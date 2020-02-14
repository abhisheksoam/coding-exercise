from typing import List


class Solution:
    """
    https://leetcode.com/problems/permutations/
    """

    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []

        def helper(list, start, end):
            if start == end:
                res.append(list.copy())
            else:
                for i in range(start, end + 1):
                    list[start], list[i] = list[i], list[start]  # The swapping
                    helper(list, start + 1, end)
                    list[start], list[i] = list[i], list[start]  # Backtracking

        helper(nums, 0, len(nums) - 1)
        return res

    """
    https://leetcode.com/problems/combination-sum/
    """

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(list, target, tmp=[]):
            if target == 0:
                res.append(tmp.copy())

            if target < 0:
                return

            tmp = []
            for i in range(0, len(list)):
                element = list[i]
                tmp.append(element)
                backtrack(list, target - element, tmp)
                tmp.remove(element)

        backtrack(candidates, target)
        return res

    """
    https://leetcode.com/problems/combination-sum-ii/
    """
    # TODO:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        dict = {}

        def backtrack(list, target, start, tmp=[]):

            if target < 0:
                return

            if target == 0 and dict.get(tuple(sorted(tmp))) is None:
                dict[tuple(sorted(tmp))] = True

            for i in range(start, len(list)):
                if i > start and list[i] == list[i - 1]:
                    continue

                element = list[i]
                tmp.append(element)
                backtrack(list, target - element, start + 1, tmp)
                del tmp[-1]

        backtrack(sorted(candidates), target, 0)
        return [list(_) for _ in dict.keys()]


s = Solution()
# print(s.permute([1, 0, 2]))
# print(s.combinationSum([2, 3, 5], 8))
print(s.combinationSum2([1, 2], 4))
