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
    https://leetcode.com/problems/permutations-ii/    
    """

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """
        Approach: Backtracking, breaking the call if finding the same element
        at previous position with the exception of handling the first starting position.
        Selecting the candidate solution when start,end index are equal.

        Using set to handle the duplicate response
        """
        result = set()

        def helper(list, start, end):
            if start == end:
                result.add(tuple(list))
                return

            for i in range(start, end + 1):
                if i != start and list[start] == list[i]:
                    continue

                list[start], list[i] = list[i], list[start]
                helper(list, start + 1, end)
                list[start], list[i] = list[i], list[start]

        helper(nums, 0, len(nums) - 1)
        return [list(_) for _ in result]

    """
    https://leetcode.com/problems/combination-sum-ii/
    """

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        result = set()

        def backtrack(list, target, start, end, tmp=[]):
            if target == 0:
                result.add(tuple(tmp))
                return

            if target < 0 or start > end:
                return

            for i in range(start, end):
                element = list[i]
                tmp.append(element)
                backtrack(list, target - element, i + 1, end, tmp)
                del tmp[-1]

        backtrack(sorted(candidates), target, 0, len(candidates))
        return [list(_) for _ in result]


s = Solution()
# print(s.permute([1, 0, 2]))
# print(s.combinationSum([2, 3, 5], 8))
# print(s.combinationSum2([], 0))
# print(s.permuteUnique([2, 2, 1, 1]))
