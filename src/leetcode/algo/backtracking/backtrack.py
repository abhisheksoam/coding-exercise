import time
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

    """
    https://leetcode.com/problems/subsets/
    """

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """
        Using hashmap to store the output.
        Approach: Add each input element to output and remove the elements from the list one by one and call
        the function on that input.
        """
        dict = {}
        dict[()] = True

        def backtrack(nums):
            if dict.get(tuple(nums)) is None:
                dict[tuple(nums)] = True

            if len(nums) == 1 or not nums:
                return

            for i in range(0, len(nums)):
                new_array = nums[0:i] + nums[i + 1:]
                if new_array:
                    backtrack(new_array)

        backtrack(nums)
        return [list(_) for _ in dict.keys()]

    # Version 2
    def subsets_v2(self, nums: List[int]) -> List[List[int]]:
        """
            Using set to store the output.
            Approach: Add each input element to output and remove the elements from the list one by one and call
            the function on that input.
            Comments: Almost same Time complexity as the Version 1
        """
        res = set()
        res.add(tuple([]))

        def backtrack(nums):
            res.add(tuple(nums))
            if len(nums) == 1 or not nums:
                return

            for i in range(0, len(nums)):
                new_array = nums[0:i] + nums[i + 1:]
                if new_array:
                    backtrack(new_array)

        backtrack(nums)
        return [list(_) for _ in res]

    # Version 3
    def subsets_v3(self, nums: List[int]) -> List[List[int]]:
        """
            Approach: Prepare a storing DS with a empty subset,
            Start iterating over result structure and start appending each element of nums of output element into it
            one by one.
            Time complexity: O(N * 2^N)
            Space complexity: O(2^N)
            Comments: This version has an awesome performance in comparison to the other two versions written above.
        """
        res = [[]]

        def helper(nums, index):
            if index > len(nums) - 1:
                return

            output = []
            for element in res:
                tmp = element.copy()
                tmp.append(nums[index])
                output.append(tmp)

            res.extend(output)
            del output
            helper(nums, index + 1)

        helper(nums, 0)
        return res


s = Solution()
# print(s.permute([1, 0, 2]))
# print(s.combinationSum([2, 3, 5], 8))
# print(s.combinationSum2([1, 2], 4))


# start_time = time.time()
# output = s.subsets_v2([1, 2, 3, 4, 5, 6, 7, 8, 10, 0])
# end_time = time.time()
# print(len(output), end_time - start_time, output)
#
# start_time = time.time()
# output = s.subsets([1, 2, 3, 4, 5, 6, 7, 8, 10, 0])
# end_time = time.time()
# print(len(output), end_time - start_time, output)

start_time = time.time()
output = s.subsets_v3([1, 2, 3, 4, 5, 6, 7, 8, 10, 0])
end_time = time.time()
print(len(output), end_time - start_time, output)
# print(s.combinationSum2([], 0))
# print(s.permuteUnique([2, 2, 1, 1]))
