from typing import List


class Solution:
    """
    https://leetcode.com/problems/move-zeroes/
    """

    def moveZeroes(self, nums: List[int]) -> None:
        left = 0
        right = len(nums) - 1
        while right >= left:
            if nums[left] == 0 and nums[right] != 0:
                nums[left] = nums[right]
                nums[right] = 0
                right -= 1
                left += 1
            elif nums[left] == 0 and nums[right] == 0:
                right -= 1
            else:
                left += 1
                right -= 1

        print(nums)

    """
    https://leetcode.com/problems/4sum-ii/
    """
    # TODO:
    def fourSumCount(
        self, A: List[int], B: List[int], C: List[int], D: List[int]
    ) -> int:
        pass

    """
    https://leetcode.com/problems/find-the-duplicate-number/
    """

    def findDuplicate(self, nums: List[int]) -> int:
        pass

    """
    https://leetcode.com/problems/count-of-smaller-numbers-after-self/
    """

    def countSmaller(self, nums: List[int]) -> List[int]:
        def binary_search(input, target, l, r):
            if l > r:
                return -1
            else:
                m = l + (r - l) // 2

                if input[m] == target:
                    index = m - 1
                    while index >= l and input[index] == target:
                        index = index - 1

                    index = index + 1
                    return index

                elif target > input[m]:
                    return binary_search(input, target, m + 1, r)
                elif target < input[m]:
                    return binary_search(input, target, l, m - 1)

        size = len(nums)
        res = [0] * (size)
        # Sort the list and remove the duplicates
        sorted_list = sorted(nums)
        for i in range(0, size):
            element = nums[i]
            index = binary_search(sorted_list, element, 0, len(sorted_list) - 1)
            if index != -1:
                res[i] = index

            sorted_list.pop(index)

        return res

    """
    https://leetcode.com/problems/subsets/
    """

    # TODO:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        pass

    """
    https://leetcode.com/problems/merge-intervals/
    """

    # # def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    #
    #     # Failed approach
    #     """
    #         while True:
    #         intervals = sorted(intervals, key=lambda x: x[0])
    #         size = len(intervals)
    #         merge = []
    #         remove_index = []
    #         for i in range(0, size - 1):
    #             current = intervals[i]
    #             next = intervals[i + 1]
    #
    #             if next[0] <= current[1] <= next[1]:
    #                 merge.append([current[0], next[1]])
    #                 if i not in remove_index:
    #                     remove_index.append(i)
    #                 if i + 1 not in remove_index:
    #                     remove_index.append(i + 1)
    #
    #             elif next[0] <= current[1] and next[1] <= current[1]:
    #                 merge.append([current[0], current[1]])
    #                 if i not in remove_index:
    #                     remove_index.append(i)
    #                 if i + 1 not in remove_index:
    #                     remove_index.append(i + 1)
    #
    #         for index in sorted(remove_index, reverse=True):
    #             del intervals[index]
    #
    #         for m in merge:
    #             intervals.append(m)
    #
    #     return intervals
    #     """

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        merge = []
        for current_interval in sorted(intervals, key=lambda e: e[0]):
            if merge:
                last_interval = merge[-1]
                if current_interval[0] > last_interval[1]:
                    # They do not overlap
                    merge.append(current_interval)
                else:
                    merge.append(
                        [last_interval[0], max(current_interval[1], last_interval[1])]
                    )
            else:
                merge.append(current_interval)

        return merge

    """
    https://leetcode.com/problems/group-anagrams/
    """

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        process = {}
        for word in strs:
            processing_list = [0] * 26
            for char in word:
                index = ord(char) - 97
                processing_list[index] = processing_list[index] + 1

            # Get the sorted string
            output_string = ""
            for index, value in enumerate(processing_list):
                if value > 0:
                    output_string = output_string + chr(97 + index) * value

            dict_value = process.get(output_string, None)
            if dict_value is not None:
                dict_value.append(word)
            else:
                process[output_string] = [word]

        res = [value for value in process.values()]
        del process
        return res

    """
    https://leetcode.com/problems/wiggle-sort-ii/
    """

    def wiggleSort(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

    """
    https://leetcode.com/problems/spiral-matrix/
    """

    # TODO:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        directions = ["R", "D", "L", "T"]
        direction_index = 0
        current_coordinate = [0, 0]

        def valid_index(current_coordinate):
            index1 = current_coordinate[0]
            index2 = current_coordinate[1]

            try:
                v = matrix[index1][index2]
            except:
                return False

            return True

        def move_next(direction, current_cordinate):
            if direction == "R":
                current_cordinate[1] += 1
                if not valid_index(current_coordinate):
                    current_coordinate[0] += 1
                    current_coordinate[1] -= 1
                    direction_index = 1

            elif direction == "D":
                current_cordinate[0] += 1
                if not valid_index(current_coordinate):
                    current_coordinate[0] -= 1
                    current_coordinate[1] -= 1
                    direction_index = 2

            elif direction == "L":
                current_cordinate[1] -= 1
                if not valid_index(current_coordinate):
                    current_coordinate[0] -= 1
                    current_coordinate[1] -= 1
                    direction_index = 3

            elif direction == "T":
                current_cordinate[0] -= 1

            return current_cordinate

        size_of_matrix = len(matrix) * len(matrix[0])
        res = []
        while True:
            if len(res) == size_of_matrix:
                break

    """
    https://leetcode.com/problems/fizz-buzz/
    """

    def fizzBuzz(self, n: int) -> List[str]:
        res = []
        for i in range(1, n + 1):
            value = str(i)
            isThreeMultiple = i % 3
            isFiveMultiple = i % 5

            if isFiveMultiple == 0 and isThreeMultiple == 0:
                value = "FizzBuzz"
            elif isFiveMultiple == 0 and isThreeMultiple != 0:
                value = "Buzz"
            elif isThreeMultiple == 0 and isFiveMultiple != 0:
                value = "Fizz"

            res.append(value)

        return res

    """
    https://leetcode.com/problems/rotate-array/
    """

    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        size = len(nums)
        bisection_point = size - k
        index = 0
        right_array = nums[bisection_point:]
        left_arrary = nums[:bisection_point]
        for value in right_array:
            nums[index] = value
            index += 1

        for value in left_arrary:
            nums[index] = value
            index += 1

    """
    https://leetcode.com/problems/implement-strstr/
    """

    # Inbuild method used
    def strStr(self, haystack: str, needle: str) -> int:
        try:
            return haystack.index(needle)
        except:
            return -1

    def strStr(self, haystack: str, needle: str) -> int:

        if haystack == "" and needle == "":
            return 0

        if haystack == "" and needle != "":
            return -1

        if haystack != "" and needle == "":
            return -1

        h_index = 0
        n_index = 0
        size = len(haystack)
        occurence_index = -1
        char_matched = 0
        needle_len = len(needle)

        while h_index < size:
            if haystack[h_index] == needle[n_index]:
                if occurence_index == -1:
                    occurence_index = h_index

                h_index += 1
                n_index += 1
                char_matched += 1
                if char_matched == needle_len:
                    break
            else:
                if occurence_index != -1 and char_matched != needle_len:
                    occurence_index = -1
                    n_index = 0
                    char_matched = 0

                h_index += 1

        return occurence_index

    """
    https://leetcode.com/problems/missing-number/
    """

    def missingNumber(self, nums: List[int]) -> int:
        size = len(nums)
        sum = size * (size + 1) / 2
        for value in nums:
            sum -= value

        return int(sum)

    """
    https://leetcode.com/problems/largest-number/
    """

    def largestNumber(self, nums: List[int]) -> str:
        if not any(nums):
            return "0"

        def helper(s1, s2):
            return str(s1) + str(s2) > str(s2) + str(s1)

        size = len(nums)
        for i in range(0, size):
            for j in range(i, size):
                if not helper(nums[i], nums[j]):
                    temp = nums[i]
                    nums[i] = nums[j]
                    nums[j] = temp

        return "".join(map(str, nums))

    """
    https://leetcode.com/problems/move-zeroes/
    """

    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) >= 2:
            l = 0
            n = 1
            size = len(nums)

            while n < size and l < size:
                l_element = nums[l]
                n_element = nums[n]
                if l_element == 0 and n_element != 0:
                    nums[l] = nums[n]
                    nums[n] = 0
                    l += 1
                    n += 1
                elif l_element == 0 and n_element == 0:
                    n += 1
                elif l_element != 0 and n_element == 0:
                    l += 1
                elif l_element != 0 and n_element != 0:
                    l += 1
                    n += 1

    """
    https://leetcode.com/problems/increasing-triplet-subsequence/
    """

    def increasingTriplet(self, nums: List[int]) -> bool:
        smallest = float("inf")
        second_smallest = float("inf")

        for element in nums:
            if element < smallest:
                smallest = element
            elif element < second_smallest and element > smallest:
                second_smallest = element
            elif element > second_smallest:
                return True

        return False

    """
    https://leetcode.com/problems/search-a-2d-matrix-ii/
    """

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        def binary_search(input, l, r, target):
            if l > r:
                return False

            mid = l + (r - l) // 2
            if input[mid] == target:
                return True
            elif input[mid] > target:
                return binary_search(input, 0, mid - 1, target)
            else:
                return binary_search(input, mid + 1, r, target)

        for i in range(0, len(matrix)):
            output = binary_search(matrix[i], 0, len(matrix[i]) - 1, target)
            if output:
                return True

        return False

    """
    https://leetcode.com/problems/number-of-islands/
    """

    # TODO
    def numIslands(self, grid: List[List[str]]) -> int:
        pass

    """
    https://leetcode.com/problems/single-number/
    """

    def singleNumber(self, nums: List[int]) -> int:
        nums.sort()
        previous = None
        previous_count = 0
        for element in nums:
            if previous is None:
                previous = element
                previous_count = 1
                continue

            if element != previous:
                if previous_count == 1:
                    return previous

                previous = element
                previous_count = 1
            elif element == previous:
                previous_count += 1

        return previous

    """
    https://leetcode.com/problems/longest-consecutive-sequence/
    """

    # TODO:

    def longestConsecutive(self, nums: List[int]) -> int:
        pass

    """
    https://leetcode.com/problems/trapping-rain-water/
    """

    # TODO:
    def trap(self, height: List[int]) -> int:
        pass

    """
    https://leetcode.com/problems/first-missing-positive/
    """

    def firstMissingPositive(self, nums: List[int]) -> int:
        process = set(nums)

        smallest_int = 1
        while True:
            if smallest_int not in process:
                return smallest_int
            smallest_int += 1

    """
    https://leetcode.com/problems/k-closest-points-to-origin/
    """
    # TODO:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
        import math
        import heapq

        def euclidean_distance(point):
            return math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2))

        processing_dict = {}
        output = []
        for point in points:
            distance = euclidean_distance(point)
            processing_dict[distance] = point
            heapq.heappush(output, distance)

        # return [value for element in heapq.nsmallest(K, output)]


"""
https://leetcode.com/problems/shuffle-an-array/
"""


# TODO:
class Solution:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.shuffling = self.nums

    def reset(self) -> List[int]:
        """
        Resets the array to its original configuration and return it.
        """
        self.shuffling = self.nums
        return self.shuffling

    def shuffle(self) -> List[int]:
        """
        Returns a random shuffling of the array.
        """
        if len(self.shuffling) > 2:
            self.shuffling[0], self.shuffling[-1] = (
                self.shuffling[-1],
                self.shuffling[0],
            )
            return self.shuffling
        else:
            return self.nums
