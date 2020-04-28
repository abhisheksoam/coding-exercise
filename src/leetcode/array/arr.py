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

    #
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

    def merge_interval(self, intervals: List[List[int]]) -> List[List[int]]:
        merge = []
        for current_interval in sorted(intervals, key=lambda e: e[0]):
            if merge:
                last_interval = merge[-1]
                current_interval_start = current_interval[0]
                current_interval_end = current_interval[1]

                last_interval_start = last_interval[0]
                last_interval_end = last_interval[1]

                if current_interval_start > last_interval_end:
                    # They do not overlap
                    merge.append(current_interval)
                else:
                    merge.pop(-1)
                    merge.append(
                        [
                            last_interval_start,
                            max(current_interval_end, last_interval_end),
                        ]
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
    https://leetcode.com/problems/sum-of-subarray-minimums/
    """

    # TODO:
    def sumSubarrayMins(self, A: List[int]) -> int:

        size = len(A)
        modulo = 1000000007

        min_el = min(A)
        for i in range(0, size):
            if A[i] == min_el:
                min_index = i
                break

        sum, i = 0, 0
        while i < size:
            j = i
            minimum_element = A[i]
            while j < size:
                if j >= min_index >= i:
                    times = size - min_index
                    sum += (min_el * times) % modulo
                    break
                else:
                    element = A[j]
                    if element < minimum_element:
                        minimum_element = element
                    sum = (sum + minimum_element) % modulo

                j = j + 1

            i = i + 1

        return sum

    """
    https://leetcode.com/problems/contains-duplicate-ii/
    Notes:
    It can be optimised further by keeping the last index of the element.
    # output = s.containsNearbyDuplicate([1,2,3,1,2,3], 2)
    # print(output)
    """

    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        size = len(nums)
        dict = {}
        for i in range(0, size):
            element = nums[i]
            value = dict.get(element, [])
            value.append(i)
            dict[element] = value

        for key, value in dict.items():
            v_size = len(value)
            if v_size == 1:
                continue
            else:
                i, j, v_size = 0, 1, len(value)
                while i < v_size and j < v_size:
                    diff = value[j] - value[i]
                    if diff <= k:
                        return True
                    elif diff > k:
                        i = i + 1
                        j = i + 1
                    else:
                        j = j + 1

        return False

    """
    https://leetcode.com/problems/product-of-array-except-self/
    # output = s.productExceptSelf([1, 2, 3, 4])
    # print(output)
    """

    def productExceptSelf(self, nums: List[int]) -> List[int]:
        size = len(nums)
        l = [0] * size
        l[0] = 1
        r = [1]
        i = size - 1
        while i > 0:
            element = nums[i]
            r = [r[0] * element] + r
            i = i - 1

        output = []
        for index, value in enumerate(nums):
            output.append(r[index] * l[index])
            try:
                l[index + 1] = l[index] * value
            except:
                pass

        return output

    """
    https://leetcode.com/problems/merge-sorted-array/
    ## Approach 1: Copy items of nums2 into the nums1 and then sort the list.
    :runtime 36ms
    :time_complexity O(nlogn)
    :space_complexity O(1) 
    """

    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        pointer = m
        i = 0
        _nums2size = len(nums2)
        _nums1size = len(nums1)

        while i < _nums2size and pointer < _nums1size:
            nums1[pointer] = nums2[i]
            i = i + 1
            pointer = pointer + 1

        nums1.sort()

    """
    https://leetcode.com/problems/plus-one/
    print(s.plusOne([4, 3, 2, 1]))
    :category easy
    
    """

    def plusOne(self, digits: List[int]) -> List[int]:
        number = "".join(map(str, digits))
        number = int(number)
        number = number + 1
        output = []
        for char in str(number):
            output.append(int(char))
        return output

    """
    https://leetcode.com/problems/single-element-in-a-sorted-array/
    :category medium
    :time_complexity O(log(n))
    :space_complexity O(1)
    
    # output = s.singleNonDuplicate([1, 1, 2, 3, 3, 4, 4, 8, 8])
    # print(output)

    """

    def singleNonDuplicate(self, nums: List[int]) -> int:
        size = len(nums)
        if size == 1:
            return nums[0]

        def helper(l, r, nums, start, end):
            if l > r:
                return 0

            mid = l + (r - l) // 2

            if (
                mid == start
                and nums[mid] != nums[mid + 1]
                or mid == end
                and nums[mid] != nums[mid - 1]
                or nums[mid] != nums[mid - 1]
                and nums[mid] != nums[mid + 1]
            ):
                return nums[mid]
            else:
                if nums[mid] == nums[mid - 1]:
                    _l_value = helper(l, mid - 2, nums, start, end)
                    _r_value = helper(mid + 1, r, nums, start, end)
                else:
                    _l_value = helper(l, mid - 1, nums, start, end)
                    _r_value = helper(mid + 2, r, nums, start, end)

                return _l_value or _r_value

        return helper(0, size - 1, nums, 0, size - 1)

    """
    https://leetcode.com/problems/lexicographical-numbers/
    One approach could be to append string in sequence into the list and sort the array. 
    :category medium
    :time_complexity 
    :space_complexity
    :runtime 144 ms
    
    print(s.lexicalOrder(50))
    """

    def lexicalOrder(self, n: int) -> List[int]:
        res = []

        def helper(i):
            if i > n:
                return

            res.append(i)
            for number in range(0, 10):
                new_number = i * 10 + number
                if new_number > n:
                    return
                helper(new_number)

        for i in range(1, 10):
            helper(i)

        return res

    """
    https://leetcode.com/problems/find-all-duplicates-in-an-array/
    :category medium
    :time_complexity O(n)
    :space_complexity O(n)
    
    Optimised Approach:
    Make the integer negative at the value of list 
    :time_complexity O(n)
    :space_complexity O(1)
    
    input = [4, 3, 2, 7, 8, 2, 3, 1]
    print(s.findDuplicates(input))

    """

    def findDuplicates(self, nums: List[int]) -> List[int]:
        size = len(nums)
        bit = [0] * (size + 1)
        for element in nums:
            bit[element - 1] = bit[element - 1] + 1

        output = []
        for i in range(0, size + 1):
            if bit[i] == 2:
                output.append(i + 1)

        return output

    """
    Last Stone Weight
    output = s.lastStoneWeight([7, 6, 7, 6, 9])
    print(output)
    """

    def lastStoneWeight(self, stones: List[int]) -> int:
        stones = sorted(stones)
        while stones:
            if len(stones) == 1:
                break

            _heavy_1 = stones.pop(len(stones) - 1)
            _heavy_2 = stones.pop(len(stones) - 1)

            if _heavy_1 == _heavy_2:
                continue
            else:
                stone_weight = abs(_heavy_1 - _heavy_2)
                if stones:
                    for i in range(0, len(stones)):
                        if stones[i] >= stone_weight:
                            break

                    stones.insert(i, stone_weight)
                else:
                    stones.append(stone_weight)
        if stones:
            return stones[0]

        return 0

    """
    Find max length, Contigous array
    """

    # TODO
    def findMaxLength(self, nums: List[int]) -> int:
        if not nums:
            return 0

        _size = len(nums)
        output = 0
        i = 0
        one_discovered = 0
        zero_discovered = 0
        while i < _size:
            current_element = nums[i]

            i += 1

        return output

    """
    https://leetcode.com/problems/squares-of-a-sorted-array/
    :solution_seen False
    :category easy
    :time_complexity O(nlog(n))
    :space_complexity O(n)
    
    # Optimised Approach
    - Using pointer approach 
    """

    def sortedSquares(self, A: List[int]) -> List[int]:
        _size = len(A)
        for i in range(0, _size):
            if A[i] < 0:
                A[i] = -A[i]

        a = sorted(A)
        for i in range(0, _size):
            a[i] = a[i] * a[i]

        return a

    """
    Search in Rotated Sorted Array
    """

    # TODO:
    def search(self, nums: List[int], target: int) -> int:
        def find_index(input, l, r):
            if l > r:
                return -1
            else:
                mid = l + (r - l) // 2
                right = mid + 1
                left = mid - 1
                if right < r:
                    if input[mid] > input[right]:
                        return mid
                elif left > l:
                    if input[mid] > input[right]:
                        return mid
                else:

                    left = find_index(input, l, mid - 1)
                    right = find_index(input, mid + 1, r)
                    if left is not -1:
                        return left

                    elif right is not -1:
                        return right
                    else:
                        return -1

        def binary_search(input, target, l, r):
            if l > r:
                return -1
            else:
                m = l + (r - l) // 2

                if input[m] == target:
                    return m

                elif target > input[m]:
                    return binary_search(input, target, m + 1, r)
                elif target < input[m]:
                    return binary_search(input, target, l, m - 1)

        index = find_index(nums, 0, len(nums) - 1)
        if index != -1:
            left = nums[0 : index + 1]
            right = nums[index + 1 :]
            left_search = binary_search(left, target, 0, len(left) - 1)
            if left_search != -1:
                return left_search

            right_search = binary_search(right, target, 0, len(right) - 1)
            if right_search != -1:
                return index + right_search + 1

            return -1

        else:
            return -1

    """
    https://leetcode.com/problems/sliding-window-maximum/
    :solution_seen False
    :space_complexity O(K)
    :time_complexity O(N^2)
    :category hard
    
    Optimised solution
    Using heap to get max
    Deque Solution
    """

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        stack = nums[0:k]
        _size = len(nums)
        output.append(max(stack))
        for i in range(k, _size):
            element = nums[i]
            stack.pop(0)
            stack.append(element)
            output.append(max(stack))

        return output

    """
    https://leetcode.com/problems/continuous-subarray-sum/
    """

    # TODO:
    def checkSubarraySum(self, nums: List[int], k: int) -> bool:
        def binary_search(input, target, l, r):
            if l > r:
                return -1
            else:
                m = l + (r - l) // 2

                if input[m] == target:
                    return m

                elif target > input[m]:
                    return binary_search(input, target, m + 1, r)
                elif target < input[m]:
                    return binary_search(input, target, l, m - 1)

        cumulative_sum = []
        _size = len(nums)
        sum = 0
        for i in range(0, _size):
            current_element = nums[i]
            sum += current_element
            cumulative_sum.append(sum)

        for i in range(0, _size):
            current_element = cumulative_sum[i]
            if binary_search(cumulative_sum, current_element + k, 0, _size - 1) != -1:
                return True

        return False

    """
    https://leetcode.com/problems/find-n-unique-integers-sum-up-to-zero/
    :time_complexity O(N)
    :space_complexity O(1)
    :category easy
    :solution_seen False
    
    """

    def sumZero(self, n: int) -> List[int]:
        if n < 0 or n == 0:
            return []
        elif n == 1:
            return [0]

        output = []
        value = n // 2
        for i in range(-value, value + 1):
            output.append(i)

        if n % 2 == 0:
            output.remove(0)

        return output

    """
    https://leetcode.com/problems/find-k-closest-elements/
    :category medium
    :time_complexity O(N)
    :space_complexity O(N)
    
    Example:
    s = Solution()
    print(s.findClosestElements([1, 2, 3, 4, 5], 4, 3))

    """

    # TODO:
    """
    This approach fails with repetitive number
    """

    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        def binary_search(input, target, l, r):
            if l > r:
                return -1
            else:
                m = l + (r - l) // 2

                if input[m] == target:
                    return m

                elif target > input[m]:
                    return binary_search(input, target, m + 1, r)
                elif target < input[m]:
                    return binary_search(input, target, l, m - 1)

        def return_array(index):
            if index > k - 1:
                return arr[index - k - 1 : index + 1]
            elif index == k:
                return arr[0:k]
            else:
                return arr[0:index] + arr[index:k]

        _size = len(arr)
        smallest_element = arr[0]
        maximum_element = arr[-1]

        if x < smallest_element:
            return arr[0:k]
        elif x > maximum_element:
            return arr[_size - k :]
        else:
            # If Element present in array
            index = binary_search(arr, x, 0, _size - 1)
            if index == -1:
                for i in range(0, _size):
                    if arr[i] > x:
                        index = i - 1
                        break

            return return_array(index)

    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        nums = []
        _size = len(arr)
        for i in range(0, _size):
            nums.append(abs(arr[i] - x))

        index = -1
        min_number = float("inf")
        for i in range(0, _size):
            if nums[i] <= min_number:
                min_number = nums[i]
                index = i

        add_elements = []
        l, r = index, index + 1
        while l >= 0 and r <= _size - 1 and len(add_elements) != k:
            l_value = nums[l]
            r_value = nums[r]

            if l_value <= r_value:
                add_elements = [arr[l]] + add_elements
                l = l - 1
            else:
                add_elements.append(arr[r])
                r = r + 1

        while l >= 0 and len(add_elements) != k:
            add_elements = [arr[l]] + add_elements
            l = l - 1

        while r <= _size - 1 and len(add_elements) != k:
            add_elements.append(arr[r])
            r = r + 1

        return add_elements

    """
    https://leetcode.com/problems/consecutive-numbers-sum/
    :category hard
    :time_complexity O(N)
    :space_complexity O(1)
    
    # Optimised Approach
    On a mathematical induction O (Log(n)) 
    """

    def consecutiveNumbersSum(self, N: int) -> int:
        numbers = 1

        l, r = 1, 2

        value_sum = l
        while r <= int(N ** 0.5 + 2):
            if value_sum > N:
                value_sum = value_sum - l
                l = l + 1
            elif value_sum < N:
                value_sum = value_sum + r
                r = r + 1

            if value_sum == N:
                numbers += 1
                value_sum = value_sum - l
                l = l + 1

        return numbers


s = Solution()
print(s.consecutiveNumbersSum(15))

"""
https://leetcode.com/problems/shuffle-an-array/
"""

# TODO:
# class Solution:
#     def __init__(self, nums: List[int]):
#         self.nums = nums
#         self.shuffling = self.nums
#
#     def reset(self) -> List[int]:
#         """
#         Resets the array to its original configuration and return it.
#         """
#         self.shuffling = self.nums
#         return self.shuffling
#
#     def shuffle(self) -> List[int]:
#         """
#         Returns a random shuffling of the array.
#         """
#         if len(self.shuffling) > 2:
#             self.shuffling[0], self.shuffling[-1] = (
#                 self.shuffling[-1],
#                 self.shuffling[0],
#             )
#             return self.shuffling
#         else:
#             return self.nums
