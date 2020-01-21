from typing import List
import math


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def findMedianSortedArrays(self, A: List[int], B: List[int]) -> float:
        a_len = len(A)
        b_len = len(B)

        total_len = a_len + b_len
        # Defining Median Values
        if total_len % 2 == 0:
            median_index_one = int(total_len / 2)
            median_index_second = int(median_index_one - 1)
        else:
            median_index_one = int(math.floor(total_len / 2))
            median_index_second = None

        a_i = 0
        b_i = 0
        output = []
        while a_i < a_len and b_i < b_len:
            a_value = A[a_i]
            b_value = B[b_i]

            if a_value > b_value:
                b_i = b_i + 1
                output.append(b_value)

            elif b_value > a_value:
                a_i = a_i + 1
                output.append(a_value)

            elif b_value == a_value:
                a_i = a_i + 1
                b_i = b_i + 1
                output.append(b_value)
                output.append(a_value)

        while a_i < a_len:
            output.append(A[a_i])
            a_i = a_i + 1

        while b_i < b_len:
            output.append(B[b_i])
            b_i = b_i + 1

        if median_index_second is not None:
            return (output[median_index_one] + output[median_index_second]) / 2
        else:
            return output[median_index_one]

    def isMatch(self, s: str, p: str) -> bool:
        pass

    # Approach 1
    def maxArea_v1(self, height: List[int]) -> int:
        max = 0
        n = len(height)
        for i in range(0, n):
            for j in range(i + 1, n):
                diff = j - i
                cell_i = height[i]
                cell_j = height[j]
                if cell_i > cell_j:
                    com_val = cell_j
                else:
                    com_val = cell_i

                computation_value = diff * (com_val)
                if computation_value > max:
                    max = computation_value
        return max

    def maxArea(self, height: List[int]) -> int:
        max = 0
        l, r = 0, len(height) - 1
        while l < r:
            diff = r - l

            cell_l = height[l]
            cell_r = height[r]

            if cell_l > cell_r:
                com_val = cell_r
                r = r - 1

            elif cell_l == cell_r:
                com_val = cell_r
                r = r - 1

            elif cell_r > cell_l:
                com_val = cell_l
                l = l + 1

            computation_value = diff * com_val
            if computation_value > max:
                max = computation_value

        return max

    def longestPalindrome(self, s: str) -> str:
        if len(s) == 0 or len(s) == 1:
            return s

        longest_palindrome = ""
        for index, char in enumerate(s):
            string1 = self.expand_around_center(s, index, index)
            string2 = self.expand_around_center(s, index, index + 1)

            if len(string1) > len(string2):
                max_s = string1
            else:
                max_s = string2

            if len(max_s) > len(longest_palindrome):
                longest_palindrome = max_s

        return longest_palindrome

    def expand_around_center(self, s, left, right):
        output = ""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            output = s[left : right + 1]
            left = left - 1
            right = right + 1

        return output

    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        current_l1_node = l1
        current_l2_node = l2
        carry_forward = None
        output, output_previous = None, None

        while current_l1_node is not None and current_l2_node is not None:
            if carry_forward:
                value = current_l1_node.val + current_l2_node.val + carry_forward
                carry_forward = None
            else:
                value = current_l1_node.val + current_l2_node.val

            # Calculate carry
            if value >= 10:
                carry_forward = int(str(value)[0])
                node_value = int(str(value)[1])
            else:
                node_value = value

            node = ListNode(node_value)
            if output is None:
                output = node
                output_previous = output
            else:
                output_previous.next = node
                output_previous = node

            current_l2_node = current_l2_node.next
            current_l1_node = current_l1_node.next

        while current_l1_node is not None:
            if carry_forward:
                value = carry_forward + current_l1_node.val
                carry_forward = None
            else:
                value = current_l1_node.val

            # Calculate carry
            if value >= 10:
                carry_forward = int(str(value)[0])
                node_value = int(str(value)[1])
            else:
                node_value = value

            node = ListNode(node_value)
            output_previous.next = node
            output_previous = node
            current_l1_node = current_l1_node.next

        while current_l2_node is not None:
            if carry_forward:
                value = carry_forward + current_l2_node.val
                carry_forward = None
            else:
                value = current_l2_node.val

            # Calculate carry
            if value >= 10:
                carry_forward = int(str(value)[0])
                node_value = int(str(value)[1])
            else:
                node_value = value

            node = ListNode(node_value)
            output_previous.next = node
            output_previous = node
            current_l2_node = current_l2_node.next

        if carry_forward:
            node = ListNode(carry_forward)
            output_previous.next = node

        return output

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        output = []

        left = 0
        right = len(nums) - 1

        while left <= right:
            l_value = nums[left]
            r_value = nums[right]
            for i in range(left + 1, right):
                if nums[i] + l_value + r_value == 0:
                    output.append([l_value, nums[i], r_value])

            left = left + 1
            right = right - 1

        return output

    # def binary_search(self, input, target, l, r):
    #     if l > r:
    #         return -1
    #     else:
    #         m = l + (r - l) / 2
    #         m = int(m)
    #
    #         if input[m] == target:
    #             return m
    #         elif target > input[m]:
    #             return self.binary_search(input, target, m + 1, r)
    #         elif target < input[m]:
    #             return self.binary_search(input, target, l, m - 1)

    """
    Question 33:
    https://leetcode.com/problems/search-in-rotated-sorted-array/
    """

    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1

        if len(nums) == 1:
            if nums[0] == target:
                return 0
            else:
                return -1

        left_array = None
        right_array = None

        previous_value = nums[0]
        for index, value in enumerate(nums[1:]):
            if value - previous_value < 0:
                right_array = nums[index + 1 :]
                left_array = nums[0 : index + 1]
                break

        if left_array is None and right_array is None:
            return self.binary_search(nums, target, 0, len(nums) - 1)

        left_result = self.binary_search(left_array, target, 0, len(left_array) - 1)
        right_result = self.binary_search(right_array, target, 0, len(right_array) - 1)

        if left_result is -1 and right_result is -1:
            return -1
        else:
            if left_result is not -1:
                return left_result
            else:
                return len(left_array) + right_result

    """
    Question 34:
    https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    """

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        size = len(nums)
        if size == 0:
            return [-1, -1]

        index = self.binary_search(nums, target, 0, len(nums) - 1)
        if index is -1:
            return [-1, -1]
        else:
            lower, upper = index, index
            l, r = index - 1, index + 1

            while l >= 0:
                if nums[l] == target:
                    lower = l
                    l = l - 1
                else:
                    break

            while r <= size - 1:
                if nums[r] == target:
                    upper = r
                    r = r + 1
                else:
                    break

        return [lower, upper]

    """
    Question 35:
    https://leetcode.com/problems/search-insert-position/
    """

    def binary_search(self, input, target, l, r):
        if l > r:
            return r - l + 1
        else:
            m = l + (r - l) / 2
            m = int(m)

            if input[m] == target:
                return m
            elif target > input[m]:
                return self.binary_search(input, target, m + 1, r)
            elif target < input[m]:
                return self.binary_search(input, target, l, m - 1)

    def searchInsert(self, nums: List[int], target: int) -> int:
        size = len(nums)
        if size == 0:
            return 0

        return self.binary_search(nums, target, 0, size - 1)

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        pass


s = Solution()
nums = [1, 3, 5, 6]
print(s.searchInsert(nums, 7))
# print(s.searchRange(nums, 6))
# print(s.longestPalindrome("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabcaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
# print(s.findMedianSortedArrays(nums1, nums2))
# print(s.isPalindrome(-121))
# h = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# print(s.maxArea(h))
# print(s.longestPalindrome("aaaaa"))
# print(s.addTwoNumbers(node, n1))
# nums = [-1, 0, 1, 2, -1, -4]
# print(s.threeSum(nums))
