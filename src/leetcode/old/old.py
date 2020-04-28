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
        count = 0
        while a_i < a_len and b_i < b_len:
            a_value = A[a_i]
            b_value = B[b_i]

            if a_value > b_value:
                b_i = b_i + 1
                count = count + 1
                # output.append(b_value)

            elif b_value > a_value:
                a_i = a_i + 1
                count = count + 1
                # output.append(a_value)

            elif b_value == a_value:
                a_i = a_i + 1
                b_i = b_i + 1
                # output.append(b_value)
                # output.append(a_value)
                count = count + 2

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

    def palindrome_check(self, s):
        if len(s) == 0 or len(s) == 1:
            return True

        i, j = 0, len(s) - 1

        while i <= j:
            if s[i] == s[j]:
                i = i + 1
                j = j - 1
            else:
                return False

        return True

    def longestPalindrome(self, s: str) -> str:
        palindromic_string = {}
        p_string = ""
        n = len(s)
        print(n)
        for i in range(0, n):
            for j in range(i + 1, n + 1):
                ss = s[i:j]
                if ss in palindromic_string:
                    pass
                else:
                    p_s = self.palindrome_check(ss)
                    if p_s:
                        if len(ss) > len(p_string):
                            p_string = ss

                        palindromic_string[ss] = True
        return p_string

    def isMatch(self, s: str, p: str) -> bool:
        pass

    def reverse(self, x: int) -> int:

        try:
            range = (-2147483648, 2147483647)
            x_str = str(x)
            reverse_string = ""
            negative = False
            for i in reversed(x_str):
                if i == "-":
                    negative = True
                else:
                    reverse_string = reverse_string + i

            value = int(reverse_string)
            if range[0] <= value <= range[1]:

                if negative:
                    return -int(value)

                return int(value)
            else:
                return 0
        except Exception as e:
            return 0

    # TODO:
    def myAtoi(self, str: str) -> int:
        try:
            str = str.lstrip()
            numerical_chars = [
                "0",
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "+",
                "-",
            ]
            int_max_value = 2147483647
            int_min_value = -2147483648

            conversion_started = False
            first_char = None
            numerical_string = ""
            negative_sign = None
            for index, char in enumerate(str):

                # First char check
                if first_char is None:
                    if char in numerical_chars:
                        if char == "-":
                            negative_sign = True
                        elif char == "+":
                            pass
                        else:
                            numerical_string = numerical_string + char

                        conversion_started = True
                        first_char = True
                    else:
                        print("here")
                        return 0

                if conversion_started:
                    if char in numerical_chars:
                        numerical_string = numerical_string + char
                    else:
                        break

            print(negative_sign)
            value = int(numerical_string)
            if negative_sign is True:
                value = -value
                print(value)

            print(value)
            if value > int_max_value:
                return int_max_value

            if value < int_min_value:
                return int_min_value

        except Exception as e:
            print(str(e))
            return 0

    def removeDuplicates(self, nums: List[int]) -> int:
        previous_value = None
        count = 0
        for index, value in enumerate(nums):
            if previous_value is None:
                count = count + 1
            else:
                if value == previous_value:
                    nums[index] = None
                else:
                    count = count + 1

            previous_value = value

        while True:
            try:
                nums.remove(None)
            except:
                break

        return count

    def removeElement(self, nums: List[int], val: int) -> int:
        count = 0
        for index, i_value in enumerate(nums):
            if i_value == val:
                nums[index] = None
            else:
                count = count + 1

        while True:
            try:
                nums.remove(None)
            except Exception as e:
                break

        return count

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

    def searchInsert(self, nums: List[int], target: int) -> int:
        size = len(nums)
        for index, number in enumerate(nums):
            if target <= number:
                return index
            elif target > nums[-1]:
                return size

        return 0

    """
    Question 39:
    https://leetcode.com/problems/combination-sum/
    """

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        pass

    """
    https://leetcode.com/problems/3sum/
    """

    def threeSum(self, nums):
        nums.sort()
        res, leng = [], len(nums)
        for i in range(leng - 2):
            if i == 0 or nums[i] != nums[i - 1]:
                l, r = i + 1, leng - 1
                while l < r:
                    s = nums[i] + nums[l] + nums[r]
                    if s == 0:
                        res.append((nums[i], nums[l], nums[r]))
                        while l < r and nums[l] == nums[l + 1]:
                            l += 1
                        while r > l and nums[r] == nums[r - 1]:
                            r -= 1
                        l += 1
                        r -= 1
                    elif s > 0:
                        r -= 1
                    else:
                        l += 1
        return res

    def countSmaller(self, nums: List[int]) -> List[int]:
        pass
