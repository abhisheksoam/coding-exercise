from typing import List


# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    https://leetcode.com/problems/merge-k-sorted-lists/
    """

    # Definition for singly-linked list.
    # class ListNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.next = None

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        def merge(left, right):
            l_pointer = left
            r_pointer = right
            if l_pointer and r_pointer:
                if l_pointer.val <= r_pointer.val:
                    head = ListNode(l_pointer.val)
                    l_pointer = l_pointer.next
                else:
                    head = ListNode(r_pointer.val)
                    r_pointer = r_pointer.next
            elif l_pointer and not r_pointer:
                head = ListNode(l_pointer.val)
                l_pointer = l_pointer.next
            elif r_pointer and not l_pointer:
                head = ListNode(r_pointer.val)
                r_pointer = r_pointer.next

            head_current = head

            while l_pointer and r_pointer:
                if l_pointer.val <= r_pointer.val:
                    head_current.next = ListNode(l_pointer.val)
                    l_pointer = l_pointer.next
                else:
                    head_current.next = ListNode(r_pointer.val)
                    r_pointer = r_pointer.next

                head_current = head_current.next

            while l_pointer:
                head_current.next = ListNode(l_pointer.val)
                l_pointer = l_pointer.next
                head_current = head_current.next

            while r_pointer:
                head_current.next = ListNode(r_pointer.val)
                r_pointer = r_pointer.next
                head_current = head_current.next

            return head

        if not lists or not any(lists):
            return None

        if len(lists) == 1:
            return lists[0]

        mid = len(lists) // 2
        left = self.mergeKLists(lists[0:mid])
        right = self.mergeKLists(lists[mid:])

        return merge(left, right)

    """
    https://leetcode.com/problems/kth-largest-element-in-an-array/
    """

    def findKthLargest(self, nums: List[int], k: int) -> int:
        """
        Learn about heap and selection algorithm for better understanding
        """
        nums.sort()
        return nums[-k]

    """
    https://leetcode.com/problems/majority-element/
    """

    def majorityElement(self, nums: List[int]) -> int:
        dict = {}
        threshold = len(nums) / 2
        for element in nums:
            element_occurence = dict.get(element, 0) + 1
            dict[element] = element_occurence
            if element_occurence > threshold:
                return element

        return -1

    """
    https://leetcode.com/problems/maximum-subarray/
    """
    # TODO:
    def maxSubArray(self, nums: List[int]) -> int:
        pass


s = Solution()
