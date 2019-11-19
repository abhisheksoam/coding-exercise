# Definition for a binary tree node.
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    """
    https://leetcode.com/problems/longest-univalue-path/
    """

    def tribonacci(self, n: int) -> int:
        if n >= 3:
            output = [-1] * (n + 1)
            output[0] = 0
            output[1] = 1
            output[2] = 1

            for i in range(3, n + 1):
                output[i] = output[i - 1] + output[i - 2] + output[i - 3]

            return output[-1]

        else:
            if n == 0:
                return 0
            elif n == 1:
                return 1
            elif n == 2:
                return 1

    def climbStairs(self, n: int) -> int:
        output = [-1] * (n + 1)

        def helper(n):
            if n == 0:
                return 1
            elif n < 0:
                return 0
            elif output[n - 1] != -1:
                return output[n - 1]
            else:
                output[n - 1] = helper(n - 1) + helper(n - 2)
                return output[n - 1]

        return helper(n)

    def fibonacci(self, n):
        output = [None] * (n + 1)
        output[0] = 1
        output[1] = 1
        for i in range(2, len(output)):
            output[i] = output[i - 1] + output[i - 2]

        print(output)
        return output

    # def longestUnivaluePath(self, root: TreeNode) -> int:
    #
    #     def helper(node):
    #         if not node:
    #             return 0
    #
    #         value = 0
    #         if node.left and node.right:
    #             if node.left.val == node.val and node.right.val == node.val:
    #                 return 2 + helper(node.left) + node(node.right)
    #         elif node.left and not node.right:
    #             if node.left.val == node.val:
    #                 return 1 + helper(node.left) + node(node.right)
    #         elif node.right and not node.left:
    #             if node.right.val == node.val:
    #                 return 1 + helper(node.left) + node(node.right)
    #
    #         return  helper(node.left) + node(node.right)
    #
    #     return helper(root)

    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost = 0
        for i in range(0, len(cost)):
            pass


s = Solution()
s.fibonacci(15)
# print(s.climbStairs(n=4))
# print(s.tribonacci(25))
# root = TreeNode(1)
# root.left = TreeNode(4)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(4)
# root.right = TreeNode(5)
# root.right.right = TreeNode(5)
#
# print(s.longestUnivaluePath(root=root))
