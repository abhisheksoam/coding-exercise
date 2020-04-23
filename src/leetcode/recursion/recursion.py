# Definition for a binary tree node.
from typing import List


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
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

        return output

    """
       https://leetcode.com/problems/expression-add-operators/
    """

    # TODO:
    def addOperators(self, num: str, target: int) -> List[str]:
        operator = ["+", "*", "-", ""]
        operands = operator[0:3]
        res = []

        def evaluate_expression(input, target):
            expression = ""
            for char in input:
                if expression and expression[-1] == "0" and char not in operands:
                    expression = expression[:-1] + char
                else:
                    expression += char

            if eval(expression) == target:
                return True

            return False

        def helper(num, target, proc=""):
            if len(num) <= 1:
                expression = proc + num
                if evaluate_expression(expression, target):
                    res.append(expression)
                return

            for j in operator:
                pro = num[0] + j
                helper(num[1:], target, proc + pro)

        helper(num, target)
        return res


s = Solution()
output = s.addOperators("00", 0)
print(output)

# s.fibonacci(15)
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
