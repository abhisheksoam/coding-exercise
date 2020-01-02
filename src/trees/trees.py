from typing import List


class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None

    def __str__(self):
        return str(self.info)


"""
Node is defined as
self.left (the left child of the node)
self.right (the right child of the node)
self.info (the value of the node)
"""


def preOrder(root):
    if root:
        print(root, end=" ")
        preOrder(root.left)
        preOrder(root.right)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def helper(node, upper=float("-inf"), lower=float("-inf")):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False

            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True

        return helper(root)

    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        def helper(node1, node2):
            if not node1 and not node2:
                return True

            if node1 and node2:
                if node1.val != node2.val:
                    return False

                left = helper(node1.left, node2.left)
                right = helper(node1.right, node2.right)

                if left and right:
                    return True
                else:
                    return False

            return False

        return helper(p, q)

    def leafSimilar(self, root1: TreeNode, root2: TreeNode) -> bool:
        def helper(node, output=[]):
            # import ipdb
            # ipdb.set_trace()
            if not node:
                return output

            if node.left is None and node.right is None:
                output.append(node.val)

            helper(node.left, output)
            helper(node.right, output)

            return output

        leaves_1 = helper(root1, output=[])
        leaves_2 = helper(root2, output=[])

        for v1, v2 in zip(leaves_1, leaves_2):
            if v1 != v2:
                return False
        return True

    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True

        left, right = [], []
        if root.left and root.right and root.left.val == root.right.val:
            left.append(root.left)
            right.append(root.right)
        elif root.left is None and root.right is None:
            return True
        else:
            return False

        while len(left) != 0 and len(right) != 0:
            l_current = left.pop(0)
            r_current = right.pop(0)

            if l_current.left and r_current.right:
                if l_current.left.val == r_current.right.val:
                    left.append(l_current.left)
                    right.append(r_current.right)
                else:
                    return False
            elif l_current.left is None and r_current.right is None:
                pass
            else:
                return False

            if l_current.right and r_current.left:
                if l_current.right.val == r_current.left.val:
                    left.append(l_current.right)
                    right.append(r_current.left)
                else:
                    return False
            elif l_current.right is None and r_current.left is None:
                pass
            else:
                return False

        return True

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        index = 0
        output = [root, None]
        while len(output) != 0:
            current = output.pop(0)
            if current is not None:
                try:
                    res[index].append(current.val)
                except:
                    res.append([current.val])

                if current.left:
                    output.append(current.left)

                if current.right:
                    output.append(current.right)
            elif current is None and len(output) != 0:
                output.append(None)
                index = index + 1

        return res

    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        index = 0
        output = [root, None]
        sequence = "Left"
        while len(output) != 0:
            current = output.pop(0)
            if current is not None:
                try:
                    if sequence == "Left":
                        res[index].append(current.val)
                    else:
                        res[index].insert(0, current.val)
                except:
                    res.append([current.val])

                if current.left:
                    output.append(current.left)

                if current.right:
                    output.append(current.right)
            elif current is None and len(output) != 0:
                output.append(None)
                index = index + 1
                if sequence == "Left":
                    sequence = "Right"
                else:
                    sequence = "Left"

        return res

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        def helper(node, sum):
            if not node:
                return False

            if node.val == sum and not node.left and not node.right:
                return True

            return helper(node.left, sum - node.val) or helper(
                node.right, sum - node.val
            )

        return helper(root, sum)

    # TODO: Complete this
    """
    https://leetcode.com/problems/path-sum-ii/
    """

    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        output = []
        index = 0

        def helper(node, sum, path=[]):
            if not node:
                return path

            output = []
            if node.val == sum and not node.left and not node.right:
                path.append(output)
                return path

            output.append(helper(node.left, sum - node.val))
            output.append(helper(node.right, sum - node.val))
            return output

        return helper(root, sum)

    """
    https://leetcode.com/problems/binary-tree-level-order-traversal-ii/ 
    """

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return []

        res = []
        index = 0
        output = [root, None]
        while len(output) != 0:
            current = output.pop(0)
            if current is not None:
                try:
                    res[index].append(current.val)
                except:
                    res.insert(0, [current.val])
                    index = 0

                if current.left:
                    output.append(current.left)

                if current.right:
                    output.append(current.right)
            elif current is None and len(output) != 0:
                output.append(None)
                index = "a"

        return res

    """
    https://leetcode.com/problems/sum-root-to-leaf-numbers/
    """

    def sumNumbers(self, root: TreeNode) -> int:
        output = []

        def helper(root, process=""):
            if not root:
                return

            if root.left is None and root.right is None:
                output.append(process + str(root.val))

            helper(root.left, process + str(root.val))
            helper(root.right, process + str(root.val))

        helper(root, "")
        return sum(int(_) for _ in output)

    """
    https://leetcode.com/problems/unique-binary-search-trees/
    """

    def numTrees(self, n: int) -> int:
        pass

    def generateTrees(self, n: int) -> List[TreeNode]:
        pass

    """
    https://leetcode.com/problems/average-of-levels-in-binary-tree/
    """

    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root:
            return []

        res = []
        index = 0
        output = [root, None]
        while output:
            current = output.pop(0)
            if current is not None:
                try:
                    print(res[index])
                    res[index][0] = (res[index][0] * res[index][1] + current.val) / (
                            res[index][1] + 1
                    )
                    res[index][1] = res[index][1] + 1
                except:
                    res.append([current.val, 1])

                if current.left:
                    output.append(current.left)

                if current.right:
                    output.append(current.right)
            elif current is None and len(output) != 0:
                output.append(None)
                index = index + 1

        return [_[0] for _ in res]

    """
    https://leetcode.com/problems/increasing-order-search-tree/
    """

    def increasingBST(self, root: TreeNode) -> TreeNode:
        output = []
        output_node = None

        def helper(node):
            if not node:
                return

            helper(node.left)
            output.append(node.val)
            helper(node.right)

        helper(node=root)
        if output:
            output_node = TreeNode(output[0])
            current = output_node
            for obj in output[1:]:
                current.right = TreeNode(obj)
                current = current.right

        return output_node

    """
    https://leetcode.com/problems/path-sum-ii/
    """

    def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
        output = []

        def helper(node, sum, process=[]):
            if not node:
                return process

            process.append(node.val)
            if not node.left and not node.right and sum == node.val:
                output.append(process)

            helper(node.left, sum - node.val, process)
            helper(node.right, sum - node.val, process)
            process.pop()

        helper(root, sum, [])
        return output

    """
    https://leetcode.com/problems/binary-tree-paths/
    """

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        output = []

        def helper(node, process=""):
            if not node:
                return

            if process:
                process = process + "->{node_val}".format(node_val=node.val)
            else:
                process = "{node_val}".format(node_val=node.val)

            if not node.left and not node.right:
                output.append(process)

            helper(node.left, process)
            helper(node.right, process)

        helper(root, "")
        return output

    """
    https://leetcode.com/problems/path-sum-iii/
    """

    def pathSum(self, root: TreeNode, sum: int) -> int:
        # TODO: DP
        pass

    """https://leetcode.com/problems/recover-binary-search-tree/"""

    def recoverTree(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """

    # """
    # https://leetcode.com/problems/all-elements-in-two-binary-search-trees/
    # """
    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:

        # Traversing the BST and adding elements into the tree
        def helper(root, output=[]):
            if not root:
                return output

            helper(root.left, output)
            output.append(root.val)
            helper(root.right, output)
            return output

        l1 = helper(root1, [])
        l2 = helper(root2, [])
        output = []
        l1_index = 0
        l2_index = 0

        while l1_index < len(l1) and l2_index < len(l2):
            l2_value = l2[l2_index]
            l1_value = l1[l1_index]
            if l1_value >= l2_value:
                output.append(l2_value)
                l2_index = l2_index + 1
            elif l2_value >= l1_value:
                output.append(l1_value)
                l1_index = l1_index + 1

        while l1_index < len(l1):
            output.append(l1[l1_index])
            l1_index += 1

        while l2_index < len(l2):
            output.append(l2[l2_index])
            l2_index += 1

        return output

    """https://leetcode.com/problems/flip-equivalent-binary-trees/"""
    # FAILED Approach
    # def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
    #
    #     # Do a level order traversal and compare the elements
    #     queue1 = [root1, None]
    #     queue2 = [root2, None]
    #     previous_set1 = set()
    #     previous_set2 = set()
    #     while queue1 and queue2:
    #         c1 = queue1.pop(0)
    #         c2 = queue2.pop(0)
    #
    #         if c1 is None and c2 is None:
    #             if not previous_set2.difference(previous_set1):
    #                 previous_set1 = set()
    #                 previous_set2 = set()
    #                 if queue1:
    #                     queue1.append(None)
    #                 if queue2:
    #                     queue2.append(None)
    #             else:
    #                 return False
    #         elif c1 and c2:
    #             previous_set1.add(c1.val)
    #             previous_set2.add(c2.val)
    #
    #             if c1.left:
    #                 queue1.append(c1.left)
    #
    #             if c1.right:
    #                 queue1.append(c1.right)
    #
    #             if c2.left:
    #                 queue2.append(c2.left)
    #
    #             if c2.right:
    #                 queue2.append(c2.right)
    #         else:
    #             return False
    #
    #     return True

    # TODO: Come back at this
    def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        pass


s = Solution()
t1 = TreeNode(1)
t1.left = TreeNode(0)
t1.right = TreeNode(3)

t2 = TreeNode(2)
t2.left = TreeNode(1)
t2.right = TreeNode(4)
