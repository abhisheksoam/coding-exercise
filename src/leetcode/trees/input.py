# from .trees import TreeNode, Solution


"""
https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
"""

# TODO:
from src.leetcode.linklist.linkll import ListNode


class Codec:
    def serialize(self, root):
        import pickle

        """Encodes a tree to a single string.

        :type root: TreeNode
        :rtype: str
        """
        return pickle.dump(root)

    def deserialize(self, data):
        """Decodes your encoded data to tree.

        :type data: str
        :rtype: TreeNode
        """
        import pickle

        return pickle.load(data)


def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(",")]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


from typing import List


class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None

    def __str__(self):
        return str(self.info)


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def create(self, val):
        if self.root == None:
            self.root = Node(val)
        else:
            current = self.root

            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break


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

    """
    https://leetcode.com/problems/maximum-depth-of-binary-tree/¬
    """

    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        depth = 1
        output = [root, None]
        while len(output) != 0:
            current = output.pop(0)
            if current is not None:
                pass

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
                depth = depth + 1

        return depth

    def maxDepth(self, root: TreeNode) -> int:
        def helper(node):
            if not node:
                return 0

            left = 1 + helper(node.left)
            right = 1 + helper(node.right)

            return max(left, right)

        return helper(root)

    """
    https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
    """

    class Node:
        def __init__(self, val=None, children=None):
            self.val = val
            self.children = children

    def maxDepth(self, root: "Node") -> int:
        if not root:
            return 0

        output = [root, None]
        if not root:
            return 0

        depth = 1
        output = [root, None]
        while len(output) != 0:
            current = output.pop(0)
            if current is not None:
                for child in current.children:
                    output.append(child)

            elif current is None and len(output) != 0:
                output.append(None)
                depth = depth + 1

        return depth

    """
    https://leetcode.com/problems/path-sum-iii/
    """

    def pathSum(self, root: TreeNode, sum: int) -> int:
        def helper(node, sum):
            if not node or sum < 0:
                return 0

            if sum == 0:
                return 1

            if node.val < sum:
                left = helper(node.left, sum - node.val)
                right = helper(node.right, sum - node.val)
                return left + right

        return helper(root, sum)

    """
    https://leetcode.com/problems/path-sum/
    """

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        def helper(node, sum):
            import ipdb

            ipdb.set_trace()
            if not node or sum < 0:
                return False

            if sum == 0:
                return True

            left = helper(node.left, sum - node.val)
            right = helper(node.right, sum - node.val)

            if left:
                return True

            if right:
                return True

        return helper(root, sum)

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

    """https://leetcode.com/problems/balanced-binary-tree/"""

    def isBalanced(self, root: TreeNode) -> bool:
        def height(node):
            if not node:
                return 0

            l_height = 1 + height(node.left)
            r_height = 1 + height(node.right)
            return max(l_height, r_height)

        if not root:
            return True

        l_height = height(root.left)
        r_height = height(root.right)
        if abs(l_height - r_height) > 1:
            return False
        else:
            l = self.isBalanced(root.left)
            r = self.isBalanced(root.right)
            if l and r:
                return True
            else:
                return False

    """https://leetcode.com/problems/minimum-depth-of-binary-tree/"""

    # TODO:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0

        l = 1 + self.minDepth(root.left)
        r = 1 + self.minDepth(root.right)

        return min(1 + self.minDepth(root.left), 1 + self.minDepth(root.right))

    """https://leetcode.com/problems/maximum-difference-between-node-and-ancestor/"""

    def maxAncestorDiff(self, root: TreeNode) -> int:
        res = float("-inf")

        def helper(node, val):
            if not node:
                return res

            return max(
                abs(val - node.val),
                res,
                helper(node.left, val),
                helper(node.right, val),
            )

        queue = [root, None]

        while queue:
            current = queue.pop(0)
            if current:
                res = max(
                    res,
                    helper(current.left, current.val),
                    helper(current.right, current.val),
                )
                if current.left:
                    queue.append(current.left)

                if current.right:
                    queue.append(current.right)

            else:
                if queue:
                    queue.append(None)
        return res

    """
    https://leetcode.com/problems/add-one-row-to-tree/
    """

    # TODO
    def addOneRow(self, root: TreeNode, v: int, d: int) -> TreeNode:
        def helper(node, v, d):
            if not node:
                return

            if d == 1:
                pass

    """
    https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/
    """

    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid = len(nums) // 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1 :])
        return node

    """
    https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/
    """

    # TODO:
    def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
        pass

    """ 
    https://leetcode.com/problems/populating-next-right-pointers-in-each-node/ 
    """
    """
    # Definition for a Node.
    class Node:
        def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
            self.val = val
            self.left = left
            self.right = right
            self.next = next
    """

    def connect(self, root: "Node") -> "Node":
        if not root:
            return root

        queue = [root, None]
        while queue:
            top = queue.pop(0)
            if top is not None:
                try:
                    next = queue[0]
                except:
                    next = None
                top.next = next

                if top.left:
                    queue.append(top.left)
                if top.right:
                    queue.append(top.right)
            else:
                if len(queue) != 0:
                    queue.append(None)
                else:
                    break

        return root

    """
    https://leetcode.com/problems/construct-string-from-binary-tree/
    """

    # TODO:
    def tree2str(self, t: TreeNode) -> str:
        def helper(node, value=""):
            if not node:
                return ""

            value = value + str(node.val) + "("
            helper(node.left, value)
            value = value + ")"
            helper(node.right, value)
            return value

        return helper(t)

    """
    https://leetcode.com/problems/linked-list-in-binary-tree/
    
    linklist = [1, 10, 3, 7, 10, 8, 9, 5, 3, 9, 6, 8, 7, 6, 6, 3, 5, 4, 4, 9, 6, 7, 9, 6, 9, 4, 9, 9, 7, 1, 5, 5, 10, 4, 4,
            10, 7, 7, 2, 4, 5, 5, 2, 7, 5, 8, 6, 10, 2, 10, 1, 1, 6, 1, 8, 4, 7, 10, 9, 7, 9, 9, 7, 7, 7, 1, 5, 9, 8,
            10, 5, 1, 7, 6, 1, 2, 10, 5, 7, 7, 2, 4, 10, 1, 7, 10, 9, 1, 9, 10, 4, 4, 1, 2, 1, 1, 3, 2, 6, 9]

    root = stringToTreeNode(
    "[4,null,8,null,5,null,7,null,5,null,2,1,3,null,null,null,6,8,9,null,null,null,3,null,2,null,10,null,7,null,8,3,4,null,null,null,3,5,1,null,null,null,3,1,7,null,null,null,4,7,7,null,null,8,3,null,null,null,6,3,1,null,null,null,1,null,8,null,2,5,5,null,null,1,3,null,null,null,5,null,3,3,5,null,null,null,7,null,10,null,7,null,6,null,8,null,4,null,10,null,6,null,6,9,3,null,null,6,5,null,null,null,5,null,2,null,7,null,5,null,4,8,2,null,null,null,2,null,10,10,8,null,null,null,7,null,2,null,5,8,6,null,null,null,5,null,7,null,3,4,5,null,null,null,4,null,8,null,8,null,8,null,2,null,5,2,9,null,null,null,2,null,3,7,1,null,null,10,1,null,null,null,7,null,6,null,6,null,7,null,7,null,4,4,2,null,null,7,4,null,null,null,7,null,3,7,5,null,null,null,5,null,4,null,9,5,2,null,null,null,4,null,9,null,5,null,5,null,5,null,2,null,5,null,2,null,5,null,7,5,5,null,null,null,6,null,1,null,7,null,3,9,8,null,null,null,4,null,7,4,8,null,null,4,2,null,null,null,3,10,2,null,null,null,7,null,10,null,3,null,1,null,2,null,5,null,9,null,8,null,5,null,9,null,3,null,7,null,10,5,2,null,null,null,2,8,10,null,null,null,4,4,7,null,null,null,5,1,4,null,null,null,10,null,9,null,4,null,9,6,5,null,null,null,7,5,4,null,null,null,8,null,8,4,9,null,null,null,6,9,1,null,null,null,3,3,6,null,null,null,6,null,7,null,2,null,1,null,8,2,9,null,null,null,8,null,3,null,1,9,1,null,null,null,2,null,6,null,1,null,6,3,9,null,null,null,10,null,1,null,9,null,9,null,10,null,2,null,6,null,3,null,7,null,2,null,2,null,2,9,5,null,null,null,5,null,6,null,6,null,2,null,5,7,9,null,null,null,6,10,4,null,null,8,4,null,null,4,2,null,null,4,7,null,null,2,5,null,null,null,4,5,1,null,null,null,3,null,1,10,6,null,null,3,2,null,null,null,6,null,9,null,7,null,5,8,5,null,null,null,5,null,5,10,6,null,null,null,7,null,1,null,6,3,7,null,null,null,9,7,1,null,null,null,7,null,4,null,4,null,9,null,4,null,1,null,10,null,1,10,10,null,null,null,6,null,3,null,1,null,9,null,7,null,6,6,1,null,null,null,9,4,7,null,null,null,3,null,10,null,4,3,3,null,null,null,4,5,10,null,null,null,1,8,10,null,null,null,6,null,9,null,10,null,4,4,9,null,null,null,3,null,3,null,3,null,10,null,10,null,6,8,1,null,null,null,9,7,1,null,null,null,5,null,3,null,10,null,5,null,9,null,5,null,8,null,6,3,2,null,null,null,8,null,8,3,9,null,null,null,9,null,10,3,8,null,null,6,6,null,null,null,6,null,8,null,2,null,9,null,4,null,6,null,4,null,4,null,6,null,9,null,7,null,10,null,1,null,3,null,6,null,7,null,4,null,9,null,1,null,3,8,10,null,null,null,2,null,10,null,4,null,8,null,10,null,7,null,8,5,1,null,null,9,3,null,null,7,8,null,null,null,1,null,1,5,4,null,null,null,1,null,4,5,7,null,null,null,3,null,6,null,6,null,9,null,4,null,1,5,10,null,null,null,3,null,7,null,10,null,8,null,9,2,5,null,null,null,3,null,9,10,6,null,null,null,8,null,7,8,6,null,null,null,6,null,3,null,5,null,4,null,4,null,9,null,6,2,7,null,null,null,9,null,6,1,9,null,null,null,4,null,9,9,9,null,null,null,7,7,1,null,null,null,5,null,5,6,10,null,null,null,4,null,4,10,10,null,null,null,7,2,7,null,null,null,2,null,4,null,5,null,5,10,2,null,null,null,7,9,5,null,null,null,8,null,6,null,10,8,2,null,null,8,10,null,null,null,1,null,1,null,6,5,1,null,null,8,8,null,null,8,4,null,null,null,7,null,10,4,9,null,null,null,7,null,9,null,9,1,7,null,null,4,7,null,null,null,7,null,1,null,5,8,9,null,null,9,8,null,null,9,10,null,null,4,5,null,null,1,1,null,null,null,7,null,6,null,1,null,2,1,10,null,null,2,5,null,null,7,7,null,null,null,7,null,2,null,4,3,10,null,null,null,1,null,7,null,10,7,9,null,null,null,5,4,9,null,null,null,10,6,4,null,null,8,4,null,null,null,1,null,2,null,1,8,1,null,null,null,3,null,2,null,6,null,9,null,2,1,10,null,null,null,5,null,8,2,1,null,null,null,2,3,10,null,null,null,8,null,9,null,5,null,4,null,1,9,10,null,null,4,9,null,null,3,5,null,null,null,6,null,6,9,1,null,null,null,5,null,2,null,2,null,6,null,1,7,9,null,null,null,6,null,8,4,4,null,null,null,2,null,10,null,1,null,2,null,9,null,8,null,2,null,1,10,4,null,null,null,10,null,8,3,2,null,null,null,10,null,3,8,1,null,null,5,3,null,null,null,6,null,8,null,7,2,5,null,null,1,6,null,null,null,8,null,6,null,3,null,8,null,9,null,5,null,2,null,9,null,2,6,10,null,null,7,10,null,null,null,6,null,8,null,7,7,4,null,null,null,3,5,2,null,null,10,4,null,null,null,4,4,3,null,null,null,5,null,1,null,10,null,10,null,5,null,9,null,3,null,8,null,3,null,2,null,4,1,1,null,null,null,7,10,8,null,null,null,9,4,8,null,null,1,2,null,null,9,7,null,null,5,8,null,null,null,9,null,7,null,4,null,4,5,3,null,null,null,2,null,4,3,10,null,null,7,7,null,null,null,2,null,2,8,8,null,null,null,2,null,4,null,5,8,4,null,null,null,9,null,4,null,10,null,4,null,5,null,5,null,1,null,5,null,8,null,5,null,5,null,1,null,10,null,9,null,10,null,2,null,7,5,9,null,null,null,6,4,6,null,null,null,2,null,10,null,1,4,3,null,null,7,8,null,null,null,3,null,3,null,8,null,10,null,6,6,10,null,null,null,1,8,5,null,null,1,3,null,null,null,8,null,9,null,10,null,8,4,9,null,null,10,1,null,null,null,2,null,8,5,2,null,null,8,6,null,null,null,4,null,7,10,1,null,null,null,3,3,3,null,null,null,3,null,5,7,3,null,null,10,9,null,null,null,2,null,8,null,10,null,7,null,10,null,3,9,10,null,null,null,6,4,9,null,null,9,3,null,null,null,7,null,2,null,10,null,10,null,7,null,4,5,7,null,null,9,8,null,null,null,6,3,1,null,null,null,9,null,7,4,4,null,null,null,6,null,1,null,9,null,9,null,3,1,1,null,null,1,8,null,null,null,1,null,2,null,7,4,6,null,null,null,1,null,3,null,8,null,10,null,3,null,10,null,10,null,10,null,10,null,10,null,6,null,7,null,3,null,9,null,7,5,4,null,null,null,5,null,5,1,3,null,null,null,6,3,4,null,null,null,3,2,10,null,null,10,5,null,null,null,5,9,1,null,null,null,8,null,7,null,9,5,3,null,null,null,2,null,7,null,10,null,2,9,4,null,null,null,4,10,10,null,null,null,6,2,6,null,null,null,4,null,5,null,7,null,7,null,2,4,1,null,null,null,7,null,5,8,8,null,null,3,6,null,null,null,1,null,5,null,8,4,6,null,null,null,6,null,9,null,4,null,4,null,3,null,2,6,9,null,null,null,6,6,8,null,null,null,7,null,5,null,5,null,9,4,3,null,null,null,10,4,6,null,null,null,9,null,3,null,10,null,9,null,1,null,6,null,1,null,4,null,5,5,3,null,null,7,8,null,null,null,6,null,6,null,5,9,4,null,null,null,9,null,7,null,7,7,5,null,null,null,7,null,3,8,3,null,null,null,1,5,4,null,null,null,2,null,3,null,4,null,5,5,6,null,null,null,2,null,2,7,9,null,null,9,5,null,null,null,9,null,9,null,8,7,6,null,null,null,2,null,9,null,2,null,7,6,4,null,null,null,1,null,7,null,2,null,7,null,3,9,2,null,null,4,5,null,null,null,3,null,2,null,8,7,8,null,null,7,7,null,null,null,10,null,9,2,7,null,null,6,3,null,null,null,10,null,5,null,7,null,9,null,3,null,1,null,9,null,2,5,2,null,null,null,4,null,8,null,6,10,10,null,null,10,3,null,null,null,3,null,1,null,3,null,8,3,2,null,null,null,5,2,8,null,null,7,5,null,null,7,7,null,null,null,1,6,5,null,null,null,2,null,4,null,7,null,5,null,3,null,7,null,10,null,10,null,7,null,9,null,5,5,5,null,null,null,9,null,4,null,7,null,6,null,2,null,3,null,3,8,10,null,null,null,1,null,3,null,6,null,10,null,8,null,6,4,5,null,null,null,6,null,3,null,3,null,8,1,3,null,null,2,3,null,null,null,7,null,10,null,2,null,10,null,2,null,7,null,10,6,7,null,null,3,4,null,null,6,2,null,null,null,9,null,8,null,7,null,10,null,9,10,1,null,null,null,5,1,10,null,null,10,2,null,null,null,2,5,8,null,null,null,9,8,8,null,null,null,8,null,4,1,3,null,null,null,4,null,4,null,9,null,4,10,7,null,null,10,4,null,null,4,5,null,null,9,2,null,null,3,7,null,null,8,7,null,null,null,5,null,10,null,3,null,8,null,3,null,5,2,9,null,null,null,10,null,3,null,10,null,7,5,1,null,null,2,4,null,null,null,5,null,2,null,6,null,8,null,9,null,10,null,9,null,6,null,2,6,7,null,null,2,7,null,null,null,3,null,6,9,5,null,null,2,6,null,null,null,8,null,4,null,8,null,2,4,9,null,null,4,7,null,null,null,9,null,5,null,3,null,8,null,6,null,5,7,4,null,null,8,7,null,null,null,9,1,2,null,null,null,9,6,7,null,null,null,8,null,6,null,6,4,6,null,null,null,3,null,5,10,4,null,null,null,5,null,8,null,8,null,7,null,10,5,10,null,null,null,10,null,10,null,10,8,2,null,null,5,3,null,null,null,8,6,6,null,null,10,8,null,null,1,8,null,null,null,9,1,6,null,null,7,6,null,null,null,10,5,4,null,null,null,10,5,4,null,null,2,5,null,null,null,4,2,5,null,null,null,3,null,4,2,8,null,null,null,5,null,9,null,3,9,3,null,null,null,5,null,7,null,7,null,5,null,10,null,3,2,7,null,null,3,8,null,null,null,10,2,3,null,null,null,7,3,3,null,null,null,6,null,4,null,8,null,3,null,3,null,1,null,9,10,1,null,null,null,1,null,6,6,5,null,null,6,3,null,null,null,6,null,4,null,2,null,10,null,9,2,5,null,null,null,10,null,10,3,5,null,null,10,6,null,null,1,9,null,null,6,7,null,null,6,5,null,null,null,8,null,8,5,6,null,null,null,6,null,8,null,8,null,4,null,6,null,9,null,2,null,1,null,10,null,9,null,9,null,4,1,6,null,null,null,1,null,3,null,4,10,8,null,null,null,7,null,5,null,10,null,1,null,9,null,9,null,9,1,8,null,null,null,1,null,9,5,1,null,null,7,1,null,null,null,8,null,1,8,6,null,null,2,9,null,null,10,5,null,null,null,2,null,10,null,10,null,9,null,10,null,7,null,7,null,5,null,8,8,2,null,null,null,9,null,10,null,1,null,1,null,1,null,10,6,1,null,null,null,9,null,2,9,9,null,null,null,9,3,8,null,null,null,1,null,10,1,10,null,null,null,8,null,7,null,8,null,8,6,5,null,null,2,5,null,null,null,7,null,1,null,10,null,4,8,5,null,null,5,2,null,null,2,3,null,null,null,6,3,10,null,null,1,8,null,null,null,9,null,8,7,10,null,null,null,10,null,5,10,6,null,null,null,5,null,6,null,5,6,6,null,null,5,8,null,null,null,7,null,8,null,10,1,1,null,null,null,10,1,2,null,null,9,5,null,null,null,7,4,5,null,null,null,10,null,3,null,5,null,8,null,2,null,9,null,9,6,7,null,null,7,1,null,null,null,5,null,2,null,8,5,3,null,null,null,7,null,6,null,6,null,7,null,5,null,1,6,7,null,null,null,6,null,8,null,8,null,5,10,10,null,null,null,10,5,2,null,null,6,5,null,null,8,1,null,null,2,3,null,null,9,3,null,null,10,7,null,null,1,4,null,null,5,10,null,null,null,7,null,6,null,1,null,9,null,8,null,2,10,7,null,null,null,5,3,9,null,null,null,2,null,7,null,3,null,7,null,7,9,2,null,null,null,5,null,6,1,2,null,null,5,10,null,null,6,9,null,null,null,10,9,8,null,null,5,9,null,null,null,10,5,6,null,null,null,10,10,1,null,null,null,7,null,10,null,3,null,2,null,6,9,9,null,null,2,5,null,null,null,1,null,8,null,2,null,4,2,9,null,null,null,10,null,6,null,5,2,3,null,null,null,1,null,7,null,10,6,10,null,null,null,2,5,9,null,null,4,7,null,null,null,2,1,1,null,null,null,9,null,5,7,7,null,null,null,3,null,4,null,10,2,6,null,null,8,6,null,null,1,10,null,null,null,10,4,4,null,null,null,7,null,8,7,5,null,null,null,2,10,6,null,null,3,6,null,null,null,10,null,8,null,8,8,9,null,null,null,7,null,8,null,1,null,5,null,8,null,7,10,6,null,null,null,3,null,5,null,6,9,10,null,null,null,10,null,6,null,2,null,2,null,2,null,9,null,7,null,4,5,9,null,null,null,4,null,4,null,3,null,10,null,3,10,3,null,null,5,7,null,null,null,6,null,3,3,4,null,null,null,7,null,6,null,10,null,5,8,8,null,null,null,4,5,5,null,null,null,2,null,10,null,2,null,1,2,8,null,null,null,5,null,8,null,3,null,4,null,8,null,1,null,5,8,1,null,null,3,9,null,null,null,3,1,1,null,null,5,9,null,null,null,6,null,9,5,6,null,null,null,5,3,5,null,null,null,9,3,1,null,null,3,5,null,null,3,10,null,null,null,10,null,8,null,1,null,7,null,4,null,1,null,7,null,1,null,3,7,9,null,null,1,2,null,null,null,8,3,7,null,null,null,8,null,1,6,6,null,null,null,9,7,4,null,null,6,10,null,null,4,5,null,null,null,1,null,7,null,6,7,3,null,null,null,6,null,9,null,8,2,6,null,null,6,8,null,null,2,7,null,null,null,8,null,8,7,5,null,null,null,4,null,9,5,3,null,null,9,5,null,null,null,5,null,1,null,5,null,6,8,6,null,null,null,5,null,4,null,2,8,5,null,null,null,9,null,5,null,9,null,3,null,5,9,3,null,null,null,2,null,7,null,8,null,8,null,8,null,10,7,2,null,null,null,6,null,2,1,10,null,null,null,6,null,8,null,4,null,6,8,5,null,null,null,3,null,1,null,6,null,6,null,2,null,9,1,9,null,null,null,3,null,7,4,7,null,null,9,6,null,null,7,8,null,null,null,1,5,1,null,null,7,10,null,null,null,6,null,8,3,2,null,null,1,5,null,null,null,8,null,3,null,3,9,1,null,null,null,8,null,1,3,5,null,null,null,9,null,8,3,4,null,null,null,9,null,1,null,3,null,7,null,3,5,1,null,null,null,4,null,1,null,5,null,1,null,3,4,8,null,null,null,1,10,7,null,null,null,1,null,9,null,7,null,3,null,10,6,9,null,null,null,3,6,8,null,null,null,8,null,3,null,4,null,10,null,2,10,7,null,null,5,4,null,null,null,4,2,6,null,null,1,10,null,null,null,4,3,7,null,null,null,4,null,1,null,6,null,10,null,7,4,9,null,null,null,10,9,4,null,null,null,6,5,9,null,null,null,7,1,7,null,null,null,4,null,4,null,4,null,6,4,3,null,null,null,4,null,5,null,10,null,2,null,1,null,1,null,2,null,2,9,4,null,null,null,9,null,9,9,4,null,null,null,5,null,6,null,2,null,3,null,10,9,10,null,null,10,2,null,null,3,9,null,null,null,9,null,10,null,9,null,3,null,1,5,6,null,null,null,6,null,2,null,9,null,3,null,9,9,3,null,null,5,3,null,null,null,2,null,3,null,8,null,2,null,9,null,3,null,4,null,3,null,4,null,8,6,7,null,null,null,6,null,3,null,1,null,9,5,1,null,null,null,2,null,7,4,7,null,null,null,2,null,9,null,7,null,10,null,6,null,7,null,1,null,4,null,5,null,2,null,7,null,3,null,7,null,4,null,5,null,10,null,1,null,9,9,8,null,null,10,10,null,null,null,6,6,10,null,null,10,4,null,null,4,6,null,null,null,4,null,3,null,5,4,8,null,null,null,5,6,3,null,null,1,7,null,null,9,4,null,null,null,9,10,2,null,null,null,5,null,6,2,5,null,null,null,10,5,1,null,null,null,8,2,2,null,null,7,6,null,null,null,9,null,4,null,4,null,2,null,4,null,8,1,10,null,null,null,8,null,3,null,1,null,5,null,2,null,9,8,5,null,null,8,6,null,null,null,1,null,6,null,2,2,9,null,null,null,9,5,7,null,null,null,4,null,5,4,5,null,null,1,1,null,null,8,3,null,null,null,10,7,10,null,null,6,5,null,null,6,3,null,null,4,1,null,null,10,1,null,null,4,2,null,null,6,3,null,null,null,2,null,9,null,10,9,9,null,null,null,2,null,8,null,8,6,2,null,null,10,7,null,null,null,10,1,3,null,null,2,3,null,null,null,10,3,1,null,null,null,9,null,4,null,3,null,4,null,7,null,2,null,1,null,9,null,1,null,7,null,9,null,7,null,6,7,9,null,null,null,10,null,6,3,2,null,null,null,4,null,4,null,5,4,1,null,null,null,3,null,3,null,6,null,5,null,4,10,5,null,null,4,6,null,null,10,4,null,null,null,7,null,10,null,1,null,1,5,6,null,null,9,7,null,null,null,3,null,6,null,8,null,2,null,4,null,2,null,7,null,8,3,10,null,null,null,6,null,3,null,7,null,4,2,3,null,null,1,9,null,null,5,6,null,null,6,6,null,null,null,7,null,8,9,9,null,null,null,9,null,1,null,9,null,5,null,1,null,5,2,6,null,null,null,9,2,4,null,null,3,6,null,null,4,2,null,null,null,9,null,6,3,3,null,null,null,7,null,9,null,6,2,9,null,null,null,8,null,5,null,4,null,7,null,4,null,8,null,5,3,2,null,null,null,1,null,1,null,1,null,1,null,1,null,1,null,4,null,6,3,7,null,null,9,7,null,null,9,2,null,null,null,4,null,1,null,5,null,8,null,2,10,1,null,null,null,10,null,1,2,7,null,null,null,5,null,8,null,7,2,6,null,null,null,10,null,3,null,7,null,3,null,6,9,4,null,null,null,2,null,8,null,8,null,1,null,8,null,8,null,9,null,7,null,5,10,9,null,null,4,4,null,null,null,7,null,6,null,8,1,3,null,null,9,3,null,null,1,10,null,null,null,9,8,2,null,null,null,8,null,4,null,4,4,1,null,null,null,7,null,8,1,9,null,null,null,10,null,3,6,7,null,null,null,5,5,2,null,null,null,4,null,5,6,6,null,null,7,7,null,null,5,10,null,null,null,6,10,6,null,null,null,2,null,5,2,5,null,null,null,7,null,7,null,7,null,4,null,9,null,4,null,8,null,1,null,5,null,9,6,4,null,null,null,8,9,2,null,null,10,2,null,null,null,3,null,4,null,10,null,6,null,10,2,9,null,null,6,1,null,null,null,7,6,6,null,null,null,2,null,4,null,10,8,9,null,null,null,7,3,9,null,null,10,4,null,null,null,10,null,6,null,2,null,5,null,1,null,8,8,3,null,null,null,2,5,10,null,null,5,8,null,null,3,10,null,null,null,5,null,8,null,5,null,4,null,5,6,2,null,null,null,7,null,5,null,10,null,8,1,5,null,null,null,1,null,1,null,5,null,9,null,6,null,1,null,5]")

    ll = ListNode(-1, linklist)
    head = ll.create()
    output = s.isSubPath(head, root)
    print(output)
    
    """

    # TODO:
    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        def helper(head, root):
            if not root:
                return False

            if head.next is None and head.val == root.val:
                print("here", head.val, root.val, root.right.val)
                return True

            if root.val == head.val:
                head = head.next

            l_value = helper(head, root.left)
            r_value = helper(head, root.right)
            # print(l_value, r_value)
            return l_value or r_value

        return helper(head, root)

    """
    https://leetcode.com/problems/invert-binary-tree/
    :time_complexity 24ms
    :category easy
    
    # Solution
    root = stringToTreeNode("[4,2,7,1,3,6,9]")
    print(s.levelOrder(root))
    root = s.invertTree(root)
    print(s.levelOrder(root))

    """

    def invertTree(self, root: TreeNode) -> TreeNode:
        def helper(root):
            if not root:
                return

            root.left, root.right = root.right, root.left
            helper(root.left)
            helper(root.right)

        helper(root)
        return root

    """
    https://leetcode.com/problems/maximum-binary-tree/
    input = [3, 2, 1, 6, 0, 5]
    root = s.constructMaximumBinaryTree(input)
    print(s.levelOrder(root))
    """

    def constructMaximumBinaryTree(self, nums: List[int]) -> TreeNode:
        def find_node(nums):
            max_index, max_element = -1, -1
            for i in range(0, len(nums)):
                element = nums[i]
                if element > max_element:
                    max_index = i
                    max_element = element

            left = nums[0:max_index]
            right = nums[max_index + 1 :]
            return max_element, left, right

        def helper(nums, root=None):
            if not nums:
                return root

            value = find_node(nums)
            if root is None:
                root = TreeNode(value[0])

            root.left = helper(value[1], root.left)
            root.right = helper(value[2], root.right)

            return root

        return helper(nums)

    """
    Diameter of binary tree
    
    """

    # TODO:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        def height(node):
            if not node:
                return 0

            if node.left:
                l = 1 + height(node.left)
            else:
                l = 0
            if node.right:
                r = 1 + height(node.right)
            else:
                r = 0
            return max(l, r)

        def helper(node, max=0):
            if not node:
                return 0

            lheight = height(node.left)
            rheight = height(node.right)
            print(node.val)
            print(lheight, rheight)
            diameter = lheight + rheight + 1
            if diameter > max:
                max = diameter

            helper(node.left, max)
            helper(node.right, max)

            return max

        return helper(root)

    """
    https://leetcode.com/problems/trim-a-binary-search-tree/
    :solution_seen True
    """

    def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
        def helper(root):
            if not root:
                return None

            elif root.val < L:
                return helper(root.right)

            elif root.val > R:
                return helper(root.left)
            else:
                root.left = helper(root.left)
                root.right = helper(root.right)
                return root

        return helper(root)

    """
    https://leetcode.com/problems/longest-univalue-path/
    """

    # TODO:
    def longestUnivaluePath(self, root: TreeNode) -> int:
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

        helper(root)
        print(output)


s = Solution()
root = stringToTreeNode("[4,4,4]")
s.longestUnivaluePath(root)
