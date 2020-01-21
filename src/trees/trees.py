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


def preOrder(root):
    if root:
        print(root, end=' ')
        preOrder(root.left)
        preOrder(root.right)


def height(root):
    pass


def levelOrder(root):
    from collections import deque
    arr = deque([root, None])

    while True:
        current_element = arr.popleft()
        if current_element is None:
            if len(arr) == 0:
                break
            arr.append(None)
        else:
            print(current_element, end=' ')
            if current_element.left is not None:
                arr.append(current_element.left)

            if current_element.right is not None:
                arr.append(current_element.right)


# tree = BinarySearchTree()
# t = int(input())
#
# arr = list(map(int, input().split(' ')))
#
# for i in range(t):
#     tree.create(arr[i])
#
# preOrder(tree.root)
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isValidBST(self, root: TreeNode) -> bool:

        def helper(node, upper=float('-inf'), lower=float('-inf')):
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

    def maxDepth(self, root: 'Node') -> int:
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

            if left: return True

            if right: return True

        return helper(root, sum)



