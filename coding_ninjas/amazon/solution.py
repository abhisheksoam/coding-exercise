from typing import List


def rearrangeString(str):
    # Write your code here.
    arr = [0] * 26
    for i in range(0, len(str)):
        zero_based_index = ord(str[i]) - 97
        arr[zero_based_index] = arr[zero_based_index] + 1


def deleteMiddle(inputStack, N):
    size = len(inputStack)
    middle = N // 2 if (N % 2 != 0) else N / 2 - 1
    one_stack = inputStack
    second_stack = [0] * (N - middle)
    while len(one_stack) != 0:
        value = one_stack.pop(0)
        second_stack.insert(0, value)

    while middle != 0:
        one_stack_value = one_stack.pop(0)


def getInversions(arr, n):
    def merge(arr, start, mid, end):
        ans = 0
        left_subarray_pointer = start
        right_subarray_pointer = mid + 1
        output = []
        while left_subarray_pointer <= mid and right_subarray_pointer <= end:
            if arr[left_subarray_pointer] > arr[right_subarray_pointer]:
                # We have the inversion
                ans = ans + mid - left_subarray_pointer + 1
                output.append(arr[right_subarray_pointer])
                right_subarray_pointer += 1
            else:
                output.append(arr[left_subarray_pointer])
                left_subarray_pointer += 1

        while left_subarray_pointer <= mid:
            output.append(arr[left_subarray_pointer])
            left_subarray_pointer = left_subarray_pointer + 1

        while right_subarray_pointer <= end:
            output.append(arr[right_subarray_pointer])
            right_subarray_pointer = right_subarray_pointer + 1

        k = 0
        while start <= end:
            arr[start] = output[k]
            k = k + 1
            start = start + 1

        return ans

    def merge_sort(arr, start, end, ans=0):
        invCount = 0
        if end > start:
            mid = start + (end - start) // 2
            invCount = merge_sort(arr, start, mid, ans)
            invCount = invCount + merge_sort(arr, mid + 1, end)
            invCount = invCount + merge(arr, start, mid, end)

        return invCount

    return merge_sort(arr, 0, n - 1)


def isAnagram(str1, str2):
    map = {}
    for char in str1:
        map[char] = map.get(char, 0) + 1

    for char in str2:
        if map.get(char, 0) < 1:
            return False
        map[char] = map.get(char) - 1

    return True


# TODO:
def findMedian(arr, n):
    # Write your code here
    pass


class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = list()

    def numChildren(self):
        return len(self.children)

    def getChild(self, index):
        if index > len(self.children):
            return None

        return self.children[index]


def countSpecialNodes(root):
    def _countSpecialNodes(root, set, ans):
        if not root:
            return ans

        if root.data not in set:
            set.add(root.data)
            ans = ans + 1

        special_node = ans
        for node in root.children:
            special_node = special_node + _countSpecialNodes(node, set, ans)

        set.remove(root.data)
        return special_node

    _countSpecialNodes(root, set(), 0)


def find_2nd_max(input_arr):
    size = len(input_arr)
    if size == 0:
        return -1

    first_max = ""
    second_max = ""

    for index in range(0, size):
        index_element = input_arr[index]
        if index_element != first_max and index_element > first_max:
            first_max = index_element

        if index_element != second_max and first_max > index_element > second_max:
            second_max = index_element

    if first_max == second_max: return -1

    if second_max == '': return -1

    return second_max


def countPositiveNegativePairs(arr, n):
    freq_map = {}
    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
    ans = 0
    for element in arr:
        element_frequency = freq_map.get(element, 0)
        negative_element_frequency = freq_map.get(-element, 0)
        ans = ans + element_frequency * negative_element_frequency
        try:
            del freq_map[element]
        except:
            pass
        try:
            del freq_map[-element]
        except:
            pass
    return ans


def pairsWithGivenSum(arr, n, target):
    freq_map = {}

    for element in arr:
        freq_map[element] = freq_map.get(element, 0) + 1
    pairs = 0
    for element in arr:
        second_element = target - element
        if second_element in freq_map and freq_map[second_element] >= 1:
            pairs += 1
            freq_map[element] = freq_map.get(element) - 1
            freq_map[second_element] = freq_map.get(second_element) - 1
    return pairs


def findIslands(mat, n, m):
    visited = n * [[-1] * m]

    def dfs(x, y):
        if x < 0 or x >= n or y < 0 or y >= m or mat[x][y] == 0:
            return

        mat[x][y] = 0
        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)
        dfs(x + 1, y + 1)
        dfs(x - 1, y - 1)
        dfs(x - 1, y + 1)
        dfs(x + 1, y - 1)

    ans = 0
    for i in range(0, n):
        for j in range(0, m):
            if mat[i][j] == 1:
                dfs(i, j)
                ans = ans + 1

    return ans


def kThCharaterOfDecryptedString(s, k):
    def convert_string(s, k):
        pass

    return convert_string(s, k)


"""

"""


def nearestSmallNumber(input):
    size = len(input)
    output = [float("-inf")] * size
    # output[0] = -1

    for i in range(1, size):
        current_element = input[i]
        previous_output = output[i - 1]
        previous_element = input[i - 1]

        if previous_output >= current_element:
            # Do a while loop to figure out the element in least direction
            j = i - 1
            while output[j] >= current_element:
                j = j - 1
            output[i] = output[j]

        else:
            if previous_element > current_element:
                output[i] = previous_output
            else:
                output[i] = max(previous_output, previous_element)

    return output


if __name__ == "__main__":
    print(nearestSmallNumber([-39, 5, 7, 8, 1, 5, 7, 54, 22, 7]))
# print(kClosest([[0, 1], [1, 0]], 2))
# print(find_2nd_max(["-214748364801", "-214748364802"]))
# print(rearrangeString("aaaz"))
