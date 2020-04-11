from typing import List


class Solution:
    def firstUniqChar(self, s: str) -> int:
        process = {}
        for char in s:
            process[char] = process.get(char, 0) + 1

        for index, char in enumerate(s):
            if process.get(char) == 1:
                return index

        return -1

    def frequencySort(self, s: str) -> str:
        pass

    """
    
    """

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if len(strs) == 0:
            return ""

        indexing_position = 0
        char_at_indexing_position = None
        looping = True
        common_prefix = ""
        while looping:
            for word in strs:
                try:
                    if char_at_indexing_position is None:
                        char_at_indexing_position = word[indexing_position]
                        continue
                    else:
                        if word[indexing_position] != char_at_indexing_position:
                            looping = False

                except IndexError:
                    looping = False

            if looping:
                indexing_position = indexing_position + 1
                common_prefix = common_prefix + char_at_indexing_position
                char_at_indexing_position = None

        return common_prefix

    """
    https://leetcode.com/problems/zigzag-conversion/
    """

    def convert(self, s: str, numRows: int) -> str:
        size = len(s)
        process = size * [[None] * size]

        zigzag = False
        index = 0
        diagonal_limit = numRows - 2
        diagonal_count, rows = 0, 0
        for i in range(0, size):
            for j in range(0, size):
                if rows == numRows:
                    zigzag = True

                if diagonal_limit == diagonal_count:
                    zigzag = False

                try:
                    if zigzag:
                        process[j - 1][i - 1] = s[index]
                        diagonal_count = diagonal_count + 1
                    else:
                        process[j][i] = s[index]
                        rows = rows + 1
                except Exception as e:
                    break

                index = index + 1
        res = ""
        for i in range(0, size):
            for j in range(0, size):
                char = process[i][j]
                if process[i][j] is not None:
                    res = res + char

        print(res)
        return res

    def multiply(self, num1: str, num2: str) -> str:
        phase = 0
        for lower in reversed(num2):
            carry_forward = 0
            sum = 0
            for upper in reversed(num2):
                product = int(lower) * int(upper)
                if carry_forward:
                    product = carry_forward + product

                if product >= 10:
                    pass

    """
    https://leetcode.com/problems/basic-calculator-ii/
    """

    # TODO:
    def calculate(self, s: str) -> int:
        operations = ["+", "-", "*", "/"]
        stack = s
        operator = None
        res = ""

    """
    https://leetcode.com/problems/minimum-window-substring/
    """

    # TODO: Complete the problem
    def minWindow(self, s: str, t: str) -> str:
        pass
        # tmapped = {}
        # for _ in t:
        #     tmapped[_] = True
        #
        # s_size = len(s)
        # t_size = len(t)
        # left_pointer = 0
        # right_pointer = 0
        # charcter_matched = 0
        # while left_pointer < s_size and right_pointer < s_size:
        #     char = s[right_pointer]
        #     if char in tmapped:
        #         charcter_matched += 1
        #
        #         if t_size == charcter_matched:
        #             pass
        #         else:
        #             left_pointer = left_pointer + 1

    """
    https://leetcode.com/problems/reverse-string/
    """

    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.

        """
        l, r = 0, len(s)
        while l < r:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1

    """
    https://leetcode.com/problems/word-search/
    """

    # TODO:
    def exist(self, board: List[List[str]], word: str) -> bool:
        pass

    """
    https://leetcode.com/problems/most-common-word/
    # paragraph = "a, a, a, a, b,b,b,c, c"
    # banned = ["a"]
    # print(s.mostCommonWord(paragraph, banned))

    """

    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        import string

        translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
        paragraph = paragraph.translate(translator)

        words = paragraph.split(" ")

        hashmap = {}
        for ban in banned:
            hashmap[ban.lower()] = True

        process = {}
        for word in words:
            if not hashmap.get(word.lower()) and not word == "":
                wd = word.lower()
                process[wd] = process.get(wd, 0) + 1

        min = 0
        most_common = ""
        for key, value in process.items():
            if value > min:
                min = value
                most_common = key

        return most_common

    """
    https://leetcode.com/problems/valid-palindrome/
    """

    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        import string

        lowercase = set(string.ascii_lowercase + string.digits)
        while l < r:
            l_char = s[l].lower()
            r_char = s[r].lower()
            if l_char in lowercase and r_char in lowercase:
                if l_char != r_char:
                    return False

                l = l + 1
                r = r - 1

            elif l_char in lowercase and r_char not in lowercase:
                r = r - 1
            elif l_char not in lowercase and r_char in lowercase:
                l = l + 1
            else:
                l = l + 1
                r = r - 1

        return True

    """
    https://leetcode.com/problems/to-lower-case/
    """

    def toLowerCase(self, str: str) -> str:
        output = ""
        for index, char in enumerate(str):
            value = ord(char)
            if value >= 97:
                output = output + char
            elif 65 <= value <= 90:
                output = output + chr(value + 32)
            else:
                output = output + char
        return output


s = Solution()
print(s.minWindow("ab", "A"))
