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

    """
    Backspace Compare
    """

    def backspaceCompare(self, S: str, T: str) -> bool:
        def get_char(word):
            r = len(word) - 1
            previous_backspace = 0
            output = ""
            while r >= 0:
                char = word[r]
                if char == "#":
                    previous_backspace = previous_backspace + 1
                else:
                    if previous_backspace == 0:
                        output = char + output
                    else:
                        previous_backspace = previous_backspace - 1

                r = r - 1
            return output

        if get_char(S) == get_char(T):
            return True

        return False

    """
    https://leetcode.com/problems/number-of-matching-subsequences/
    """

    # TODO:
    def numMatchingSubseq(self, S: str, words: List[str]) -> int:
        def is_subsequence(s, word):
            s_pointer = 0
            w_pointer = 0
            s_size = len(s)
            w_size = len(word)

            while s_pointer < s_size and w_pointer < w_size:
                s_char = s[s_pointer]
                w_char = word[w_pointer]

                if s_char == w_char:
                    w_pointer += 1

                s_pointer += 1

            if w_pointer == w_size:
                return True

            return False

        count = 0
        process = {}
        for word in words:
            if process.get(word, False):
                count += 1
            else:
                if is_subsequence(S, word):
                    count += 1
                    process[word] = True

        return count

    """
    https://leetcode.com/problems/is-subsequence/
    """

    def isSubsequence(self, s: str, t: str) -> bool:
        s, t = t, s
        s_pointer = 0
        t_pointer = 0
        s_size = len(s)
        t_size = len(t)
        while s_pointer < s_size and t_pointer < t_size:
            s_char = s[s_pointer]
            t_char = t[t_pointer]
            if s_char == t_char:
                t_pointer += 1

            s_pointer += 1

        if t_pointer == t_size:
            return True

        return False

    """
    Check Valid string
    """

    # TODO:
    def checkValidString(self, s: str) -> bool:
        if not s:
            return True

        def is_parenthesis_valid(top, current_char):
            if top == "(" and current_char == ")":
                return True
            return False

        def helper(stack, parenthesis_string):
            if not parenthesis_string:
                if stack:
                    return False
                return True

            if not stack:
                return True

            current_char = parenthesis_string[0]
            top = stack[-1]
            if current_char == "*":
                blank = helper(stack, parenthesis_string[1:])
                if blank:
                    return True

                l_p_valid = is_parenthesis_valid(top, "(")
                if l_p_valid:
                    stack.pop(-1)
                else:
                    stack.append("(")

                left_parenthesis = helper(stack, parenthesis_string[1:])
                if l_p_valid:
                    stack.append(top)
                else:
                    stack.pop(-1)

                if left_parenthesis:
                    return True

                r_p_valid = is_parenthesis_valid(top, ")")
                if r_p_valid:
                    stack.pop(-1)
                else:
                    stack.append(")")

                right_parenthesis = helper(stack, parenthesis_string[1:])

                if r_p_valid:
                    stack.append(top)
                else:
                    stack.pop(-1)

                if right_parenthesis:
                    return True

            else:
                if is_parenthesis_valid(top, current_char):
                    stack.pop(-1)
                else:
                    stack.append(current_char)

                return helper(stack, parenthesis_string[1:])

        return helper([s[0]], parenthesis_string=s[1:])

    """
    https://leetcode.com/problems/rotate-string/
    :solution_seen False
    :time_complexity O(N^2)
    :space_complexity O(1)
    
    Optimized Approach:
    KNP Algorithm
    Rolling Hash
    
    Example
    output = s.rotateString("bbbacddceeb", "ceebbbbacdd")
    print(output)
    """

    def rotateString(self, A: str, B: str) -> bool:

        if len(A) != len(B):
            return False

        if A == B:
            return True

        def rotate(input):
            left = input[0]
            right = input[1:]
            return right + left

        value = B
        for i in range(0, len(B)):
            value = rotate(value)
            if value == A:
                return True

        return False

    """
    https://leetcode.com/problems/longest-substring-without-repeating-characters/
    """

    def lengthOfLongestSubstring(self, s: str) -> int:
        i = 0
        hashmap = {}
        max_len = 0
        p_len = 0
        while i < len(s):
            if hashmap.get(s[i]):
                max_len = max(max_len, p_len)
                i = int(hashmap.get(s[i])) + 1
                p_len = 0
                hashmap = {}
            else:
                p_len += 1
                hashmap[s[i]] = str(i)
                i = i + 1

        max_len = max(max_len, p_len)
        return max_len

    """
    https://leetcode.com/problems/longest-palindromic-substring/
    """

    # TODO:
    def longestPalindrome(self, s: str) -> str:
        l = 0
        r = len(s) - 1
        max_palindrome = ""
        max_palindrome_len = 1

        while l < r:

            if s[l] == s[r]:
                l = l + 1
                r = r - 1
                max_palindrome = max_palindrome +
            else:



s = Solution()
print(s.lengthOfLongestSubstring(" "))
