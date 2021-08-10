class Solution:
    """
    1.1
    """

    def is_unique(self, s):
        s = sorted(s)
        size = len(s)
        index = 1
        previous_char = s[0]
        while index < size:
            import pdb

            pdb.set_trace()
            current_char = s[index]
            if previous_char == current_char:
                return False

            index = index + 1
        return True


s = Solution()
print(s.is_unique("bab"))
