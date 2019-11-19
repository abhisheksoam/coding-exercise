class Solution:
    def reverse(self, s):
        return

    def reverseStr(self, s: str, k: int) -> str:
        window = 2*k
        size = len(s)
        starting_index = 0
        while window < size:
            process = s[0:window]

            window = window + 2*k



s = Solution()
s.reverseStr("abcdefg", 2)

