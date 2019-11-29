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

