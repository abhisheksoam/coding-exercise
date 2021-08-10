class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        size, i, temp_dict, temp_string, substring = len(s), 0, {}, '', ''
        while i < size:
            current_char = s[i]
            if current_char in temp_dict:
                index = temp_dict[current_char]
                i = index
                temp_dict = {}
                substring = temp_string if len(temp_string) > len(substring) else substring
                temp_string = ''

            else:
                temp_string = temp_string + current_char
                temp_dict[current_char] = i

            i = i + 1
        substring = temp_string if len(temp_string) > len(substring) else substring

        return len(substring)


if __name__ == "__main__":
    s = Solution()
    print(s.lengthOfLongestSubstring("abcabcbb"))
