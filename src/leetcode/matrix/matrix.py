from typing import List


class Solution:
    """
    https://leetcode.com/problems/word-search/
    """

    # TODO:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word:
            return False

        def dfs(matrix, i, j, row, column, word, l):
            if i >= row or i < 0 or j >= column or j < 0:
                return False

            if l == len(word) - 1:
                return True

            char = matrix[i][j]
            matching_char = word[l]

            if char != matching_char:
                return False

            return (
                dfs(matrix, i + 1, j, row, column, word, l + 1)
                or dfs(matrix, i - 1, j, row, column, word, l + 1)
                or dfs(matrix, i, j + 1, row, column, word, l + 1)
                or dfs(matrix, i, j - 1, row, column, word, l + 1)
            )

        input_coordinates = []
        row = len(board)
        column = len(board[0])
        first_char = word[0]
        for i in range(0, row):
            for j in range(0, column):
                if first_char == board[i][j]:
                    input_coordinates.append((i, j))

        for source in input_coordinates:
            if dfs(board, source[0], source[1], row, column, word, 0):
                return True

        return False


s = Solution()
board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
print(s.exist(board, "ABCB"))
