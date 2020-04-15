from typing import List


class Solution:
    """
    https://leetcode.com/problems/battleships-in-a-board/
    :solution_seen
    """

    # TODO:
    def countBattleships(self, board: List[List[str]]) -> int:
        pass

    #     count = 0
    #     row = len(board)
    #     column = len(board[0])
    #     visited = [[False] * column] * row
    #
    #     def dfs(i, j, row, column, battle_ship_found=False):
    #         if i >= row or i < 0 or j >= column or j < 0:
    #             return
    #
    #         if visited[i][j]:
    #             return
    #
    #         element = board[i][j]
    #         visited[i][j] = True
    #
    #
    #     return dfs(0, 0)

    """
    https://leetcode.com/problems/word-search/
    :solution_seen False

    Output
    board = [["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]]
    print(s.exist(board, "ABCB"))
    board = [["a", "b"], ["c", "d"]]
    print(s.exist(board, "cdba"))
    """

    def exist(self, board: List[List[str]], word: str) -> bool:
        if not word:
            return False

        def dfs(matrix, i, j, row, column, word, l, visited):
            if l == len(word):
                return True

            if i >= row or i < 0 or j >= column or j < 0 or visited[i][j] == 1:
                return False

            char = matrix[i][j]
            matching_char = word[l]

            if char != matching_char:
                return False

            visited[i][j] = 1
            if dfs(matrix, i + 1, j, row, column, word, l + 1, visited):
                return True
            if dfs(matrix, i - 1, j, row, column, word, l + 1, visited):
                return True
            if dfs(matrix, i, j + 1, row, column, word, l + 1, visited):
                return True

            if dfs(matrix, i, j - 1, row, column, word, l + 1, visited):
                return True

            visited[i][j] = 0
            return False

        row = len(board)
        column = len(board[0])

        first_char = word[0]
        for i in range(0, row):
            for j in range(0, column):
                if first_char == board[i][j]:
                    visited = [[0 for col in range(column)] for r in range(row)]
                    if dfs(board, i, j, row, column, word, 0, visited):
                        return True
        return False


s = Solution()
