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

    """
    https://leetcode.com/problems/as-far-from-land-as-possible/
    """

    # TODO:

    def maxDistance(self, grid: List[List[int]]) -> int:
        if not grid:
            return -1

        row = len(grid)
        column = len(grid[0])
        max_path = -1

        def dfs(matrix, i, j, visited, path=0, output=[]):
            if i < 0 or i >= row or j >= column or j < 0 or visited[i][j] == 1:
                return path

            if matrix[i][j] == 1:
                return output.append(path)

            visited[i][j] = 1
            l1 = dfs(matrix, i + 1, j, visited, path + 1)
            l2 = dfs(matrix, i - 1, j, visited, path + 1)
            l3 = dfs(matrix, i, j + 1, visited, path + 1)
            l4 = dfs(matrix, i, j - 1, visited, path + 1)
            visited[i][j] = 0
            return output

        for r in range(0, row):
            for c in range(0, column):
                element = grid[r][c]
                if element == 0:
                    visited = [[0 for col in range(column)] for rr in range(row)]
                    path = dfs(grid, r, c, visited)
                    print(path)
                    # max_path = max(path, max_path)

        return max_path


s = Solution()
output = s.maxDistance([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
print(output)
