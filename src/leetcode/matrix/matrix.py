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
    output = s.maxDistance([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    print(output)

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

    """
    https://leetcode.com/problems/set-matrix-zeroes/
    :category medium
    :time_complexity O(M * N)
    :space_complexity O(M + N)
    :solution_seen False
    
    Example:
    # For Set zero
    matrix = [[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]
    s.setZeroes(matrix)
    print(matrix)    
    """

    def setZeroes(self, matrix: List[List[int]]) -> None:
        rows = len(matrix)
        columns = len(matrix[0])

        def mark_row(row):
            for i in range(0, columns):
                if matrix[row][i] == 0:
                    continue

                matrix[row][i] = None

        def mark_col(col):
            for i in range(0, rows):
                if matrix[i][col] == 0:
                    continue

                matrix[i][col] = None

        row_marked = set()
        col_marked = set()

        for i in range(0, rows):
            for j in range(0, columns):
                if matrix[i][j] == 0:
                    if i not in row_marked:
                        mark_row(i)
                        row_marked.add(i)
                    if j not in col_marked:
                        mark_col(j)
                        col_marked.add(j)

        for i in range(0, rows):
            for j in range(0, columns):
                if matrix[i][j] is None:
                    matrix[i][j] = 0

    """
    Flood Fill
    """

    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        rows = len(image)
        columns = len(image[0])
        old_colr = image[sr][sc]
        if old_colr == newColor:
            return image

        visited = [[0 for col in range(columns)] for rr in range(rows)]

        def dfs(i, j):
            if i < 0 or i >= rows or j < 0 or j >= columns or visited[i][j] == 1:
                return

            if image[i][j] == old_colr:
                image[i][j] = newColor
                visited[i][j] = 1
                dfs(i + 1, j)
                dfs(i - 1, j)
                dfs(i, j + 1)
                dfs(i, j - 1)
                visited[i][j] = 0

            return

        dfs(sr, sc)

        return image


s = Solution()
matrix = [[0, 0, 0], [0, 1, 1]]

s.floodFill(matrix, 1, 1, 1)
print(matrix)
