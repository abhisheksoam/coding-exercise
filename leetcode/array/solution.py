import time

from typing import List


class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0 or len(intervals) == 1:
            return intervals

        merge = []
        for _interval in sorted(intervals, key=lambda x: x[0]):
            if merge:
                last_interval = merge[-1]
                last_interval_start = last_interval[0]
                last_interval_end = last_interval[1]

                current_interval_start = _interval[0]
                current_interval_end = _interval[1]

                if current_interval_start <= last_interval_end:
                    merge.pop(-1)
                    merge.append([last_interval_start, max(current_interval_end, last_interval_end)])
                else:
                    merge.append(_interval)
            else:
                merge.append(_interval)

        return merge

    def numIslands(self, grid: List[List[str]]) -> int:
        row = len(grid)
        col = len(grid[0])

        def dfs(grid, r, c):
            if r < 0 or c < 0 or r >= row or c >= col or grid[r][c] == '0':
                return

            grid[r][c] = '0'
            dfs(grid, r + 1, c)
            dfs(grid, r - 1, c)
            dfs(grid, r, c + 1)
            dfs(grid, r, c - 1)

        def bfs(i, j):
            queue = [[i, j]]
            while queue:
                top_element = queue.pop(0)
                _row, _col = top_element[0], top_element[1]
                grid[_row][_col] = '0'

                if _row - 1 >= 0 and grid[_row - 1][_col] == '1':
                    queue.append([_row - 1, _col])
                    grid[_row - 1][_col] = '0'

                if _row + 1 < row and grid[_row + 1][_col] == '1':
                    queue.append([_row + 1, _col])
                    grid[_row + 1][_col] = '0'

                if _col - 1 >= 0 and grid[_row][_col - 1] == '1':
                    queue.append([_row, _col - 1])
                    grid[_row][_col - 1] = '0'

                if _col + 1 < col and grid[_row][_col + 1] == '1':
                    queue.append([_row, _col + 1])
                    grid[_row][_col + 1] = '0'

        def union():
            pass

        def call_dfs():
            island_count = 0
            for i in range(0, row):
                for j in range(0, col):
                    if grid[i][j] == '1':
                        island_count = island_count + 1
                        dfs(grid=grid, r=i, c=j)

            return island_count

        def call_bfs():
            island_count = 0
            for i in range(0, row):
                for j in range(0, col):
                    if grid[i][j] == '1':
                        island_count = island_count + 1
                        bfs(i, j)

            return island_count

        return call_dfs()

    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        grid = m * [[0] * n]
        output = []
        for _position in positions:
            _row, _col = _position[0], _position[1]
            ans = 0
            if _row - 1 >= 0 and grid[_row - 1][_col] == 1:
                pass

    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 1 or len(intervals) == 0:
            return len(intervals)

        merge = []
        meetingRooms = 1
        print(sorted(intervals, key=lambda x: x[1] or x[0]))
        for interval in sorted(intervals, key=lambda x: x[1] or x[0]):
            if not merge:
                merge.append(interval)
            else:
                previous = merge[-1]
                if previous[1] > interval[0]:  # Overlap
                    meetingRooms = meetingRooms + 1
                merge.append(interval)
        return meetingRooms


if __name__ == "__main__":
    s = Solution()
    print(s.minMeetingRooms([[2, 11], [6, 16], [11, 16]]))
    s.numIslands2(3, 2, [])
    # print(s.numIslands())
    # print(s.merge([[2, 3], [4, 5], [6, 7], [8, 9], [1, 10]]))
