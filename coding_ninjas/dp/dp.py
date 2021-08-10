## Knapsack problem

def knapsack():
    def input_for_question():
        # 3
        # 50
        # 60 10
        # 100 20
        # 120 30
        n = int(input())
        w = int(input())
        arr = []
        for i in range(0, n):
            pair = list(map(int, input().split(' ')))
            arr.append(pair)

        return arr, w

    def recursive_solution(arr, weight):
        if len(arr) == 0:
            return 0

        if weight <= 0:
            return 0

        current_obj_weight = arr[0][1]
        current_obj_value = arr[0][0]

        if weight < current_obj_weight:
            return recursive_solution(arr[1:], weight)

        return max(
            recursive_solution(arr[1:], weight),
            current_obj_value + recursive_solution(arr[1:], weight - current_obj_weight)
        )

    def dp_solution(arr, weight, dp, n):
        if len(arr) == 0:
            return 0

        if weight <= 0:
            return 0

        if dp[n] > -1:
            return dp[n]

        current_obj_weight = arr[0][1]
        current_obj_value = arr[0][0]

        if weight < current_obj_weight:
            return dp_solution(arr[1:], weight, dp, n + 1)

        output = max(
            dp_solution(arr[1:], weight, dp, n + 1),
            current_obj_value + dp_solution(arr[1:], weight - current_obj_weight, dp, n + 1)
        )
        dp[n] = output
        return output

    arr, weight = input_for_question()
    # return recursive_solution(arr, weight)
    return dp_solution(arr, weight, [-1] * len(arr), 0)


def getMinimumStrength(mat, n, m):
    def _getMinimumStrength(matrix, n, m, i, j):
        if i >= n or j >= m:
            return float("inf")

        if i == n - 1 and j == m - 1:
            return 1

        horizontal_health = 1
        vertical_health = 1
        if i + 1 < n:
            horizontal_health = _getMinimumStrength(matrix, n, m, i + 1, j) - matrix[i + 1][j]
        if j + 1 < m:
            vertical_health = _getMinimumStrength(matrix, n, m, i, j + 1) - matrix[i][j + 1]

        return max(min(horizontal_health, vertical_health), 1)

    return _getMinimumStrength(mat, n, m, 0, 0)


def maxMoneyLooted(houses, n):
    dp = [-1] * n

    def maxmize_loot(house, k, n):
        if k >= n:
            return 0

        if k == n - 1:
            return house[k]

        if dp[k] != -1:
            return dp[k]

        loot = house[k] + maxmize_loot(house, k + 2, n)
        loot1 = maxmize_loot(house, k + 1, n)
        dp[k] = max(loot, loot1)
        return dp[k]

    return maxmize_loot(houses, 0, n)


if __name__ == "__main__":
    print(maxMoneyLooted([2, 3, 1000, 2000], 4))
    # print(getMinimumStrength([[0, 1, -3], [1, -2, 0]], 2, 3))
