# """
# Min. Absolute Difference In Array
# Given an integer array A of size N, find and return the minimum absolute difference between any two elements in the array.
# We define the absolute difference between two elements ai, and aj (where i != j ) is |ai - aj|.
#
# """


def minimum_absolute_difference():
    def input_for_question():
        n = int(input())
        arr = input()
        arr = arr.split(' ')
        del arr[len(arr) - 1]
        arr = list(map(int, arr))
        return arr

    arr = input_for_question()
    arr = sorted(arr)
    ans = arr[1] - arr[0]
    for i in range(2, len(arr)):
        if arr[i] - arr[i - 1] < ans:
            ans = arr[i] - arr[i - 1]
    return ans


# """ Nikunj and Donuts:
# Nikunj loves donuts, but he also likes to stay fit. He eats n donuts in one sitting,
# and each donut has a calorie count, ci. After eating a donut with k calories, he must walk at least 2^j x k(where j
# is the number donuts he has already eaten) miles to maintain his weight. Given the individual calorie counts for
# each of the n donuts, find and print a long integer denoting the minimum number of miles Nikunj must walk to
# maintain his weight. Note that he can eat the donuts in any order. """


def nikunj_donuts_minimum_value():
    def input_for_question():
        n = int(input())
        arr = input()
        arr = arr.split(' ')
        arr = list(map(int, arr))
        return arr

    arr = input_for_question()
    arr = sorted(arr, reverse=True)
    eaten = 0
    ans = 0
    for i in range(0, len(arr)):
        ans = ans + (arr[i] * pow(2, eaten))
        eaten = eaten + 1

    return ans


"""
Activity Selection:
You are given n activities with their start and finish times. 
Select the maximum number of activities that can be performed by a single person, 
assuming that a person can only work on a single activity at a time.
"""


def activity_selection():
    def input_for_questions():
        n = int(input())
        arr = []
        for i in range(0, n):
            pair = list(map(int, input().split(' ')))
            arr.append(pair)

        return arr

    arr = input_for_questions()
    ## Sort this particular array by the ending time
    arr = sorted(arr, key=lambda x: x[1])
    previous_pair = arr[0]
    ans = 1
    for i in range(1, len(arr)):
        current_pair = arr[i]
        starting_time = current_pair[0]
        ending_time = current_pair[1]

        previous_pair_st = previous_pair[0]
        previous_pair_et = previous_pair[1]

        if previous_pair_et <= starting_time:
            ans = ans + 1
            previous_pair = current_pair

    return ans


"""
Fractional Knapsack
"""


def fractional_knapscak():
    def input_for_question():
        parameter = list(map(int, input().split(' ')))
        n, w = parameter[0], parameter[1]

        arr = []
        for i in range(0, n):
            pair = list(map(int, input().split(' ')))
            arr.append(pair)

        return arr, w

    arr, weight = input_for_question()
    current_weight = 0
    maximum_value = 0
    arr = sorted(arr, key=lambda x: x[0] / x[1], reverse=True)
    for i in range(0, len(arr)):
        item_weight = arr[i][1]
        item_value = arr[i][0]
        if current_weight + item_weight <= weight:
            current_weight += item_weight
            maximum_value += item_value
        else:
            r_weight = weight - current_weight
            maximum_value += item_value * (r_weight / item_weight)
            break

    return maximum_value


def weighted_job_scheduling():
    def input_for_question():
        n = int(input())
        arr = []
        for i in range(0, n):
            pair = list(map(int, input().split(' ')))
            arr.append(pair)
        return arr

    def binary_search():
        pass

    arr = input_for_question()
    arr = sorted(arr, key=lambda x: x[1])

    dp = [0] * len(arr)
    dp[0] = arr[0][2]
    for i in range(1, len(arr)):
        including, non_conflicting_index = arr[i][2], -1
        j = i - 1
        while j >= 0:
            if arr[j][1] <= arr[i][0]:
                non_conflicting_index = j
                break

            j = j - 1
        dp[i] = max((including + dp[non_conflicting_index]), dp[i - 1])

    return dp[len(arr) - 1]


if __name__ == "__main__":
    print(weighted_job_scheduling())
    # print(fractional_knapscak())
    # print(activity_selection())

    # print(nikunj_donuts_minimum_value())

    """
        Minimum Absolute Difference
    """
    # print(minimum_absolute_difference())
