## array of non-negative numbers, they are consecutive.


#          0 1 2 3 4
#
# 5 7 8 9 10
# 0 1 2 3 4
#
#


def find_element(input):
    size = len(input)
    if size == 0:
        return -1

    start = 0
    end = size - 1

    starting_element = input[0]
    output = -1
    while start <= end:
        mid_index = start + (end - start) // 2
        mid_element = input[mid_index]
        expected_value = starting_element + mid_index

        if mid_element == expected_value:
            start = mid_index + 1
        else:
            output = expected_value
            end = mid_index - 1

    return output


# for arr in [[5],
#             [5, 6],
#             [4, 6],  ##
#             [3, 4, 5, 6, 8, 9],
#             [3, 4, 5, 6, 7, 9],  ##
#             [3, 4, 5, 6, 7, 8, 9],
#             [3, 5, 6, 7, 8],
#             [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14]]:
#     ans = find_element(arr)
#     print(ans)


# ans = find_element([4, 6])
# print(ans)


# 1. For array ["3", "-2"] should return "-2"
# 2. For array ["5", "5", "4", "2"] should return "4"
# 3. For array ["4", "4", "4"] should return "-1" (duplicates are not considered as the second max)
# 4. For [] (empty array) should return "-1".
# 5. For ["-214748364801","-214748364802"] should return "-214748364802".


def find_2nd_max(input_arr):
    size = len(input_arr)
    if size == 0:
        return -1

    first_max = ""
    second_max = ""

    for index in range(0, size):
        index_element = input_arr[index]
        if index_element != first_max and index_element > first_max:
            first_max = index_element

        if index_element != second_max and first_max > index_element > second_max:
            second_max = index_element

    if first_max == second_max: return -1

    if second_max == '': return -1

    return second_max


input_arr = ["-214748364801", "-214748364802"]
print(find_2nd_max(input_arr))


