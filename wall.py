# """
# A wall is made of two types of bricks (porous and non-porous). Grey bricks are non porous, and the white bricks are porous.
# A porous brick allows water to pass through in three directions, left and right diagonals, as well as straight down.
# A wall is said to be safe if the water does not reach the ground when it rains. Given example is an unsafe wall.
# Write a program to check if any given wall is safe or not.
# """


def check_porous(row, col, i, j, matrix):
    if i >= row:
        return True

    if j < 0 or j >= col:
        return False

    if matrix[i][j] == 0:
        return False

    return check_porous(row, col, i + 1, j, matrix) or \
           check_porous(row, col, i + 1, j - 1, matrix) or \
           check_porous(row, col, i + 1, j + 1, matrix)


def is_porous(matrix, row, col):
    for i in range(0, col):
        current_value = matrix[0][i]
        if current_value == 1:
            if check_porous(row=row, col=col, i=0, j=i, matrix=matrix):
                return True
        else:
            continue

    return False


matrix = [
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0]
]
output = is_porous(matrix=matrix, row=6, col=5)
print(output)
