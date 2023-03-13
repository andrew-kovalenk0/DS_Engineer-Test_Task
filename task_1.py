def is_valid(i, j, matrix, visited):
    # Row number is in range, column number
    # is in range, not yet visited and matrix value is 1
    return (0 <= i < len(visited) and 0 <= j < len(visited[0]) and
            not visited[i][j] and matrix[i][j])


def dfs(i, j, matrix, visited):
    # Indexes increments from current position to dots that are neighbours.
    row_ind = [-1, 0, 0, 1]
    col_ind = [0, -1, 1, 0]

    # Mark this cell as visited
    visited[i][j] = True

    # Recurrent search for all connected neighbours
    for k in range(len(row_ind)):
        if is_valid(i + row_ind[k], j + col_ind[k], matrix, visited):
            dfs(i + row_ind[k], j + col_ind[k], matrix, visited)


def islands_counter(matrix):
    # Visited points matrix
    visited = [[False for _ in range(len(matrix[0]))] for _ in
               range(len(matrix))]
    count = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # True case - new island found
            if not visited[i][j] and matrix[i][j] == 1:
                # Mark all cells on this island as visited
                dfs(i, j, matrix, visited)
                count += 1
    return count


if __name__ == '__main__':
    matrix_1 = [[0, 1, 0],
                [0, 0, 0],
                [0, 1, 1]]

    matrix_2 = [[0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]]

    matrix_3 = [[0, 0, 0, 1],
                [0, 0, 1, 1],
                [0, 1, 0, 1]]

    print(islands_counter(matrix_1))
