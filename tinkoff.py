# n = int(input())
#
# if not 1 <= n <= 10 ** 5:
#     print("Некорректное количество оценок.")
#     exit()
#
# grades = list(map(int, input().split()))
#
# if any(grade < 2 or grade > 5 for grade in grades):
#     print("Оценки должны быть в диапазоне от 2 до 5.")
#     exit()
#
# max_fives_count = 0
# for i in range(n - 6):
#     if 2 not in grades[i:i + 7] and 3 not in grades[i:i + 7]:
#         max_fives_count = max(max_fives_count, grades[i:i + 7].count(5))
#
# print(max_fives_count if max_fives_count > 0 else -1)


##########################################################################
# n, m = map(int, input().split())
#
# if not (1 <= n <= 10 ** 3 and 1 <= m <= 10 ** 3):
#     print("Размеры матрицы должны быть от 1 до 10^3")
#     exit()
#
# matrix = [list(map(int, input().split())) for _ in range(n)]
#
# transposed_matrix = [[matrix[j][i] for j in range(n)] for i in range(m)]
#
# for row in transposed_matrix:
#     print(*row[::-1])


##########################################################################
# n = int(input())
# directories = [input() for _ in range(n)]
#
# directory_levels = {}
#
# for directory in directories:
#     path = directory.split('/')
#     level = len(path) - 1
#     directory_levels[directory] = level
#
# for directory in sorted(directories):
#     level = directory_levels[directory]
#     print('  ' * level + directory.split('/')[-1])


###################################################################
# n, rotate = input().split()
# n = int(n)
# mat = [list(map(int, input().split())) for _ in range(n)]
#
# result = []
#
# if rotate == "L":
#     for i in range(n // 2):
#         for j in range(i, n - i - 1):
#             result.extend([
#                 [[i, j], [j, n - i - 1]],
#                 [(j, n - i - 1), (n - i - 1, n - j - 1)],
#                 [(n - i - 1, n - j - 1), (n - j - 1, i)]
#             ])
# else:
#     for i in range(n // 2):
#         for j in range(i, n - i - 1):
#             result.extend([
#                 [[i, j], [n - j - 1, i]],
#                 [(j, n - i - 1), (n - i - 1, n - j - 1)],
#                 [(n - i - 1, n - j - 1), (n - j - 1, i)]
#             ])
#
# print(len(result))
# for item in result:
#     ind1, ind2 = item
#     print(*ind1, *ind2)


##################################################################
# def max_white_mushrooms(n, forest):
#     prev_dp = [0] * 3
#
#     for j in range(3):
#         if forest[0][j] == 'C':
#             prev_dp[j] = 1
#
#     for i in range(1, n):
#         curr_dp = [0] * 3
#         for j in range(3):
#             if forest[i][j] == 'W':
#                 curr_dp[j] = 0
#             else:
#                 max_previous = 0
#                 for prev_j in range(max(0, j - 1), min(3, j + 2)):
#                     max_previous = max(max_previous, prev_dp[prev_j])
#                 curr_dp[j] = max_previous + (forest[i][j] == 'C')
#         prev_dp = curr_dp
#
#     max_mushrooms = max(prev_dp)
#
#     return max_mushrooms
#
# n = int(input())
# forest = [input().strip() for _ in range(n)]
#
# print(max_white_mushrooms(n, forest))


#################################################################
# from collections import deque
#
#
# def bfs():
#     queue = deque([(start, "K")])
#     distance = 0
#     while queue:
#         distance += 1
#         for _ in range(len(queue)):
#             position, state = queue.popleft()
#             current_i, current_j = position
#             for di, dj in directions[state]:
#                 i, j = current_i + di, current_j + dj
#                 if not (0 <= i < n and 0 <= j < n) or (i, j) in visited[state]:
#                     continue
#                 if (i, j) == end:
#                     return distance
#                 elif board[i][j] not in (".", "S"):
#                     queue.append(((i, j), board[i][j]))
#                     visited[board[i][j]].add((i, j))
#                 else:
#                     queue.append(((i, j), state))
#                     visited[state].add((i, j))
#
#     return -1
#
#
# n = int(input())
# board = []
# start = (-1, -1)
# end = (-1, -1)
#
# for i in range(n):
#     row = input()
#     if "S" in row:
#         start = (i, row.index("S"))
#     elif "F" in row:
#         end = (i, row.index("F"))
#     board.append(row)
#
# directions = {
#     "K": {
#         (-2, -1), (-1, -2), (1, -2), (2, -1),
#         (2, 1), (1, 2), (-1, 2), (-2, 1)
#     },
#     "G": {
#         (-1, -1), (0, -1), (1, -1), (1, 0),
#         (1, 1), (0, 1), (-1, 1), (-1, 0)
#     }
# }
# visited = {
#     "K": set([start]),
#     "G": set()
# }
#
# result = bfs()
# print(result)
