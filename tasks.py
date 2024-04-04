import asyncio
import collections
import heapq
import itertools
import operator
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from itertools import groupby, product, permutations
import math
import random
from collections import defaultdict
from collections import deque


# def maxDepth(s):
#     current_depth = 0
#     max_depth = 0
#     for i in s:
#         if i == '(':
#             current_depth += 1
#             max_depth = max(max_depth, current_depth)
#         elif i == ')':
#             current_depth -= 1
#
#     return max_depth
#
#
# print(maxDepth("(1+(2*3)+((8)/4))+1"))
# print(maxDepth("(1)+((2))+(((3)))"))


# def exist(board, word):
#     if not board:
#         return False
#
#     def dfs(i, j, word_index):
#         if word_index == len(word):
#             return True
#
#         if i < 0 or i >= len(board) or \
#             j < 0 or j >= len(board[0]) or \
#                 board[i][j] != word[word_index]:
#             return False
#
#         temp = board[i][j]
#         board[i][j] = ''  # Помечаем текущую ячейку как посещенную
#
#         # Проверяем соседние ячейки в четырех направлениях
#         found = \
#             dfs(i + 1, j, word_index + 1) or \
#             dfs(i - 1, j, word_index + 1) or \
#             dfs(i, j + 1, word_index + 1) or \
#             dfs(i, j - 1, word_index + 1)
#
#         board[i][j] = temp  # Восстанавливаем текущую ячейку
#
#         return found
#
#     # Начинаем поиск слова с каждой ячейки сетки
#     for i in range(len(board)):
#         for j in range(len(board[0])):
#             if dfs(i, j, 0):
#                 return True
#
#     return False
#
#
# print(exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCCED"))
# print(exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "SEE"))
# print(exist([["A", "B", "C", "E"], ["S", "F", "C", "S"], ["A", "D", "E", "E"]], "ABCB"))


# def isIsomorphic(s, t):
#     # result_s = [(s.count(k), len(list(v))) for k, v in groupby(s)]
#     # result_t = [(t.count(k), len(list(v))) for k, v in groupby(t)]
#     # return result_s == result_t
#
#     # но круче так
#
#     return len(set(s)) == len(set(t)) == len(set(zip(s, t)))
#
#
# print(isIsomorphic("egg", "add"))
# print(isIsomorphic("foo", "bar"))
# print(isIsomorphic("paper", "title"))
# print(isIsomorphic("bbbaaaba", "aaabbbba"))
# print(isIsomorphic("badc", "baba"))


# def lengthOfLastWord(s):
#     result = s.split()
#     return len(result[-1])
#
#
# print(lengthOfLastWord("Hello World"))
# print(lengthOfLastWord("   fly me   to   the moon  "))
# print(lengthOfLastWord("luffy is still joyboy"))


# def countSubarrays(nums, minK, maxK):
#     ans = 0
#     min_i = max_i = waste_i = -1
#     for i in range(len(nums)):
#         if nums[i] < minK or nums[i] > maxK:
#             waste_i = i
#         if nums[i] == minK:
#             min_i = i
#         if nums[i] == maxK:
#             max_i = i
#         temp = min(max_i, min_i) - waste_i
#         ans += max(0, temp)
#     return ans
#
#
# print(countSubarrays([1, 3, 5, 2, 7, 5], 1, 5))  # Output: 2
# print(countSubarrays([1, 1, 1, 1], 1, 1))  # Output: 10


# class Solution(object):
#     def subarraysWithKDistinct(self, nums, k):
#         return self.subarraysWithAtMostKDistinct(nums, k) - self.subarraysWithAtMostKDistinct(nums, k - 1)
#
#     def subarraysWithAtMostKDistinct(self, nums, k):
#         ans = 0
#         count = [0] * (len(nums) + 1)
#
#         left = 0
#         for right in range(len(nums)):
#             count[nums[right]] += 1
#             if count[nums[right]] == 1:
#                 k -= 1
#             while k == -1:
#                 count[nums[left]] -= 1
#                 if count[nums[left]] == 0:
#                     k += 1
#                 left += 1
#             ans += right - left + 1
#         return ans
#
#
# solution = Solution()
# print(solution.subarraysWithKDistinct([1, 2, 1, 2, 3], 2))
# print(solution.subarraysWithKDistinct([1, 2, 1, 3, 4], 3))


# def countSubarrays(nums, k):
#     max_num = max(nums)
#     max_count = 0
#     count, left, right = 0, 0, 0
#
#     while right < len(nums):
#         if nums[right] == max_num:
#             max_count += 1
#
#         while max_count >= k:
#             count += len(nums) - right
#             if nums[left] == max_num:
#                 max_count -= 1
#
#             left += 1
#         right += 1
#     return count
#
#
# print(countSubarrays([1, 3, 2, 3, 3], 2))
# print(countSubarrays([1, 4, 2, 1], 3))


# def maxSubarrayLength(nums, k):
#     left, result = 0, 0
#     freq = defaultdict(int)
#
#     for right in range(len(nums)):
#         freq[nums[right]] += 1
#         while freq[nums[right]] > k:
#             freq[nums[left]] -= 1
#             left += 1
#         result = max(result, right - left + 1)
#     return result
#
#
# print(maxSubarrayLength([1, 2, 3, 1, 2, 3, 1, 2], 2))
# print(maxSubarrayLength([1, 2, 1, 2, 1, 2, 1, 2], 1))
# print(maxSubarrayLength([5, 5, 5, 5, 5, 5, 5], 4))


# def numSubarrayProductLessThanK(nums, k):
#     if k <= 1:
#         return 0
#
#     product = 1
#     result = 0
#     left = 0
#
#     for right in range(len(nums)):
#         product *= nums[right]
#         while product >= k:
#             product /= nums[left]
#             left += 1
#         result += right - left + 1
#
#     return result
#
#
# print(numSubarrayProductLessThanK([10, 5, 2, 6], 100))
# print(numSubarrayProductLessThanK([1, 2, 3], 0))


# def firstMissingPositive(nums):
#     nums = set(nums)
#     for i in range(1, len(nums) + 2):
#         if i not in nums:
#             return i
#
#
# print(firstMissingPositive([1]))
# print(firstMissingPositive([1, 2, 0]))
# print(firstMissingPositive([3, 4, -1, 1]))
# print(firstMissingPositive([7, 8, 9, 11, 12]))


# def findDuplicates(nums):
#     seen = set()
#     result = []
#     for i in nums:
#         if i in seen:
#             result.append(i)
#         seen.add(i)
#     return result
#
#
# print(findDuplicates([4, 3, 2, 7, 8, 2, 3, 1]))
# print(findDuplicates([1, 1, 2]))
# print(findDuplicates([1]))


# def findDuplicate(nums):
#     seen = set()
#     for num in nums:
#         if num in seen:
#             return num
#         seen.add(num)
#
#
# print(findDuplicate([1, 3, 4, 2, 2]))
# print(findDuplicate([3, 1, 3, 4, 2]))
# print(findDuplicate([3, 3, 3, 3, 3]))


# class ListNode:
#     def __init__(self, value=0, next=None):
#         self.value = value
#         self.next = next
#
#
# class Solution(object):
#     def reorderList(self, head):
#         if not head or not head.next or not head.next.next:
#             return head
#
#         # Шаг 1. Найти середину списка
#         slow = fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#
#         # Шаг 2. Реверсировать вторую половину списка
#         prev = None
#         current = slow
#         while current:
#             next_temp = current.next
#             current.next = prev
#             prev = current
#             current = next_temp
#
#         # Шаг 3. Слить две половины списка
#         first, second = head, prev
#         while second.next:
#             temp1 = first.next
#             first.next = second
#             first = temp1
#
#             temp2 = second.next
#             second.next = first
#             second = temp2
#
#
# def createLinkedList(lst):
#     dummy = ListNode(0)
#     current = dummy
#     for value in lst:
#         current.next = ListNode(value)
#         current = current.next
#     return dummy.next
#
#
# def printList(head):
#     while head:
#         print(head.value, end=" -> " if head.next else "")
#         head = head.next
#     print()
#
#
# solution = Solution()
# head = createLinkedList([1, 2, 3, 4, 5])
# solution.reorderList(head)
# printList(head)  # Ожидаемый вывод: 1 -> 5 -> 2 -> 4 -> 3


# class ListNode:
#     def __init__(self, value=0, next=None):
#         self.value = value
#         self.next = next
#
#
# class Solution(object):
#     def isPalindrome(self, head):
#         if head is None or head.next is None:
#             return True
#
#         # Найти середину списка
#         slow = fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#
#         # Реверсировать вторую половину списка
#         prev = None
#         while slow:
#             next_temp = slow.next
#             slow.next = prev
#             prev = slow
#             slow = next_temp
#
#         # Сравнить две половины списка
#         left, right = head, prev
#         while right:  # Проверяем только вторую половину, так как она может быть короче
#             if left.value != right.value:
#                 return False
#             left = left.next
#             right = right.next
#
#         return True
#
#
# # Вспомогательная функция для создания связного списка из списка
# def createLinkedList(lst):
#     dummy = ListNode()
#     current = dummy
#     for value in lst:
#         current.next = ListNode(value)
#         current = current.next
#     return dummy.next
#
#
# solution = Solution()
# head = createLinkedList([1, 2, 2, 1])
# print(solution.isPalindrome(head))  # Должен вывести: True


# class ListNode:
#     def __init__(self, value=0, next=None):
#         self.value = value
#         self.next = next
#
#
# class Solution(object):
#     def reverseList(self, head):
#         prev = None
#         curr = head
#         while curr:
#             next_temp = curr.next  # Сохраняем следующий узел
#             curr.next = prev  # Меняем указатель текущего узла на предыдущий
#             prev = curr  # Перемещаем prev на один шаг вперед
#             curr = next_temp  # Перемещаем curr на один шаг вперед
#         return prev  # В конце prev будет указывать на новую голову списка
#
#
# def createLinkedList(lst):
#     dummy = ListNode()
#     current = dummy
#     for value in lst:
#         current.next = ListNode(value)
#         current = current.next
#     return dummy.next
#
#
# def printLinkedList(head):
#     current = head
#     while current:
#         print(current.value, end=" -> " if current.next else "")
#         current = current.next
#     print()
#
#
# solution = Solution()
# list1 = createLinkedList([1, 2, 3, 4, 5])
# reversed_list = solution.reverseList(list1)
# printLinkedList(reversed_list)  # Ожидаемый вывод: 5 -> 4 -> 3 -> 2 -> 1


# class ListNode:
#     def __init__(self, value=0, next=None):
#         self.value = value
#         self.next = next
#
#
# def mergeInBetween(list1, a, b, list2):
#     # Шаг 1: Найти узел перед `a`
#     dummy = ListNode(-1)
#     dummy.next = list1
#     prev = dummy
#     for i in range(a):
#         prev = prev.next
#
#     # Шаг 2: Найти узел на позиции `b` и сохранить узел после `b`
#     afterB = prev
#     for i in range(b - a + 2):
#         afterB = afterB.next
#
#     # Шаг 3: Соединить `prev` с головой `list2`
#     prev.next = list2
#
#     # Шаг 4: Найти хвост `list2`
#     while list2.next:
#         list2 = list2.next
#
#     # Шаг 5: Соединить хвост `list2` с `afterB`
#     list2.next = afterB
#
#     return dummy.next
#
#
# # Вспомогательная функция для создания связного списка из списка
# def createLinkedList(lst):
#     dummy = ListNode()
#     current = dummy
#     for value in lst:
#         current.next = ListNode(value)
#         current = current.next
#     return dummy.next
#
#
# # Вспомогательная функция для печати связного списка
# def printLinkedList(head):
#     current = head
#     while current:
#         print(current.value, end=" -> " if current.next else "")
#         current = current.next
#     print()
#
#
# # Пример использования
# list1 = createLinkedList([10, 1, 13, 6, 9, 5])
# list2 = createLinkedList([1000000, 1000001, 1000002])
# result = mergeInBetween(list1, 3, 4, list2)
# printLinkedList(result)  # Ожидаемый результат: 10 -> 1 -> 13 -> 1000000 -> 1000001 -> 1000002 -> 5


# def leastInterval(tasks, n):
#     task_counts = collections.Counter(tasks).values()
#     max_val = max(task_counts)
#     max_count = sum(count == max_val for count in task_counts)
#
#     return max(len(tasks), (n + 1) * (max_val - 1) + max_count)
#
#
# print(leastInterval(["A", "A", "A", "B", "B", "B"], 2))  # Выведет: 8
# print(leastInterval(["A", "C", "A", "B", "D", "B"], 1))  # Выведет: 6
# print(leastInterval(["A", "A", "A", "B", "B", "B"], 3))  # Выведет: 10


# def findMinArrowShots(points):
#     if not points:
#         return 0
#
#     points.sort(key=lambda x: x[1])
#     arrows = 1
#     current_arrow_position = points[0][1]
#
#     for start, end in points:
#         if start <= current_arrow_position:
#             continue
#         arrows += 1
#         current_arrow_position = end
#
#     return arrows
#
#
# print(findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]))
# print(findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]))


# def insert(intervals, newInterval):
#     start, end = newInterval[0], newInterval[-1]
#     left, right = [], []
#     for i in intervals:
#         if i[-1] < start:
#             left.append(i)
#         elif i[0] > end:
#             right.append(i)
#         else:
#             start = min(i[0], start)
#             end = max(i[-1], end)
#
#     return left + [[start, end]] + right
#
#
# print(insert([[1, 3], [6, 9]], [2, 5]))
# print(insert([[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]], [4, 8]))


# def findMaxLength(nums):
#     count = 0
#     max_length = 0
#     table = {0: -1}
#     for index, num in enumerate(nums):
#         count += -1 if num == 0 else 1
#
#         if count in table:
#             max_length = max(max_length, index - table[count])
#         else:
#             table[count] = index
#
#     return max_length
#
#
# print(findMaxLength([0, 1]))
# print(findMaxLength([0, 1, 0]))


# def productExceptSelf(nums):
#     left_products = [1] * len(nums)
#     right_products = [1] * len(nums)
#     result = [1] * len(nums)
#
#     for i in range(1, len(nums)):
#         left_products[i] = nums[i - 1] * left_products[i - 1]
#
#     for i in range(len(nums) - 2, -1, -1):
#         right_products[i] = nums[i + 1] * right_products[i + 1]
#
#     for i in range(len(nums)):
#         result[i] = left_products[i] * right_products[i]
#
#     return result
#
#
# print(productExceptSelf([1, 2, 3, 4]))
# print(productExceptSelf([-1, 1, 0, -3, 3]))


# def numSubarraysWithSum(nums, goal):
#     count = 0
#     current_sum = 0
#     sum_counts = collections.Counter({0: 1})
#     for num in nums:
#         current_sum += num
#         count += sum_counts[current_sum - goal]
#         sum_counts[current_sum] += 1
#     return count
#
#
# print(numSubarraysWithSum([1, 0, 1, 0, 1], 2))
# print(numSubarraysWithSum([0, 0, 0, 0, 0], 0))


# def pivotInteger(n):
#     left_sum = 0
#     right_sum = sum(range(1, n + 1))
#
#     for i in range(n, 0, - 1):
#         left_sum += i
#         right_sum -= i
#         if left_sum == right_sum + i:
#             return i
#     return - 1
#
#
# print(pivotInteger(8))
# print(pivotInteger(4))
# print(pivotInteger(1))


# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# def removeZeroSumSublists(head):
#     dummy = ListNode(0)
#     dummy.next = head
#     prefix_sum = 0
#     sum_to_node = {0: dummy}
#
#     current = head
#     while current:
#         prefix_sum += current.val
#         sum_to_node[prefix_sum] = current
#         current = current.next
#
#     current = dummy
#     prefix_sum = 0
#     # Второй проход: Используем отображение, чтобы пропустить узлы, формирующие последовательность с нулевой суммой
#     while current:
#         prefix_sum += current.val
#         # Напрямую соединяем текущий узел с последним узлом, который имел такую же префиксную сумму
#         current.next = sum_to_node[prefix_sum].next
#         current = current.next
#
#     return dummy.next
#
#
# # Вспомогательная функция для печати списка
# def printList(node):
#     while node:
#         print(node.val, end=' ')
#         node = node.next
#     print()
#
#
# # Пример использования
# head = ListNode(1, ListNode(2, ListNode(-3, ListNode(3, ListNode(1)))))
# result = removeZeroSumSublists(head)
# printList(result)  # Ожидается: 3 1
#
# head = ListNode(1, ListNode(2, ListNode(3, ListNode(-3, ListNode(4)))))
# result = removeZeroSumSublists(head)
# printList(result)  # Ожидается: 1 2 4
#
# head = ListNode(1, ListNode(2, ListNode(3, ListNode(-3, ListNode(-2)))))
# result = removeZeroSumSublists(head)
# printList(result)  # Ожидается: 1


# def customSortString(order, s):
#     count = dict(collections.Counter(s))
#     result = ''
#     for i in order:
#         if i in count:
#             result += i * count[i]
#             del count[i]
#
#     for i in count:
#         result += i * count[i]
#
#     return result
#
#
# print(customSortString("bcafg", "abcd"))  # bcad
# print(customSortString("cba", "abcd"))  # cbad
# print(customSortString("kqep", "pekeq"))  # kqeep


# def intersection(nums1, nums2):
#     return list(set(nums1) & set(nums2))
#
#
# print(intersection([4, 9, 5], [9, 4, 9, 8, 4]))
# print(intersection([1, 2, 2, 1], [2, 2]))


# def getCommon(nums1, nums2):
#     result = set(nums1) & set(nums2)
#     return min(result) if len(result) else - 1
#
#
# print(getCommon([1, 2, 3], [2, 4]))
# print(getCommon([1, 2, 3, 6], [2, 3, 4, 5]))
# print(getCommon([2, 4], [1, 2]))


# def maxFrequencyElements(nums):
#     result = collections.Counter(nums).values()
#     max_res = max(result)
#     return sum([i for i in result if i == max_res])
#
#
# print(maxFrequencyElements([1, 2, 2, 3, 1, 4]))
# print(maxFrequencyElements([1, 2, 3, 4, 5]))
# print(maxFrequencyElements([10, 12, 11, 9, 6, 19, 11]))
# print(maxFrequencyElements([17, 17, 2, 12, 20, 17, 12]))


# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# class Solution(object):
#     def middleNode(self, head):
#         slow = fast = head
#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next
#         return slow
#
#
# # Вспомогательная функция для создания связного списка из списка значений
# def createLinkedList(elements):
#     head = ListNode(elements[0]) if elements else None
#     current = head
#     for element in elements[1:]:
#         current.next = ListNode(element)
#         current = current.next
#     return head
#
#
# # Функция для печати списка, начиная с заданного узла
# def printList(node):
#     current = node
#     while current:
#         print(current.val, end=" ")
#         current = current.next
#     print()
#
#
# head = createLinkedList([1, 2, 3, 4, 5])
# solution = Solution()
# middleNode = solution.middleNode(head)
# printList(middleNode)


# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
#
#
# class Solution(object):
#     def hasCycle(self, head):
#         slow = fast = head
#         while fast and fast.next:
#             slow, fast = slow.next, fast.next.next
#             if fast == slow:
#                 return True
#         return False


# def minimumlength(s):
#     left, right = 0, len(s) - 1
#
#     while left < right and s[left] == s[right]:
#         char = s[left]
#         while left <= right and s[left] == char:
#             left += 1
#         while right >= left and s[right] == char:
#             right -= 1
#
#     return right - left + 1

# или

# def minimumlength(s):
#     while len(s) >= 2 and s[0] == s[-1]:
#         s = s.strip(s[0])
#     return len(s)
#
#
# print(minimumlength('aabccabba'))
# print(minimumlength('cabaabac'))
# print(minimumlength('ca'))


# def maxScore(tokens, power):
#     tokens.sort()  # Сортируем токены по возрастанию
#     left, right = 0, len(tokens) - 1
#     score = 0
#     maxScore = 0
#
#     while left <= right:
#         if power >= tokens[left]:
#             # Играем токен лицевой стороной вверх
#             power -= tokens[left]
#             score += 1
#             maxScore = max(maxScore, score)
#             left += 1
#         elif score > 0:
#             # Играем токен лицевой стороной вниз
#             power += tokens[right]
#             score -= 1
#             right -= 1
#         else:
#             break  # Не можем сыграть ни одним способом
#
#     return maxScore
#
#
# print(maxScore([100], 50))  # Вывод: 0
# print(maxScore([200, 100], 150))  # Вывод: 1
# print(maxScore([100, 200, 300, 400], 200))  # Вывод: 2


# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
#
#
# class Solution:
#     def removeNthFromEnd(self, head, n):
#         slow, fast = head, head
#         for _ in range(n):
#             fast = fast.next
#         if not fast:
#             return head.next
#         while fast.next:
#             slow = slow.next
#             fast = fast.next
#         slow.next = slow.next.next
#         return head
#
#
# def printList(head):
#     """Функция для печати элементов связного списка"""
#     current = head
#     while current:
#         print(current.val, end=' ')
#         current = current.next
#     print()
#
#
# # Создание связного списка [1, 2, 3, 4, 5]
# head = ListNode(1)
# head.next = ListNode(2)
# head.next.next = ListNode(3)
# head.next.next.next = ListNode(4)
# head.next.next.next.next = ListNode(5)
#
# solution = Solution()
# n = 2
# head = solution.removeNthFromEnd(head, n)
# printList(head)


# def sortedSquares(nums):
#     result = [i ** 2 for i in nums]
#     return sorted(result)
#
#
# print(sortedSquares([-4, -1, 0, 3, 10]))
# print(sortedSquares([-7, -3, 2, 3, 11]))


# class Solution(object):
#     def maximumOddBinaryNumber(self, s):
#         """
#         :type s: str
#         :rtype: str
#         """
#         ones = s.count('1')
#         zeros = s.count('0')
#         if ones == 1:
#             return zeros * '0' + str(ones)
#         return (ones - 1) * '1' + zeros * '0' + '1'
#
#
# solution = Solution()
# print(solution.maximumOddBinaryNumber('010'))
# print(solution.maximumOddBinaryNumber('0101'))
# print(solution.maximumOddBinaryNumber('1'))


# def remove_duplicates(dicts):
#     seen = set()
#     unique_dicts = []
#
#     for d in dicts:
#         # Преобразование словаря в кортеж (ключ, значение)
#         hashable = tuple(d.items())
#         # Проверяем, был ли уже такой "хешируемый" словарь
#         if hashable not in seen:
#             seen.add(hashable)
#             unique_dicts.append(d)
#
#     return unique_dicts
#
#
# dicts = [
#     {"key1": "value1"},
#     {"k1": "v1", "k2": "v2", "k3": "v3"},
#     {},
#     {},
#     {"key1": "value1"},
#     {"key1": "value1"},
#     {"key2": "value2"}
# ]
#
# print(remove_duplicates(dicts))


# class Solution:
#     def canTraverseAllPairs(self, nums):
#         if len(nums) == 1: return True
#         nums_set = set(nums)
#         if 1 in nums_set: return False
#         nums_sorted = sorted(nums_set, reverse=True)
#
#         for i in range(len(nums_sorted) - 1):
#             for j in range(i + 1, len(nums_sorted)):
#                 if self.gcd(nums_sorted[i], nums_sorted[j]) - 1:
#                     nums_sorted[j] *= nums_sorted[i]
#                     break
#             else:
#                 return False
#         return True
#
#     def gcd(self, a, b):
#         while b:
#             a, b = b, a % b
#         return a
#
#
# sol = Solution()
# print(sol.canTraverseAllPairs([2, 3, 6]))  # Вывод: true
# print(sol.canTraverseAllPairs([3, 9, 5])) # Вывод: false
# print(sol.canTraverseAllPairs([4, 3, 12, 8]))  # Вывод: true


# def findallpeople(n, meetings, firstperson):
#     graph = collections.defaultdict(list)
#
#     for x, y, t in meetings:
#         graph[x].append((y, t))
#         graph[y].append((x, t))
#
#     heap = [(0, firstperson)]
#     for nei, time in graph[0]:
#         heapq.heappush(heap, (time, nei))
#
#     visited = set([0])
#
#     while heap:
#         time, person = heapq.heappop(heap)
#         if person in visited:
#             continue
#
#         visited.add(person)
#
#         for nei, t in graph[person]:
#             if t >= time:
#                 heapq.heappush(heap, (t, nei))
#
#     return visited
#
#
# print(findallpeople(6, [[1, 2, 5], [2, 3, 8], [1, 5, 10]], 1))  # [0, 1, 2, 3, 5]


# def findcheapestprice(n, flights, src, dst, k):
#     cost = [[float('inf')] * n for _ in range(k + 2)]
#     cost[0][src] = 0
#
#     for i in range(1, k + 2):
#         cost[i][src] = 0
#         for from_city, to_city, price in flights:
#             cost[i][to_city] = min(cost[i][to_city], cost[i - 1][from_city] + price)
#             print(cost[i][to_city])
#
#     return cost[k + 1][dst] if cost[k + 1][dst] != float('inf') else -1
#
#
# print(findcheapestprice(4, [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]], 0, 3, 1))


# def findоudge(n, trust):
#     trusts_count = [0] * (n + 1)
#     trusted_by_count = [0] * (n + 1)
#
#     for a, b in trust:
#         trusts_count[a] += 1
#         trusted_by_count[b] += 1
#
#     for i in range(1, n + 1):
#         if trusts_count[i] == 0 and trusted_by_count[i] == n - 1:
#             return i
#
#     return -1
#
#
# print(findоudge(3, [[1, 2],[2, 3]]))
# print(findоudge(3, [[1, 3],[2, 3]]))
# print(findоudge(3, [[1, 3],[2, 3],[3, 1]]))


# def rangebitwiseand(left, right):
#     shift = 0
#     # Сдвигаем, пока числа не станут равны
#     while left < right:
#         left >>= 1
#         right >>= 1
#         shift += 1
#     # Сдвигаем результат обратно влево
#     return left << shift
#
#
# print(rangebitwiseand(5, 7))


# def roomwithmostmeetings(n, meetings):
#     meetings.sort()
#     hm = [0] * n
#     heap, occupied_heap = [i for i in range(n)], []
#     heapq.heapify(heap)
#
#     for i in range(len(meetings)):
#         start, end = meetings[i]
#
#         while occupied_heap and occupied_heap[0][0] <= start:
#             end_time, room_num = heapq.heappop(occupied_heap)
#             heapq.heappush(heap, room_num)
#         if heap:
#             unused_room = heapq.heappop(heap)
#             hm[unused_room] += 1
#             heapq.heappush(occupied_heap, (end, unused_room))
#         else:
#             end_time, room_num = heapq.heappop(occupied_heap)
#             hm[room_num] += 1
#             heapq.heappush(occupied_heap, (end_time + end - start, room_num))
#
#     return hm.index(max(hm))
#
#
# print(roomwithmostmeetings(2, [[0, 10], [1, 5], [2, 7], [3, 4]]))  # 0
# print(roomwithmostmeetings(3, [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]]))  # 1


# def furthestbuilding(heights, bricks, ladders):
#     heap = []
#     for i in range(len(heights) - 1):
#         diff = heights[i + 1] - heights[i]
#         if diff > 0:
#             heapq.heappush(heap, diff)
#         if len(heap) > ladders:
#             bricks -= heapq.heappop(heap)
#         if bricks < 0:
#             return i
#     return len(heights) - 1
#
#
# print(furthestbuilding([4, 12, 2, 7, 3, 18, 20, 3, 19], 10, 2))  # 7
# print(furthestbuilding([4, 2, 7, 6, 9, 14, 12], 5, 1))  # 4
# print(furthestbuilding([14, 3, 19, 3], 17, 0))  # 3


# def findLeastNumOfUniqueInts(arr, k):
#     freq = collections.Counter(arr)
#     sorted_items = sorted(freq.items(), key=lambda x: x[1])
#     for num, count in sorted_items:
#         if k >= count:
#             k -= count
#             del freq[num]
#         else:
#             break
#
#     return len(freq)
#
#
# print(findLeastNumOfUniqueInts([5, 5, 4], 1))  # 1
# print(findLeastNumOfUniqueInts([4, 3, 1, 1, 3, 3, 2], 3))  # 2


# def largestperimeter(nums):
#     nums.sort()
#     elements_sum = 0
#     ans = -1
#     for num in nums:
#         if num < elements_sum:
#             ans = num + elements_sum
#         elements_sum += num
#     return ans
#
#
# print(largestperimeter([5, 5, 5]))
# print(largestperimeter([5, 5, 50]))
# print(largestperimeter([1, 12, 1, 2, 5, 50, 3]))


# def rearrangearray(nums):
#     pos, neg = [], []
#     result = []
#
#     res = [pos.append(i) if i >= 0 else neg.append(i) for i in nums]
#     for k, v in list(zip(pos, neg)):
#         result.append(k)
#         result.append(v)
#     return result
#
#
# print(rearrangearray([3, 1, -2, -5, 2, -4]))  # [3, -2, 1, -5, 2, -4]
# print(rearrangearray([-1, 1]))  # [1, -1]


# def increment_string(strng):
#     num_part = ''
#     found_number = False
#     for k, v in enumerate(strng[::-1]):
#         if v.isdigit():
#             num_part = v + num_part
#             found_number = True
#         elif found_number:
#             break
#     if not found_number:
#         return strng + '1'
#
#     index = len(strng) - len(num_part)
#     incremented = str(int(num_part) + 1).zfill(len(num_part))
#     return strng[:index] + incremented

# или

# def increment_string(strng):
#     head = strng.rstrip('0123456789')
#     tail = strng[len(head):]
#     if tail == "": return strng + "1"
#     return head + str(int(tail) + 1).zfill(len(tail))
#
#
# print(increment_string("foobar001"))
# print(increment_string("fo99obar99"))
# print(increment_string("foobar"))


# def firstpalindrome(words):
#     result = [i for i in words if i == i[::-1]][:1]
#     return ''.join(result) if result else ''
#
#
# print(firstpalindrome(["abc", "car", "ada", "racecar", "cool"]))
# print(firstpalindrome(["notapalindrome", "racecar"]))
# print(firstpalindrome(["def", "ghi"]))


# def twosum(nums, target):
#     result = {}
#     for k, v in enumerate(nums):
#         complement = target - v
#         if complement in result:
#             return [result[complement], k]
#         result[v] = k
#     return []
#
#
# print(twosum([2, 7, 11, 15], 9))
# print(twosum([3, 2, 4], 6))
# print(twosum([3, 3], 6))


# def majorityelement(nums):
#     result = collections.Counter(nums)
#     return max(result.keys(), key=result.get)

# или

# def majorityelement(nums):
#     nums.sort()
#     n = len(nums)
#     return nums[n // 2]
#
#
# print(majorityelement([2, 2, 1, 1, 1, 2, 2]))
# print(majorityelement([3, 2, 3]))
# print(majorityelement([3, 3, 4]))


# def ip_to_int(ip):
#     result = map(int, ip.split('.'))
#     return sum(
#         part << (8 * (3 - index))
#         for index, part in enumerate(result)
#     )
#
#
# def ips_between(start, end):
#     first_int = ip_to_int(start)
#     last_int = ip_to_int(end)
#     return last_int - first_int

# или

# from ipaddress import ip_address
#
# def ips_between(start, end):
#     return int(ip_address(end)) - int(ip_address(start))
#
#
# print(ips_between("10.0.0.0", "10.0.0.50"))
# print(ips_between("20.0.0.10", "20.0.1.0"))


# def cherrypickup(grid):
#     n, m = len(grid), len(grid[0])
#     memo = {}
#
#     def dp(r, c1, c2):
#         if (r, c1, c2) in memo:
#             return memo[(r, c1, c2)]
#         if c1 < 0 or c1 >= m or c2 < 0 or c2 >= m:
#             return float('-inf')
#
#         result = grid[r][c1] + (grid[r][c2] if c1 != c2 else 0)
#         if r != n - 1:
#             result += max(
#                 dp(r + 1, newc1, newc2)
#                 for newc1 in [c1 - 1, c1, c1 + 1]
#                 for newc2 in [c2 - 1, c2, c2 + 1]
#             )
#             memo[(r, c1, c2)] = result
#         return result
#     return dp(0, 0, m - 1)
#
#
# print(cherrypickup([[3, 1, 1], [2, 5, 1], [1, 5, 5], [2, 1, 1]]))
# print(cherrypickup([[1, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 3, 0], [2, 0, 9, 0, 0, 0, 0], [0, 3, 0, 5, 4, 0, 0], [1, 0, 2, 3, 0, 0, 6]]))


# def countsubstrings(s):
#     def helper(left, right):
#         count = 0
#         while left >= 0 and right < len(s) and s[left] == s[right]:
#             count += 1
#             left -= 1
#             right += 1
#         return count
#
#     result = 0
#     for i in range(len(s)):
#         result += helper(i, i)
#         result += helper(i, i + 1)
#
#     return result
#
#
# print(countsubstrings('a'))
# print(countsubstrings('aba'))
# print(countsubstrings('abc'))
# print(countsubstrings('aaa'))


# def largestdivisiblesubset(nums):
#     n = len(nums)
#     if n == 0:
#         return []
#     nums.sort()
#     dp = [[i] for i in nums]
#
#     for i in range(n):
#         for j in range(i):
#             if nums[i] % nums[j] == 0 and len(dp[j]) + 1 > len(dp[i]):
#                 dp[i] = dp[j] + [nums[i]]
#
#     return max(dp, key=len)
#
#
# print(largestdivisiblesubset([1, 2, 4, 8]))
# print(largestdivisiblesubset([1, 2, 3]))


# def firstuniqchar(s):
#     result = collections.Counter(s)
#     for k, v in enumerate(s):
#         if result[v] == 1:
#             return k
#     return -1
#
#
# print(firstuniqchar('z'))
# print(firstuniqchar('aabb'))
# print(firstuniqchar('loveleetcode'))
# print(firstuniqchar('leetcode'))


# def numsquares(n):
#     dp = [float('inf')] * (n + 1)
#     dp[0] = 0
#     for i in range(1, n + 1):
#         j = 1
#         while j*j <= i:
#             dp[i] = min(dp[i], dp[i - j*j] + 1)
#             j += 1
#     return dp[n]
#
#
# print(numsquares(12))  # Вывод: 3


# def frequencysort(s):
#
#     res = []
#     result = collections.Counter(s[::-1]).most_common()
#     for i, j in result:
#         res.append(i * j)
#
#     return ''.join(res)
#
#
# print(frequencysort('tree'))  # eert
# print(frequencysort('cccaaa'))  # aaaccc
# print(frequencysort('Aabb'))  # bbAa
# print(frequencysort('raaeaedere'))  # "eeeeaaarrd"


# def anagram(strs):
#     anagrams = {}
#     for i in strs:
#         key = tuple(sorted(i))
#         if key in anagrams:
#             anagrams[key].append(i)
#         else:
#             anagrams[key] = [i]
#     return list(anagrams.values())
#
#
# print(anagram(["eat", "tea", "tan", "ate", "nat", "bat"]))  # [["bat"],["nat","tan"],["ate","eat","tea"]]


# def domain_name(url):
#     url = re.sub(r'https?://', '', url)
#     url = re.sub(r'www\.', '', url)
#     result = re.split(r'/|\.', url)[0]
#     return result

# или

# def domain_name(url):
#     return url.split("//")[-1].split("www.")[-1].split(".")[0]
#
#
# print(domain_name("http://google.com"))
# print(domain_name("http://google.co.jp"))
# print(domain_name("https://123.net"))
# print(domain_name("icann.org"))
# print(domain_name("https://hyphen-site.org"))


# def min_path(grid, x, y):
#     s = [float('inf')] * (x + 2)
#     s[1] = 0
#
#     for i in range(y + 1):
#         for j in range(x + 1):
#             s[j + 1] = min(s[j], s[j + 1]) + grid[i][j]
#
#     return s[-1]
#
#
# square = [
#     [1, 2, 3, 6, 2, 8, 1],
#     [4, 8, 2, 4, 3, 1, 9],
#     [1, 5, 3, 7, 9, 3, 1],
#     [4, 9, 2, 1, 6, 9, 5],
#     [7, 6, 8, 4, 7, 2, 6],
#     [2, 1, 6, 2, 4, 8, 7],
#     [8, 4, 3, 9, 2, 5, 8]]
#
# print(min_path(square, 6, 6))


# def cakes(recipe, available):
# 	return min(available.get(k, 0) // recipe[k] for k in recipe)

# или

# def cakes(recipe, available):
#     result = []
#     for i in recipe:
#         if available.get(i) is None:
#             return 0
#         result.append(available[i] // recipe[i])
#     return min(result)
#
#
# r = {"flour": 500, "sugar": 200, "eggs": 1}
# a = {"flour": 1200, "sugar": 1200, "eggs": 5, "milk": 200}
# r = {"apples": 3, "flour": 300, "sugar": 150, "milk": 100, "oil": 100}
# a = {"sugar": 500, "flour": 2000, "milk": 2000}
# print(cakes(r, a))


# PIN = {
#     '1': ('1', '2', '4'),
#     '2': ('1', '2', '3', '5'),
#     '3': ('2', '3', '6'),
#     '4': ('1', '4', '5', '7'),
#     '5': ('2', '4', '5', '6', '8'),
#     '6': ('5', '6', '9', '3'),
#     '7': ('4', '7', '8'),
#     '8': ('7', '5', '8', '9', '0'),
#     '9': ('6', '8', '9'),
#     '0': ('0', '8')
# }
#
#
# def get_pins(observed):
#     return [''.join(a) for a in product(*(PIN[b] for b in observed))]
#
#
# print(get_pins('76'))


# def find_reverse_number(n):
#     if n < 11:
#         return n - 1
#     n_zeros = len(str(n // 11))
#     left = str(n - 10 ** n_zeros)
#     right = left[n_zeros - 1::-1]
#     return int(left + right)
#
#
# print(find_reverse_number(100))


# def longest_slide_down(pyramid):
#     for row in range(len(pyramid)-2, -1, -1):
#         for col in range(len(pyramid[row])):
#             pyramid[row][col] += max(pyramid[row + 1][col], pyramid[row + 1][col + 1])
#     return pyramid[0][0]
#
#
# print(longest_slide_down([[3], [7, 4], [2, 4, 6], [8, 5, 9, 3]]))


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#
# def isEvenOddTree(root):
#     if not root:
#         return True
#
#     queue = deque([root])
#
#     level = 0
#     while queue:
#         level_size = len(queue)
#         prev_value = None
#
#         for _ in range(level_size):
#             node = queue.popleft()
#
#             # Проверка для уровней с четным индексом
#             if level % 2 == 0:
#                 # Значения должны быть нечетными и строго возрастающими
#                 if node.val % 2 == 0 or (prev_value is not None and prev_value >= node.val):
#                     return False
#             else:
#                 # Проверка для уровней с нечетным индексом
#                 # Значения должны быть четными и строго убывающими
#                 if node.val % 2 != 0 or (prev_value is not None and prev_value <= node.val):
#                     return False
#
#             prev_value = node.val
#
#             if node.left:
#                 queue.append(node.left)
#             if node.right:
#                 queue.append(node.right)
#
#         level += 1
#
#     return True
#
#
# root = TreeNode(1, TreeNode(10, TreeNode(3, TreeNode(12), TreeNode(8)), TreeNode(7, None, TreeNode(6))),
#                 TreeNode(4, TreeNode(9, None, TreeNode(2))))
# print(isEvenOddTree(root))


# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
# class Solution(object):
#     def diameterOfBinaryTree(self, root):
#         self.max_diameter = 0
#
#         def depth(node):
#             if not node:
#                 return 0
#             left_depth = depth(node.left)
#             right_depth = depth(node.right)
#             self.max_diameter = max(self.max_diameter, left_depth + right_depth)
#             return max(left_depth, right_depth) + 1
#
#         depth(root)
#         return self.max_diameter
#
#
# # Создаем узлы дерева
# root = TreeNode(1)
# root.left = TreeNode(2)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# root.right = TreeNode(3)
#
# solution = Solution()
# print(solution.diameterOfBinaryTree(root))


# class Node:
#     def __init__(self, L, R, n):
#         self.left = L
#         self.right = R
#         self.value = n
#
#
# def tree_by_sum(node):
#     if node is None:
#         return 0
#
#     max_l = tree_by_sum(node.left)
#     max_r = tree_by_sum(node.right)
#     return max_l + max_r + node.value
#
#
# res = Node(Node(None, Node(None, None, -4), -2), Node(Node(None, None, 5), Node(None, None, 6), 3), 1)
# print(tree_by_sum(res))


# class Node:
#     def __init__(self, L, R, n):
#         self.left = L
#         self.right = R
#         self.value = n
#
#
# def tree_by_sum(root):
#     if root is None:
#         return 0
#
#     max_l = tree_by_sum(root.left)
#     max_r = tree_by_sum(root.right)
#     return max(max_l, max_r) + root.value
#
#
# node = Node(Node(Node(None, None, 4), Node(None, None, 5), 2), Node(None, None, 3), 1)
# print(tree_by_sum(node))


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
# class Solution:
#     def isSameTree(self, p, q):
#         # Если оба узла равны None, значит на этом этапе они одинаковы
#         if not p and not q:
#             return True
#         # Если один из узлов равен None, или их значения не совпадают, деревья не одинаковы
#         if not p or not q or p.val != q.val:
#             return False
#         # Рекурсивно проверяем левых и правых потомков
#         return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
#
#
# p = TreeNode(1)
# p.left = TreeNode(2)
# p.right = TreeNode(3)
#
# q = TreeNode(1)
# q.left = TreeNode(2)
# q.right = TreeNode(3)
#
# solution = Solution()
# print(solution.isSameTree(p, q))


# class Node:
#     def __init__(self, L, R, n):
#         self.left = L
#         self.right = R
#         self.value = n
#
#
# def tree_by_levels(root):
#     if root is None:
#         return []
#
#     queue = [root]
#     level_order = []
#
#     while queue:
#         current_root = queue.pop(0)
#         level_order.append(current_root.value)
#
#         if current_root.left:
#             queue.append(current_root.left)
#
#         if current_root.right:
#             queue.append(current_root.right)
#
#     return level_order
#
#
# node = Node(Node(None, Node(None, None, 4), 2), Node(Node(None, None, 5), Node(None, None, 6), 3), 1)
# print(tree_by_levels(node))


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#
# def findBottomLeftValue(root):
#     if not root:
#         return None
#
#     queue = deque([(root, 0)])  # (узел, уровень)
#     current_level = 0
#     leftmost_value = root.val
#
#     while queue:
#         node, level = queue.popleft()
#
#         # Когда достигаем нового уровня, обновляем самое левое значение
#         if level > current_level:
#             current_level = level
#             leftmost_value = node.val
#
#         if node.left:
#             queue.append((node.left, level + 1))
#         if node.right:
#             queue.append((node.right, level + 1))
#
#     return leftmost_value
#
#
# root = TreeNode(1)
# root.left = TreeNode(2, TreeNode(4))
# root.right = TreeNode(3, TreeNode(5, TreeNode(7)), TreeNode(6))
#
# print(findBottomLeftValue(root))  # Вывод: 7


# def strip_comments(string, markers):
#     lines = string.split('\n')
#     for i, line in enumerate(lines):
#         min_index = len(line)
#         for marker in markers:
#             index = line.find(marker)
#             if index != -1 and index < min_index:
#                 min_index = index
#         lines[i] = line[:min_index].rstrip()
#     return "\n".join(lines)
#
#
# print(strip_comments("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"]))

# или

# def strip_comments(string, markers):
#     for num, marker in enumerate(markers):
#         string = '\n'.join([i.split(marker)[0].rstrip() for i in string.split('\n')])
#
#     return string
#
#
# print(strip_comments("apples, pears # and bananas\ngrapes\nbananas !apples", ["#", "!"]))


# def permutations(s):
#     result = set()
#     for i in itertools.permutations(s):
#         if ''.join(i) not in result:
#             result.add(''.join(i))
#     return list(result)
#
#
# print(permutations('aabb'))

# или

# def permutations(s):
#     return list("".join(i) for i in set(itertools.permutations(s)))
#
#
# print(permutations('aabb'))


# def sum_strings(x, y):
#     x, y = x[::-1], y[::-1]
#     result = []
#     carry = 0
#
#     for i in range(max(len(x), len(y))):
#         dig_x = int(x[i]) if i < len(x) else 0
#         dig_y = int(y[i]) if i < len(y) else 0
#         total = dig_x + dig_y + carry
#         carry = total // 10
#         result.append(total % 10)
#     if carry:
#         result.append(carry)
#
#     return ''.join(map(str, result))[::-1].lstrip('0') or '0'
#
#
# print(sum_strings("123", "456999"))

# или

# from gmpy2 import mpz
#
# def sum_strings(x, y):
#     return str(mpz(x or '0') + mpz(y or '0'))
#
#
# print(sum_strings("123", "456999"))


# def next_smaller(n):
#     digits = list(str(n))
#     for i in range(len(digits) - 2, -1, -1):
#         if digits[i] > digits[i + 1]:
#             break
#     else:
#         return - 1
#
#     for j in range(len(digits) - 1, i, -1):
#         if digits[j] < digits[i]:
#             break
#
#     digits[i], digits[j] = digits[j], digits[i]
#     digits[i + 1:] = sorted(digits[i + 1:], reverse=True)
#     result = int(''.join(digits))
#
#     if str(result) != ''.join(map(str, digits)):
#         return - 1
#
#     return result
#
#
# print(next_smaller(1027))


# def next_bigger(n):
#     digits = list(str(n))
#
#     # Шаг 1: Найти первую цифру, которая меньше цифры рядом с ней, справа налево
#     for i in range(len(digits) - 2, -1, -1):
#         if digits[i] < digits[i + 1]:
#             break
#     else:
#         return -1
#
#     # Шаг 2: Найти наименьшую цифру справа от (i-й цифры), которая больше digits[i]
#     for j in range(len(digits) - 1, i, -1):
#         if digits[j] > digits[i]:
#             break
#
#     digits[i], digits[j] = digits[j], digits[i]
#     digits[i + 1:] = sorted(digits[i + 1:])
#     return int("".join(digits))
#
#
# print(next_bigger(2017))


# def snail(snail_map):
#     result = []
#     while snail_map:
#         result += snail_map.pop(0)
#         if snail_map and snail_map[0]:
#             for row in snail_map:
#                 result.append(row.pop())
#         if snail_map:
#             result += snail_map.pop()[::-1]
#         if snail_map and snail_map[0]:
#             for row in snail_map[::-1]:
#                 result.append(row.pop(0))
#
#     return result
#
#
# array = [[1, 2, 3],
#          [8, 9, 4],
#          [7, 6, 5]]
#
# print(snail(array))  #=> [1,2,3,4,5,6,7,8,9]

# или

# def snail(snail_map):
#     result = []
#     while len(snail_map):
#         result += snail_map.pop(0)
#         snail_map = list(zip(*snail_map))[::-1]
#     return result
#
#
# array = [[1, 2, 3],
#          [8, 9, 4],
#          [7, 6, 5]]
#
# print(snail(array))  #=> [1,2,3,4,5,6,7,8,9]


# res = [i for i in range(1000) if i % 3 == 0 and i % 5 != 0]
# for i in res:
#     if i % 100 // 10 + i % 10 + i // 100 < 10:
#         print(i, end=' ')

# u = [1, 2, 4, 6, 5, 1]
# a = True if len(u) != len(set(u)) else False
# print(a)

# x = '1234'
# if not isinstance(x, str):
#     res = [int(i) for i in str(x)]
#     print(sum(res))
# else:
#     print("Че, самый умный?")

# r = sum(map(int, str(x)))
# print(r)

# res = list(map(int, str(x)))
# result_1 = res[0] * res[1] * res[2] * res[3]
# result_2 = reduce(lambda x, y: x * y, res)
# print(f'Сумма: {sum(res)}, произведение: {result_1}')
# print(f'Сумма: {sum(res)}, произведение: {result_2}')

# def convert(n):
#     return ''.join(map(str, range(n + 1)))
#
#
# print(convert(10))

# def merge(a, b):
#     return dict(sorted(zip(a, b)))
#
#
# d = ['a', 'v', 'b', 'c']
# c = [1, 3, 2, 3, 4]
# s = merge(d, c)

# for k, v in s.items():
#     print(k, v)

# for j in enumerate(s):
#     print(j)

# num = int(input("Введите целое число: "))
# summ = [num * i for i in range(1, 5 + 1)]
# print(*summ, sep='---')


# while a != 123:
#     a = int(input('Введите пароль: '))
# print("Пароль верный")

# a = int(input("Введите пароль: "))
# b = int(input("Повторите пароль: "))
# print('Пароль принят' if a == b else 'Пароль не принят')

# t = int(input("Введите число: "))
# result = ('t - четное') if t % 2 == 0 else ('t - не четное')
# print(result)

# my_list = [1, 3, 4, 6, 10, 11, 15, 12, 14]
# new_list_1 = list(filter(lambda x: x % 2 == 0, my_list))
# new_list_2 = [i for i in my_list if i % 2 == 0]
# print(new_list_1)
# print(new_list_2)

# current_list = [5, 15, 20, 30, 50, 55, 75, 60, 70]
# summa = reduce((lambda x, y: x + y), current_list)
# print(summa)
# print(sum(current_list))

# c = map(int, input("Введите число: ").split())
# print(min(c))

# a, b, c, d = int(input('введите 1: ')), int(input('введите 2: ')), int(input('введите 3: ')), int(input('введите 4: '))
# if a > b:
#     a = b
# elif c > d:
#     c = d
# elif a > c:
#     print(c)
#
# print(min(a, b, c, d))

# c = list(map(int, input("Введите число: ").split()))
# v = filter(lambda x: x % 2 == 0, c)
# n = filter(lambda x: x > 0, c)
# print(sum(v), sum(n))


# a = int(input("Введите 1 число: "))
# m = int(input("Введите 2 число: "))
# w = input("Введите действие: ")


# def calc(a, b):
#     if w == "+":
#         return a + b
#     elif w == '-':
#         return a - b
#     elif w == '*':
#         return a * b
#     elif w == '/':
#         if b == 0:
#             return 'На ноль делить нельзя!'
#         return a / b
#     else:
#         return 'Неверная операция'
#
#
# res = calc(a, m)
# print(res)

# def func(w):
#     match w:
#         case "+":
#             print(f'Результат ввода {w} = {a + m}')
#         case "-":
#             print(a - m)
#         case "/":
#             print('На ноль делить нельзя!' if m == 0 else a / m)
#         case "*":
#             print(a * m)
#         case _:
#             print('Неверная операция')


# a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
# d = [i for i in a if i < 5]
# f = list(filter(lambda x: x < 5, a))
# print(d)
# print(f)

# a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
# b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# c = list(set(a) & set(b))
# r = list(filter(lambda x: x in b, a))
# t = [i for i in a if i in b]
# print(c)
# print(r)
# print(t)

# d = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
#
# res = dict(sorted(d.items(), key=lambda x: x[1]))
# print(res)
#
# result = dict(sorted(d.items(), key=operator.itemgetter(1)))
# result = dict(sorted(d.items(), key=operator.itemgetter(1), reverse=True))
# print(result)
#
# dict_a = {1: 10, 2: 20}
# dict_b = {3: 30, 4: 40}
# dict_c = {5: 50, 6: 60}
#
# result = {**dict_a, **dict_b, **dict_c}
# print(result)

# my_dict = {'a': 500, 'b': 5874, 'c': 560, 'd': 400, 'e': 5874, 'f': 20}
# res = sorted(my_dict.items(), key=lambda x: -x[1])[:3]
# result = sorted(my_dict, key=my_dict.get, reverse=True)[:3]
# print(*res)
# print(result)

# data = 'Казак'.lower()
# s = data[::-1]
# print("Палиндром" if data == s else "Не палиндром")


# def is_palindrome(string):
#     return string == string[::-1]

# def is_palindrome(string):
#     return string == ''.join(reversed(string))
#
# print(is_palindrome('казак'))

# num = int(input("Введите количесто сек: "))
# day = 86400
# print(f'Дни: {num // day}, часы: {num // 3600}, минуты: {num // 60}, секунды: {num}')
#
#
# def convert(seconds):
#     days = seconds // (24 * 3600)
#     seconds %= 24 * 3600
#     hours = seconds // 3600
#     seconds %= 3600
#     minutes = seconds // 60
#     seconds %= 60
#     print(f'{days}:{hours}:{minutes}:{seconds}')
#
#
# convert(86399)


# data = input("Введите несколько чисел через запятую: ").split(',')
# num = map(int, data)
# num_list = list(num)
# num_tupl = tuple(num_list)
# print(num_tupl)
# print(num_list)

# def factor(n):
#     match n:
#         case 0 | 1:
#             return 1
#         case _:
#             return n * factor(n - 1)


# print(factor(6))

# data = "rexment.txt"
# res = re.findall(r'\b\w{3}\b', data)
# print(*res)


# def func(file):
#     res = file.split('.')
#     if len(res) < 2:
#         raise ValueError("Не верное расширение")
#     f, *m, l = res
#     if not l or not f and not m:
#         raise ValueError("Что то не то")
#     return res[-1]
# 5
# print(func("rexment.txt"))

# r = int(input("Введите число: "))
# print(r + int(str(r) * 2) + int(str(r) * 3))

# a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 237, 568, 823]
# for i in a:
#     if i % 2 == 0:
#         print(i)
#     elif i == 237:
#         break

# string = "проплльаьлддыь"
# print(string.count("л"))
#
# a, b = 2, 4
# a, b = b, a
# print(a, b)
#
# nums = [45, 55, 60, 37, 100, 105, 220]
# res = list(filter(lambda x: x % 15 == 0, nums))
# print(res)


# text_in_mp3 = 'lorem ipsum dolor sit amet amet amet'
# words = text_in_mp3.split()
# counter = collections.Counter(words)
# most_common, occurrences = counter.most_common()[0]
# longest = max(words, key=len)
# print(most_common, longest)


# def get_count(sentence):
#     data = 'aeiouy'
#     res = [len(i) for i in sentence if i in data]
#     return sum(res)
#
#
# print(get_count('tyueindksljokaeb'))


# def disemvowel(string_):
#     data = 'aeiouy'
#     return ''.join([i for i in string_ if i.lower() not in data])
#
#
# print(disemvowel('This website is for losers LOL!'))


# def spin_words(sentence):
#     return ' '.join([i[::-1] if len(i) >= 5 else i for i in sentence.split()])
#
#
# print(spin_words("Hey fellow warriors"))


# def find_it(seq):
#     return [i for i in seq if seq.count(i) % 2 != 0][0]

# def find_it(seq):
#     return reduce(operator.xor, seq)


# print(find_it([1, 2, 2, 3, 3, 3, 4, 3, 3, 3, 2, 2, 1]))
#
#
# def square_digits(num):
#     return int(''.join([str(int(i) ** 2) for i in str(num)]))
#
#
# print(square_digits(9119))


# def digits(n):
#     return n if n < 10 else digits(sum(map(int, str(n))))
#
#
# print(digits(129))

# или


# def digital_root(n):
#     return n % 9 or n and 9
#
#
# print(digital_root(125))


# def create_phone_number(n):
#     res = ''.join(map(str, n))
#     return f'({res[:3]}) {res[3:6]}-{res[6:]}'


# def create_phone_number(n):
# 	return "({}{}{}) {}{}{}-{}{}{}{}".format(*n)
#
#
# print(create_phone_number([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))


# []                                -->  "no one likes this"
# ["Peter"]                         -->  "Peter likes this"
# ["Jacob", "Alex"]                 -->  "Jacob and Alex like this"
# ["Max", "John", "Mark"]           -->  "Max, John and Mark like this"
# ["Alex", "Jacob", "Mark", "Max"]  -->  "Alex, Jacob and 2 others like this"


# def likes(m):
#     match m:
#         case m if len(m) == 1:
#             return (f'{m[0]} likes this')
#         case m if len(m) == 2:
#             return (f'{m[0]} and {m[1]} like this')
#         case m if len(m) == 3:
#             return (f'{m[0]}, {m[1]} and {m[2]} like this')
#         case m if len(m) >= 4:
#             return (f'{m[0]}, {m[1]} and {len(m) - 2} others like this')
#         case _:
#             return ('no one likes this')


# m = ["Alex", "Jacob", "Mark", "Max"]
# print(likes(m))


# def array_diff(a, b):
#     res = [i for i in a if i not in b]
#     return res
#
#
# print(array_diff([1, 2], [1]))


# def high_and_low(numbers):
#     res = [int(i) for i in numbers.split()]
#     return f'{max(res)} {min(res)}'
#
#
# print(high_and_low("1 9 3 4 -5"))


# def unique_in_order(n):
#     return [i for i, _ in groupby(n)]
#     if not isinstance(n, str):
#         return list(set(n))
#
#
# print(unique_in_order('ABBCcAD'))
# print(unique_in_order(["a", "b", "b", "a"]))
# print(unique_in_order([1, 2, 2, 3, 3]))


# def move_zeros(n):
#     res = ''.join(map(str, n))
#     count = res.count('0')
#     add = res.replace('0', '') + '0' * count
#     return [int(i) for i in add]


# def move_zeros(array):
#     return [x for x in array if x] + [0] * array.count(0)
#     return sorted(array, key=lambda x: not x)
#
#
# print(move_zeros([1, 0, 1, 0, 2, 0, 1, 3]))


# def positive_sum(arr):
#     return sum(list(filter(lambda x: x > 0, arr)))
#
#
# a = [1, -4, 7, 12]
# print(positive_sum(a))


# def opposite(number):
#     return number - number * 2 if number > 0 else abs(number)
#     return - number
#
#
# print(opposite(-34))


# def summation(num):
#     return sum(range(1, num + 1))
#
#
# print(summation(8))
#
#
# def no_space(s):
#     return s.replace(' ', '')


# print(no_space("8 j 8   mBliB8g  imjB8B8  jl  B"))

# s = "8 j 8   mBliB8g  imjB8B8  jl  B"
# print(re.sub(r'\s', '', s))


# def pig_it(s):
#     if s.split()[-1] in '!?':
#         res = s[-1]
#         s = s[0:-1]
#         return ' '.join([i[1:] + i[0] + 'ay' for i in s.split()] + list(res))
#     return ' '.join([i[1:] + i[0] + 'ay' for i in s.split()])
#
#
# print(pig_it('Pig latin is cool ?'))  # igPay atinlay siay oolcay


# def zero(f=None): return 0 if not f else f(0)
# def one(f=None): return 1 if not f else f(1)
# def two(f=None): return 2 if not f else f(2)
# def three(f=None): return 3 if not f else f(3)
# def four(f=None): return 4 if not f else f(4)
# def five(f=None): return 5 if not f else f(5)
# def six(f=None): return 6 if not f else f(6)
# def seven(f=None): return 7 if not f else f(7)
# def eight(f=None): return 8 if not f else f(8)
# def nine(f=None): return 9 if not f else f(9)
# def times(y): return lambda x: x * y
# def plus(y): return lambda x: x + y
# def minus(y): return lambda x: x - y
# def divided_by(y): return lambda x: x // y
#
#
# print(seven(times(five())))


# def generate_hashtag(s):
#     if len(s) == 0: return False
#     res = ''.join(['#'] + [i.capitalize() for i in s.split()])
#     if len(res) > 140: return False
#     return res
#
#
# print(generate_hashtag(" Hello there thanks for trying my Kata"))  # "#HelloThereThanksForTryingMyKata"
# print(generate_hashtag("    Hello     World   "))
# print(generate_hashtag(""))


# def count_sheeps(sheep):
#     r = len(list(filter(lambda x: x == True, sheep)))
#     return r
#
#
# a = [True,  True,  True,  False,
#   True,  True,  True,  True ,
#   True,  False, True,  False,
#   True,  False, False, True ,
#   True,  True,  True,  True ,
#   False, False, True,  True]
#
# print(count_sheeps(a))


# def litres(time):
#     return math.floor(time * 0.5)
#
#
# print(litres(11.8))


# def century(year):
#     return year // 100 if year % 100 == 0 else year // 100 + 1
#
#
# print(century(401))
#
#
# def abbrev_name(name):
#     return '.'.join(s[0] for s in name.upper().split())
#
#
# print(abbrev_name('patrick feeney'))
#
#
# def digitize(n):
#     return list(map(int, list(reversed(str(n)))))
#
#
# print(digitize(35231))


# def find_needle(haystack):
#     for k, v in enumerate(haystack):
#         if v == 'needle':
#             return f"found the needle at position {k}"
#
#
# print(find_needle(["hay", "junk", "hay", "hay", "moreJunk", "needle", "randomJunk"]))


# def find_average(numbers):
#     return 0 if numbers == [] or numbers == [0] else sum(numbers) / len(numbers)
#
#
# print(find_average([]))


# def invert(lst):
#     return [-x for x in lst]
#     return [-i if i > 0 else abs(i) for i in lst]
#
#
# print(invert([1, -2, 3, -4, 5]))


# def count_positives_sum_negatives(arr):
#     if arr == 0 or arr == []: return []
#     res_1 = len([str(i) for i in arr if i > 0])
#     res_2 = sum([abs(i) for i in arr if i < 0])
#     return [res_1, res_2 * - 1]
#
#
# print(count_positives_sum_negatives([]))


# def fake_bin(x):
#     res = [int(i) * 0 if int(i) < 5 else int(i) // int(i) for i in x]
#     return ''.join(map(str, res))
#     return ''.join('0' if c < '5' else '1' for c in x)
#
#
# print(fake_bin('45385593107843568'))


# def grow(arr):
#     return reduce(lambda x, y: x * y, arr)
#
#
# print(grow([1, 2, 3, 4]))


# def descending_order(num):
#     return int(''.join(sorted(str(num), reverse=True)))
#
#
# print(descending_order(42145))


# def get_middle(s):
#     return ''.join([s[len(s) // 2 - 1], s[len(s) // 2]]) if len(s) % 2 == 0 else s[len(s) // 2]
#     return s[(len(s) - 1) // 2:len(s) // 2 + 1]
#
#
# print(get_middle('testing'))


# d = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
# data = {"IV": 2, "IX": 2, "XL": 20, "XC": 20, "CD": 200, "CM": 200}
#
#
# def func(n):
#     return sum(d[i] for i in n) - sum(v for k, v in data.items() if k in n)
#
#
# print(func('MCMXCIV'))  # 1994


# def accum(s):
#     return '-'.join([(i + 1) * u for i, u in enumerate(s)]).title()
#
#
# print(accum("RqaEzty"))  # "R-Qq-Aaa-Eeee-Zzzzz-Tttttt-Yyyyyyy"

# def filter_list(l):
#     # return sorted(i for i in l if isinstance(i, int))
#     return [i for i in l if isinstance(i, int)]
#
#
# print(filter_list([1, 2, 'aasf', 0, '1', '123', 123]))

# def is_isogram(string):
#     return False if len(set(string.lower())) != len(string) else True
#
#
# print(is_isogram("hgH"))

# def to_jaden_case(string):
#     return ' '.join([i.capitalize() for i in string.split()])
#
#
# print(to_jaden_case("How can mirrors be real if our eyes aren't real"))

# def find_short(s):
#     return len(min(s.split(), key=len))
#
# print(find_short("Let's travel abroad shall we"))

# def make_readable(seconds):
#     h = seconds // 3600
#     m = seconds // 60 % 60
#     c = seconds % 60
#     h_t = h if h > 9 else f'{0}{h}'
#     m_t = m if m > 9 else f'{0}{m}'
#     c_t = c if c > 9 else f'{0}{c}'
#     return f'{h_t}:{m_t}:{c_t}'

# def make_readable(s):
#     return '{:02}:{:02}:{:02}'.format(s // 3600, s // 60 % 60, s % 60)
#
#
# print(make_readable(86399)) # "23:59:59"

# def maskify(s):
#     return '#' * (len(s) - 4) + s[-4:] if len(s) > 4 else s
#
#
# print(maskify("55553"))

# def sum_two_smallest_numbers(numbers):
#     return sorted(numbers)[0] + sorted(numbers)[1]
#
#
# print(sum_two_smallest_numbers([19, 5, 42, 2, 77]))

# def friend(x):
#     return [i for i in x if len(i) == 4]
#
#
# print(friend(["Ryan", "Kieran", "Mark"]))

# def validate_pin(pin):
#     return pin.isdigit() if len(pin) == 4 or len(pin) == 6 else False
#     return len(pin) in [4, 6] and pin.isdigit()
#
#
# print(validate_pin('9i345'))

# def find_outlier(integers):
#     res = [i for i in integers if i % 2 != 0]
#     return res[0] if len(res) == 1 else [i for i in integers if i % 2 == 0][0]
#
#
# print(find_outlier([2, 4, 0, 100, 4, 11, 2602, 36]))
# print(find_outlier([160, 3, 1719, 19, 11, 13, -21]))


# def duplicate_count(text_in_mp3):
#     count = collections.Counter(text_in_mp3.lower())
#     return len([k for k, v in count.most_common() if v > 1])
#     return len([c for c in set(text_in_mp3.lower()) if text_in_mp3.lower().count(c) > 1])
#
#
# print(duplicate_count('abcde'))
# print(duplicate_count('aabBcde'))
# print(duplicate_count('aA11'))
# print(duplicate_count('indivisibility'))
# print(duplicate_count('Indivisibilities'))

# while True:
#     x = input('Вводи все в одну стоку: ')
#     print(eval(x))


# def elevator_distance(array):
#     return sum([abs(array[i] - array[i + 1]) for i in range(len(array) - 1)])
#     return sum([abs(x - y) for x, y in zip(array, array[1:])])
#     return sum(abs(x - y) for x, y in itertools.pairwise(array))
#
#
# a = elevator_distance([7, 1, 7, 1])
# print(a)


# def max_sequence(arr):
#     if not arr or max(arr) < 0:
#         return 0
#
#     curr = max_sub = arr[0]
#     for num in arr[1:]:
#         curr = max(num, curr + num)
#         max_sub = max(max_sub, curr)
#     return max_sub
#
#
# print(max_sequence([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
# print(max_sequence([7, 4, 11, -11, 39, 36, 10, -6, 37, -10, -32, 44, -26, -34, 43, 43]))

# def get_sum(a, b):
#     return sum(range(min(a, b), max(a, b) + 1))
#
#
# print(get_sum(0, -2))
# print(get_sum(-4259, -178))
# print(get_sum(4251, 312))

# def first_non_repeating_letter(s):
#     if not s: return ''
#     count = collections.Counter(s.lower())
#     res = [k for k, v in count.items() if v == 1]
#     if not res: return ''
#     return res[0].upper() if res[0].upper() in s else res[0]
#
#
# print(first_non_repeating_letter('stress'))
# print(first_non_repeating_letter('~><#~><'))
# print(first_non_repeating_letter(''))
# print(first_non_repeating_letter('sTreSS'))
# print(first_non_repeating_letter('abba'))

# class Answer:
#     def fizzbuzztest(self):
#         return ['FizzBuzz' if i % 3 == 0 and i % 5 == 0
#         else 'Fizz' if i % 3 == 0 else 'Buzz' if i % 5 == 0 else i for i in range(1, 100 + 1)]
#
#
# result = Answer()
# print(result.fizzbuzztest())

# a, *_ = zip((1, 'Elon Mask', 'good'), (2, 'Bill Gates', 'good'), (3, 'Tim Cook', 'good'))
# print(a)

# class Answer:
#     def PureIntersection(self, arr1, arr2):
#         res = [set(arr1) & set(arr2)]
#         return list(*res)
#
#
# arr1 = [1, 2, 3]
# arr2 = [6, 3, 5]
#
# a = Answer()
# print(a.PureIntersection(arr1, arr2))

# def odd_or_even(arr):
#     return "even" if sum(arr) % 2 == 0 else "odd"
#
#
# print(odd_or_even([1]))

# def reverse_words(text):
#     return ' '.join([i[::-1] for i in text.split(' ')])
#
#
# print(reverse_words("This is an example!"))  # "sihT si na !elpmaxe"
# print(reverse_words("elbuod  secaps"))


# class A:
#     foo = "a"
#
#     def __init__(self):
#         self.foo = "b"
#
#     def __getattribute__(self, name):
#         if name != 'bar':
#             return super().__getattribute__(name)
#         return "c"
#
#     def __getattr__(self, name):
#         if name != 'bar':
#             return super().__getattribute__(name)
#         return "d"
#
#
# a = A()
# print(f'{A.foo=}, {a.foo=}, {a.bar=}')


# def zeros(n):
#     if n == 0: return 0
#     elif n == 1: return 1
#     res = reduce(lambda x, y: x * y, range(1, n + 1))
#     result = [list(v) for k, v in groupby(str(res)[::-1])]
#     return len(result[0])
#
#
# print(zeros(100))


# def zeros(n):
#     return 0 if int(n/5) < 1 else int(n/5) + int(zeros(n/5))
#
# print(zeros(100))

# def zeros(n):
#     count = 0
#     while n:
#         n = n // 5
#         count += n
#     return count
#
#
# print(zeros(5))


# def zeros(n):
#     start = 1
#     for i in range(1, n + 1):
#         start *= i
#         yield start
#
#
# for x in zeros(10000):
#     print(x)

# def sum_two_smallest_numbers(numbers):
#     res = sorted(numbers)
#     return res[0] + res[1]


# print(sum_two_smallest_numbers([19, 5, 42, 2, 77]))


# def fibonacci(n):
#     if n in (1, 2):
#         return 1
#     return fibonacci(n - 1) + fibonacci(n - 2)
#
#
# print(fibonacci(10))


# class Fibonacci:
#     '''Итератор последовательности Фибоначчи из N элементов'''
#
#     def __init__(self, number):
#         self.number = number
#         self.count = 0
#         self.cur_n = 0
#         self.next_n = 1
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         self.count += 1
#         if self.count > self.number:
#             raise StopIteration
#         self.cur_n, self.next_n = self.next_n, self.cur_n + self.next_n
#         return self.cur_n
#
#
# fib_iterator = Fibonacci(10)
# print(*fib_iterator)

# def fibonacci(number):
#     cur_val = 1
#     next_val = 1
#     for _ in range(number):
#         yield cur_val
#         cur_val, next_val = next_val, cur_val + next_val
#
#
# print(*fibonacci(10))

# colors = ['красный', 'синий', 'зеленый', 'желтый']
# for item in itertools.permutations(colors, 4):
#    print(item)
#
# print('=' * 40)
# for item in itertools.combinations(colors, 2):
#    print(item)
#
#
# my_cycle = itertools.cycle(['раз', 'два', 'три'])
# print(next(my_cycle))
# print(next(my_cycle))
# print(next(my_cycle))
# print(next(my_cycle))
# print(next(my_cycle))


# def s():
#    global a
#    a = 30
#
#
# a = [1, 2, 4, 5, 6]
# s()
# print(a, 'global')


# @contextmanager
# def next_num(num):
#    print ('Входим в функцию')
#    try:
#        yield num + 1
#    except ZeroDivisionError as exc:
#        print('Обнаружена ошибка:', type(exc))
#    finally:
#        print('Здесь код выполнится в любом случае')
#    print('Выход из функции')
#
#
# with next_num(10) as next:
#    print('Следующее число = {}'.format(next))
#    print(10 / next)


# def decorator(func):
#     def wrapper(*args, **kwargs):
#         print(f"Вызов функции {func.__name__}() с аргументами {args}")
#         now = datetime.now()
#         result = func(*args, **kwargs)
#         end = datetime.now() - now
#         print(f"Функция {func.__name__}() вернула результат: {result}", end)
#         return result
#     return wrapper
#
#
# @decorator
# def summ(a, b):
#     return a + b
#
#
# @decorator
# def add(a, b):
#     return a - b
#
#
# result_1 = summ(5, 9)
# result_2 = add(9, 7)

# class X:
#     def x(self):
#         print('текст X')
#
#
# class A(X):
#     def a(self):
#         print('текст A')
#
#
# class B(X):
#     def b(self):
#         print('текст B')
#
#
# class C(X):
#     def c(self):
#         print('текст C')
#
#
# class D(X):
#     def d(self):
#         print('текст D')
#
#
# class E(A, B):
#     def e(self):
#         print('текст E')
#
#
# class F(B, C):
#     def f(self):
#         print('текст F')
#
#
# class G(B, C, D):
#     def g(self):
#         print('текст G')
#
#
# class H(C, D):
#     def h(self):
#         print('текст H')
#
#
# class J(E, G):
#     def j(self):
#         print('текст J')
#
#
# class K(F, G, H):
#     def k(self):
#         print('текст K')
#
#
# class Z(J, K):
#     def z(self):
#         print('текст Z')
#
#
# z = Z()
# print(z.__class__.__mro__)