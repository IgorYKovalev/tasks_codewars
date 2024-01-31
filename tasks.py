import asyncio
import collections
import itertools
import operator
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from functools import reduce
from itertools import groupby
import math
import random


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

# def sum_strings(x, y):
#     return str(mpz(x or '0') + mpz(y or '0'))
#
#
# print(sum_strings("123", "456999"))


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