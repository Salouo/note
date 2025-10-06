## 1. defaultdict(type)

A default dictory which is commonly used in figuring algorithm questions.

If we access a key that does not exist, defaultdict will automatically create it and assign a default value of the type we specified.

```python
from collections import defaultdict

my_int_dict = defaultdict(int)   # Default value is 0
my_list_dict = defaultdict(list)  # Default value is an empty list []

my_int_dict['a']+= 1	# my_int_dict['a'] becomes 1
my_list_dict['b'].append(1) # my_list_dict['b'] becomes [1]
```



## 2. str.join(iterable)

str.join(iterable) returns a single string by joining the elements of an interable, using the str as a separator.

```python
a = '-'.join(['a', 'b', 'c'])  # 'a-b-c'

word = 'dca'
b = ''.join(sorted(word))	# 'acd'
```



## 3. for num in nums

```python
nums = [1, 2, 3, 4]

for num in nums:
    num += 1

# nums = [1, 2, 3, 4]
```

This operation will not change the value of the list.

Instead, we should do this.

```python
n = len(nums)
for i in range(n):
    nums[i] += 1

# nums = [2, 3, 4, 5]
```



## 3. fast pointer and low pointer

We can use a fast pointer which moves 2 steps at an iteration and a slow pointer which moves 1 step at an iteration  to find a midpoint. For example, using fast pointer and slow pointer can find the midpoint of a linked list.

```python
fast = slow = head
while fast and fast.next:
  fast = fast.next.next
  slow = slow.next
# `slow` is the midpoint (round up) when the loop ends.

fast = slow = head
while fast.next and fast.next.next:
  fast = fast.next.next
  slow = slow.next
# `slow` is the midpoint (round down) when the loop ends.
```



## 4. Linked List Cycle ll

![142. Linked List Cycle II](/Users/cheesberry/Desktop/note/algorithm/142. Linked List Cycle II.jpg)

We first need to use fast and low pointers to check if there exists a cycle. (easy)

The difficult part is to find the entrance of the cycle.

```text
本题难点在第二步找入口。我认为我这个理解应该是最简单直接的。

f：快指针走的距离

s：慢指针走的距离

d：慢指针在圈里走的距离，注意d < b，因为在慢指针在完成一圈之前一定会被快指针追上

c：从起点到入口的距离，也是我们要求的

b：绕环一圈的长度

n：常数，用来标记绕环走了几圈，如nb就是绕了n圈

下面开始推导。

f = 2s

-> c + nb = 2(c + d)

-> c = nb - d    

注意这里nb其实就是无限绕圈回到你现在所在的位置，可以当成0来看，当然你保留也无所谓，你在跑道上你n=10000都不影响你现在站在-d的位置。

-> c = -d 

注意看这个什么意思，c是起点到入口的距离，而-d说明慢指针此时离入口还有距离d。他们两个恰好绝对值相等。


```





