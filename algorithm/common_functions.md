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



## 3.

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



