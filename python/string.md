## 1. string.strip()

`string.strip()` removes any **leading** and **trailing** whitespace characters from a string.
 These include spaces (' '), tabs ('\t'), and newline characters ('\n').\

### Example

```python
s1 = 'hello world  \d'
print(s.strip())

s = '##hello##a#a'
print(s.strip('#'))
```

**output**

```text
hello world

hello##a#a
```



## 2. string.split()

`string.split()` splits a string into a list wherever a specific character appears.

### Example

```python
s = "apple,banana,orange"
result = s.split(',')
print(result)
```

**output**

```text
['apple', 'banana', 'orange']
```

