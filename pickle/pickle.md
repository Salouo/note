# Pickle

Pickle is a library used for the serialization and deserialization of Python variables.

```python 
import pickle
```



## 1. Serialization

The process of converting variables from memory into a binary format that can be stored or transmitted.



```python
a = "hello, world!"
b = {'Gender': 'man', 'Age': 20, 'Money': 'none'}
c = ['happy', 32, [2, 10]]

# Create a channel to modify data.txt (if not existing, create a new one)
# wb: write mode and binary format
with open("data.pkl", "wb") as f:   # with open: we do not need to use f.close() to close the channle
    pickle.dump(a, f)
    pickle.dump(b, f)
    pickle.dump(c, f)
    pickle.dump((a, b, c), f)
```

So we wil have a .pkl file which stores variables a, b, c, and (a, b, c).



## 2. Deserialization

The process of converting data from a stored or transmitted format back into variables in memory.



```python
# Create a channel to modify data.txt
with open("data.pkl", "rb") as f:
    # deserialization
    a = pickle.load(f)
    b = pickle.load(f)
    c = pickle.load(f)
    d, e, f = pickle.load(f)


print(a)
print(b)
print(c)
print(d, e, f)
```

**output**

```pytho
hello, world!
{'Gender': 'man', 'Age': 20, 'Money': 'none'}
hello, world! {'Gender': 'man', 'Age': 20, 'Money': 'none'} ['happy', 32, [2, 10]]
```

