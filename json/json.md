

# JSON

## 1. Basic Conception

```python
import json
```

**JSON's function: serialization and deserialization**



## 2. Serialization

**Serilization** is a process of converting variables from memory into a format that can be stored or transmitted. (python object -> json string)

Here, we have two Python dictionary objects `person1` and  `person2`,  and we try to serialize them.

```python
person1 = {'Name': 'Jack', 'Tel': ['10095', '87321'], 'Gender': 'male', 'Age': 30, 'Is_Only': True}
person2 = {'Name': 'Kitty', 'Tel': ['12345', '87321'], 'Gender': 'female', 'Age': 25, 'Is_Only': True}
```



### 2.1 json.dump()



#### 2.1.1

We can use `json.dump()` to serialize only one object.

```python
# Serialize variable person1 and store it in .json file
with open('data.json', 'w') as f:
  	# indent: make data more readable
    # sort_keys parameter controls whether the keys in a Python dictionary should be sorted alphabetically.
    json.dump(person1, f, indent=4, sort_keys=True)
```

`data.json` (not `data.jsonl`, because here is only one JSON object):

```text
{
    "Age": 30,
    "Gender": "male",
    "Is_Only": true,
    "Name": "Jack",
    "Tel": [
        "10095",
        "87321"
    ]
}
```



#### 2.1.2

We can also use `json.dump()` to serialize a list which stores several Python dictionaries.  **(the most commonly used method)**

```python
with open('data.json', 'w') as f:
    json.dump([person1, person2], f, indent=4, sort_keys=True)
```

`data.json` (only one JSON list) :

```text
[
    {
        "Age": 30,
        "Gender": "male",
        "Is_Only": true,
        "Name": "Jack",
        "Tel": [
            "10095",
            "87321"
        ]
    },
    {
        "Age": 25,
        "Gender": "female",
        "Is_Only": true,
        "Name": "Kitty",
        "Tel": [
            "12345",
            "87321"
        ]
    }
]
```



### 2.2 json.dumps()



#### 2.2.1

`json.dumps()` converts a Python object into a JSON-formatted string. It returns a JSON-formatted string.

```python
# Serialize variable person
# python dict {k1: v1, k2: v2} -> json string '{k1: v1, k2: v2}'
json_s = json.dumps(person1, indent=4, sort_keys=True)
print(json_s)
```



`data.json` :

```text
{
    "Age": 30,
    "Gender": "male",
    "Is_Only": true,
    "Name": "Jack",
    "Tel": [
        "10095",
        "87321"
    ]
}
```



#### 2.2.2

We can also use `json.dumps()` to serialize Python dictionaries into separate JSON objects.

```python
with open('data.json', 'w') as f:
    f.write(json.dumps(person1))
    f.write(json.dumps(person2))
```

`data.jsonl` **(Notice that we should us `.jsonl` to indicate that this file contains several JSON objects)**:

```text
{"Name": "Jack", "Tel": ["10095", "87321"], "Gender": "male", "Age": 30, "Is_Only": true}
{"Name": "Kitty", "Tel": ["12345", "87321"], "Gender": "female", "Age": 25, "Is_Only": true}
```



## 3. Deserialization



### 3.1

When there is **only one JSON object** in `data.json`, we can use `json.load()` to obtain a corresponding Python dictionary.

`data.json`

```text
# data.json
{
    "Age": 30,
    "Gender": "male",
    "Is_Only": true,
    "Name": "Jack",
    "Tel": [
        "10095",
        "87321"
    ]
}
```

**code**

```python
with open('data.json', 'r') as f:
  x = json.load(f)

print(f'The type of x is: {typye(x)}\n')
print(x)
```

**output**

```text
The type of x is: <class 'dict'>

{
    "Age": 30,
    "Gender": "male",
    "Is_Only": true,
    "Name": "Jack",
    "Tel": [
        "10095",
        "87321"
    ]
}
```



### 3.2

When there is only one JSON list which includes several JSON objects, we can use `json.load()` to convert the JSON list to Python list.

`data.json`

```text
[
    {
        "Age": 30,
        "Gender": "male",
        "Is_Only": true,
        "Name": "Jack",
        "Tel": [
            "10095",
            "87321"
        ]
    },
    {
        "Age": 25,
        "Gender": "female",
        "Is_Only": true,
        "Name": "Kitty",
        "Tel": [
            "12345",
            "87321"
        ]
    }
]
```

**code**

```python
with open('data.json', 'r') as f:
    python_dict_list = json.load(f)

for e in python_dict_list:
    print(e)
```

**output**

```text
{'Age': 30, 'Gender': 'male', 'Is_Only': True, 'Name': 'Jack', 'Tel': ['10095', '87321']}
{'Age': 25, 'Gender': 'female', 'Is_Only': True, 'Name': 'Kitty', 'Tel': ['12345', '87321']}
```



### 3.3

When there are several JSON objects in `data.json`, we use `json.loads()` to load each each JSON objects and convert them into Python dictionaries.

`data.jsonl` **(not **`data.json`**)**

```text
{"Name": "Jack", "Tel": ["10095", "87321"], "Gender": "male", "Age": 30, "Is_Only": true}
{"Name": "Kitty", "Tel": ["12345", "87321"], "Gender": "female", "Age": 25, "Is_Only": true}
```

**code**

```python
with open('data.jsonl', 'r') as f:
  
  	# line.strip(): line.strip() removes all leading and trailing whitespace characters from a string.
    # This includes: spaces(' '), newlines('\n'), Tabs('\t'), Carriage returns('\r'), etc.
    python_dict_list = [json.loads(line.strip()) for line in f]
    
print(python_dict_list)
```

**output**

```text
[{'Name': 'Jack', 'Tel': ['10095', '87321'], 'Gender': 'male', 'Age': 30, 'Is_Only': True}, {'Name': 'Kitty', 'Tel': ['12345', '87321'], 'Gender': 'female', 'Age': 25, 'Is_Only': True}]
```

