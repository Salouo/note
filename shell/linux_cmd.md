# Linux Command

## 1. Basic Conception

A **Shell** is a command-line interpreter that allows users to interact with the operating system by typing commands. It servers as an interface between the user and the system's kernel.



Shell scripts usually start with a **shebang line** to specify which shell should interpret the script, for example:

```sh
#!/bin/bash
```

This line tells the system to use the **Bash** shell to run the script.



A normal shell script should be like:

```sh
#!/bin/bash

# Print string.
echo "hello world!"

# Print current time.
date

# Print current user.
whoami

# Run Python scripts as a pipeline using the default Python environment.
python goodmorning.py
python goodnight.py

```



## 2. Execution Permission

Sometimes we need to grant execution permission to a shell script in order to run it.

```sh
chmod a+x test.sh
```

This command grants all users permission to execute this script.



Also, we can just grant execution permission to the current user (ourselves).

```sh
chmod u+x test.sh
```



## 3. `Tab`

Press `Tab` to  auto-complete the long file name.

```sh
ls
main.py
cd m (use Tab)	-> cd main.py
```



## 4. `Ctrl` + `A`

Press `Ctrl` + `A` to back to the head of inputting command.



## 5. nano

`nano` is a text modifier. Use the command below to use nano to edit files.

```sh
nano README.md
```



## 6. echo

`echo` is used to print something. Put a `$` mark ahead to dedicate it is a variable.

```sh
h="hello"

echo $h
hello

echo "abc$h"
abchello

echo "abc-$hasd"
abc-

echo "abc-${h}asd"
abc-helloasd
```



## 7.  for loop



```sh
for i in {1..7};
for> do
for> mkdir "folder_$i"
for> done


for f in folder??
for> echo $f
for> done

folder_1
folder_2
folder_3
folder_4
folder_5
folder_6
folder_7

for f in folder*
do
mv $f chapter${f#folder_}		# `#` delete the head elements; `%`delete the tail elements
done

for f in cha*
for> do
for> echo $f
for> done
chapter1
chapter2
chapter3
chapter4
chapter5
chapter6
chapter7
```

