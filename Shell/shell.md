# Shell

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



