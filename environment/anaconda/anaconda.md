## **1. Initialize the environment**

We should use the following command line to load the environment configuration again.

```shell
source ~/.bashrc
```



## 2. Create a new virtual environment

Create a new virtual environment with python 3.10 version.

```shell
conda create -n env_name python=3.10
```

Create a new virtual environment with python the lastest version.

```shell
conda create -n env_name
```



## 3. Show all the virtual environments

```shell
conda env list
```



## 4. Delete a specific virtual environment

```shell
conda env remove -n env_name
```



## 5. Clone a virtual environment 

```shell
conda create -n new_env --clone old_env
```



## 6. Remove a virtual environment

```shell
conda remove -n env_name --all
```

