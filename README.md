# youtiao
A repository for noise separation experiments. Youtiao, or fried dough stick, is eaten by first splitting into two, just like how we are trying to split speech and noise signals from a mixed signal. Yes. The analogy totally makes sense. Don't quote me. 

## Build Docker Image
To set up this repository on your local machine, follow the steps below.

```shell
# clone the folder
>> git clone https://github.com/test-dan-run/youtiao.git
>> cd youtiao
# build docker image
# userid and gid added to bypass permission problems when working with container-created files
>> sudo docker build -t youtiao:v0.1.0 --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) .
```