## Introduction

This is repo contains the work than can be cloned directly on your ros_installation/src folder. 

## Folder structure
```json
gyku8924_service
├── CMakeLists.txt
├── include
│   └── gyku8294_service 
├── package.xml
├── src
└── srv
	└── Gyku8294Service.srv #custom srv file containing the input and output

4 directories, 3 files
```

```json
Gyku8294_hw2
├── gyku8294_hw2
│   ├── client_member.py # client for service
│   ├── __init__.py
│   ├── server_member.py # server for service
│   ├── topic_pub.py # topic publisher for service
│   └── topic_sub.py # topic subscriber for service
├── package.xml
├── resource
│   └── gyku8294_hw2
├── setup.cfg
├── setup.py
└── test
	├── test_copyright.py
	├── test_flake8.py
	└── test_pep257.py

3 directories, 12 files
```

## Installation + testing
To run the following:
```json
cd ros2_humble/
colcon build --packages-select gyku8294_hw2 gyku8294_service
```
You may see some warning depending on your setup, but it would compiling the builds into the install folder inside your ros2_humble location. 

Open 4 terminals - use terminator if possible 

for Part B
```json
ros2 run gyku8294_hw2 service_server 
ros2 run gyku8294_hw2 service_client
```
Histogram will be generated in your default file explorer location.

for Part C
```json
(optional) $ros2 run gyku8294_hw2 service_server 
ros2 run gyku8294_hw2 topic_pub
ros2 run gyku8294_hw2 topic_sub
```
Histogram will be generated in your default file explorer location.


## Implementation
The service_client uses directly array reversing [::-1] to reverse the input_string. 
For this, my input string is set defaults as "I am gyanig". you can add logger to get the input for 400 times the input_string is read as well.



