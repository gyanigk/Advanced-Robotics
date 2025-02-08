from setuptools import find_packages, setup
import glob
package_name = 'gyku8294_service'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/srv', ['srv/Gyku8294_service.srv']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gyanig_ros2',
    maintainer_email='gyanig_ros2@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
