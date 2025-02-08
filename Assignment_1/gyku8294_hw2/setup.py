from setuptools import find_packages, setup

package_name = 'gyku8294_hw2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='gyanig_ros2',
    maintainer_email='gyanig.kumar@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'service_server = gyku8294_hw2.server_member:main',
            'service_client = gyku8294_hw2.client_member:main',
            'topic_sub = gyku8294_hw2.topic_sub:main',
            'topic_pub = gyku8294_hw2.topic_pub:main',
        ],
    }, 
)
