from setuptools import find_packages, setup

package_name = 'ekf_slam_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/slam_test.py']),
        ('share/' + package_name + '/config', ['config/slam_toolbox_params.yaml']),    
        ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='raspi-legoracers',
    maintainer_email='raspi-legoracers@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odometry = ekf_slam_robot.odometry:main',
            'ekf_slam_node = ekf_slam_robot.ekf_slam_node:main',
            'ekf_slam_nodev2 = ekf_slam_robot.ekf_slam_nodev2:main',
            'odometryv2 = ekf_slam_robot.odometryv2:main',
            'slam_diagnostics = ekf_slam_robot.slam_diagnostics:main',
        ],
    },
)
