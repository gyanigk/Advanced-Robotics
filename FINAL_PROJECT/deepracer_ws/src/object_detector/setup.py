from setuptools import find_packages, setup

package_name = 'object_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mo Zhou',
    maintainer_email='mozh7931@colorado.edu',
    description='Stop sign detection node',
    license='Apache License 2.0',
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'stop_sign_detector = object_detector.stop_sign_detector:main',
            'school_zone_detector = object_detector.school_zone_detector:main',
            'target_finder = object_detector.target_finder:main',
            'stop_sign_line_following = object_detector.stop_sign_line_following:main',
            'line_follower = object_detector.line_follower:main',
            'line_followerv2 = object_detector.line_followerv2:main',
            'line_followerv3 = object_detector.line_followerv3:main',
            'line_followerv4 = object_detector.line_followerv4:main',
            'telemetry= object_detector.telemetry:main',
        ],
    },
)