import os
from glob import glob
from setuptools import setup

package_name = 'farm_navigation'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chinmay',
    maintainer_email='chinmaychinmay2003@gmail.com',
    description='Runtime marker reading, perception-driven coverage mission, Nav2 config.',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'marker_reader_node = farm_navigation.marker_reader_node:main',
            'mission_node = farm_navigation.mission_node:main',
            'static_map_odom_node = farm_navigation.static_map_odom_node:main',
        ],
    },
)
