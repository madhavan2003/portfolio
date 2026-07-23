import os
from glob import glob
from setuptools import setup

package_name = 'farm_bringup'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chinmay',
    maintainer_email='chinmaychinmay2003@gmail.com',
    description='Bringup, world, RViz and run recorders for the farm coverage mission.',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_recorder_node = farm_bringup.video_recorder_node:main',
            'path_recorder_node = farm_bringup.path_recorder_node:main',
        ],
    },
)
