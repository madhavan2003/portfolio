from setuptools import setup

package_name = 'farm_perception'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='chinmay',
    maintainer_email='chinmaychinmay2003@gmail.com',
    description='Depth-geometric lane detection and plant counting from one RGB-D camera.',
    license='BSD',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_perception_node = farm_perception.lane_perception_node:main',
            'plant_counter_node = farm_perception.plant_counter_node:main',
        ],
    },
)
