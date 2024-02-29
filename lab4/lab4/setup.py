from setuptools import setup
import os
from glob import glob
package_name = 'lab4'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
           (os.path.join('share', package_name), glob('urdf/*')),
           (os.path.join('share', package_name), glob('launch/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akshat',
    maintainer_email='akshat@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'controller = lab4.move_robot:main',
            'move = lab4.move_robot_reaal:main',
            'gesture = lab4.gesture_node:main'
        ],
    },
)