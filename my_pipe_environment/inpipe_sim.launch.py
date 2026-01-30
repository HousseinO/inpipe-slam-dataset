from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import TimerAction
from launch_ros.actions import Node


import os

def generate_launch_description():
    world_default = '/home/oho/gazebo_models/my_pipe_environment/pipe_world.world'
    urdf_default  = '/home/oho/gazebo_models/my_pipe_environment/inpipe_sensors_vlp16_l515.urdf' #l515_lidar.sdf'

    world_arg = DeclareLaunchArgument('world', default_value=world_default)
    urdf_arg  = DeclareLaunchArgument('urdf',  default_value=urdf_default)

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join('/opt/ros/humble/share/gazebo_ros/launch', 'gazebo.launch.py')
        ),
        launch_arguments={
        'world': LaunchConfiguration('world'),
        'use_sim_time': 'true'
        }.items()
    )

    # Robot State Publisher reads the URDF file path directly
    rsp = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'robot_state_publisher', 'robot_state_publisher',
            LaunchConfiguration('urdf'),
            '--ros-args', '-p', 'use_sim_time:=true'
        ],
        output='screen'
    )

    # Spawn FROM /robot_description (published by RSP), not from -file
    spawn_now = ExecuteProcess(
        cmd=[
            'ros2', 'run', 'gazebo_ros', 'spawn_entity.py',
            '-entity', 'inpipe_bot_sensors',
            '-topic', '/robot_description',
            '-x', '0', '-y', '0', '-z', '0.5',
            '--ros-args', '-p', 'use_sim_time:=true'
        ],
        output='screen'
    )
    spawn = TimerAction(period=2.0, actions=[spawn_now])
    
    
    


    return LaunchDescription([world_arg, urdf_arg, gazebo, rsp, spawn])

