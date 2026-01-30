from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    world_arg = DeclareLaunchArgument('world', default_value='/home/oho/gazebo_models/my_pipe_environment/pipe_world.world')
    urdf_arg  = DeclareLaunchArgument('urdf',  default_value='/mnt/data/inpipe_sensors_vlp16_l515.urdf')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join('/opt/ros/humble/share/gazebo_ros/launch', 'gazebo.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    rsp = ExecuteProcess(
        cmd=['ros2', 'run', 'robot_state_publisher', 'robot_state_publisher', LaunchConfiguration('urdf'), '--ros-args', '-p', 'use_sim_time:=true'],
        output='screen'
    )

    spawn = ExecuteProcess(
        cmd=['ros2', 'run', 'gazebo_ros', 'spawn_entity.py', '-entity', 'inpipe_bot_sensors', '-topic', '/robot_description'],
        output='screen'
    )

    return LaunchDescription([world_arg, urdf_arg, gazebo, rsp, spawn])
