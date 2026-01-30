oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment$ ros2 launch inpipe_sim.launch.py 
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment/segmentation$ python3 fit_ellipse.py  
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment$ python3 trajectory_set.py --entity inpipe_bot_sensors
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment/segmentation$ python3 icp.py 


oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment/slam$ python3 gtsam_depth_imu.py 
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment$ ros2 launch inpipe_sim.launch.py  
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment$ python3 trajectory_set.py --entity inpipe_bot_sensors
oho@oho-ROG-Strix-G531GT-G531GT:~/gazebo_models/my_pipe_environment/segmentation$ python3 icp.py

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom



