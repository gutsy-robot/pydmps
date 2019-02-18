pydmps
======

The original repository is a Python implementation of DMPs, with accompanying tutorials and applications that can be found at http://studywolf.wordpress.com/category/robotics/dynamic-movement-primitive/

This is a ROS package built on top of the original repo to quickly record trajectories on a turtlesim simulator and then visualise the DMP under different number of basis functions,



Dependencies
-----------


1. ros turtlesim simulator
2. ros turtlesim teleop_key

Make sure that both of these are present on your ROS package path.



Running the demo
----------------

    roslauch pydmps record_trajectory.launch <name_of_bag_file> <name_of_csv_file>
    
    
This will open the turtlesim simulator and the teleoperation node. You can specify the name of the bag file and the csv file as atguments.
Once the simulator is up, move around the robot to record a trajectory. Once you're done, go to the pydmps directory inside the package and do:
    
       python imitate_path.py <path_to_csv> <start_state> <goal_state>
       
       
You should now be able to see the DMPs with the different number of basis alongside the original path.