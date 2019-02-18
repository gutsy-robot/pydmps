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

    roslauch pydmps record_trajectory.launch
    
    
This will open the turtlesim simulator and the teleoperation node. Go to the bag directory of the package and do:
    
    rosbag record turtle1/pose -o <name_of_bag_file>


Now move the turtle around by switching back to the previous tab of the terminal, so that the trajectory gets recorded in the bag.
Once you're done kill the nodes in both of the terminals.


Once you're done, go to the pydmps directory inside the package and do:
    
       python imitate_turtlesim.sh <path_to_bag_file>
       
       
You should now be able to see the DMPs with the different number of basis alongside the original path.