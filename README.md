pydmps
======

This is a fork from the following repo: 

    http://studywolf.wordpress.com/category/robotics/dynamic-movement-primitive/

which has the python implementation of Dynamic Motion Primitives. This repo builds on top of this functionality.
Here, I have tried some methods by which the path given by a dynamic motion primitive can be further adapted online 
to make it feasible in the given new workspace.


Dependencies
-----------

1. shapely (can be installed with pip)
2. ros turtlesim simulator
3. ros turtlesim teleop_key

Make sure that 2 and 3 are present on your ROS package path. 2 and 3 are only needed if you wish to
record new actions. I have already provided an example action in the csv subdirectory in data.csv. If you
record a new 2D action, just create a csv file in the similar fashion.



Running the demo
----------------

Recording a new Action Replicating it with DMP
-----

    roslauch pydmps record_trajectory.launch
    
    
This will open the turtlesim simulator and the teleoperation node. Go to the bag directory of the package and do:
    
    rosbag record turtle1/pose -o <name_of_bag_file>


Now move the turtle around by switching back to the previous tab of the terminal, so that the trajectory gets recorded in the bag.
Once you're done kill the nodes in both of the terminals.


Once you're done, go to the pydmps directory inside the package and do:
    
       python imitate_turtlesim.sh <path_to_bag_file>
       
       
You should now be able to see the DMPs with the different number of basis alongside the original path.


Working with the example action 
----


For this the best way is to use the jupyter notebook given in the repo, which is quite self-explainatory. 
In the notebook, I have given some additional basic actions like moving in a straight line as well. The cells starting
from the 3D grid search are not fully updated right now, but eveything before that works.