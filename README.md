# bird_eye_view_tracking
Bird Eye View visualisation in ROS1 for MOTS problem

## Installation

Command sequence:

```bash
cd catkin_ws/src
git clone https://github.com/Ilyabasharov/bird_eye_view_tracking.git
git clone https://github.com/eric-wieser/ros_numpy.git
cd bird_eye_view_tracking && pip install -r requirements.txt
cd ../..
source /opt/ros/melodic/setup.bash
catkin_make
source devel/setup.bash
```

## Execution

Input messages like ```ObjectArray``` and ```PointCloud2``` are required for execution. Also you must specify a path to odometry dataset in KITTI-like format of [http://www.cvlibs.net/datasets/kitti/eval_odometry.php](data) and ID of sequence in 2 numbers format with the first zero if necessary.

```bash
roslaunch bird_eye_view_tracking main.launch \
	path2odometry:=/root/dataset \
	sequence:=01 \
    points:=/depth_registered/points \
    objects:=/stereo/objects \
    markers:=/bird_eye_view/visualisation
```