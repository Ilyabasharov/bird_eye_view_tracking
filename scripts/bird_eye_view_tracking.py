#!/usr/bin/python

import rospy
import pykitti

import numpy as np

from sensor_msgs.msg import PointCloud2
from camera_objects_msgs.msg import ObjectArray
from visualization_msgs.msg import MarkerArray

from utils import (
    colorGenerator,
    clastering,
    create_marker_array
)

class BirdEyeView:

	def __init__(self):

		rospy.init_node('bird_eye_view_tracking')

        self.synchronizer = rospy.TimeSynchronizer(
            [
            	rospy.Subscriber('point_cloud', PointCloud2),
            	rospy.Subscriber('objects', ObjectArray),
            ],
            queue_size=10,
        )

        self.synchronizer.registerCallback(self.process)

        self.centers_pub = rospy.Publisher(
        	'visualisation',
        	MarkerArray,
        	queue_size=10,
        )

        path2odometry = rospy.get_param('~odometry', '')
        sequence = rospy.get_param('~sequence', '')

        self.datainfo = pykitti.odometry(path2odometry, sequence)
        self.frame_id = 0
        self.track_current, self.track_prev = {}, {}

        self.keep_alive = 30
        self.colors = colorGenerator()

    def run(self):
        rospy.spin()

   	def process(self, pc_msg, objects_msg):

        pc = pointcloud2_to_array(pc_msg)

        num_mask = np.isfinite(point_cloud['x']) \
                 & np.isfinite(point_cloud['y']) \
                 & np.isfinite(point_cloud['z'])

        self.track_prev = {}

        for track_id in self.track_current:
            if self.track_current[track_id]['frame_id'] > self.frame_id - self.keep_alive:
                del self.track_current[track_id]

        for object_msg in objects_msg.objects:

            obj_mask = rle_msg_to_mask(object_msg.rle_msg)
            obj_pc = s2u(pc[num_mask & obj_mask][['x', 'y', 'z']])

            if len(obj_pc) < 50:
                continue

            local_center = np.append(clastering(obj_pc), 1)

            center = (local_center @ self.datainfo.poses[self.frame_id])[:3]

            if object_msg.track_id in self.track_current:

                self.track_prev[object_msg.track_id] = self.track_current[object_msg.track_id]
                self.track_current[object_msg.track_id]['center'] = center
                self.track_current[object_msg.track_id]['frame_id'] = self.frame_id

            else:

                self.track_current[object_msg.track_id] = {
                    'frame_id': self.frame_id,
                    'center':   center,
                }

        self.frame_id += 1

        all_markers = []
        for desc, dev_by in zip((self.track_current, self.track_prev,), (1., 2.,)):
            all_markers.extend(create_marker_array(desc, self.colors, pc_msg.header, dev_by))

        self.centers_pub.publish(MarkerArray(all_markers))


def main():
    
    np.random.seed(100)

    denoiser = BirdEyeView()
    denoiser.run()

if __name__ == '__main__':
    main()
