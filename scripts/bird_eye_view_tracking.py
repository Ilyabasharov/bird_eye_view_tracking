#!/usr/bin/python

import rospy
import message_filters
from ros_numpy.point_cloud2 import pointcloud2_to_array

import pykitti
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured as s2u

from sensor_msgs.msg import PointCloud2
from camera_objects_msgs.msg import ObjectArray
from visualization_msgs.msg import MarkerArray

from utils import (
    rle_msg_to_mask,
    colorGenerator,
    clastering,
    create_marker_array,
)

class BirdEyeView:

    def __init__(self):

        rospy.init_node('bird_eye_view_tracking')

        self.synchronizer = message_filters.TimeSynchronizer(
            [
                message_filters.Subscriber('point_cloud', PointCloud2),
                message_filters.Subscriber('objects', ObjectArray),
            ],
            queue_size=10,
        )

        self.synchronizer.registerCallback(self.process)

        self.centers_pub = rospy.Publisher(
            'visualisation',
            MarkerArray,
            queue_size=10,
        )

        path2odometry = rospy.get_param('~odometry', '/home/docker_solo/dataset')
        sequence = rospy.get_param('~sequence', '01')
        if isinstance(sequence, int):
            sequence = str(sequence).zfill(2)

        print(path2odometry, sequence)

        self.datainfo = pykitti.odometry(path2odometry, sequence)
        self.frame_id = 0
        self.track_current, self.track_prev = {}, {}

        self.colors = colorGenerator()

    def run(self):
        rospy.spin()

    def process(self, pc_msg, objects_msg):

        pc = pointcloud2_to_array(pc_msg)

        num_mask = np.isfinite(pc['x']) \
                 & np.isfinite(pc['y']) \
                 & np.isfinite(pc['z'])
        
        transform = self.datainfo.poses[self.frame_id]

        self.track_prev = {}
        for track_id in self.track_current:
            self.track_prev[track_id] = self.track_current[track_id]
        self.track_current = {}
        
        for object_msg in objects_msg.objects:
            
            if object_msg.track_id < 1:
                continue

            obj_mask = rle_msg_to_mask(object_msg.rle).astype(np.bool)
            obj_pc = s2u(pc[num_mask & obj_mask][['x', 'y', 'z']])

            if len(obj_pc) < 50:
                continue

            local_center = np.append(clastering(obj_pc), 1)
            center = np.dot(transform, local_center)[:3]

            self.track_current[object_msg.track_id] = {
                'frame_id': self.frame_id,
                'center':   center,
            }

        self.frame_id += 1

        all_markers = []
        for desc, dev_by, start_id in zip(
                (self.track_current, self.track_prev,),
                (1., 1.5,), (0, len(self.track_current))):
            all_markers.extend(create_marker_array(desc, self.colors, pc_msg.header, dev_by, start_id))

        self.centers_pub.publish(MarkerArray(all_markers))


def main():
    
    np.random.seed(100)

    denoiser = BirdEyeView()
    denoiser.run()

if __name__ == '__main__':
    main()
