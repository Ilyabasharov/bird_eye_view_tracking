#!/usr/bin/python

import rospy
import tf
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
    ColorGenerator,
    clastering,
    create_marker_array,
)

def future_odom(future_m, current_m, x):
    
    inv_f = np.linalg.inv(future_m)
    transform = np.dot(inv_f, current_m)

    return np.dot(transform, x)


class TransformBroadcaster:

    def __init__(self, path2odometry, sequence):
        
        if isinstance(sequence, int):
            sequence = str(sequence).zfill(2)

        self.datainfo = pykitti.odometry(path2odometry, sequence)
        self.br = tf.TransformBroadcaster()

    def publish(self, n_frame, header):
        _, _, rpy_angles, translation_vector, _ = \
        tf.transformations.decompose_matrix(self.datainfo.poses[n_frame])
        
        self.br.sendTransform(
            translation_vector,
            tf.transformations.quaternion_from_euler(*rpy_angles),
            header.stamp,
            'odom',
            header.frame_id,
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

        #self.br = TransformBroadcaster(path2odometry, sequence)
        self.datainfo = pykitti.odometry(path2odometry, sequence)
        self.n_frame = 0
        self.track_current, self.track_future = {}, {}

        self.colors = ColorGenerator()

    def run(self):
        rospy.spin()

    def process(self, pc_msg, objects_msg):

        pc = pointcloud2_to_array(pc_msg)

        num_mask = np.isfinite(pc['x']) \
                 & np.isfinite(pc['y']) \
                 & np.isfinite(pc['z'])
        
        to_viz = {}
        self.track_current = {}

        for track_id in self.track_future:
            to_viz[track_id] = self.track_future[track_id]
        self.track_future = {}
        
        for object_msg in objects_msg.objects:
            
            if object_msg.track_id < 1:
                continue

            obj_mask = rle_msg_to_mask(object_msg.rle).astype(np.bool)
            obj_pc = s2u(pc[num_mask & obj_mask][['x', 'y', 'z']])

            if len(obj_pc) < 50:
                continue

            local_center = np.append(clastering(obj_pc), 1)

            self.track_current[object_msg.track_id] = {
                'n_frame': self.n_frame,
                'center':  local_center,
            }
        
        for track_id in self.track_current:
            self.track_future[track_id] = {
                'n_frame': self.n_frame,
                'center':  future_odom(
                    self.datainfo.poses[self.n_frame + 1], 
                    self.datainfo.poses[self.n_frame + 0],
                    self.track_current[track_id]['center'],
                ),
            }


        all_markers = create_marker_array(self.track_current, self.colors, pc_msg.header, 1., 0)
        all_markers.extend(
            create_marker_array(to_viz, self.colors, pc_msg.header, 1.2, len(self.track_current))
        )

        self.centers_pub.publish(MarkerArray(all_markers))
        #self.br.publish(self.n_frame, pc_msg.header)

        self.n_frame += 1


def main():
    
    np.random.seed(100)

    viz = BirdEyeView()
    viz.run()

if __name__ == '__main__':
    main()
