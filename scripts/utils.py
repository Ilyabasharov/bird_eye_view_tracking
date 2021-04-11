#!/usr/bin/python

import rospy
import colorsys
import numpy as np
import pycocotools.mask as mask_utils

from ros_numpy.point_cloud2 import (
    pointcloud2_to_array,
)
from numpy.lib.recfunctions import structured_to_unstructured as s2u
from sklearn.cluster import DBSCAN

from visualization_msgs.msg import (
    Marker,
)
from std_msgs.msg import (
    ColorRGBA,
)
from geometry_msgs.msg import (
    Vector3,
    Pose,
    Point,
    Quaternion,
)


class colorGenerator:

    def __init__(self, basis=30):
        brightness = 0.7
        hsv = [(float(i) / basis, 1. / (i + 1), brightness) for i in range(basis)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        
        self.colors = np.random.permutation(colors)
        self.basis = basis

    def get_color(self, track_id):
        return self.colors[track_id % self.basis]


def rle_msg_to_mask(rle_msg):
    mask = mask_utils.decode(
        {
            'counts': rle_msg.data,
            'size':   [rle_msg.height, rle_msg.width],
        }
    )
    return mask


def clastering(pc):

    n_sampling = 500 if len(pc) > 500 else len(pc)

    indexes = np.linspace(0, len(pc), n_sampling, dtype=np.int32, endpoint=False)
    points_to_fit = pc[indexes]

    db = DBSCAN(eps=3, min_samples=2).fit(points_to_fit)

    values, counts = np.unique(db.labels_, return_counts=True)
    biggest_subcluster_id = values[np.argmax(counts)]

    mask_biggest_subcluster = db.labels_ == biggest_subcluster_id
    points_of_biggest_subcluster = points_to_fit[mask_biggest_subcluster]

    return points_of_biggest_subcluster.mean(axis=0)


def create_marker_array(desc, colormap, header, dev_by=1, start_id=0):

    markers = []

    for track_id in desc:

        color = colormap.get_color(track_id)

        marker = Marker(
            id       = start_id,
            header   = header,
            type     = Marker.SPHERE,
            action   = Marker.ADD,
            lifetime = rospy.Duration(1.),
            color    = ColorRGBA(
                r = color[0],
                g = color[1],
                b = color[2],
                a = 1.,
            ),
            scale    = Vector3(
                x = 4. / dev_by,
                y = 4. / dev_by,
                z = 4. / dev_by,
            ),
            pose     = Pose(
                position    = Point(
                    x = desc[track_id]['center'][0],
                    y = desc[track_id]['center'][1],
                    z = desc[track_id]['center'][2],
                ),
                orientation = Quaternion(
                    x = 0.,
                    y = 0.,
                    z = 0.,
                    w = 1.,
                ),
            ),
        )
        
        start_id += 1
        markers.append(marker)

    return markers
