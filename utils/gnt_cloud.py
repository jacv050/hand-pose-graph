# -*- coding: utf-8 -*-
""" Point Cloud Auxiliary Generator

This module is responsible for generating 3D colored point clouds in PLY format.
It takes an RGB image and the corresponding depth map of a frame and makes use
of the camera calibration intrinsics to project a point cloud in 3D space. In
addition, a binary mask can be provided to produce a point cloud of a certain
class or instance pixels.

Todo:
    * Nothing

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html

"""

import numpy as np
from plyfile import (PlyData, PlyElement)

# Deprecated function that is no longer used since we don't need it anymore after
# we changed the way we generate depth information with UnrealROX. Anyways, the
# function will be left here for the record in case it is useful for any other
# situation or problem.
#
# def camera_depth_to_plane_depth(depth, f):

#     h_ = depth.shape[0]
#     w_ = depth.shape[1]

#     i_c_ = np.float(h_) / 2 - 1
#     j_c_ = np.float(w_) / 2 - 1

#     cols_, rows_ = np.meshgrid(np.linspace(0, w_ - 1, num = w_), np.linspace(0, h_ - 1, num = h_))
#     dist_from_center_ = ((rows_ - i_c_)**2 + (cols_ - j_c_)**2)**(0.5)
#     plane_depth_ = depth / ((1 + dist_from_center_ / f)**2)**(0.5)

#     return plane_depth_

def rgbd_to_rgb_cloud(color, depth, camera, max_depth=float('inf'), binary_mask=None):
    """ Generates a set of 3D points and a set of colors for those points given
    an RGB image and its corresponding depth map.

    Args:
        color: The RGB image of the frame (HxWx3[uchar]).
        depth: The corresponding depth map (HxWx1[float]).
        camera: The camera used to capture the frame which is a container for all
            its calibration values, aspect ratio, and resolution.
        depthmax: A maximum threshold for the depth values to be included in the
            point cloud. Any point above it will be omitted.
        binary_mask: A mask the size of the RGB and depth frames to decide which
            of such pixels will be mapped to the 3D space and which will be left
            outside of the cloud. It is useful to segment certain classes or
            instances from the cloud.

    Returns:
        A tuple contaning an array of 3D points (HWx3[float]) and an array of
        RGB colors for each point (HWx3[uchar]).

    """

    points_ = []
    colors_ = []

    for i in range(0, depth.shape[1]-1):
        for j in range(0, depth.shape[0]-1):

            point_z_ = depth[j][i] / 1000.0

            if((point_z_ <= max_depth) and (binary_mask is None or binary_mask[j][i])):

                point_x_ = (i - camera.cx) * (point_z_ / camera.fx)
                point_y_ = (j - camera.cy) * (point_z_ / camera.fy)

                point_color_ = [color[j][i][0], color[j][i][1], color[j][i][2]]

                points_.append([point_x_, point_y_, point_z_])
                colors_.append(point_color_)

    return np.array(points_), np.array(colors_)

def save_ply_cloud(points, colors, filename):
    """ Creates a PLY point cloud from a set of 3D points and their colors.

    Args:
        points: An array of 3D points (Nx3[float]).
        colors: An array of corresponding RGB colors (Nx3[uchar])
        filename: The name of the output PLY file.

    Returns:
        Nothing at all really.
    """

    vertex_ = np.zeros(points.shape[0],
                       dtype=[('x', 'f4'),
                              ('y', 'f4'),
                              ('z', 'f4'),
                              ('red', 'u1'),
                              ('green', 'u1'),
                              ('blue', 'u1')])

    for i in range(points.shape[0]):
        vertex_[i] = (points[i][0], points[i][1], points[i][2],
                      colors[i][0], colors[i][1], colors[i][2])

    ply_out = PlyData([PlyElement.describe(vertex_, 'vertex', comments=['vertices'])])
    ply_out.write(filename)

def generate_cloud(color, depth, camera, filename, binary_mask=None):
    """ Generates and saves to disk (in PLY format) a colored 3D point cloud
    given an RGB image and its corresponding depth map.

    Args:
        color: The RGB image of the frame (HxWx3[uchar]).
        depth: The corresponding depth map (HxWx1[float]).
        camera: The camera used to capture the frame which is a container for all
            its calibration values, aspect ratio, and resolution.
        filename: The name of the output PLY file.
        binary_mask: A mask the size of the RGB and depth frames to decide which
            of such pixels will be mapped to the 3D space and which will be left
            outside of the cloud. It is useful to segment certain classes or
            instances from the cloud.

    Returns:
        Nothing at all, the point cloud is saved to disk.
    """

    points_, colors_ = rgbd_to_rgb_cloud(color, depth, camera, camera.depthmax, binary_mask)
    save_ply_cloud(points_, colors_, filename)
