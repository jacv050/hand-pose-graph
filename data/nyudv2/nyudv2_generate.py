__author__      = "Alberto Garcia-Garcia"
__copyright__   = "Copyright 2019, 3D Perception Lab"
__credits__     = ["Alberto Garcia-Garcia"]

__license__     = "MIT"
__version__     = "1.0"
__maintainer__  = "Alberto Garcia-Garcia"
__email__       = "agarcia@dtic.ua.es"
__status__ = "Development"

import h5py
import logging
import numpy as np
import numpy.linalg
import pathlib
from PIL import Image
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import png
import sys

# RGB Intrinsic Parameters
fx_rgb = 5.1885790117450188e+02
fy_rgb = 5.1946961112127485e+02
cx_rgb = 3.2558244941119034e+02
cy_rgb = 2.5373616633400465e+02
 
# RGB Distortion Parameters
k1_rgb =  2.0796615318809061e-01
k2_rgb = -5.8613825163911781e-01
p1_rgb = 7.2231363135888329e-04
p2_rgb = 1.0479627195765181e-03
k3_rgb = 4.9856986684705107e-01
 
# Depth Intrinsic Parameters
fx_d = 5.8262448167737955e+02
fy_d = 5.8269103270988637e+02
cx_d = 3.1304475870804731e+02
cy_d = 2.3844389626620386e+02

# RGB Distortion Parameters
k1_d = -9.9897236553084481e-02
k2_d = 3.9065324602765344e-01
p1_d = 1.9290592870229277e-03
p2_d = -1.9422022475975055e-03
k3_d = -5.1031725053400578e-01

classes_40_ = {
"void" : 0,
"wall" : 1,
"floor" : 2,
"cabinet" : 3,
"bed" : 4,
"chair" : 5,
"sofa" : 6,
"table" : 7,
"door" : 8,
"window" : 9,
"bookshelf" : 10,
"picture" : 11,
"counter" : 12,
"blinds" : 13,
"desk" : 14,
"shelves" : 15,
"curtain" : 16,
"dresser" : 17,
"pillow" : 18,
"mirror" : 19,
"floor mat" : 20,
"clothes" : 21,
"ceiling" : 22,
"books" : 23,
"refridgerator" : 24,
"television" : 25,
"paper" : 26,
"towel" : 27,
"shower curtain" : 28,
"box" : 29,
"whiteboard" : 30,
"person" : 31,
"night stand" : 32,
"toilet" : 33,
"sink" : 34,
"lamp" : 35,
"bathtub" : 36,
"bag" : 37,
"otherstructure" : 38,
"otherfurniture" : 39,
"otherprop" : 40,
}

map_classes_ = [40,40,3,22,5,40,12,38,40,40,2,39,40,40,26,40,24,40,7,40,1,40,40,34,38,29,40,8,40,40,40,40,38,40,40,14,40,38,40,40,40,15,39,40,30,40,40,39,40,39,38,40,38,40,37,40,38,38,9,40,40,38,40,11,38,40,40,40,40,40,40,40,40,40,40,40,40,40,38,13,40,40,6,40,23,40,39,10,16,40,40,40,40,38,40,40,40,40,40,40,40,40,40,38,40,39,40,40,40,40,39,38,40,40,40,40,40,40,18,40,40,19,28,33,40,40,40,40,40,40,40,40,40,38,27,36,40,40,40,40,21,40,20,35,40,40,40,40,40,40,40,40,38,40,40,40,4,32,40,40,39,40,39,40,40,40,40,40,17,40,40,25,40,39,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,39,38,38,40,40,39,40,39,40,38,39,38,40,40,40,40,40,40,40,40,40,40,39,40,38,40,40,38,38,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,38,40,40,39,40,40,38,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,31,40,40,40,40,40,40,40,38,40,40,38,39,39,40,40,40,40,40,40,40,40,40,38,40,39,40,40,39,40,40,40,38,40,40,40,40,40,40,40,40,38,39,40,40,40,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,38,39,40,40,40,40,40,40,40,39,40,40,40,40,40,40,38,40,40,40,38,40,39,40,40,40,39,39,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,39,39,40,40,39,39,40,40,40,40,38,40,40,38,39,39,40,39,40,39,38,40,40,40,40,40,40,40,40,40,40,40,39,40,38,40,39,40,40,40,40,40,39,39,40,40,40,40,40,40,39,39,40,40,38,39,39,40,40,40,40,40,40,40,40,40,39,39,40,40,40,40,39,40,40,40,40,40,39,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,38,40,40,40,40,40,40,40,39,38,39,40,38,39,40,39,40,39,40,40,40,40,40,40,40,40,38,40,40,40,40,40,38,40,40,39,40,40,40,39,40,38,40,40,40,40,40,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,38,40,40,38,40,40,38,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,40,38,40,40,40,40,40,40,40,40,40,40,40,38,38,38,40,40,40,38,40,40,40,38,38,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,38,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,39,40,40,40,40,38,38,40,40,40,38,40,40,40,40,40,40,40,40,40,40,40,40,40,40,39,40,40,39,40,40,39,39,40,40,40,40,40,40,40,40,39,39,39,40,40,40,40,39,40,40,40,40,40,40,40,40,39,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,40,40,40,40,40,38,40,40,40,40,40,40,40,39,40,40,38,40,39,40,40,40,40,38,40,40,40,40,40,38,40,40,40,40,40,40,40,39,40,40,40,40,40,40,40,40,40,39,40,40,]

palette_ = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (191, 0, 0),
    (0, 191, 0),
    (0, 0, 191),
    (191, 191, 0),
    (191, 0, 191),
    (0, 191, 191),
    (191, 162, 155),
    (197, 168, 161),
    (203, 174, 167),
    (209, 180, 173),
    (215, 186, 179),
    (221, 192, 185),
    (227, 198, 191),
    (233, 204, 197),
    (239, 210, 203),
    (245, 216, 209),
    (251, 222, 215),
    (1, 228, 221),
    (7, 234, 227),
    (13, 240, 233),
    (19, 246, 239),
    (25, 252, 245),
    (31, 2, 251),
    (37, 8, 1),
    (43, 14, 7),
    (49, 20, 13),
    (55, 26, 19),
    (61, 32, 25),
    (67, 38, 31),
    (73, 44, 37),
    (79, 50, 43),
    (85, 56, 49),
    (91, 62, 55),
    (97, 68, 61),
]

def rgbd_to_rgb_cloud(depth, color, labels, cx, cy, fx, fy):

    points_ = []
    colors_ = []
    labels_ = []

    for i in range(0, depth.shape[1]):
        for j in range(0, depth.shape[0]):

            z_ = depth[j][i]
            x_ = (i - cx) * (z_ / fx)
            y_ = (j - cy) * (z_ / fy)

            r_ = color[j][i][0]
            g_ = color[j][i][1]
            b_ = color[j][i][2]

            points_.append([x_, y_, z_])
            colors_.append([r_, g_, b_])
            labels_.append(labels[j][i])

    return np.array(points_), np.array(colors_), np.array(labels_)

def save_ply_cloud(points, colors, labels, filename):

    vertex = np.zeros(points.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('label', 'u1')])

    for i in range(points.shape[0]):
        vertex[i] = (points[i][0], points[i][1], points[i][2], colors[i][0], colors[i][1], colors[i][2], labels[i])
        
    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices'])])
    ply_out.write(filename)

def crop_projection_mask(img):

    if (len(img.shape) == 2):
        return img[45:471, 41:601]
    elif (len(img.shape) == 3):
        return img[45:471, 41:601, :]

def relabel(labels):

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            id_ = labels[i][j]
            class_ = map_classes_[id_-1]

            if (class_ in classes_40_.values()):
                labels[i][j] = class_
            else:
                labels[i][j] = classes_40_["void"]

    return labels

if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    log_ = logging.getLogger(__name__)

    log_.info("Loading nyu_depth_v2_labeled.mat")
    f_ = h5py.File('nyu_depth_v2_labeled.mat', 'r')

    log_.info("Getting array of RGB images...")
    rgb_images_ = np.array(f_.get('images'))
    log_.info(rgb_images_.shape)

    log_.info("Getting array of depth images...")
    depth_images_ = np.array(f_.get('depths'))
    log_.info(depth_images_.shape)

    log_.info("Getting array of label images...")
    label_images_ = np.array(f_.get('labels'))
    log_.info(label_images_.shape)

    out_path_ = "raw/"
    out_dirs_ = ["rgb", "depth", "label", "cloud", "cloud_gt"]
    out_paths_ = [out_path_ + d for d in out_dirs_]

    log_.info("Generating output directories {0}".format(out_paths_))

    for d in out_paths_:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True) 

    for i in range(rgb_images_.shape[0]):

        log_.info("Processing image {0}".format(i))

        # Fetch RGB image in HxWxC order and crop it to projection mask
        rgb_ = np.transpose(rgb_images_[i, :, :, :], (2, 1, 0))
        rgb_ = crop_projection_mask(rgb_)

        # Save RGB image in PNG format
        img_rgb_ = Image.fromarray(rgb_, 'RGB')
        img_rgb_.save("{}rgb/rgb_{:04d}.png".format(out_path_, i))

        # Fetch depth map in HxW order and crop it to projection mask
        depth_ = np.transpose(depth_images_[i], (1, 0)) * 1000.0
        depth_ = crop_projection_mask(depth_)

        # Convert depth in meters to 16-bit representation and save it as a single
        # channel PNG with a bit depth of 16 per channel
        depth_img_ = (65535 *((depth_ - depth_.min())/depth_.ptp())).astype(np.uint16)
        with open("{}depth/depth_{:04d}.png".format(out_path_, i), "wb") as depth_file_:
            writer_ = png.Writer(width=depth_img_.shape[1], height=depth_img_.shape[0], bitdepth=16, greyscale=True)
            depth_list_ = depth_img_.tolist()
            writer_.write(depth_file_, depth_list_)

        # Fetch labels image in HxW, crop it to projection mask, and remap the 895
        # classes to the 40-class setting
        img_label_ = np.transpose(label_images_[i], (1,0))
        img_label_ = crop_projection_mask(img_label_)
        img_label_ = relabel(img_label_)

        # Save label image as an indexed single-channel PNG with 8 bits per channel
        with open("{}label/label_{:04d}.png".format(out_path_, i), "wb") as label_file_:
            writer_ = png.Writer(width=img_label_.shape[1], height=img_label_.shape[0], palette=palette_, bitdepth=8)
            img_label_list_ = img_label_.tolist()
            writer_.write(label_file_, img_label_list_)

        # Project the depth map on the RGB image to generate a colored point cloud and
        # save it as a binary PLY with color (RGB) and label per vertex
        points_, colors_, labels_ = rgbd_to_rgb_cloud(depth_, rgb_, img_label_, cx_d, cy_d, fx_d, fy_d)
        save_ply_cloud(points_, colors_, labels_, "{}cloud/cloud_{:04d}.ply".format(out_path_, i))

        # Get the ground-truth colored point cloud and save it as a binary PLY too
        colors_gt_ = []
        for k in range(len(labels_)):
            colors_gt_.append(palette_[labels_[k]])
        save_ply_cloud(points_, colors_gt_, labels_, "{}cloud_gt/cloud_{:04d}.ply".format(out_path_,i))