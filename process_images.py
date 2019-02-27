# Copyrights. All rights reserved.
# ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Space Center (eSpace), 2018
# See the LICENSE.TXT file for more details.

import torch

from practical_deep_stereo import pds_network

import numpy as np
import imageio
import shutil


def save_image(filename, img_th):
    img_np = img_th.cpu().detach().numpy()
    img_np = (img_np * 255.).astype(np.uint8)
    imageio.imwrite(filename, np.transpose(img_np, (1,2,0)))

def read_image(filename):
    img_np = imageio.imread(filename)
    img_np = np.transpose(img_np, (2,0,1))[None,:,:,:].astype(np.float32) / 255.
    img_th = torch.tensor(img_np)
    # print(img_th.size())
    return img_th

def get_images(dataset, scene_id=0, sample_id=0):
    if dataset == "mp3d":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/mp3d/stereo/train/{:05}/{:03}_{}"
    elif dataset == "suncg_f":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/suncg/stereo/furnished/train1/{:05}/{:03}_{}"
    elif dataset == "suncg_e":
        file_template = "/home/adosovit/work/toolboxes/2019/Revisiting_Single_Depth_Estimation/data/suncg/stereo/empty/train1/{:05}/{:03}_{}"
    else:
        raise Exception("Unknown dataset", dataset)

    left_file = file_template.format(scene_id, sample_id, "color_l.jpg")
    right_file = file_template.format(scene_id, sample_id, "color_r.jpg")
    depth_file = file_template.format(scene_id, sample_id, "depth.png")

    return left_file, right_file, depth_file

class PDSNet:
    def __init__(self, max_disparity=127, checkpoint_file="experiments/flyingthings3d/010_checkpoint.bin"):
        self.max_disparity = max_disparity
        checkpoint = torch.load(checkpoint_file)
        self.network = pds_network.PdsNetwork(max_disparity).cuda()
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()

    def compute_disparity(self, left_image, right_image):
        return self.network(left_image, right_image)

def process_images():

    out_template = "results/{}_{:05}_{:03}_{{}}"
    dataset = "suncg_e" # "mp3d" "suncg_f" "suncg_e"
    model = PDSNet()

    for scene_id in range(10):
        print("=Scene {}".format(scene_id))
        for sample_id in range(1):
            print("   Sample {}".format(sample_id))
            left_file, right_file, depth_file = get_images(dataset, scene_id, sample_id)
            left_image = read_image(left_file).cuda()
            right_image = read_image(right_file).cuda()
            out_template_curr =  out_template.format(dataset, scene_id, sample_id)
            shutil.copyfile(depth_file, out_template_curr.format("depth_gt.png"))

            disparity = model.compute_disparity(left_image, right_image)

            save_image(out_template_curr.format("right.png"), right_image[0])
            save_image(out_template_curr.format("left.png"), left_image[0])
            save_image(out_template_curr.format("disparity.png"), disparity / float(model.max_disparity))
            depth_pred = 5. / torch.clamp(disparity, 5., model.max_disparity)
            save_image(out_template_curr.format("depth_pred.png"), depth_pred)

if __name__ == "__main__":
    process_images()
