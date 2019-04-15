import _init_paths
import argparse
import os
import copy
import random
import numpy as np
from PIL import Image
import scipy.io as scio
import scipy.misc
import numpy.ma as ma
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable
from datasets.ycb.dataset import PoseDataset
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import euler_matrix, quaternion_matrix, quaternion_from_matrix
from vanilla_segmentation.segnet import SegNet as segnet
from PIL import ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir')
parser.add_argument('--model', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--seg_model', type=str, default = '',  help='resume SegNet model')
parser.add_argument('--save_processed_image', type=bool, default = False,  help='Save image with model points')
opt = parser.parse_args()

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])
cam_cx = 312.9869
cam_cy = 241.3109
cam_fx = 558.341390
# cam_fx = 1066.778
cam_fy = 558.387543
# cam_fy = 1067.487
cam_scale = 10000.0
num_obj = 21
img_width = 480
img_length = 640
num_points = 1000
num_points_mesh = 500
iteration = 2
# iteration = 0
bs = 1
dataset_config_dir = 'datasets/ycb/dataset_config'
ycb_toolbox_dir = 'YCB_Video_toolbox'
result_wo_refine_dir = 'experiments/eval_result/ycb/Densefusion_wo_refine_result'
result_refine_dir = 'experiments/eval_result/ycb/Densefusion_iterative_result'
colors = [0xC0C0C0, 0x708069, 0xFFFFFF, 0xFAEBD7, 0xF0FFFF, 0xFFFFCD, 0xFF0000,
          0x9C661F, 0x872657, 0xFFC0CB, 0xFF4500, 0xFF00FF, 0xFFFF00, 0x802A2A,
          0x0000FF, 0x03A89E, 0x00FFFF, 0x00FF00, 0xA020F0, 0x00FF7F, 0xDA70D6,
          0xDDA0DD]
label_strings = ['ground', 'master_chef_can', 'cracker_box', 'sugar_box', 'tomato_soup_can',
                 'mustard_bottle', 'tuna_fish_can', 'pudding_box', 'gelatin_box',
                 'potted_meat_can', 'banana', 'pitcher_base', 'bleach_cleanser',
                 'bowl', 'mug', 'power_drill', 'wood_block', 'scissors', 'large_marker',
                 'large_clamp', 'extra_large_clamp', 'foam_brick']

def projection(point, cx, cy, fx, fy):
    tx = point[0] * cam_scale
    ty = point[1] * cam_scale
    tz = point[2] * cam_scale
    x = fx * tx / tz + cx
    y = fy * ty / tz + cy
    if x < 0 or x > img_length:
        x = 0
    if y < 0 or y > img_width:
        y = 0
    return int(x), int(y)


def get_bbox(posecnn_rois):
    rmin = int(posecnn_rois[idx][3]) + 1
    rmax = int(posecnn_rois[idx][5]) - 1
    cmin = int(posecnn_rois[idx][2]) + 1
    cmax = int(posecnn_rois[idx][4]) - 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def get_bbox(rmin, rmax, cmin, cmax):
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


seg = segnet()      # seg模型
seg = seg.cuda()
seg.load_state_dict(torch.load(opt.seg_model))
seg.eval()

estimator = PoseNet(num_points = num_points, num_obj = num_obj)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points = num_points, num_obj = num_obj)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

testlist = []
input_file = open('{0}/test_data_list.txt'.format(dataset_config_dir))
while 1:
    input_line = input_file.readline()
    if not input_line:
        break
    if input_line[-1:] == '\n':
        input_line = input_line[:-1]
    testlist.append(input_line)
input_file.close()
print(len(testlist))

class_file = open('{0}/classes.txt'.format(dataset_config_dir))
class_id = 1
cld = {}
while 1:
    class_input = class_file.readline()
    if not class_input:
        break
    class_input = class_input[:-1]

    input_file = open('{0}/models/{1}/points.xyz'.format(opt.dataset_root, class_input))
    cld[class_id] = []
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        input_line = input_line[:-1]
        input_line = input_line.split(' ')
        cld[class_id].append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
    input_file.close()
    cld[class_id] = np.array(cld[class_id])
    class_id += 1

norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
# for now in range(0, 448):
for now in range(0, 50):
    # 图片读取
    img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, testlist[now]))
    depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, testlist[now])))
    output_img = copy.deepcopy(img)     # 投影过后的图片

    # 语义分割
    rgb = np.array(trancolor(img).convert("RGB"))
    rgb = np.transpose(rgb, (2, 0, 1))
    rgb = norm(torch.from_numpy(rgb.astype(np.float32)))
    rgb = Variable(rgb.unsqueeze(0)).cuda()     # segnet是按batch处理的，所以添加一个维度
    seg_outputs = seg(rgb)
    _, seg_predicted = torch.max(seg_outputs, 1)
    label = seg_predicted.cpu().numpy().astype(np.int8)
    label = label.squeeze(0)        # 去除添加的维度
    lst = np.unique(label)      # 图片中有的类别
    lst = lst[lst.nonzero()]    # 去除0

    label_img = Image.fromarray(label, mode='L')
    drawObject_label = ImageDraw.Draw(label_img)
    drawObject_label.ink = 255

    drawObject_point = ImageDraw.Draw(output_img)
    drawObject_point.ink = 255

    # posecnn_meta = scio.loadmat('{0}/results_PoseCNN_RSS2018/{1}.mat'.format(ycb_toolbox_dir, '%06d' % now))
    # label = np.array(posecnn_meta['labels'])
    # posecnn_rois = np.array(posecnn_meta['rois'])

    # lst = posecnn_rois[:, 1:2].flatten()
    my_result_wo_refine = []
    my_result = []
    
    for idx in range(len(lst)):
        itemid = lst[idx]
        # if itemid != 12:
        #     continue
        try:
            # rmin, rmax, cmin, cmax = get_bbox(posecnn_rois)
            # bbox
            item_index = np.where(label == itemid)   # 类别像素索引
            rmin = item_index[0].min()
            rmax = item_index[0].max()
            cmin = item_index[1].min()
            cmax = item_index[1].max()
            # rmin, rmax, cmin, cmax = get_bbox(rmin, rmax, cmin, cmax)

            # label画bounding box
            drawObject_label.line([cmin, rmin, cmax, rmin])
            drawObject_label.line([cmin, rmin, cmin, rmax])
            drawObject_label.line([cmax, rmax, cmin, rmax])
            drawObject_label.line([cmax, rmax, cmax, rmin])
            drawObject_label.text([cmin, rmin], label_strings[itemid])

            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, itemid))
            mask = mask_label * mask_depth

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) / ((rmax - rmin) * (cmax - cmin)) > 0.4:
                if len(choose) > num_points:
                    c_mask = np.zeros(len(choose), dtype=int)
                    c_mask[:num_points] = 1
                    np.random.shuffle(c_mask)
                    choose = choose[c_mask.nonzero()]
                elif len(choose) > 300:
                    choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')
                else:
                    continue
            else:
                continue

            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)

            img_masked = np.array(img)[:, :, :3]
            img_masked = np.transpose(img_masked, (2, 0, 1))
            img_masked = img_masked[:, rmin:rmax, cmin:cmax]

            cloud = torch.from_numpy(cloud.astype(np.float32))
            choose = torch.LongTensor(choose.astype(np.int32))
            img_masked = norm(torch.from_numpy(img_masked.astype(np.float32)))
            index = torch.LongTensor([itemid - 1])

            cloud = Variable(cloud).cuda()
            choose = Variable(choose).cuda()
            img_masked = Variable(img_masked).cuda()
            index = Variable(index).cuda()

            cloud = cloud.view(1, num_points, 3)        # 选出来的点
            img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

            pred_r, pred_t, pred_c, emb = estimator(img_masked, cloud, choose, index)
            pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)

            pred_c = pred_c.view(bs, num_points)
            how_max, which_max = torch.max(pred_c, 1)
            pred_t = pred_t.view(bs * num_points, 1, 3)
            points = cloud.view(bs * num_points, 1, 3)

            my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
            my_t = (points + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
            my_pred = np.append(my_r, my_t)
            my_result_wo_refine.append(my_pred.tolist())

            for ite in range(0, iteration):
                T = Variable(torch.from_numpy(my_t.astype(np.float32))).cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)
                my_mat = quaternion_matrix(my_r)
                R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
                my_mat[0:3, 3] = my_t
                
                new_cloud = torch.bmm((cloud - T), R).contiguous()
                pred_r, pred_t = refiner(new_cloud, emb, index)
                pred_r = pred_r.view(1, 1, -1)
                pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
                my_r_2 = pred_r.view(-1).cpu().data.numpy()
                my_t_2 = pred_t.view(-1).cpu().data.numpy()
                my_mat_2 = quaternion_matrix(my_r_2)

                my_mat_2[0:3, 3] = my_t_2

                my_mat_final = np.dot(my_mat, my_mat_2)
                my_r_final = copy.deepcopy(my_mat_final)
                my_r_final[0:3, 3] = 0
                my_r_final = quaternion_from_matrix(my_r_final, True)
                my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

                my_pred = np.append(my_r_final, my_t_final)
                my_r = my_r_final
                my_t = my_t_final

            # Here 'my_pred' is the final pose estimation result after refinement ('my_r': quaternion, 'my_t': translation)

            my_result.append(my_pred.tolist())

            my_r = quaternion_matrix(my_r)[:3, :3]
            pred = np.dot(cld[itemid], my_r.T) + my_t  # 旋转后的点云

            if opt.save_processed_image and how_max.cpu().data > 0.1:
                drawObject_point.text([cmin, rmin], label_strings[itemid] + ' : ' + str(how_max.cpu().data.numpy()))
                # 绘制模型的点在二维图上
                for my_t in pred:
                    x, y = projection(my_t, cam_cx, cam_cy, cam_fx, cam_fy)
                    output_img.putpixel((x, y), colors[int(itemid)])

        except ZeroDivisionError:
            print("PoseCNN Detector Lost {0} at No.{1} keyframe".format(itemid, now))
            my_result_wo_refine.append([0.0 for i in range(7)])
            my_result.append([0.0 for i in range(7)])

    label_img.save('img/%04d_pre_label.png' % now)
    output_img.save('img/%04d_projected_rgb.png' % now)

    scio.savemat('{0}/{1}.mat'.format(result_wo_refine_dir, '%04d' % now), {'poses':my_result_wo_refine})
    scio.savemat('{0}/{1}.mat'.format(result_refine_dir, '%04d' % now), {'poses':my_result})
    print("Finish No.{0} keyframe".format(now))