import os
import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from compute_iou import my_compute_IoU
from Discrimintor_Net import *
from mydataloader import Mydataload
from torch.utils.data import DataLoader
from Discriminator_D import myresnet50
# from deeplab import myresnet50_1

parser = argparse.ArgumentParser(description='inpainting and forensics training')
parser.add_argument('--origin_path', type=str,
                    default='/media/work/Joker/datasets/test_data/origin_img/circle/circle_5/', help='testing dataset')
parser.add_argument('--img_path', type=str,
                    default='/media/work/Joker/datasets/generative-inpainting-dataset/inpainting_test/circle/circle_5/',
                    help='testing dataset')
parser.add_argument('--label_path', type=str,
                    default='/media/work/Joker/datasets/test_data/mask_and_label/circle/circle_5/image_label/',
                    help='testing label dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--netD', default='checkpoints/forensics_se_iou/forensics_50.pth',                    help="path to netD weight")  # 加载判成器
parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--comparison_dir', type=str, default='result_dir/pre_and_result/', help='save the runing result')
parser.add_argument('--pre_dir', type=str, default='result_dir/prediction/', help='save the prediction result')
parser.add_argument('--target_dir', type=str, default='result_dir/target/', help='save the origin')
parser.add_argument('--result_dir', type=str, default='result_dir/', help='save data...')
parser.add_argument('--shuffle', type=bool, default=True, help='whether or not shuffle the data')


def main(opt):
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available...')
    device = torch.device('cuda:0')
    print('using GPU: ', torch.cuda.get_device_name())

    if not os.path.exists(opt.result_dir):
        os.mkdir(opt.result_dir)
        print("create save data dir successful...")

    if not os.path.exists(opt.pre_dir):
        os.mkdir(opt.pre_dir)
        print("create prediction dir successful...")

    if not os.path.exists(opt.target_dir):
        os.mkdir(opt.target_dir)
        print("create target dir successful...")

    if not os.path.exists(opt.comparison_dir):
        os.mkdir(opt.comparison_dir)
        print("create comparison dir successful...")

    metric = my_compute_IoU().to(device)
    datasets = Mydataload(opt)
    test_dataset = DataLoader(datasets, batch_size=opt.batch_size, shuffle=not opt.shuffle, num_workers=opt.workers)

    dataset_size = len(test_dataset)
    print('testing image dataset size = %d' % dataset_size)

    # model_D = Forensics_Base_ResNet50().to(device)
    model_D = myresnet50().to(device)
    model_D.load_state_dict(torch.load(opt.netD))  # 加载取证网络权重
    model_D.eval()

    tbar = tqdm(test_dataset)
    print('\n Start testing...')
    start_time = time.time()

    for i, data in enumerate(tbar):
        image = data['img']
        label = data['lab']
        path = data['img_path']
        [batch, channel, height, width] = image.size()

        image = image.to(device)
        label = label.to(device)

        output = model_D(image)
        output = torch.softmax(output, 1)

        for p in range(batch):
            path_without_format, _ = path[p].split(".")
            Path_ori = opt.comparison_dir + path_without_format + "ori" + ".png"
            Path_out = opt.comparison_dir + path_without_format + "out" + ".png"
            result = torch.argmax(output, axis=1)
            result = torch.unsqueeze(result, 1).detach().cpu().numpy()[p]
            label_ = torch.squeeze(label, 0)
            label_ = label_.unsqueeze(dim=1).detach().cpu().numpy()[p]
            result = np.transpose(result, (1, 2, 0))
            label_ =np.transpose(label_, (1, 2, 0))

            # save origin and result in a directory
            cv2.imwrite(Path_out, np.uint8(result * 255))  # 保存测试的结果
            cv2.imwrite(Path_ori, np.uint8(label_ * 255))  # save origin

            # seperately save result and target label
            target_dir = opt.target_dir + path[p]
            prediction_dir = opt.pre_dir + path[p]
            cv2.imwrite(target_dir,np.uint8(label_ * 255))
            cv2.imwrite(prediction_dir,np.uint8(result * 255))

    metric_iou = metric(opt.target_dir,opt.pre_dir)
    print('iou metric:',metric_iou)
    end_time = time.time() - start_time
    print('all testing time:%d' % end_time)
    print('\n End testing...')


if __name__ == '__main__':

    print('------------testing-options------------')
    for k, v in sorted(vars(parser.parse_args()).items()):
        print('%s: %s' % (str(k), str(v)))
    print('----------------end---------------------')

    opt = parser.parse_args()
    main(opt)
