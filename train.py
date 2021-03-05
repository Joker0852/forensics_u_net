import os
import time
import torch
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from mydataloader import Mydataload
from torch.utils.data import DataLoader
from utils import adjust_learnig_rate
from loss_function import IOU_loss
from forensics_net import forensics_net

parser = argparse.ArgumentParser(description='inpainting and forensics training')
parser.add_argument('--img_path', type=str, default='inpainting_path/train_small/', help='training dataset')
parser.add_argument('--label_path', type=str, default='label/train_small/',
                    help='traing and testing label dataset')
parser.add_argument('--batchsize', type=int, default=8, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--num_workers', type=int, default=4, help='multi threads to process data')
parser.add_argument('--shuffle', type=bool, default=True, help='whether or not shuffle the data')
parser.add_argument('--lr', type=float, default=1e-3, help='initializing the learn rate for adam')
parser.add_argument('--gpu_ids', type=int, default=0, help='gpu ids:0,1,2 cpu:-1')
parser.add_argument('--seed', type=int, default=1, help='random seeds')
parser.add_argument('--model_file',type=str,default='checkpoints/',help=' ')
parser.add_argument('--log_dir', type=str, default='log_dir', help='save the current running log')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--epoch_num', type=int, default=100, help='the number of training')
parser.add_argument('--cosine_max_value', type=int, default=10,help='the number of interation for training Generator')


def main(opt):
    if not torch.cuda.is_available():
        raise Exception('At least one gpu must be available...')
    device = torch.device('cuda:0')
    print('using GPU: ', torch.cuda.get_device_name(opt.gpu_ids))

    if not os.path.exists(opt.log_dir):
        os.mkdir(opt.log_dir)
        print("create log dir successfully...")

    writer = SummaryWriter(logdir='{}/{}'.format(opt.log_dir, 'log'), comment='inpainting-forensics')

    # loading training dataset
    print('loading dataset...(it may take a few minute...)')
    datasets = Mydataload(opt)
    dataset = DataLoader(datasets, batch_size=opt.batchsize, shuffle=not opt.shuffle, num_workers=opt.num_workers)

    dataset_size = len(dataset)
    print('training image dataset size = %d' % dataset_size)

    #load forensics net
    print('load the forensics net...')
    model = forensics_net()
    # model_D.load_state_dict(torch.load(opt.netD))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1.0, 5.0])).to(device)
    criterion1 = IOU_loss().to(device)

    # 安装优化器
    opt_model = torch.optim.Adam(model.parameters(), lr=opt.lr,betas=(opt.beta2, 0.999))

    # 调整学习率
    CosineLR_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_model, T_max=opt.cosine_max_value,eta_min=0)

    # 先训练判别器
    tbar = tqdm(dataset)
    total_iteration = 0
    start_time = time.time()  # 训练开始的时间
    for epoch in range(opt.epoch_num):
        print('\n Training epoch: %d' % epoch)
        for i, data in enumerate(tbar):
            total_iteration += 1

            image = data['img']
            label = data['lab']

            image = image.to(device)
            label = label.to(device)
            label_iou = label
            label = torch.squeeze(label,1).long()

            out = model(image)
            loss_ce = criterion(out, label)
            # loss_iou = criterion1(out,label_iou)
            # loss = loss_iou + loss_ce
            opt_model_D.zero_grad()  # 将梯度清零
            loss_ce.backward()
            iter_loss1 = loss_ce.item()
            opt_model_D.step()
            # CosineLR_D.step()
            if total_iteration % opt.print_freq == 0:
                print("[ epoch", epoch, "]  loss = ", iter_loss1)
                writer.add_scalar('loss_ce', iter_loss1, dataset_size * (epoch) + i)
                # writer.add_scalar('loss_iou', loss_iou.item(), dataset_size * (epoch) + i)
                # writer.add_scalar('learning_rate',CosineLR_D,dataset_size * (epoch) + i)

        tbar.close()
        if np.mod(epoch,10) == 0:
            print('adjust discriminator learning rate...')
            adjust_learnig_rate(opt, model, epoch)

        # 保存模型
        if np.mod(epoch, 5) == 0:
            torch.save(model_D.state_dict(), 'checkpoints/base_net/forensics_' + str(epoch) + '.pth')

    dura_epoch_time = time.time() - start_time
    print("dura_epoch_time", dura_epoch_time)
    writer.close()
    print('\nEnd training')


if __name__ == '__main__':

    print('------------training-options------------')
    for k, v in sorted(vars(parser.parse_args()).items()):
        print('%s: %s' % (str(k), str(v)))
    print('----------------end---------------------')

    opt = parser.parse_args()
    main(opt)
