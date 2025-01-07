import os
import cv2
import numpy as np
from tqdm import tqdm
import argparser
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import logging
from datetime import datetime

from utils.dataset import iaanetDataset
from utils.box_generate import get_box
from utils.loss import ComputerLoss, DetectLoss, SegLoss
from utils.calculateF1Measure import calculateF1Measure
from models.model import get_model
from models.backbone import backbone

class Trainer():
    def __init__(self, args):
        # 配置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cudnn.benchmark, cudnn.deterministic = True, False

        # 设置日志记录
        log_dir = os.path.join(args.save_path, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger()

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)

        # 对数据集预测标签生成标注框
        if not os.path.exists(args.box_save_path):
            get_box(args.mask_path, args.box_save_path, 4)

        # 加载数据集
        self.train_dataset = iaanetDataset(args.image_path, args.mask_path, args.box_save_path, args.stride)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            shuffle=True,
                                                            batch_size=args.batch_size,
                                                            num_workers=8,
                                                            collate_fn=self.train_dataset.collate_fn)

        # 获取模型
        self.Model = get_model(self.device, args.nel, args.hidden_dim, args.pos_mode)

        # 加载优化器
        self.param_dicts = [
            {"params": [v for k, v in self.Model.named_parameters() if "region_module" in k and v.requires_grad],
             "lr": args.d_lr},
            {"params": [v for k, v in self.Model.named_parameters() if "region_module" not in k and v.requires_grad],
             "lr": args.s_lr}
        ]

        self.optimizer = torch.optim.SGD(self.param_dicts)
        # 学习率调整
        self.fun_1 = lambda epoch: 1
        self.fun_2 = lambda epoch: 0 if epoch < 1 else 1
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=[self.fun_1, self.fun_2])
        # 损失函数
        self.Dcriterion = DetectLoss(obj_pw=args.obj_posw, device=self.device)
        self.Scriterion = SegLoss(posw=args.seg_posw, mode=args.loss_mode, device=self.device)
        self.criterion = ComputerLoss(self.Dcriterion, self.Scriterion)

        self.Loss_list = []
        self.Loss_box = []
        self.Loss_obj = []
        self.Loss_d = []
        self.Loss_s = []

    def train(self, args):
        # 训练
        for epoch in range(args.epochs):
            self.Model.train()
            print(f'Epoch: {epoch} / {args.epochs}')
            self.logger.info(f'Epoch: {epoch} / {args.epochs}')

            mloss = np.zeros(5, dtype=np.float32)
            for i, (input, seg_targets, bbox_targets) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
                detect_output, reg_output, mask_maps, target_boxes = self.Model(input.to(self.device), max_det_num=args.max_det_num, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk)
                loss, loss_items = self.criterion(Dp=detect_output, Dtarget=bbox_targets.to(self.device),
                                                  anchor=backbone().anchor, stride=backbone().stride,
                                                  Sp=reg_output, Starget=seg_targets.to(self.device), mask_maps=mask_maps,
                                                  target_boxes=target_boxes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mloss = (mloss * i + loss_items.cpu().numpy()) / (i + 1)

            self.Loss_box.append(mloss[0])
            self.Loss_obj.append(mloss[1])
            self.Loss_d.append(mloss[2])
            self.Loss_s.append(mloss[3])
            self.Loss_list.append(mloss[4])
            print(f'{reg_output.sigmoid().max()}')
            self.logger.info(f'{reg_output.sigmoid().max()}')

            # TensorBoard记录
            self.writer.add_scalar('Loss/Total', mloss[4], epoch)
            self.writer.add_scalar('Loss/Box', mloss[0], epoch)
            self.writer.add_scalar('Loss/Object', mloss[1], epoch)
            self.writer.add_scalar('Loss/Detect', mloss[2], epoch)
            self.writer.add_scalar('Loss/Segmentation', mloss[3], epoch)

            self.lr_scheduler.step()

            torch.save(self.Model, os.path.join(args.save_path, f'model_{epoch+1}.pth'))

            self.eval(epoch,args.val_image, args.val_mask, self.Model, self.device, args.conf_thres, args.iou_thres, args.expand, args.topk, args.fast)

    def eval(self, epoch, image_path, mask_path, model, device, conf_thres, iou_thres, expand, topk, fast):
        average_F1 = 0
        average_prec = 0
        average_recall = 0

        model.eval()
        with torch.no_grad():
            num = len(os.listdir(image_path))
            for i, img_name in enumerate(os.listdir(image_path)):
                print(f'{i+1}/{num}', end='\r', flush=True)
                self.logger.info(f'{i+1}/{num}')
                input_img = cv2.imread(os.path.join(image_path, img_name), 0) / 255.0
                target = cv2.imread(os.path.join(mask_path, img_name), 0) / 255.0
                input = torch.from_numpy(input_img).to(torch.float)
                input = input[None, None, :]
                _, _, h, w = input.shape

                output, mask_maps, _, _ = model(input.to(device), max_det_num=0, conf_thres=conf_thres, iou_thres=iou_thres, expand=expand, topk=topk, fast=fast)

                probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
                if output is not None:
                    output = output.squeeze()
                    output = output.sigmoid()
                    mask_maps = mask_maps.squeeze()

                    probability_map[~mask_maps] = output

                probability_map = probability_map.cpu().numpy()
                prec, recall, F1 = calculateF1Measure(probability_map, target, 0.5)
                average_F1 = (average_F1 * i + F1) / (i + 1)
                average_prec = (average_prec * i + prec) / (i + 1)
                average_recall = (average_recall * i + recall) / (i + 1)
        print(f'prec:{average_prec}  recall:{average_recall}  F1: {average_F1} ')
        self.logger.info(f'prec:{average_prec}  recall:{average_recall}  F1: {average_F1} ')

        self.writer.add_scalar('F1', average_F1, epoch)
        self.writer.add_scalar('Precision', average_prec, epoch)
        self.writer.add_scalar('Recall', average_recall, epoch)

        return average_F1


if __name__ == '__main__':
    args = argparser.parse_args()

    trainer = Trainer(args)
    trainer.train(args)