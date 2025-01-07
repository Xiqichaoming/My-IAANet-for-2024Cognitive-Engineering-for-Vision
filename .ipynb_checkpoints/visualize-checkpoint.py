import os
import random
import numpy as np
import torch
import argparser

import cv2

if __name__ == '__main__':
    args = argparser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = torch.load('./outputs/model_10.pth').to(device)
    Model.eval()

    img_path = ['./data/test/MDvsFA/image', './data/test/SIRST/image']
    mask_path = ['./data/test/MDvsFA/mask', './data/test/SIRST/mask']

    num_images = 3

    for img, mask in zip(img_path, mask_path):
        print('当前加载的数据集为：', img.split('/')[-2])
        selected_images = random.sample(os.listdir(img), num_images)
        print('随机选择的图片为：', selected_images)

        input_images = [cv2.imread(os.path.join(img, img_name), 0) / 255.0 for img_name in selected_images]
        if img.split('/')[-2] == 'MDvsFA':
            target_images = [cv2.imread(os.path.join(mask, img_name), 0) / 255.0 for img_name in selected_images]
        else:
            target_images = [cv2.imread(os.path.join(mask, img_name.replace('.png', '_pixels0.png')), 0) / 255.0 for img_name in selected_images]
        
        output_images = []
        with torch.no_grad():
            for input in input_images:
                input = torch.from_numpy(input).to(torch.float)
                input = input[None, None, :]
                _, _, h, w = input.shape

                output, mask_maps, _, _ = Model(input.to(device), max_det_num=0, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk, fast=args.fast)

                probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
                if output is not None:
                    output = output.squeeze()
                    output = output.sigmoid()
                    mask_maps = mask_maps.squeeze()

                    probability_map[~mask_maps] = output

                probability_map = probability_map.cpu().numpy()

                output_images.append(probability_map)
    

        for i in range(num_images):
            input_image = input_images[i]
            target_image = target_images[i]
            output_image = output_images[i].astype('float64')

            white_space = 255 * np.ones((input_images[i].shape[0], input_images[i].shape[1]//20), dtype='float64')

            combined_image = cv2.hconcat([input_image, white_space, target_image, white_space, output_image])

            cv2.imwrite(f'./visualize_re/combined_{selected_images[i]}', combined_image * 255)


