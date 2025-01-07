import os
import cv2
import torch 
import argparser

from utils.calculateF1Measure import calculateF1Measure

def eval(image_path, mask_path, model, device, conf_thres, iou_thres, expand, topk, fast):
    average_F1 = 0
    average_prec = 0
    average_recall = 0

    with torch.no_grad():
        num = len(os.listdir(image_path))
        for i, img_name in enumerate(os.listdir(image_path)):
            print(f'{i+1}/{num}', end='\r', flush=True)
            input_img = cv2.imread(os.path.join(image_path, img_name), 0) / 255.0
            if os.path.exists(os.path.join(mask_path, img_name)):
                target = cv2.imread(os.path.join(mask_path, img_name), 0) / 255.0
            else:
                target = cv2.imread(os.path.join(mask_path, img_name.replace('.png', '_pixels0.png')), 0) / 255.0
            input = torch.from_numpy(input_img).to(torch.float)
            input = input[None, None, :]
            _, _, h, w = input.shape
            
            output, mask_maps, _, _= model(input.to(device), max_det_num=0, conf_thres=conf_thres, iou_thres=iou_thres, expand=expand, topk=topk, fast=fast)

            probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                probability_map[~mask_maps] = output

            probability_map = probability_map.cpu().numpy()
            try:
                prec, recall, F1 = calculateF1Measure(probability_map, target, 0.5)
            except:
                print(f'error in {img_name}')
                continue
            average_F1 = (average_F1 * i + F1) / (i + 1)
            average_prec = (average_prec * i + prec) / (i + 1)
            average_recall = (average_recall * i + recall) / (i + 1)
    print(f'prec:{average_prec}  recall:{average_recall}  F1: {average_F1} ')

    return average_F1

if __name__ == '__main__':
    
    args = argparser.parse_args()    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = torch.load('./outputs/model.pth').to(device)
    Model.eval()

    img_path = ['./data/test/MDvsFA/image','./data/test/SIRST/image']
    mask_path = ['./data/test/MDvsFA/mask','./data/test/SIRST/mask']

    for img, mask in zip(img_path, mask_path):
        print(f'evaluating {img}...')
        eval(img, mask, Model, device, args.conf_thres, args.iou_thres, args.expand, args.topk,args.fast)
    
    print('done')
