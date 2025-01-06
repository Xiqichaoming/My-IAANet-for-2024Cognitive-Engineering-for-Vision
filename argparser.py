import argparse
import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/config.yaml', \
                        help="path to config file")
    parser.add_argument("--image_path", type=str, default='./data/train/image',\
                        help="path to load training image")
    parser.add_argument("--mask_path", type=str, default='./data/train/mask', \
                        help="path to load training masks")
    parser.add_argument("--val_image", type=str, default="./data/test/MDvsFA/image", \
                        help="path to load validation image")
    parser.add_argument("--val_mask", type=str, default="./data/test/MDvsFA/mask", \
                        help="path to load validation masks")
    parser.add_argument("--save_path", type=str, default='./outputs/demo', \
                        help="path to save model")
    parser.add_argument("--box_save_path", type=str, default='./data/train_box_gt.csv', \
                        help="path to save box ground truth")
    parser.add_argument("--pos_mode", type=str, default='cosin', \
                        help="position embedding type, ['cosin'] & [''learned] are available")
    parser.add_argument("--iou_thres", type=float, default=0.6, \
                        help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.2, \
                        help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=5, \
                        help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--nel", type=int, default=4, \
                        help="number of encoder layer")
    parser.add_argument("--stride", type=int, default=16, \
                        help="stride of backbone")
    parser.add_argument("--loss_mode", default=None, \
                        help="['focal'], None")
    parser.add_argument("--seg_posw", type=int, default=3, \
                        help="Positive weights of BCEwithLogitLoss for segmentation")
    parser.add_argument("--obj_posw", type=int, default=10, \
                        help="Positive weights of BCEwithLogitLoss for object detection")
    parser.add_argument("--max_det_num", type=int, default=5, \
                        help="Maximum number of region boxes proposed by detect head")
    parser.add_argument("--hidden_dim", type=int, default=512, \
                        help="hidden_dim of transformer")
    parser.add_argument("--d_lr", type=float, default=0.01, \
                        help="Learning Rate for detection")
    parser.add_argument("--s_lr", type=float, default=0.001, \
                        help="Learning Rate for segmentation")
    parser.add_argument("--batch_size", type=int, default=8, \
                        help="Batch Size")
    parser.add_argument("--epochs", type=int, default=10, \
                        help="Epochs")
    parser.add_argument('--rpn_pretrained', type=str, default=None, \
                        help='load pretrained rpn weights')
    parser.add_argument("--expand", type=int, default=8, \
                        help="The additonal length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', \
                        help='fast inference')
    
    if parser.parse_args().config:
        args = parser.parse_args()
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        args = argparse.Namespace(**config)
        return args

    return parser.parse_args()
        



