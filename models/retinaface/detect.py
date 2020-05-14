from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg
from prior_box import PriorBox
from py_cpu_nms import py_cpu_nms
import cv2
from retinaface import RetinaFace
from box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Final_Retinaface.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.05, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = RetinaFace(phase="test")
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    resize = 1
    # testing begin
    # for i in range(1):
    video_capture = cv2.VideoCapture(0)
    root_path = "C:/Users/ASUS/Desktop/Face_Recognition-master/dataset/"
    name = input("Enter your name for registration")
    input_directory = os.path.join(root_path, name)
    if not os.path.exists(input_directory):
        os.makedirs(input_directory, exist_ok = 'True')
    no_of_frame = 1
    while True:
        ret,frame = video_capture.read()
        img = np.float32(frame)
        im_height,im_width,_ = img.shape
        scale = torch.Tensor([img.shape[1],img.shape[0],img.shape[1],img.shape[0]])
        img -= (104,117,123)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)
        print('net forward time: {:.4f}'.format(time.time() - tic))

        
        priorbox = PriorBox(cfg, image_size=(im_height,im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data 
        boxes = decode(loc.data.squeeze(0),prior_data,cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:,1]
        landms = decode_landm(landms.data.squeeze(0),prior_data,cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 /resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes,scores[:,np.newaxis])).astype(np.float32,copy=False)
        keep =py_cpu_nms(dets,args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k,:]
        landms = landms[:args.keep_top_k,:]
        
        dets = np.concatenate((dets,landms),axis=1)

        # show image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(frame, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(frame, (b[5], b[6]), 4, (0, 0, 255), 4)
                cv2.circle(frame, (b[7], b[8]), 4, (0, 255, 255), 4)
                cv2.circle(frame, (b[9], b[10]), 4, (255, 0, 255), 4)
                cv2.circle(frame, (b[11], b[12]), 4, (0, 255, 0), 4)
                cv2.circle(frame, (b[13], b[14]), 4, (255, 0, 0), 4)

            cv2.imshow("Capture",frame)
            cv2.imwrite(os.path.join(input_directory,str(name + str(no_of_frame) + '_' + '.jpg')),frame)
            no_of_frame+= 1
            key = cv2.waitKey(2)
            if no_of_frame > 100:
                break
            else:  
                if key == ord('q'):
                    break

    video_capture.release() 
    cv2.destroyAllWindows()
