import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision import transforms
from coco_utils import get_coco
import transforms as T
from plot import plot_poses
import numpy as np

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(train_weight=None):
    print("Creating model")
    # num_classes: background is first class, object is second
    model = torchvision.models.detection.__dict__['keypointrcnn_resnet50_fpn'](num_classes=2,
                                                              pretrained=False)
    
    if train_weight: 
        model.load_state_dict(torch.load(train_weight))
    return model

def main():
    device = torch.device('cuda', 8) #use gpu first gpu to train
    torch.cuda.set_device(device)

    # Data loading code
    print("Loading data")
    # dataset_test, num_classes = get_dataset(args.dataset, "train", get_transform(train=True))
    dataset_test = get_coco('./data/plot_img/', image_set="val", transforms=get_transform(train=False))
    model = get_model(train_weight='training_weights.pth')
    model.to(device)
    model.eval()
    detect_threshold = 0.7
    keypoint_score_threshold = 2
    with torch.no_grad():
        for i in range(2645): 
            print(i)
            img, _ = dataset_test[i]
            prediction = model([img.to(device)])
            keypoints = prediction[0]['keypoints'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()
            keypoints_scores = prediction[0]['keypoints_scores'].cpu().numpy()
            idx = np.where(scores>detect_threshold)
            keypoints = keypoints[idx]
            keypoints_scores = keypoints_scores[idx]
            for j in range(keypoints.shape[0]):
                for num in range(17):
                    if keypoints_scores[j][num]<keypoint_score_threshold:
                        keypoints[j][num]=[0,0,0]
            img = img.mul(255).permute(1, 2, 0).byte().numpy()
            plot_poses(img,keypoints,save_name='./plot_result/'+str(i)+'.jpg')
    print('Finish!')
    
if __name__ == "__main__":
    main()
