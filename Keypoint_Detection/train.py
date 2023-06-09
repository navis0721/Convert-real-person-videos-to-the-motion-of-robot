import datetime
import time
import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from keypoint_rcnn import keypointrcnn_resnet50_fpn
from torchvision import transforms as Trans

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model(modelType, train_weight=None):
    print("Loading model")
    # num_classes: background is first class, object is second
    model = torchvision.models.detection.__dict__[modelType](num_classes=2,
                                                              pretrained=False)
    if train_weight: 
        model.load_state_dict(torch.load(train_weight))
    return model

def main(args):
    print(args.device)
    
    device = torch.device('cuda', 0) #use gpu first gpu to train
    torch.cuda.set_device(device)
    
    # Data loading code
    print("Loading data")
    dataset = get_coco(args.data_path, image_set= "train", transforms=get_transform(train=True), dataNum=20000) # training data
    dataset_test = get_coco(args.data_path, image_set="val", transforms=get_transform(train=False)) #validation data
    
    num = list(range(1, 4000, 2))
    num = num+list(range(4001, 5000, 1))
    dataset_test = torch.utils.data.Subset(dataset_test, num)
    
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.RandomSampler(dataset_test)
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    print(len(train_sampler))
    print(len(test_sampler))
    
    print("Creating data loaders")
    # for training Dataloader can use batch to train
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler,
        collate_fn=utils.collate_fn)
    # for validation
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler,
        collate_fn=utils.collate_fn)
    
    
    model = get_model(modelType=args.model)
   
    model.to(device)    

    # optimize the model
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #adjust learning rate every step_size lr = lr*gamma
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) 

    print("Start training")
    start_time = time.time()
    i = 0
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=device) # evaluate after every epoch
        torch.save(model.state_dict(), 'training_weights'+str(i)+'.pth') # save model /epoch
        i = i+1
    # save the model weights
    torch.save(model.state_dict(), 'training_weights.pth')
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Keypoint Detection Training')
    parser.add_argument('--data-path', default='./data/COCO2017/') 
    parser.add_argument('--dataset', default='coco_kp')
    parser.add_argument('--model', default='keypointrcnn_resnet50_fpn')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001, type=float) 
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, dest='weight_decay')
    parser.add_argument('--lr-step-size', default=5, type=int) # decrease lr every step-size epochs
    parser.add_argument('--lr-gamma', default=0.1, type=float) # decrease lr by a factor of lr-gamma
    parser.add_argument('--print-freq', default=20, type=int)
    
    args = parser.parse_args()

    main(args)
