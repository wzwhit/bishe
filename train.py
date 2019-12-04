import argparse
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.cityscapes import Cityscapes
import os
from model.build_fast_isa import ISA
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss

def val(args, model, dataloader):
    print('start val!')
    with torch.no_grad():
        model.eval()
        precision_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        for i, (data, label_dice, _) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label_dice = label_dice.cuda()
                #  label_ce = label_ce.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())

            # get RGB label image
            label_dice = label_dice.squeeze()
            label_dice = reverse_one_hot(label_dice)
            label_dice = np.array(label_dice.cpu())

            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label_dice)
            hist += fast_hist(label_dice.flatten(), predict.flatten(), args.num_classes)

            precision_record.append(precision)
        precision = np.mean(precision_record)
        miou_list = per_class_iu(hist)[:-1]
        miou = np.mean(miou_list)
        print('precision per pixel for test: %.3f' % precision)
        print('mIoU for validation: %.3f' % miou)
        return precision, miou

def train(args, model, optimizer, dataloader_train, dataloader_val, start_epoch, start_step):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    #  if args.loss == 'dice':
        #  loss_func = DiceLoss()
    #  elif args.loss == 'crossentropy':
        #  loss_func = torch.nn.CrossEntropyLoss()
    loss_func_dice = DiceLoss()
    loss_func_ce = torch.nn.CrossEntropyLoss()
    max_miou = 0
    step = start_step
    for epoch in range(start_epoch, args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, (data, label_dice, label_ce) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label_dice = label_dice.cuda()
                label_ce = label_ce.cuda()
            output = model(data)
            loss = loss_func_ce(output, label_ce) - math.log(loss_func_dice(output, label_dice))

            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))

        state = {
            'net': model.module.state_dict(),
            'loss': loss_train_mean,
            'epoch': epoch,
            'step': step,
        }
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(state,
                       os.path.join(args.save_model_path, 'latest_dice_loss.pth'))
        if epoch % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            if miou > max_miou:
                max_miou = miou
                torch.save(state,
                           os.path.join(args.save_model_path, 'best_dice_loss.pth'))
            writer.add_scalar('epoch/precision_val', precision, epoch)
            writer.add_scalar('epoch/miou val', miou, epoch)

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=1024, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='dice', help='loss function, dice or crossentropy')
    parser.add_argument('--resume', type=bool, default=False, help='whether to load weights from ckpt')

    args = parser.parse_args(params)

    #  # create dataset and dataloader
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset_train = Cityscapes(args.data, csv_path, scale=(args.crop_height, args.crop_width),
                                loss=args.loss, mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    dataset_val = Cityscapes(args.data, csv_path, scale=(args.crop_height, args.crop_width),
                         loss=args.loss, mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    model = ISA(num_classes=args.num_classes, out_channels=256, isa_H=128, isa_W=64, isa_P_h=16, isa_P_w=8)
    if torch.cuda.is_available() and args.use_gpu:
        print('use gpu: %s'%(args.use_gpu))
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    start_epoch = 0
    start_step = 0
    # load pretrained model if exists
    if args.pretrained_model_path is not None and args.resume:
        print(args.resume)
        print('load model from %s ...' % args.pretrained_model_path)
        checkpoint = torch.load(args.pretrained_model_path)
        model.module.load_state_dict(checkpoint['net']) #, strict=False)
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, start_epoch, start_step)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '1000',
        '--learning_rate', '1e-3',  #2.5e-2
        '--data', '/bigdata/wzw/Cityscapes',
        '--num_workers', '8',
        '--num_classes', '20',
        '--batch_size', '16',  # 6 for resnet101, 12 for resnet18
        '--pretrained_model_path', './ckpt/latest_dice_loss.pth',
        '--save_model_path', './ckpt',
        '--context_path', 'resnet18',  # only support resnet18 and resnet101
        '--optimizer', 'adam',
        '--resume', False,
    ]
    main(params)
