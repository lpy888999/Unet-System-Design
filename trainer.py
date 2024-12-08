import logging
import os
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader,random_split
from torchvision import transforms
from tqdm import tqdm

from utils import DiceLoss


def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10
    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), desc=f"Train: {epoch_num}", total=len(train_loader),
                                           leave=False):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            # logging.info('Train: iteration : %d/%d, lr : %f, loss : %f, loss_ce: %f, loss_dice: %f' % (
            #     iter_num, epoch_num, lr_, loss.item(), loss_ce.item(), loss_dice.item()))
            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()
            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)
        batch_ce_loss /= len(train_loader)
        batch_dice_loss /= len(train_loader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        logging.info('Train epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
            epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_loader), desc=f"Val: {epoch_num}",
                                                   total=len(val_loader), leave=False):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch[:].long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_loader)
                batch_dice_loss /= len(val_loader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                logging.info('Val epoch: %d : loss : %f, loss_ce: %f, loss_dice: %f' % (
                    epoch_num, batch_loss, batch_ce_loss, batch_dice_loss))
                if batch_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Training Finished!"
def trainer_liver(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from datasets.dataset_mine import LiverImageDataset
    # 设置日志记录
    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # 数据集和随机划分
    dataset = LiverImageDataset(root_dir=args.root_path,
                                 transform=transforms.Compose([RandomGenerator(output_size=(args.img_size, args.img_size))]))
    val_split = int(len(dataset) * args.val_ratio)
    train_split = len(dataset) - val_split
    train_dataset, val_dataset = random_split(dataset, [train_split, val_split],
                                              generator=torch.Generator().manual_seed(args.seed))
    logging.info(f"训练集大小: {train_split}, 验证集大小: {val_split}")

    # # 记录一下训练时的数据集划分
    # list_mine_dir = "lists/list_mine"
    # os.makedirs(list_mine_dir, exist_ok=True)
    # # 将训练集文件名记录到train.txt文件
    # train_file_path = os.path.join(list_mine_dir, "train.txt")
    # with open(train_file_path, 'w') as train_file:
    #     for sample in train_dataset:
    #         file_name = sample['case_name'] + ".jpg"  # 假设文件名是加上.png后缀的形式，可根据实际情况修改
    #         train_file.write(file_name + "\n")
    # # 将验证集文件名记录到val.txt文件
    # val_file_path = os.path.join(list_mine_dir, "val.txt")
    # with open(val_file_path, 'w') as val_file:
    #     for sample in val_dataset:
    #         file_name = sample['case_name'] + ".jpg"  # 假设文件名是加上.png后缀的形式，可根据实际情况修改
    #         val_file.write(file_name + "\n")

    # DataLoader设置
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    # 模型、损失函数和优化器
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

    ce_loss = nn.CrossEntropyLoss()
    # ce_loss = nn.BCELoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(os.path.join(snapshot_path, 'log'))

    # 训练循环
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = max_epoch * len(train_loader)
    logging.info(f"{len(train_loader)} 每个epoch的迭代数，总迭代数: {max_iterations}")

    best_loss = float('inf')
    for epoch_num in range(max_epoch):
        model.train()
        epoch_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch_num}"):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.2 * loss_ce + 0.8 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss.item(), iter_num)
            writer.add_scalar('info/loss_ce', loss_ce.item(), iter_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iter_num)

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        logging.info(f"Epoch {epoch_num}: 训练损失: {epoch_loss:.4f}")

        # 验证循环
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for sampled_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch_num}"):
                    image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch.long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    val_loss += 0.2 * loss_ce.item() + 0.8 * loss_dice.item()

                val_loss /= len(val_loader)
                logging.info(f"Epoch {epoch_num}: 验证损失: {val_loss:.4f}")

                if val_loss < best_loss:
                    save_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_path)
                    best_loss = val_loss
                    logging.info(f"最佳模型保存于epoch {epoch_num}")

                save_path = os.path.join(snapshot_path, 'last_model.pth')
                torch.save(model.state_dict(), save_path)

    writer.close()
    logging.info("训练结束！")