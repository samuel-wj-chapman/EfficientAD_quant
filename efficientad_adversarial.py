#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
import itertools
import os
import random
import shutil
from tqdm import tqdm
from common import get_pdn_small, get_pdn_medium, Autoencoder_improved , \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score


# this version of the model does not use the teacher, instead relies on the autoencoder to learn the features accurately
# from the treacher. we use various methods to help this stage.

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='/dataset2/mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

class Discriminator(torch.nn.Module):
    def __init__(self, feature_dim):
        super(Discriminator, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(feature_dim, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config = get_argparse()

    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path
    else:
        raise Exception('Unknown config.dataset')

    pretrain_penalty = True
    if config.imagenet_train_path == 'none':
        pretrain_penalty = False

    # create output dir
    train_output_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_output_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    

    if os.path.exists(train_output_dir):
        shutil.rmtree(train_output_dir)
    os.makedirs(train_output_dir)
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)
    os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    if config.dataset == 'mvtec_ad':
        # mvtec dataset paper recommend 10% validation set
        train_size = int(0.9 * len(full_train_set))
        validation_size = len(full_train_set) - train_size
        rng = torch.Generator().manual_seed(seed)
        train_set, validation_set = torch.utils.data.random_split(full_train_set,
                                                           [train_size,
                                                            validation_size],
                                                           rng)
    elif config.dataset == 'mvtec_loco':
        train_set = full_train_set
        validation_set = ImageFolderWithoutTarget(
            os.path.join(dataset_path, config.subdataset, 'validation'),
            transform=transforms.Lambda(train_transform))
    else:
        raise Exception('Unknown config.datasetpenalty_loader_infinite')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)

    if pretrain_penalty:
        # load pretraining data for penalty
        penalty_transform = transforms.Compose([
            transforms.Resize((2 * image_size, 2 * image_size)),
            transforms.RandomGrayscale(0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                                  0.225])
        ])
        penalty_set = ImageFolderWithoutTarget(config.imagenet_train_path,
                                               transform=penalty_transform)
        penalty_loader = DataLoader(penalty_set, batch_size=1, shuffle=True,
                                    num_workers=4, pin_memory=True)
        penalty_loader_infinite = InfiniteDataloader(penalty_loader)
    else:
        penalty_loader_infinite = itertools.repeat(None)
        # Create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = Autoencoder_improved(out_channels)

    # Initialize the discriminator
    discriminator = Discriminator(feature_dim=out_channels)

    # Set models to appropriate modes
    teacher.eval()
    student.train()
    autoencoder.train()
    discriminator.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()
        discriminator.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    
    # Optimizers and schedulers
    optimizer_student = torch.optim.Adam(student.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler_student = torch.optim.lr_scheduler.StepLR(
        optimizer_student, step_size=int(0.95 * config.train_steps), gamma=0.1)

    optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler_ae = torch.optim.lr_scheduler.StepLR(
        optimizer_ae, step_size=int(0.95 * config.train_steps), gamma=0.1)

    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=1e-5, weight_decay=1e-6)
    scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
        optimizer_discriminator, step_size=int(0.95 * config.train_steps), gamma=0.1)

    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()

        # -----------------------------------
        # Update Discriminator
        # -----------------------------------
        # Get teacher and autoencoder outputs
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        ae_output = autoencoder(image_ae)

        # Discriminator predictions
        pred_teacher = discriminator(teacher_output_ae.detach())
        pred_ae = discriminator(ae_output.detach())

        # Labels for discriminator
        real_labels = torch.ones_like(pred_teacher)
        fake_labels = torch.zeros_like(pred_ae)

        # Compute discriminator loss
        loss_disc_real = torch.nn.functional.binary_cross_entropy(pred_teacher, real_labels)
        loss_disc_fake = torch.nn.functional.binary_cross_entropy(pred_ae, fake_labels)
        loss_discriminator = (loss_disc_real + loss_disc_fake) / 2

        # Backpropagate discriminator loss
        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()
        scheduler_discriminator.step()

        # -----------------------------------
        # Update Autoencoder and Student
        # -----------------------------------
        # Adversarial loss for autoencoder
        pred_ae_for_ae = discriminator(ae_output)
        adversarial_loss = torch.nn.functional.binary_cross_entropy(pred_ae_for_ae, real_labels)

        # Reconstruction loss
        distance_ae = (teacher_output_ae - ae_output) ** 2
        loss_reconstruction = torch.mean(distance_ae)

        # Total autoencoder loss
        loss_ae_total = loss_reconstruction + adversarial_loss  

        # Backpropagate autoencoder loss
        optimizer_ae.zero_grad()
        loss_ae_total.backward()
        optimizer_ae.step()
        scheduler_ae.step()

        # Student update remains the same
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std

        student_output_ae = student(image_ae)
        ae_output_detached = ae_output.detach()
        distance_stae = (ae_output_detached - student_output_ae) ** 2
        loss_stae = torch.mean(distance_stae)

        optimizer_student.zero_grad()
        loss_stae.backward()
        optimizer_student.step()
        scheduler_student.step()

        # -----------------------------------
        # Logging
        # -----------------------------------
        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Iter: {}, Loss AE: {:.4f}, Adv Loss: {:.4f}, Disc Loss: {:.4f}".format(
                    iteration, loss_reconstruction.item(), adversarial_loss.item(), loss_discriminator.item()))


        if iteration % 1000 == 0:
            torch.save(teacher, os.path.join(train_output_dir,
                                             'teacher_tmp.pth'))
            torch.save(student, os.path.join(train_output_dir,
                                             'student_tmp.pth'))
            torch.save(autoencoder, os.path.join(train_output_dir,
                                                 'autoencoder_tmp.pth'))

        if iteration % 10000 == 0 and iteration > 0:
            # run intermediate evaluation
            teacher.eval()
            student.eval()
            autoencoder.eval()

            q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
                test_output_dir=None, desc='Intermediate inference')
            print('Intermediate image auc: {:.4f}'.format(auc))

            # teacher frozen
            teacher.eval()
            student.train()
            autoencoder.train()

    teacher.eval()
    student.eval()
    autoencoder.eval()

    torch.save(teacher, os.path.join(train_output_dir, 'teacher_final.pth'))
    torch.save(student, os.path.join(train_output_dir, 'student_final.pth'))
    torch.save(autoencoder, os.path.join(train_output_dir,
                                         'autoencoder_final.pth'))

    q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    auc = test(
        test_set=test_set, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))
    filename = "accuracy.txt"
    # Open the file in append mode ('a' mode creates the file if it doesn't exist)
    with open(filename, 'a') as file:
        # Write the accuracy and label, separated by a comma or tab
        file.write(f"{auc}\t{config.subdataset}\n")

def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        if on_gpu:
            image = image.cuda()
        map_ae, teach_ae= predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = map_ae
        teacher_map = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        teacher_map = torch.nn.functional.interpolate(
            teacher_map, (orig_height, orig_width), mode='bilinear')

        teacher_map = teacher_map[0, 0].cpu().numpy()

        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')

        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))

        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            file_teacher = os.path.join(test_output_dir, defect_class, img_nm + '_teacher.tiff')
            tifffile.imwrite(file, map_combined)
            tifffile.imwrite(file_teacher, teacher_map)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
             q_ae_start=None, q_ae_end=None):

    student_output = student(image)
    teacher_output = teacher(image)
    autoencoder_output = autoencoder(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    map_ae = torch.mean((autoencoder_output -
                         student_output)**2,
                        dim=1, keepdim=True)
    teach_ae = torch.mean((teacher_output - autoencoder_output)**2, dim=1, keepdim=True)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    return map_ae, teach_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):

    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        if on_gpu:
            image = image.cuda()
        map_ae, teach_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_ae.append(map_ae)
    maps_ae = torch.cat(maps_ae)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        if on_gpu:
            train_image = train_image.cuda()
            
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        if on_gpu:
            train_image = train_image.cuda()
        teacher_output = teacher(train_image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
