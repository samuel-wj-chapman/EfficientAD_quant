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
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score

import torch
import model_compression_toolkit as mct
from typing import Iterator, List


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
                        default='./mvtec_anomaly_detection',
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

def main():
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)

    config = get_argparse()


    seed = 42
    out_channels = 384
    image_size = 256
    from common import get_pdn_small, get_autoencoder, UnifiedAnomalyDetectionModel


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
    test_output_dir_quant = os.path.join(config.output_dir, 'anomaly_maps_quant',
                                   config.dataset, config.subdataset, 'test')
    

    #if os.path.exists(train_output_dir):
    #    shutil.rmtree(train_output_dir)
    #os.makedirs(train_output_dir)
    #if os.path.exists(test_output_dir):
    #    shutil.rmtree(test_output_dir)
    #os.makedirs(test_output_dir)

    # load data
    full_train_set = ImageFolderWithoutTarget(
        os.path.join(dataset_path, config.subdataset, 'train'),
        transform=transforms.Lambda(train_transform))
    test_set = ImageFolderWithPath(
        os.path.join(dataset_path, config.subdataset, 'test'))
    
    print("Size of the test dataset:", len(test_set))

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
        raise Exception('Unknown config.dataset')


    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
    train_loader_infinite = InfiniteDataloader(train_loader)
    validation_loader = DataLoader(validation_set, batch_size=1)



    # Assuming the paths to the trained model weights are defined
    teacher_weights = 'output/1/trainings/mvtec_ad/bottle/teacher_final.pth'
    student_weights = 'output/1/trainings/mvtec_ad/bottle/student_final.pth'
    autoencoder_weights = 'output/1/trainings/mvtec_ad/bottle/autoencoder_final.pth'

    # Create model instances
    #teacher_model = get_pdn_small(out_channels)  # or get_pdn_medium based on your configuration
    #student_model = get_pdn_small(2 * out_channels)  # Adjust according to your needs
    #autoencoder_model = get_autoencoder(out_channels)


    # Load trained weights with map_location
    teacher = torch.load(teacher_weights, map_location=torch.device('cpu'))
    student = torch.load(student_weights, map_location=torch.device('cpu'))
    autoencoder = torch.load(autoencoder_weights, map_location=torch.device('cpu'))

    # Load trained weights
    #teacher_model.load_state_dict(torch.load(teacher_weights))
    #student_model.load_state_dict(torch.load(student_weights))
    #autoencoder_model.load_state_dict(torch.load(autoencoder_weights))
    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
    # Create the unified model


    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=train_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    
    unified_model = UnifiedAnomalyDetectionModel(teacher, student, autoencoder, out_channels, teacher_mean, teacher_std, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None)




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
    '''
    # create models
    if config.model_size == 'small':
        teacher = get_pdn_small(out_channels)
        student = get_pdn_small(2 * out_channels)
    elif config.model_size == 'medium':
        teacher = get_pdn_medium(out_channels)
        student = get_pdn_medium(2 * out_channels)
    else:
        raise Exception()
    state_dict = torch.load(config.weights, map_location='cpu')
    teacher.load_state_dict(state_dict)
    autoencoder = get_autoencoder(out_channels)

    # teacher frozen
    teacher.eval()
    student.train()
    autoencoder.train()

    if on_gpu:
        teacher.cuda()
        student.cuda()
        autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    optimizer = torch.optim.Adam(itertools.chain(student.parameters(),
                                                 autoencoder.parameters()),
                                 lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=int(0.95 * config.train_steps), gamma=0.1)   
    tqdm_obj = tqdm(range(config.train_steps))
    for iteration, (image_st, image_ae), image_penalty in zip(
            tqdm_obj, train_loader_infinite, penalty_loader_infinite):
        if on_gpu:
            image_st = image_st.cuda()
            image_ae = image_ae.cuda()
            if image_penalty is not None:
                image_penalty = image_penalty.cuda()
        with torch.no_grad():
            teacher_output_st = teacher(image_st)
            teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
        student_output_st = student(image_st)[:, :out_channels]
        distance_st = (teacher_output_st - student_output_st) ** 2
        d_hard = torch.quantile(distance_st, q=0.999)
        loss_hard = torch.mean(distance_st[distance_st >= d_hard])

        if image_penalty is not None:
            student_output_penalty = student(image_penalty)[:, :out_channels]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty
        else:
            loss_st = loss_hard

        ae_output = autoencoder(image_ae)
        with torch.no_grad():
            teacher_output_ae = teacher(image_ae)
            teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
        student_output_ae = student(image_ae)[:, out_channels:]
        distance_ae = (teacher_output_ae - ae_output)**2
        distance_stae = (ae_output - student_output_ae)**2
        loss_ae = torch.mean(distance_ae)
        loss_stae = torch.mean(distance_stae)
        loss_total = loss_st + loss_ae + loss_stae

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        scheduler.step()

        if iteration % 10 == 0:
            tqdm_obj.set_description(
                "Current loss: {:.4f}  ".format(loss_total.item()))

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

            q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
                validation_loader=validation_loader, teacher=teacher,
                student=student, autoencoder=autoencoder,
                teacher_mean=teacher_mean, teacher_std=teacher_std,
                desc='Intermediate map normalization')
            auc = test(
                test_set=test_set, teacher=teacher, student=student,
                autoencoder=autoencoder, teacher_mean=teacher_mean,
                teacher_std=teacher_std, q_st_start=q_st_start,
                q_st_end=q_st_end, q_ae_start=q_ae_start, q_ae_end=q_ae_end,
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
    '''

    unified_model.eval()
    import json
    torch.save(unified_model.state_dict(), 'combined_model.pth')
    params = {
        'out_channels': unified_model.out_channels,
        'teacher_mean': unified_model.teacher_mean.tolist(),  # Convert tensors to lists
        'teacher_std': unified_model.teacher_std.tolist(),
        'q_st_start': unified_model.q_st_start.item() if unified_model.q_st_start is not None else None,
        'q_st_end': unified_model.q_st_end.item() if unified_model.q_st_end is not None else None,
        'q_ae_start': unified_model.q_ae_start.item() if unified_model.q_ae_start is not None else None,
        'q_ae_end': unified_model.q_ae_end.item() if unified_model.q_ae_end is not None else None
    }


    
    params_save_path = 'unified_model_params.json'
    with open(params_save_path, 'w') as f:
        json.dump(params, f)
    
    if q_st_end is not None: 
        print('warning code wont work', q_st_start)

    auc = test(
        test_set=test_set, unified_model=unified_model,  q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))

    
    # Change the function name and parameters to reflect its new functionality
    def train_dataset_generator(train_loader: DataLoader) -> Iterator[List]:
        while True:
            for data, _ in train_loader:
                yield [data.numpy()]

    # Update the representative dataset generator to use the new train dataset generator
    def get_representative_dataset(n_iter: int, dataset_loader: Iterator[List]) -> Iterator[List]:
        def representative_dataset() -> Iterator[List]:
            ds_iter = iter(dataset_loader)
            for _ in range(n_iter):
                yield next(ds_iter)
        return representative_dataset

    # Usage example with the modified generator
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)  # Ensure this matches your batch size and other DataLoader settings
    train_dataset = train_dataset_generator(train_loader)
    representative_dataset_gen = get_representative_dataset(n_iter=20, dataset_loader=train_dataset)



    # Set target platform capabilities
    tpc = mct.get_target_platform_capabilities(fw_name="pytorch", target_platform_name='imx500', target_platform_version='v1')

    # Configuration for mixed precision quantization
    mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=5, use_hessian_based_scores=False)
    config = mct.core.CoreConfig(mixed_precision_config=mp_config,
                           quantization_config=mct.core.QuantizationConfig(shift_negative_activation_correction=True))

    # Define target Resource Utilization for mixed precision weights quantization (75% of 'standard' 8bits quantization)
    #resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=unified_model,
    #                                                                       representative_data_gen=representative_dataset_gen,
    #                                                                       core_config=config,
    #                                                                       target_platform_capabilities=tpc)
    #resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * 0.75)

    # Perform post training quantization
    quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=unified_model,
                                                                representative_data_gen=representative_dataset_gen,#target_resource_utilization=resource_utilization,
                                                                core_config=config,
                                                                target_platform_capabilities=tpc)


    auc = test(test_set=test_set, unified_model=quant_model,  q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir_quant, desc='Final inference')
    print('Final image auc quant: {:.4f}'.format(auc))
    

def test(test_set,unified_model, q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
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
        map_combined = unified_model(image)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')

        #map_combined = map_combined[0, 0].cpu().numpy() #.cpu()
        map_combined = map_combined[0, 0].detach().cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(path))
        if test_output_dir is not None:
            img_nm = os.path.split(path)[1].split('.')[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def test(test_set, unified_model, q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    for image, target, path in tqdm(test_set, desc=desc):
        orig_width, orig_height = image.size
        image_tensor = default_transform(image)[None]  # Add batch dimension
        if on_gpu:
            image_tensor = image_tensor.cuda()
        map_combined = unified_model(image_tensor)
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear', align_corners=False)
        map_combined = map_combined[0, 0].detach().cpu().numpy()

        # Convert map_combined to a heatmap
        heatmap = cm.jet(map_combined)  # Apply colormap
        heatmap = np.uint8(cm.ScalarMappable(cmap='jet').to_rgba(map_combined) * 255)
        heatmap_pil = Image.fromarray(heatmap, 'RGBA').convert('RGB')  # Convert RGBA to RGB

        # Ensure the original image is in RGB
        image_pil = image.convert('RGB')

        # Combine the original image and the heatmap side by side
        combined_image = Image.new('RGB', (orig_width * 2, orig_height))
        combined_image.paste(image_pil, (0, 0))
        combined_image.paste(heatmap_pil, (orig_width, 0))

        defect_class = os.path.basename(os.path.dirname(path))
        demo_output_dir = os.path.join(test_output_dir, "demo", defect_class)
        if not os.path.exists(demo_output_dir):
            os.makedirs(demo_output_dir)
        img_nm = os.path.split(path)[1].split('.')[0]
        file_path = os.path.join(demo_output_dir, img_nm + '_combined.jpg')
        combined_image.save(file_path, 'JPEG')

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100


def test_onnx(test_set, unified_model, q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    for image_path in tqdm(test_set, desc=desc):
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        image = default_transform(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        if on_gpu:
            image = image.cuda()
        # Ensure the model is in evaluation mode
        unified_model.eval()
        with torch.no_grad():
            map_combined = unified_model(torch.from_numpy(image).float())
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')

        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = os.path.basename(os.path.dirname(image_path))
        if test_output_dir is not None:
            img_nm = os.path.splitext(os.path.basename(image_path))[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output, defect_class, img_nm + '.tiff')
            tifffile.imwrite(file, map_combined)

        y_true_image = 0 if defect_class == 'good' else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100
'''
@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def predictv2(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, out_channels:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae


@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for image, _ in tqdm(validation_loader, desc=desc):
        #if on_gpu:
        #    image = image.cuda()
        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for train_image, _ in tqdm(train_loader, desc='Computing mean of features'):
        #if on_gpu:
            #train_image = train_image.cuda()
        #print("Train image shape:", train_image.shape)
#        print("Expected input shape by the model:", teacher.input_shape)
        teacher_output = teacher(train_image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for train_image, _ in tqdm(train_loader, desc='Computing std of features'):
        #if on_gpu:
        #    train_image = train_image.cuda()
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
