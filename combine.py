seed = 42
out_channels = 384
image_size = 256
from common import get_pdn_small, get_autoencoder, UnifiedAnomalyDetectionModel
import torch
from efficientad2 import teacher_normalization
from torch.utils.data import DataLoader
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
import argparse

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


#on_gpu = torch.cuda.is_available()
# Assuming the paths to the trained model weights are defined
teacher_weights = 'output/1/trainings/mvtec_ad/bottle/teacher_final.pth'
student_weights = 'output/1/trainings/mvtec_ad/bottle/student_final.pth'
autoencoder_weights = 'output/1/trainings/mvtec_ad/bottle/autoencoder_final.pth'

# Create model instances
teacher_model = get_pdn_small(out_channels)  # or get_pdn_medium based on your configuration
student_model = get_pdn_small(2 * out_channels)  # Adjust according to your needs
autoencoder_model = get_autoencoder(out_channels)

config = get_argparse()

if config.dataset == 'mvtec_ad':
    dataset_path = config.mvtec_ad_path
elif config.dataset == 'mvtec_loco':
    dataset_path = config.mvtec_loco_path
else:
    raise Exception('Unknown config.dataset')
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
    raise Exception('Unknown config.dataset')
# Load trained weights with map_location
teacher = torch.load(teacher_weights, map_location=torch.device('cpu'))
student = torch.load(student_weights, map_location=torch.device('cpu'))
autoencoder = torch.load(autoencoder_weights, map_location=torch.device('cpu'))
train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              num_workers=4, pin_memory=True)
# Load trained weights
#teacher_model.load_state_dict(torch.load(teacher_weights))
#student_model.load_state_dict(torch.load(student_weights))
#autoencoder_model.load_state_dict(torch.load(autoencoder_weights))
teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)
teacher_mean = teacher_mean.cpu()
teacher_std = teacher_std.cpu()

# Create the unified model
unified_model = UnifiedAnomalyDetectionModel(teacher, student, autoencoder, out_channels, teacher_mean, teacher_std)
unified_model.to('cpu')


dummy_input = torch.randn(4, 3, 256, 256)  # Batch size of 4, 3 channels, 256x256 images


import torch
import model_compression_toolkit as mct
from typing import Iterator, List

# Constants
BATCH_SIZE = 4
n_iters = 20
image_size = 256  # Assuming image size from your model configuration
out_channels = 384  # Assuming out_channels from your model configuration
'''
# Generate random noise dataset
def random_noise_dataset_generator(batch_size: int, n_channels: int, height: int, width: int) -> Iterator[List]:
    while True:
        # Generate random noise images
        noise_images = torch.randn(batch_size, n_channels, height, width)
        yield [noise_images.numpy()]

# Define representative dataset generator
def get_representative_dataset(n_iter: int, dataset_loader: Iterator[List]) -> Iterator[List]:
    def representative_dataset() -> Iterator[List]:
        ds_iter = iter(dataset_loader)
        for _ in range(n_iter):
            yield next(ds_iter)
    return representative_dataset

# Create random noise dataset generator
random_noise_dataset = random_noise_dataset_generator(BATCH_SIZE, 3, image_size, image_size)
representative_dataset_gen = get_representative_dataset(n_iter=n_iters, dataset_loader=random_noise_dataset)
'''

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

print('Quantized model is ready')


mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./q_anom.onnx',
                                  repr_dataset=representative_dataset_gen)


