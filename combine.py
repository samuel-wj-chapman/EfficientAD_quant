seed = 42
out_channels = 384
image_size = 256
from common import get_pdn_small, get_autoencoder, UnifiedAnomalyDetectionModel
import torch


on_gpu = torch.cuda.is_available()
# Assuming the paths to the trained model weights are defined
teacher_weights = 'output/1/trainings/mvtec_ad/bottle/teacher_final.pth'
student_weights = 'output/1/trainings/mvtec_ad/bottle/student_final.pth'
autoencoder_weights = 'output/1/trainings/mvtec_ad/bottle/autoencoder_final.pth'

# Create model instances
teacher_model = get_pdn_small(out_channels)  # or get_pdn_medium based on your configuration
student_model = get_pdn_small(2 * out_channels)  # Adjust according to your needs
autoencoder_model = get_autoencoder(out_channels)


# Load trained weights with map_location
teacher_model = torch.load(teacher_weights, map_location=torch.device('cpu'))
student_model = torch.load(student_weights, map_location=torch.device('cpu'))
autoencoder_model = torch.load(autoencoder_weights, map_location=torch.device('cpu'))

# Load trained weights
#teacher_model.load_state_dict(torch.load(teacher_weights))
#student_model.load_state_dict(torch.load(student_weights))
#autoencoder_model.load_state_dict(torch.load(autoencoder_weights))

# Create the unified model
unified_model = UnifiedAnomalyDetectionModel(teacher_model, student_model, autoencoder_model, out_channels)


dummy_input = torch.randn(4, 3, 256, 256)  # Batch size of 4, 3 channels, 256x256 images


import torch
import model_compression_toolkit as mct
from typing import Iterator, List

# Constants
BATCH_SIZE = 4
n_iters = 20
image_size = 256  # Assuming image size from your model configuration
out_channels = 384  # Assuming out_channels from your model configuration

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

# Set target platform capabilities
tpc = mct.get_target_platform_capabilities(fw_name="pytorch", target_platform_name='imx500', target_platform_version='v1')

# Configuration for mixed precision quantization
mp_config = mct.core.MixedPrecisionQuantizationConfig(num_of_images=5, use_hessian_based_scores=False)
config = mct.core.CoreConfig(mixed_precision_config=mp_config,
                             quantization_config=mct.core.QuantizationConfig(shift_negative_activation_correction=True))

# Define target Resource Utilization for mixed precision weights quantization (75% of 'standard' 8bits quantization)
resource_utilization_data = mct.core.pytorch_resource_utilization_data(in_model=unified_model,
                                                                       representative_data_gen=representative_dataset_gen,
                                                                       core_config=config,
                                                                       target_platform_capabilities=tpc)
resource_utilization = mct.core.ResourceUtilization(weights_memory=resource_utilization_data.weights_memory * 0.75)

# Perform post training quantization
quant_model, _ = mct.ptq.pytorch_post_training_quantization(in_module=unified_model,
                                                            representative_data_gen=representative_dataset_gen,
                                                            target_resource_utilization=resource_utilization,
                                                            core_config=config,
                                                            target_platform_capabilities=tpc)

print('Quantized model is ready')




mct.exporter.pytorch_export_model(model=quant_model,
                                  save_model_path='./q_anom.onnx',
                                  repr_dataset=representative_dataset_gen)


