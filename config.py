import torch
import torchvision.transforms as tt


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 1
IMAGE_SIZE = 128
NUM_RESIDUALS = 6
EPOCHS = 200
LEARNING_RATE = 0.0002
CYCLE_LOSS_LAMBDA = 10
IDENTITY_LAMBDA = 5
SCHEDULER_STEPS = 100
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
PUMA_ROOT = 'datasets/puma'
LION_ROOT = 'datasets/lion'
ORIGINAL_PUMA_ROOT = 'original/puma_original'
ORIGINAL_LION_ROOT = 'original/lion_original'
TEST_LEN = 16


image_transform = tt.Compose(
    [
        tt.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        tt.ToTensor(),
        tt.Normalize(*STATS)
    ]
)
