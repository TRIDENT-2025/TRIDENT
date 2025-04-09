import os
import argparse
import torch
import random
import csv
import numpy as np

from dataset.get_transform import get_img_transform, get_rf_transform, get_audio_transform
from dataset.get_dataset import DroneFusionDataset, get_loader
from training.unimodal_train import test_uni_track_acc
from models.get_model import get_model

def append_to_csv(file_path, data):
    is_empty = not os.path.exists(file_path) or os.stat(file_path).st_size == 0
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if is_empty:
            writer.writerow(["Modality", "Model", "Top-1 Acc", "Top-5 Acc", "Recall", "Precision", "Average Precision", "F1-score", "F1-macro", "Avg Prediction Time"])
        writer.writerow(data)
    return file_path

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--modal_num', type=int, default=0, help='The choice of the modality | 0: audio, 1: video, 2: RF', choices=[0, 1, 2])
    parser.add_argument('--model_name_1', type=str, help='Audio model choice', choices=['lenet', 'vgg'])
    parser.add_argument('--model_name_2', type=str, help='Visual model choice', choices=['resnet10', 'resnet18', 'resnet34', 'mobilenet'])
    parser.add_argument('--model_name_3', type=str, help='RF model choice', choices=['resnet10', 'resnet18', 'resnet34', 'mobilenet'])
    parser.add_argument("--dataset-dir", type=str, default=r"TRIDENT/Dataset")
    parser.add_argument('--parallel', help='use several GPUs', action='store_true', default=False)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size of train/test dataset")
    parser.add_argument("--crop-size", type=int, default=112, help="Crop size of video/RF images")
    parser.add_argument("--scale-size", type=int, default=640, help="Scale size of video/RF images")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu_id')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.modal_num == 0: # Audio
        args.batch_size = 112
        model = get_model(name=args.model_name_1, n_classes=2)
        args.name = args.model_name_1+'_audio'
        target_model = args.model_name_1
        modality = 'audio'
    elif args.modal_num == 1: # Video
        args.batch_size = 112
        model = get_model(name=args.model_name_2, n_classes=2, m=7)
        args.name = args.model_name_2+'_video'
        target_model = args.model_name_2
        modality = 'video'
    else: # RF
        args.batch_size = 112
        model = get_model(name=args.model_name_3, n_classes=2, m=1)
        args.name = args.model_name_3+'_rf'
        target_model = args.model_name_3
        modality = 'rf'

    audio_root = os.path.join(args.dataset_dir, 'Audio')
    video_root = os.path.join(args.dataset_dir, 'Video')
    rf_root = os.path.join(args.dataset_dir, 'RF_Spectrograms')
    
    transform_train = {'audio': get_audio_transform(args, is_training=True),
                       'video': get_img_transform(args, is_training=True),
                       'rf': get_rf_transform(args, is_training=True)}
    
    transform_test = {'audio': get_audio_transform(args, is_training=False),
                      'video': get_img_transform(args, is_training=False),
                      'rf': get_rf_transform(args, is_training=False)}
    
    test_dataset = DroneFusionDataset(audio_root+'/Test', video_root+'/Test', rf_root+'/Test', dataset_type='Test', transform=transform_test)
    test_loader, test_sampler = get_loader(args, test_dataset, distributed=False, is_train=False)
        
    datasets = {'test': test_dataset}
    dataloaders = {'test': test_loader}

    # Check if CUDA is available
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.path = 'TRIDENT/save_unimodals'
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print(f"Number of parameters in the model: {num_params/1e6}")

    criterion = torch.nn.BCELoss()
    
    checkpoint = torch.load(os.path.join(args.path, f"{args.name}.pt"), map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    del checkpoint
    
    results = test_uni_track_acc(model, criterion, dataloaders, device=device, phase='test', status='eval', modal_num=args.modal_num)

    data = [
        modality, target_model, results['top-1_acc'], results['top-5_acc'],
        results['recall'], results['precision'], results ['average_precision'], results['f1_score'],
        results['f1_macro'], results['avg_prediction_time']
    ]

    append_to_csv('./uni_results.csv', data)

    print(results)

