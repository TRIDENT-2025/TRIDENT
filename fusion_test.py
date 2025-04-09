import os
import argparse
import torch, os, random, csv
import numpy as np

from dataset.get_transform import get_img_transform, get_rf_transform, get_audio_transform
from dataset.get_dataset import DroneFusionDataset, get_loader

from training.multimodal_train import test_multi_track_acc
from models.av_fusion import LateFusion, GMU, LateFusion3Modal, GMU3Modal
from models.get_model import get_model

def append_to_csv(file_path, data):
    is_empty = os.path.exists(file_path) and os.stat(file_path).st_size == 0
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if is_empty:  # Write header only if the file is empty
            writer.writerow(["Fusion", "Model_1", "Model_2", "Model_3", "Top-1 Acc", "Top-5 Acc", "Recall", "Precision", "Average Precision", "F1-score", "F1-macro", "Avg Prediction Time"])  # Added Avg Prediction Time
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
    parser.add_argument('--type', type=str, default='late', help='Fusion type', choices=['late', 'gmu', 'late_3_modal', 'gmu_3_modal'])
    parser.add_argument('--model_1', type=str, help='Audio model choice', choices=['lenet', 'vgg'])
    parser.add_argument('--model_2', type=str, help='Visual model choice', choices=['resnet10', 'resnet18', 'resnet34','mobilenet'])
    parser.add_argument('--model_3', type=str, help='RF model choice', choices=['resnet10', 'resnet18', 'resnet34', 'mobilenet'])
    parser.add_argument("--dataset-dir", type=str, default=r"TRIDENT/Dataset")
    parser.add_argument("--distributed-train", action='store_true', default=False, help="Enable distributed training")
    parser.add_argument("--distributed-test", action='store_true', default=False, help="Enable distributed test")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
    parser.add_argument('--parallel', help='use several GPUs', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, help='training epochs', default=15)
    parser.add_argument("--batch-size", type=int, default=112, help="Batch size of train/test dataset")
    parser.add_argument("--crop-size", type=int, default=112, help="Crop size of video/RF images")
    parser.add_argument("--scale-size", type=int, default=640, help="Scale size of video/RF images")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, help='gpu_id')

    args = parser.parse_args()

    set_seed(args.seed)

    model_1_name = args.model_1
    model_2_name = args.model_2
    model_3_name = args.model_3

    if args.type == 'late':
        model_1 = get_model(name=model_1_name, n_classes=2)
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model = LateFusion(model_1, model_2, num_classes=2)
    elif args.type == 'gmu':
        model_1 = get_model(name=model_1_name, n_classes=2)
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model = GMU(model_1, model_2, args, num_classes=2)
    elif args.type == 'late_3_modal':
        model_1 = get_model(name=model_1_name, n_classes=2)
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model_3 = get_model(name=model_3_name, n_classes=2, m=1)
        model = LateFusion3Modal(model_1, model_2, model_3, num_classes=2)
    elif args.type == 'gmu_3_modal':
        model_1 = get_model(name=model_1_name, n_classes=2)
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model_3 = get_model(name=model_3_name, n_classes=2, m=1)
        model = GMU3Modal(model_1, model_2, model_3, num_classes=2)
    else:
        raise NotImplementedError
    
    # args.name = args.model_1+'_'+'audio_'+args.model_2+'_'+'visual_'+args.type ## Use it for two modalities combinations

    args.name = args.model_1+'_'+'audio_'+args.model_2+'_'+'visual_'+args.model_3+'_'+'rf_'+args.type ## Use it for three modalities combinations

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
    test_loader, test_sampler = get_loader(args, test_dataset, distributed=args.distributed_test, is_train=False)

    datasets = {'test': test_dataset}
    dataloaders = {'test': test_loader}

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    args.path = 'save_tri-modals' ## In case of dual-modals is path = 'save_dual-modals' 
        
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    print(f"Number of parameters in the model: {num_params/1e6}")

    criterion = torch.nn.BCELoss()
    
    checkpoint = torch.load(args.path+'/'+args.name+'.pt')

    model.load_state_dict(checkpoint, strict=False)

    model.to(device)

    del checkpoint

    # results = test_multi_track_acc(model, criterion, dataloaders, device=device, args=args, phase='test') # It is used in two modalities
    results = test_multi_track_acc(model, criterion, dataloaders, device=device, phase='test') ## It is uses in three modalities


    data = [args.type, args.model_1, args.model_2, args.model_3, results['top-1_acc'], results['top-5_acc'], results['recall'],  results['precision'], results ['average_precision'], results['f1_score'], results['f1_macro'], results['avg_prediction_time']] 

    append_to_csv('./multi_results.csv', data)
    print(results)
