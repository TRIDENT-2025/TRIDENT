import argparse, logging
import torch, sys, os, random
import utils.scheduler as sc
import numpy as np
import torch.optim as op

from dataset.get_transform import get_img_transform, get_rf_transform, get_audio_transform
from dataset.get_dataset import DroneFusionDataset, get_loader

from training.multimodal_train import train_multi_track_acc, Architect
from models.av_fusion import LateFusion, LateFusion3Modal, GMU, GMU3Modal
from models.get_model import get_model

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
    parser.add_argument('--model_2', type=str, help='Visual model choice', choices=['resnet10', 'resnet18', 'mobilenet'])
    parser.add_argument('--model_3', type=str, help='RF model choice', choices=['resnet10', 'resnet18', 'resnet34', 'mobilenet'])
    parser.add_argument('--name', type=str, help='Multimodal model name', default='')
    parser.add_argument("--dataset-dir", type=str, default=r"TRIDENT/Dataset")
    parser.add_argument("--distributed-train", action='store_true', default=False, help="Enable distributed training")
    parser.add_argument("--distributed-test", action='store_true', default=False, help="Enable distributed test")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for arch encoding')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
    parser.add_argument('--parallel', help='use several GPUs', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, help='training epochs', default=20)
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
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model_3 = get_model(name=model_3_name, n_classes=2, m=1)
        model = LateFusion(model_2, model_3, num_classes=2)
    elif args.type == 'gmu':
        model_2 = get_model(name=model_2_name, n_classes=2, m=7)
        model_3 = get_model(name=model_3_name, n_classes=2, m=1)
        model = GMU(model_2, model_3, args, num_classes=2)
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

    model.cuda()
    model.eval()

    # 2 Modalities
    # args.name = args.model_2+'_'+'visual_'+args.model_3+'_'+'rf_'+args.type

    # 3 Modalities
    args.name = args.model_1+'_'+'audio_'+args.model_2+'_'+'visual_'+args.model_3+'_'+'rf_'+args.type


    audio_root = os.path.join(args.dataset_dir, 'Audio')
    video_root = os.path.join(args.dataset_dir, 'Video')
    rf_root = os.path.join(args.dataset_dir, 'RF_Spectrograms')

    transform_train = {'audio': get_audio_transform(args, is_training=True),
                       'video': get_img_transform(args, is_training=True),
                       'rf': get_rf_transform(args, is_training=True)}

    transform_test = {'audio': get_audio_transform(args, is_training=True),
                      'video': get_img_transform(args, is_training=True),
                      'rf': get_rf_transform(args, is_training=True)}

    train_dataset = DroneFusionDataset(audio_root+'/Train', video_root+'/Train', rf_root+'/Train', dataset_type='Train', transform=transform_train)
    test_dataset = DroneFusionDataset(audio_root + '/Validation', video_root + '/Validation', rf_root + '/Validation', dataset_type='Validation', transform=transform_test)

    train_loader, train_sampler = get_loader(args, train_dataset, distributed=args.distributed_train, is_train=True)
    test_loader, test_sampler = get_loader(args, test_dataset, distributed=args.distributed_test, is_train=False)

    datasets = {'train': train_dataset, 'test': test_dataset}
    dataloaders = {'train': train_loader, 'test': test_loader}

    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

    path = 'save_tri-modals' ## In case of dual-modals is path = 'save_dual-modals' 
    if not os.path.exists(path):
        os.makedirs(path)
    args.save = path

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params/1e6}")

    checkpoint_path = args.save+'/'+args.name+'.pt'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint)
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")

    model.to(device)

    params = model.parameters()
    criterion = torch.nn.BCELoss()
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'test']}
    num_batches_per_epoch = dataset_sizes['train'] / args.batch_size

    eta_min, eta_max = 0.001, 0.01

    optimizer = op.Adam(params, lr=eta_max, weight_decay=args.weight_decay)
    scheduler = sc.LRCosineAnnealingScheduler(eta_max, eta_min, 1, 2, num_batches_per_epoch)

    arch_optimizer = op.Adam(model.parameters(), lr=eta_max, betas=(0.5, 0.999), weight_decay=args.weight_decay)
    architect = Architect(model, args, criterion, arch_optimizer)

    model.to(device)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, args.name+'.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    best_test, top1_train, top5_train, top1_test, top5_test = train_multi_track_acc(
        model, architect, criterion, optimizer, scheduler,
        dataloaders, dataset_sizes, device=device, num_epochs=args.epochs, parallel=args.parallel,
        logger=logger, args=args, init_acc=0.0)

    print("Training is done")


