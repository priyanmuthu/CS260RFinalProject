from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# dist
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import sys

bucket_dict = {}


def setup(taskinfo):
    os.environ['MASTER_PORT'] = '12355'

    taskinfo['gpu_num'] = taskinfo['rank'] % torch.cuda.device_count()

    print(f"rank {taskinfo['rank']}, gpu {taskinfo['gpu_num']}/{torch.cuda.device_count()}")

    dist.init_process_group("nccl", rank=taskinfo['rank'], world_size=taskinfo['world_size'])



def cleanup():
    dist.destroy_process_group()

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def check_val_acc(model, dataset_name, taskinfo, dataloader, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    phase = 'val'
    best_acc = 0.0

    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    num_processed_items = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(taskinfo['gpu_num'])
        labels = labels.to(taskinfo['gpu_num'])

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
         
            _, preds = torch.max(outputs, 1)

        # statistics
        running_loss += loss.item() * inputs.size(0) # inputs.size(0) is batch size
        running_corrects += torch.sum(preds == labels.data)
        num_processed_items += inputs.size(0)

    epoch_loss = running_loss / num_processed_items 
    epoch_loss_tensor = torch.Tensor([epoch_loss]).to(taskinfo['gpu_num'])
    sync_epoch_loss = reduce_tensor(epoch_loss_tensor, taskinfo['world_size'])
    sync_epoch_loss_item = to_python_float(sync_epoch_loss)

    epoch_acc = running_corrects.double() / num_processed_items 
    epoch_acc_tensor = torch.Tensor([epoch_acc]).to(taskinfo['gpu_num'])
    sync_epoch_acc = reduce_tensor(epoch_acc_tensor, taskinfo['world_size'])
    sync_epoch_acc_item = to_python_float(sync_epoch_acc)

    if taskinfo['rank'] == 0:
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, sync_epoch_loss_item, sync_epoch_acc_item))
        print()

    # Cleanup distributed execution
    cleanup()
 
    time_elapsed = time.time() - since
    if taskinfo['rank'] == 0:
        print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, val_acc_history


def train_model(model, dataset_name, taskinfo, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, start_epoch=0):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    if start_epoch >= num_epochs and taskinfo['rank'] == 0:
        print("starting epoch is less than or equal to number of epochs. exiting...")
        return model, val_acc_history

    for epoch in range(start_epoch, num_epochs):
        
        if taskinfo['rank'] == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # Set epoch for distributed sampler
            try:
                dataloaders[phase].sampler.set_epoch(epoch)
            except AttributeError:
                pass

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            num_processed_items = 0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad(set_to_none=True)

                t = time.time()

                # Do profiling only after some time into training.
                if i == 10 and epoch == 2:

                    p = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,])

                    p.__enter__()

                # Todo: don't do a allocation every iteration
                inputs = inputs.to(taskinfo['gpu_num'])
                labels = labels.to(taskinfo['gpu_num'])

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) # inputs.size(0) is batch size
                running_corrects += torch.sum(preds == labels.data)
                num_processed_items += inputs.size(0)

                if i == 10 and epoch == 2:
                    p.__exit__(None, None, None)

                    #if taskinfo['rank'] == 0:
                    p.export_chrome_trace(f"profile_{taskinfo['rank']}.trace")

                    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

                    exit()

                print(f"Time taken: {time.time() - t}")

            epoch_loss = running_loss / num_processed_items 
            epoch_loss_tensor = torch.Tensor([epoch_loss]).to(taskinfo['gpu_num'])
            sync_epoch_loss = reduce_tensor(epoch_loss_tensor, taskinfo['world_size'])
            sync_epoch_loss_item = to_python_float(sync_epoch_loss)

            epoch_acc = running_corrects.double() / num_processed_items 
            epoch_acc_tensor = torch.Tensor([epoch_acc]).to(taskinfo['gpu_num'])
            sync_epoch_acc = reduce_tensor(epoch_acc_tensor, taskinfo['world_size'])
            sync_epoch_acc_item = to_python_float(sync_epoch_acc)

            if taskinfo['rank'] == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, sync_epoch_loss_item, sync_epoch_acc_item))

            # deep copy the model
            if phase == 'val' and sync_epoch_acc_item > best_acc:
                best_acc = sync_epoch_acc_item
                best_model_wts = copy.deepcopy(model.state_dict())
                # save and checkpoint model
                if taskinfo['rank'] == 0:
                    # model
                    path = "./saved_models_and_checkpoints/" + dataset_name + "_state_dict_model.pt"
                    torch.save(model.state_dict(), path)
                    # checkpoint
                    path = "./saved_models_and_checkpoints/" + dataset_name + "_checkpoint.pt"
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            }, path)
            if phase == 'val':
                val_acc_history.append(sync_epoch_acc_item)

        if taskinfo['rank'] == 0:
            print()


    # Cleanup distributed execution
    cleanup()
 
    time_elapsed = time.time() - since
    if taskinfo['rank'] == 0:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save and checkpoint model
    if taskinfo['rank'] == 0:
        # model
        path = "./saved_models_and_checkpoints/" + dataset_name + "_state_dict_model.pt"
        torch.save(model.state_dict(), path)
        # checkpoint
        path = "./saved_models_and_checkpoints/" + dataset_name + "_checkpoint.pt"
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, path)

    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def main(taskinfo, data_dir, dataset_name, num_classes):

    if taskinfo['rank'] == 0:
        print("PyTorch Version: ",torch.__version__)
        print("Torchvision Version: ",torchvision.__version__)
        print("data_dir:",data_dir)
        print("dataset_name:",dataset_name)
        print("num_classes:",num_classes)

    setup(taskinfo)

    # Top level data directory. Here we assume the format of the directory conforms
    #   to the ImageFolder structure
    #data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/utku1/code/Users/utku/datasets/hymenoptera_data"
    #data_dir = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/utku1/code/Users/utku/datasets/imagenet/imagenet_training_8class_subset"
    #data_dir = "/n/home04/usirin/datasets/imagenet_training_8class_subset"
    #data_dir = "/n/home04/usirin/datasets/imagenet_training_20class_subset"
    #data_dir= "/n/home04/usirin/datasets/blood-cell-images/dataset2-master/dataset2-master/images/TRAIN"
    #data_dir = "/n/home04/usirin/datasets/imagenet_subsets/imagenet_training_10class_subset0"   
    #dataset_name = "imagenet_10class_subset0"

    # Models to choose from [resnet, resnet50, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet50"

    # Number of classes in the dataset
    #num_classes = 20

    # Batch size for training (change depending on how much memory you have)
    batch_size = 16

    # Number of epochs to train for
    num_epochs = 200

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False

    # Single-GPU execution
    # Initialize the model for this run
    #model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    ## Detect if we have a GPU available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ## Send the model to GPU
    #model_ft = model_ft.to(device)

    # Multi-GPU execution using DataParallel -- slower than single-GPU
    ## Initialize the model for this run
    #model_ft_unpar, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    #
    ## parallelize
    #model_ft = nn.DataParallel(model_ft_unpar, device_ids = [0,1,2,3,4,5,6])

    # Multi-GPU execution using DistributedDataParallel
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)
    model_ft = model_ft.to(taskinfo['gpu_num'])
    

    

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model_ft = DDP(model_ft, device_ids=[taskinfo['gpu_num']], output_device=taskinfo['gpu_num'], find_unused_parameters=True, bucket_cap_mb=1)

    # Print the model we just instantiated
    if taskinfo['rank'] == 0:
        print(model_ft)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if taskinfo['rank'] == 0:
        print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    # Create distributed samplers
    image_samplers = {x: DistributedSampler(image_datasets[x], num_replicas=taskinfo['world_size'], rank=taskinfo['rank'], shuffle=True, drop_last=False) for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=4, drop_last=False, persistent_workers=True) for x in ['train', 'val']}

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    if taskinfo['rank'] == 0:
        print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                if taskinfo['rank'] == 0:
                    print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                if taskinfo['rank'] == 0:
                    print("\t",name)

    def thenf(fut, idx):
        print(fut.value()[0][0]);
        bucket_dict[idx]["end_times"].append(time.perf_counter())
        bucket_dict[idx]["dt"].append(bucket_dict[idx]["end_times"][-1] - bucket_dict[idx]["start_times"][-1])
        return fut.value()[0]

    def _allreduce_fut( process_group, tensor, bucket):
        "Averages the input gradient tensor by allreduce and returns a future."
        group_to_use = process_group if process_group is not None else dist.group.WORLD

        # Apply the division first to avoid overflow, especially for FP16.
        tensor.div_(group_to_use.size())

        return (
            dist.all_reduce(tensor, group=group_to_use, async_op=True)
            .get_future()
            .then(lambda x: thenf(x, bucket.index()))
        )


    def allreduce_hook(process_group, bucket):
        """
        This DDP communication hook just calls ``allreduce`` using ``GradBucket``
        tensors. Once gradient tensors are aggregated across all workers, its ``then``
        callback takes the mean and returns the result. If user registers this hook,
        DDP results is expected to be same as the case where no hook was registered.
        Hence, this won't change behavior of DDP and user can use this as a reference
        or modify this hook to log useful information or any other purposes while
        unaffecting DDP behavior.

        Example::
            >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
        """
        if bucket.index() not in bucket_dict:
            bucket_dict[bucket.index()] = {
                "start_times": [],
                "end_times": [],
                "dt": [],
                "size": []
            }

        bucket_dict[bucket.index()]["start_times"].append(time.perf_counter())


        return _allreduce_fut(process_group, bucket.buffer(), bucket)

    model_ft.register_comm_hook(None, allreduce_hook)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataset_name, taskinfo, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), start_epoch=0)

    # load existing model and check val acuracy
    #path = "./saved_models_and_checkpoints/" + dataset_name + "_state_dict_model.pt"
    #model_ft.load_state_dict(torch.load(path))
    #check_val_acc(model_ft, world_size, dataset_name, rank, dataloaders_dict['val'], criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

    # load an existing checkpoint and continue training from epoch
    #path = "./saved_models_and_checkpoints/" + dataset_name + "_checkpoint.pt"
    #checkpoint = torch.load(path)
    #model_ft.load_state_dict(checkpoint['model_state_dict'])
    #optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    #last_epoch = checkpoint['epoch']
    #train_model(model_ft, world_size, dataset_name, rank, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"), start_epoch=(last_epoch+1))

if __name__ == '__main__':
    # suppose we have world_size-many gpus
    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    num_classes = int(sys.argv[3])

    ngpus_per_node = torch.cuda.device_count()

    print(int(os.environ['SLURM_PROCID']), int(os.environ["WORLD_SIZE"]),data_dir, dataset_name, num_classes)

    taskinfo = {}
    taskinfo['world_size'] = int(os.environ["WORLD_SIZE"])
    taskinfo['rank'] = int(os.environ['SLURM_PROCID'])

    main(taskinfo,data_dir, dataset_name, num_classes)
