import os
from torchvision.utils import save_image
from torchvision import datasets
try:
    import output as out
    import inverted_obj as io
except:
    raise ValueError('Run as a module to trigger __init__.py, ie '
                     'run as python -m human_nn_alignment.reg_free_loss')


def get_classes_names(dataset, data_path):
    if dataset == 'imagenet':
        dset = datasets.ImageNet(data_path, split='val')
        ordered_class_labels = []
        for idx, x in enumerate(dset.classes):
            all_names = []
            for i in range(len(x)):
                all_names.append('_'.join(x[i].split()))
            if idx == 134:
                all_names.append('bird')
            elif idx == 517:
                all_names.append('machine')
            ordered_class_labels.append('-'.join(all_names))
        return ordered_class_labels
    elif dataset == 'cifar10':
        dset = datasets.CIFAR10(data_path, train=False)
        return dset.classes
    elif dataset == 'cifar100':
        dset = datasets.CIFAR100(data_path, train=False)
        return dset.classes
    else:
        return ['none']


def save_tensor_images(path, img_indices, seed_name, results, seed, targets, labels, classes_name):
    path = os.path.abspath(path)
    for _d in out.recursive_create_dir(path):
        out.create_dir(_d)
    path_target = os.path.join(path, 'target')
    out.create_dir(path_target)
    path_result = os.path.join(path, 'result')
    out.create_dir(path_result)

    save_image(seed, os.path.join(path, f'{seed_name}.png'))
    # io.save_object(0, seed, 0, os.path.join(path, f'{seed_name}.pkl'))

    for idx, result, target, label in zip(img_indices, results, targets, labels):
        img_name = f'{int(idx)}_{classes_name[int(label)]}_seed_{seed_name}'
        img_target = f'{path_target}/{img_name}.png'
        img_result = f'{path_result}/{img_name}.png'

        save_image(target, img_target)
        save_image(result, img_result)

        # io.save_object(int(idx), target, label.item(), f'{path_target}/{img_name}.pkl')
        # io.save_object(int(idx), result, label.item(), f'{path_result}/{img_name}.pkl')

    print(f'=> Saved images in {path}')

