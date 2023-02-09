from human_nn_alignment.reg_free_loss import main
import argparse, glob
import copy

parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--source_dataset', type=str, default=None)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument('--append_path', type=str, default='')
parser.add_argument('--inversion_loss', type=str, default='reg_free')
parser.add_argument('--trans_robust', type=bool, default=False)
parser.add_argument('--fft', type=bool, default=False)
parser.add_argument('--step_size', type=float, default=1.)
parser.add_argument('--seed_type', type=str, default='super-noise')
parser.add_argument('--iters', type=int, default=None)
args = parser.parse_args()

print (glob.glob(f'{args.checkpoint_dir}/*_rand_seed_420.ckpt'))
original_path = copy.copy(args.append_path)
for checkpoint_path in glob.glob(f'{args.checkpoint_dir}/*_rand_seed_420.ckpt'):
    checkpoint_name = checkpoint_path.split('/')[-1].split('_')[0]
    epoch_num = checkpoint_name.split('epoch=')[1]
    args.__setattr__('append_path', f'{original_path}_{epoch_num}')
    args.__setattr__('checkpoint_path', checkpoint_path)
    print (args)
    main(args)
