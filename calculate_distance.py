import argparse
import time
import torch
import sys

from constants.metric_vo import PerceptualMetric, LpMetric
from dataset.dataset_factory import DatasetFactory
from metrics.metric_factory import get_metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--metric_name', type=str, required=True, default='l2',
                        help='l2, linf, lpips, r_lpips, SimCLR')
    parser.add_argument('--dataset', type=str, required=True, default='Cifar-10',
                        help='Cifar-10, Imagenet-100, Cifar-100, Imagenet-1K')
    parser.add_argument('--split', type=str, required=True, default='val', help='train, val')
    parser.add_argument('--data_path', type=str, required=False, help='path for data')
    parser.add_argument('--model_path', type=str, required=True, help='pretrained model for perceptual metrics')

    args = parser.parse_args()
    embedding, similarity_metric = get_metric(args.metric_name, args.model_path)
    selected_dataset = DatasetFactory(args.dataset, args.split, args.data_path).get_dataset()

    start_time = time.time()
    distance_matrix = torch.zeros((selected_dataset.shape[0], selected_dataset.shape[0]))

    for idx, (data, _) in enumerate(selected_dataset):
        if PerceptualMetric.is_member(metric=args.metric_name):
            distance_matrix[idx, idx:] = similarity_metric(embedding(data.cuda()),
                                                           embedding(selected_dataset[idx:].cuda())).cpu()
        else:
            p = LpMetric.get_p(metric=args.metric_name)
            distance_matrix[idx, idx:] = similarity_metric(data, selected_dataset[idx:], p=p,
                                                           dim=tuple(range(1, len(data.shape))))

        if idx % 1000 == 999:
            print('Data index: {idx}, time: {time}'.format(idx=idx + 1, time=(time.time() - start_time) / 60))
            sys.stdout.flush()

    torch.save(distance_matrix, '{dataset_name}-{split}-{metric_name}-matrix.pt'.format(
        dataset_name=args.dataset, split=args.split, metric_name=args.metric_name))
