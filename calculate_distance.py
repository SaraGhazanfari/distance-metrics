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
    parser.add_argument('--dataset', type=str, required=True, default='cifar-10',
                        help='cifar-10, imagenet-100, cifar-100, imagenet-1K')
    parser.add_argument('--split', type=str, required=True, default='val', help='train, val')
    parser.add_argument('--data_path', type=str, required=False, help='path for data')
    parser.add_argument('--model_path', type=str, required=True, help='pretrained model for perceptual metrics')
    parser.add_argument('--batch_size', type=int, required=False, help='batch size', default=1000)

    args = parser.parse_args()
    embedding, similarity_metric = get_metric(args.metric_name, args.model_path)
    selected_dataset, data_loader = DatasetFactory(dataset_name=args.dataset.lower(), split=args.split,
                                                   data_path=args.data_path,
                                                   batch_size=args.batch_size).get_dataset()

    start_time = time.time()
    distance_matrix = torch.zeros((selected_dataset.shape[0], selected_dataset.shape[0]))
    first_idx = 0
    second_idx = 0
    for data_batch in data_loader:
        for data in data_batch:
            if PerceptualMetric.is_member(metric=args.metric_name):
                distance_matrix[first_idx, second_idx:second_idx + len(data_batch)] = similarity_metric(
                    embedding(data.cuda()),
                    embedding(
                        data_batch.cuda())).cpu()
            else:
                p = LpMetric.get_p(metric=args.metric_name)
                distance_matrix[first_idx, second_idx:second_idx + len(data_batch)] = similarity_metric(data,
                                                                                                        data_batch, p=p,
                                                                                                        dim=tuple(
                                                                                                            range(1,
                                                                                                                  len(data.shape))))
            first_idx += 1
        second_idx += len(data_batch)
        print('Data index: {idx}, time: {time}'.format(idx=second_idx + 1, time=(time.time() - start_time) / 60))
        sys.stdout.flush()

    torch.save(distance_matrix, '{dataset_name}-{split}-{metric_name}-matrix.pt'.format(
        dataset_name=args.dataset, split=args.split, metric_name=args.metric_name))
