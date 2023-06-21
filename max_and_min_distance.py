import argparse

import torch

from constants.dataset_vo import DatasetConstants


def preprocess_distance_matrix(distance_matrix):
    distance_matrix = distance_matrix.T + distance_matrix
    for idx in range(0, distance_matrix.shape[0]):
        distance_matrix[idx, idx] = float('inf')

    return distance_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True, default='Cifar-10',
                        help='Cifar-10, Imagenet-100, Cifar-100, Imagenet-1K')
    parser.add_argument('--data_path', type=str, required=False, help='path for distance matrix')
    parser.add_argument('--split', type=str, default='train', required=True, help='train or test')
    parser.add_argument('--operator', type=str, required=True, default='min',
                        help='min or max')
    args = parser.parse_args()

    distance_matrix = torch.load(args.data_path)
    selected_operator = torch.max
    if args.operator == 'min':
        distance_matrix = preprocess_distance_matrix(distance_matrix)
        selected_operator = torch.min

    dataset_constants = DatasetConstants(args.dataset)
    compact_distance_matrix = torch.zeros((dataset_constants.CLASS_NUMBER, dataset_constants.CLASS_NUMBER))
    row_idx, col_idx = 0, 0

    for mat_row_idx, row_class_sample_size in dataset_constants.get_per_class_size(args.split).items():
        for mat_col_idx, col_class_sample_size in dataset_constants.get_per_class_size(args.split).items():
            compact_distance_matrix[mat_row_idx, mat_col_idx] = selected_operator(
                distance_matrix[row_idx:row_idx + row_class_sample_size,
                col_idx:col_idx + col_class_sample_size]).item()
            col_idx += col_class_sample_size
        row_idx += row_class_sample_size
        col_idx = 0

    print(compact_distance_matrix)
