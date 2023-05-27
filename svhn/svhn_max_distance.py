import argparse

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    path = parser.add_argument('--path', type=int, required=True)
    args = parser.parse_args()

    class_sample_size_dict = {0: 4948, 1: 13861, 2: 10585, 3: 8497, 4: 7458, 5: 6882, 6: 5727, 7: 5595, 8: 5045,
                              9: 4659}
    lpips_matrix = torch.load(args['path'])

    max_distance = torch.zeros((10, 10))
    row_idx = 0
    col_idx = 0
    for mat_row_idx, row_class_sample_size in class_sample_size_dict.items():
        for mat_col_idx, col_class_sample_size in class_sample_size_dict.items():
            max_distance[mat_row_idx, mat_col_idx] = torch.max(
                lpips_matrix[row_idx:row_idx + row_class_sample_size, col_idx:col_idx + col_class_sample_size]).item()
            print(row_idx, row_idx + row_class_sample_size, col_idx, col_idx + col_class_sample_size)
            col_idx += col_class_sample_size
        row_idx += row_class_sample_size
        col_idx = 0

    print(max_distance)
