import torch

from metrics.lpips_metric import AlexNetFeatureModel, LPIPS_Metric


def get_metric(metric_name, model_path):
    if metric_name in ['lpips', 'r-lpips']:
        alex_net = AlexNetFeatureModel(model_path).eval().cuda()
        lpips_metric = LPIPS_Metric().eval().cuda()
        return alex_net, lpips_metric
    elif metric_name in ['linf', 'l2']:
        return None, torch.norm
