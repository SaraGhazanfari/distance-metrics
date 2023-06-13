from enum import Enum


class PerceptualMetric(Enum):
    LPIPS = 'lpips'
    R_LPIPS = 'r-lpips'
    SimCLR = 'SimCLR'

    @classmethod
    def is_member(cls, metric):
        for _, member in cls.__members__.items():
            if metric == member.value:
                return True
        return False


class LpMetric(Enum):
    Linf = ('linf', float('inf'))
    L2 = ('l2', 2)
    L1 = ('l1', 1)

    def __init__(self, metric_name, p):
        self.metric_name = metric_name
        self.p = p

    @classmethod
    def is_member(cls, metric):
        for _, member in cls.__members__.items():
            if metric == member.metric_name:
                return True
        return False

    @classmethod
    def get_p(cls, metric):
        for _, member in cls.__members__.items():
            if metric == member.metric_name:
                return member.p
