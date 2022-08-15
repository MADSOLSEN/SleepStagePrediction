import math
from matplotlib import pyplot as plt

def OSCILLATOR_EXP_DECAY(lr_min=1e-5, lr_max=4e-3, lr_phase=10, warmup=15):
    def oscillator_exp_decay(epoch, lr):

        cos_exp_dec = lr_max
        if epoch > warmup:
            epoch -= warmup
            cos_pos = (math.cos(2 * math.pi * 1/lr_phase * (epoch)) + 1) / 2
            cos_sca_pos = (lr_max - lr_min) * cos_pos + lr_min
            cos_exp_dec = cos_sca_pos * math.exp(-(epoch) / (3 * lr_phase))

        return cos_exp_dec

    return oscillator_exp_decay