import math

def OSCILLATOR_EXP_DECAY(lr_min=1e-6, lr_max=1e-4, lr_phase=10):
    def oscillator_exp_decay(epoch, lr):

        cos_pos = (math.cos(2 * math.pi * 1/lr_phase * (epoch)) + 1) / 2
        cos_sca_pos = (lr_max - lr_min) * cos_pos + lr_min
        cos_exp_dec = cos_sca_pos * math.exp(-(epoch) / (3 * lr_phase))

        return cos_exp_dec

    return oscillator_exp_decay


