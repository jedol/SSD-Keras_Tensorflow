import numpy as np


def multi_step_learning_rate_decay(epoch, base_lr, gamma=0.1, steps=[]):
    return base_lr*gamma**np.sum(epoch >= np.array(steps))


def step_learning_rate_decay(epoch, base_lr, gamma=0.1, step=1):
    return base_lr*gamma**(int(epoch)/int(step))