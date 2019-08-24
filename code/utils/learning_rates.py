import math
# from proto import learning_rate_pb2


class LRScheduler():
    def __init__(self, opt):
        self.opt = opt
        self.lr = 0

    def step(self):
        lr = self.get()
        if self.lr == lr:
            return
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr
        self.lr = lr

    def learningrate(self):
        return self.lr

    def get(self):
        raise NotImplementedError

class exponential_decay(LRScheduler):

    def __init__(self, opt, learning_rate, decay_steps, decay_rate, global_step=1, staircase=False, name=None):
        super(exponential_decay, self).__init__(opt)
        self.learning_rate_ = learning_rate
        self.global_step = global_step

        if staircase:
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate
        else:
            self.decay_steps = 1
            self.decay_rate = decay_rate ** (1 / decay_steps)

        self.learning_rate = self.learning_rate_ * self.decay_rate** (self.global_step // self.decay_steps)

    def get(self):
        if self.global_step % self.decay_steps == 0:
            self.learning_rate *= self.decay_rate
        self.global_step += 1
        return self.learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.learning_rate = self.learning_rate_ * self.decay_rate ** (self.global_step // self.decay_steps)

class piecewise_constant(LRScheduler):

    def __init__(self, opt, boundaries, values, global_step=1):
        super(piecewise_constant, self).__init__(opt)
        assert len(values) == len(boundaries) + 1
        self.global_step = global_step
        self.values = [v for v in values]
        self.boundaries = [b for b in boundaries]
        self.learning_rate = values[0]
        for i, b in enumerate(boundaries):
            if self.global_step >= b:
                self.learning_rate = values[i + 1]

    def get(self):
        if self.global_step in self.boundaries:
            self.learning_rate = self.values[self.boundaries.index(self.global_step) + 1]
        self.global_step += 1
        return self.learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.learning_rate = self.values[0]
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                self.learning_rate = self.values[i + 1]

class polynomial_decay(LRScheduler):
    def __init__(self, opt, learning_rate, decay_steps,
                 end_learning_rate=0.0001, power=1.0,
                 global_step=1, cycle=False):
        super(polynomial_decay, self).__init__(opt)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate=end_learning_rate
        self.power = power
        self.global_step = global_step
        self.cycle = cycle

    def get(self):
        if self.cycle:
            decay_steps = self.decay_steps * math.ceil(self.global_step / self.decay_steps)
            global_step = self.global_step
        else:
            decay_steps = self.decay_steps
            global_step = min(self.global_step, self.decay_steps)
        decayed_learning_rate = (self.learning_rate - self.end_learning_rate) * \
            (1 - global_step / decay_steps) ** (self.power) + self.end_learning_rate
        self.global_step += 1
        return decayed_learning_rate

    def set_global_step(self, global_step):
        self.global_step = global_step

class mix(LRScheduler):

    def __init__(self, opt, config, global_step=1):
        """
        :param config: A dictionary.
        for example:
        config = ={0: polynomial_decay(1e-8, 5000, 0.01),
                   5001: exponential_decay(0.01, 300, 0.9)}
        means learning rate increase linearly in first 5000 steps and decay exponentially later.
        """
        super(mix, self).__init__(opt)
        self.boundaries, self.values = zip(*config.items())
        self.global_step = global_step

        self.curr = self.values[0]
        self.curr.set_global_step(global_step)
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                global_step -= b
                self.curr = self.values[i]
                self.curr.set_global_step(global_step)

    def get(self):
        self.global_step += 1
        if self.global_step in self.boundaries:
            self.curr = self.values[self.boundaries.index(self.global_step)]
        return self.curr.get()

    def set_global_step(self, global_step):
        self.global_step = global_step
        self.curr = self.values[0]
        self.curr.set_global_step(global_step)
        for i, b in enumerate(self.boundaries):
            if self.global_step >= b:
                global_step -= b
                self.curr = self.values[i]
                self.curr.set_global_step(global_step)


# def get_exp(opt, cfg):
#     return exponential_decay(opt=opt,
#                              learning_rate=cfg.start_lr,
#                              decay_steps=cfg.decay_steps,
#                              decay_rate=cfg.decay_rate,
#                              staircase=cfg.staircase)
#
#
# def get_pie(opt, cfg):
#     return piecewise_constant(opt, cfg.boundary, cfg.value)
#
#
# def get_pol(opt, cfg):
#     return polynomial_decay(opt=opt,
#                             learning_rate=cfg.learning_rate,
#                             decay_steps=cfg.decay_steps,
#                             end_learning_rate=cfg.end_learning_rate,
#                             power=cfg.power,
#                             cycle=cfg.cycle)
#
#
# def get_mix(opt, cfg):
#     mix_cfg = {}
#     for item in cfg.item:
#         mix_cfg[item.boundary] = get_lr(item.lr, opt)
#     return mix(opt, mix_cfg)
#
#
# def get_lr(opt, cfg):
#
#     if cfg.type == learning_rate_pb2.LearningRate.EXPONENTIAL:
#         s = get_exp(opt, cfg.exponential_decay)
#         end = cfg.start_lr
#     elif cfg.type == learning_rate_pb2.LearningRate.PIECEWISE:
#         s = get_pie(opt, cfg.piecewise_constant)
#         end = cfg.piecewise_constant.value[0]
#     elif cfg.type == learning_rate_pb2.LearningRate.POLYNOMIAL:
#         s = get_pol(opt, cfg.polynomial_decay)
#         end = cfg.learning_rate
#     elif cfg.type == learning_rate_pb2.LearningRate.MIX:
#         s = get_mix(opt, cfg.mix)
#         cfg.warmup = 0
#
#     if cfg.warmup:
#         cfg = {0: polynomial_decay(opt, 1e-8, cfg.warmup, end),
#                cfg.warmup: s}
#         return mix(opt, cfg)
#     else:
#         return s
def get_exp(opt, cfg):
    return exponential_decay(opt=opt,
                             learning_rate=cfg['start_lr'],
                             decay_steps=cfg['decay_steps'],
                             decay_rate=cfg['decay_rate'],
                             staircase=cfg['staircase'])


def get_pie(opt, cfg):
    return piecewise_constant(opt, cfg['boundary'], cfg['value'])


def get_pol(opt, cfg):
    return polynomial_decay(opt=opt,
                            learning_rate=cfg['learning_rate'],
                            decay_steps=cfg['decay_steps'],
                            end_learning_rate=cfg['end_learning_rate'],
                            power=cfg['power'],
                            cycle=cfg['cycle'])


# def get_mix(opt, cfg):
#     mix_cfg = {}
#     for item in cfg.item:
#         mix_cfg[item.boundary] = get_lr(item.lr, opt)
#     return mix(opt, mix_cfg)

def get_lr(opt, cfg, warmup=0):

    if cfg['type'] == 'EXPONENTIAL':
        s = get_exp(opt, cfg['exponential_decay'])
        end = cfg['start_lr']
    elif cfg['type'] == 'PIECEWISE':
        s = get_pie(opt, cfg['piecewise_constant'])
        end = cfg['piecewise_constant']['value'][0]
    elif cfg['type'] == 'POLYNOMIAL':
        s = get_pol(opt, cfg['polynomial_decay'])
        end = cfg['learning_rate']
    elif cfg['type'] == 'MIX':
        s = get_mix(opt, cfg['mix'])
        cfg.warmup = 0

    if warmup:
        cfg = {0: polynomial_decay(opt, 1e-8, warmup, end),
               warmup: s}
        return mix(opt, cfg)
    else:
        return s
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from proto import config_pb2
    from google.protobuf import text_format

    cfg = config_pb2.Config()
    with open('../test.cfg', "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, cfg)
    l = get_lr(None, cfg.lr)

    lrs = []
    for i in range(200000):
        lrs.append(l.get())
    x = list(range(200000))
    plt.plot(x, lrs)
    plt.show()


