""" SharedAdam optimizer for a3c """
import torch


class SharedAdam(torch.optim.Adam):
    """ Original Adam optimizer in shared memory """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0, amsgrad=True):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas,
                                         eps=eps, weight_decay=weight_decay,
                                         amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                amsgrad = group['amsgrad']
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        state['max_exp_avg_sq'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()