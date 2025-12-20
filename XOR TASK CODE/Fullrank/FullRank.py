from low_rank_rnns.helpers import *
import torch.nn as nn
from math import sqrt, floor
import random
import time


def loss_mse(output, target, mask):
    """
    Mean squared error loss
    :param output: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :return: float
    """
    # Compute loss for each (trial, timestep) (average accross output dimensions)
    loss_tensor = (mask * (target - output)).pow(2).mean(dim=-1)
    # Account for different number of masked values per trial
    loss_by_trial = loss_tensor.sum(dim=-1) / mask[:, :, 0].sum(dim=-1)
    return loss_by_trial.mean()


def accuracy_general(output, targets, mask):
    good_trials = (targets != 0).any(dim=1).squeeze()
    target_decisions = torch.sign(
        (targets[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    decisions = torch.sign(
        (output[good_trials, :, :] * mask[good_trials, :, :]).mean(dim=1).squeeze()
    )
    return (target_decisions == decisions).type(torch.float32).mean()


def train(
    net,
    _input,
    _target,
    _mask,
    n_epochs,
    lr=1e-2,
    batch_size=32,
    plot_learning_curve=False,
    plot_gradient=False,
    mask_gradients=False,
    clip_gradient=None,
    early_stop=None,
    keep_best=False,
    cuda=False,
    resample=False,
):
    """
    Train a network
    :param net: nn.Module
    :param _input: torch tensor of shape (num_trials, num_timesteps, input_dim)
    :param _target: torch tensor of shape (num_trials, num_timesteps, output_dim)
    :param _mask: torch tensor of shape (num_trials, num_timesteps, 1)
    :param n_epochs: int
    :param lr: float, learning rate
    :param batch_size: int
    :param plot_learning_curve: bool
    :param plot_gradient: bool
    :param mask_gradients: bool, set to True if training the SupportLowRankRNN_withMask for reduced models
    :param clip_gradient: None or float, if not None the value at which gradient norm is clipped
    :param early_stop: None or float, set to target loss value after which to immediately stop if attained
    :param keep_best: bool, if True, model with lower loss from training process will be kept (for this option, the
        network has to implement a method clone())
    :param resample: for SupportLowRankRNNs, set True
    :return: nothing
    """
    print("Training...")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    num_examples = _input.shape[0]
    all_losses = []
    if plot_gradient:
        gradient_norms = []

    # CUDA management
    if cuda:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    net.to(device=device)
    input = _input.to(device=device)
    target = _target.to(device=device)
    mask = _mask.to(device=device)

    # Initialize setup to keep best network
    with torch.no_grad():
        initial_loss = loss_mse(net(input), target, mask)
        print("initial loss: %.3f" % (initial_loss.item()))
        if keep_best:
            best = net.clone()
            best_loss = initial_loss.item()

    # Training loop
    for epoch in range(n_epochs):
        begin = time.time()
        
        losses = []  # losses over the whole epoch
        for i in range(num_examples // batch_size):
            optimizer.zero_grad()#先将梯度缓存清零
            random_batch_idx = random.sample(range(num_examples), batch_size)#打乱抽取batchsize
            batch = input[random_batch_idx]
            output = net(batch)
            if epoch >0 and i == 0:
                # print(batch.shape,'input.....')
                print(batch[2, 15, :], 'input.....')
                print(target[random_batch_idx][2, 60], 'target.....')
                print(output[2, 65:], 'output.....')
            loss = loss_mse(output, target[random_batch_idx], mask[random_batch_idx])
            losses.append(loss.item())
            all_losses.append(loss.item())
            loss.backward()#backpropogation
            if mask_gradients:
                net.m.grad = net.m.grad * net.m_mask
                net.n.grad = net.n.grad * net.n_mask
                net.wi.grad = net.wi.grad * net.wi_mask
                net.wo.grad = net.wo.grad * net.wo_mask
                net.unitn.grad = net.unitn.grad * net.unitn_mask
                net.unitm.grad = net.unitm.grad * net.unitm_mask
                net.unitwi.grad = net.unitwi.grad * net.unitwi_mask
            #Backward Pass:loss.backward() computes the gradients of the loss with respect to all model parameters by backpropagation.


                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_gradient)#防止梯度爆炸
            #这段代码用于记录每一步的梯度L2范数，便于后续画出“梯度随训练进程的变化曲线”，帮助判断训练是否稳定：
            if plot_gradient:
                tot = 0
                for param in [p for p in net.parameters() if p.requires_grad]:
                    tot += (param.grad**2).sum()
                gradient_norms.append(sqrt(tot))
            optimizer.step()
            # These 2 lines important to prevent memory leaks
            loss.detach_()
            output.detach_()
            if resample:
                net.resample_basis()
        if keep_best and np.mean(losses) < best_loss:
            best = net.clone()
            best_loss = np.mean(losses)
            print(
                "epoch %d:  loss=%.3f  (took %.2f s) *"
                % (epoch, np.mean(losses), time.time() - begin)
            )
        else:
            print(
                "epoch %d:  loss=%.3f  (took %.2f s)"
                % (epoch, np.mean(losses), time.time() - begin)
            )
        if early_stop is not None and np.mean(losses) < early_stop:
            break

    if plot_learning_curve:
        plt.plot(all_losses)
        plt.title("Learning curve")
        plt.show()

    if plot_gradient:
        plt.plot(gradient_norms)
        plt.title("Gradient norm")
        plt.show()

    if keep_best:
        net.load_state_dict(best.state_dict())

    print('RECURRENT WEIGHT MATRIX:')
    print(net.wrec)

import torch
import torch.nn as nn
from math import sqrt

class FullRankRNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        noise_std,
        alpha=0.2,
        rho=1,
        train_wi=False,
        train_wo=False,
        train_wrec=True,
        train_h0=False,
        wi_init=None,
        wo_init=None,
        wrec_init=None,
        si_init=None,
        so_init=None,
    ):
        super(FullRankRNN, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.noise_std   = noise_std
        self.alpha       = alpha
        self.rho         = rho
        self.train_wi    = train_wi
        self.train_wo    = train_wo
        self.train_wrec  = train_wrec
        self.train_h0    = train_h0
        self.non_linearity = torch.tanh

        # ——— 定义参数 ———
        self.wi   = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.si   = nn.Parameter(torch.Tensor(input_size))
        if train_wi:
            self.si.requires_grad = False
        else:
            self.wi.requires_grad = False

        self.wrec = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        if not train_wrec:
            self.wrec.requires_grad = False

        self.wo   = nn.Parameter(torch.Tensor(hidden_size, output_size))
        self.so   = nn.Parameter(torch.Tensor(output_size))
        if train_wo:
            self.so.requires_grad = False
        else:
            self.wo.requires_grad = False

        self.h0   = nn.Parameter(torch.Tensor(hidden_size))
        if not train_h0:
            self.h0.requires_grad = False

        # self.bias = nn.Parameter(torch.zeros(hidden_size))  # trainable by default

        # 参数初始化
        with torch.no_grad():
            if wi_init is None:
                self.wi.normal_()
            else:
                self.wi.copy_(wi_init)

            if si_init is None:
                self.si.set_(torch.ones_like(self.si))
            else:
                self.si.copy_(si_init)

            if wrec_init is None:
                self.wrec.normal_(std=rho / sqrt(hidden_size))
            else:
                self.wrec.copy_(wrec_init)

            if wo_init is None:
                self.wo.normal_(std=1/hidden_size)
            else:
                self.wo.copy_(wo_init)

            if so_init is None:
                self.so.set_(torch.ones_like(self.so))
            else:
                self.so.copy_(so_init)

            self.h0.zero_()

        self.wi_full, self.wo_full = [None, None]
        self._define_proxy_parameters()


    def _define_proxy_parameters(self):
        # 把 wi, si 拼成 wi_full；把 wo, so 拼成 wo_full
        self.wi_full = (self.wi.t() * self.si).t()
        self.wo_full = self.wo * self.so


    def forward(self, input, return_dynamics=False, return_all_states=False):
        """
        :param input: 形状 (batch_size, seq_len, input_dim) 的输入张量
        :param return_dynamics: 如果 True，返回 output 和 hid_states
        :param return_all_states: 如果 True，返回 output、hid_states 和 activations
                                  （注意：return_all_states=True 时会覆盖 return_dynamics）
        :return:
            - 如果两者都 False，只 return output，形状 (batch_size, seq_len, output_dim)
            - 如果 return_dynamics=True，则 return (output, hid_states)，
              hid_states 形状 (batch_size, seq_len, hidden_size)
            - 如果 return_all_states=True，则 return (output, hid_states, activations)，
              activations 形状也为 (batch_size, seq_len, hidden_size)
        """
        batch_size = input.shape[0]
        seq_len    = input.shape[1]

        # ——— 初始化隐藏状态 h(0) 及激活 r(0) ———
        h = self.h0.unsqueeze(0).expand(batch_size, -1).clone()  # (batch_size, hidden_size)               # shape = (hidden_size,)
        r = self.non_linearity(h)

        # 拼 wi_full, wo_full
        self._define_proxy_parameters()

        # 造噪声张量 (batch_size, seq_len, hidden_size)
        noise = torch.randn(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # 用来存 output 的张量 (batch_size, seq_len, output_dim)
        output = torch.zeros(batch_size, seq_len, self.output_size, device=self.wrec.device)

        # 如果要记录隐藏状态 h(t)，就先分配空间
        if return_dynamics:
            hid_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # 如果要记录激活 r(t)，也分配空间
        if return_all_states:
            activations = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.wrec.device)

        # ——— 循环算每个时间步 ———
        for t in range(seq_len):
            # （1）h 的更新：h ← h + noise + alpha * ( -h + r @ W_rec^T + x_t @ W_i^T )
            h = (
                h
                + self.noise_std * noise[:, t, :]
                + self.alpha * (
                    -h
                    + r.matmul(self.wrec.t())
                    + input[:, t, :].matmul(self.wi_full)  # 新增：加上 bias
                )
            )
            # （2）激活 r(t) = tanh(h)
            r = self.non_linearity(h)

            # （3）读出 output(t) = r @ W_o^T
            output[:, t, :] = r.matmul(self.wo_full)
           
            # （4）如果要记录隐状态，就把 h 存进 hid_states[:, t, :]
            if return_dynamics or return_all_states:
                hid_states[:, t, :] = h

            # （5）如果要记录激活，就把 r 存进 activations[:, t, :]
            if return_all_states:
                activations[:, t, :] = r

        # 最后根据标志 return 不同的东西
        if return_all_states:
            return output, hid_states, activations
        elif return_dynamics:
            return output, hid_states
        else:
            return output


    def clone(self):
        new_net = FullRankRNN(
            self.input_size,
            self.hidden_size,
            self.output_size,
            self.noise_std,
            self.alpha,
            self.rho,
            self.train_wi,
            self.train_wo,
            self.train_wrec,
            self.train_h0,
            self.wi,
            self.wo,
            self.wrec,
            self.si,
            self.so,
        )
        return new_net