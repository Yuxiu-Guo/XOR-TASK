import torch
import numpy as np
import matplotlib.pyplot as plt
from math import floor

---
deltaT = 20.0
t_ctx   = 400    # context cue duration (ms)
t_stim  = 800    # checkerboard stimulus duration (ms)
t_dec   = 300    # decision window (ms)
t_delay = 0      # no delay between stim and decision for simplicity

t_ctx_steps  = floor(t_ctx   / deltaT)
t_stim_steps = floor(t_stim  / deltaT)
t_dec_steps  = floor(t_dec   / deltaT)
t_total      = t_ctx_steps + t_stim_steps + t_dec_steps



# 
pos_vals = 0.5 * (0.5 ** np.arange(0, 4, dtype=float))  
#     pos_vals = [0.5, 0.25, 0.125, 0.0625]

# 
neg_vals = -pos_vals[::-1]  
coherences = np.sort(np.concatenate([neg_vals, pos_vals])  )
#     coherences = [-0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5]

print("8 specific coherence：")
print(coherences)


def make_single_xor_trial(context, coherence, std=0, coh_noise_std=0):#idealized
    
    x = std * torch.randn(1, t_total, 3)
    if context == -1:
        x[0, :t_ctx_steps, 0] = 1
        x[0, :t_ctx_steps, 1] = 0 
    else:
        x[0, :t_ctx_steps, 0] = 0
        x[0, :t_ctx_steps, 1] = 1

    
        stim_slice = slice(t_ctx_steps, t_ctx_steps + t_stim_steps)
    x[0, stim_slice, 2] = coherence
    
    x[0, stim_slice, 2] += coh_noise_std * torch.randn(t_stim_steps)

    # color_choice: 1 if coherence > 0 else 0
    color_choice = 1 if coherence > 0 else 0

    if (context == 1 and color_choice == 1) or (context == -1 and color_choice == 0):
        action = -1
    else:
        action = 1

    y = torch.zeros(1, t_total, 1)
    mask = torch.zeros_like(y)
    start = t_ctx_steps + t_stim_steps
    y[0, start:start + t_dec_steps, 0]   = float(action)
    mask[0, start:start + t_dec_steps, 0] = 1.0

    return x, y, mask

def generate_xor_batch(n_trials, std=0.05):
    X = torch.zeros(n_trials, t_total, 3)
    Y = torch.zeros(n_trials, t_total, 1)
    M = torch.zeros_like(Y)
    params = []
    for i in range(n_trials):
        ctx = np.random.choice([-1,1])
        coh = float(np.random.choice(coherences))
        x, y, m = make_single_xor_trial(ctx, coh, std)
        X[i], Y[i], M[i] = x, y, m
        params.append((ctx, coh, int(y[0,-1,0])))
    return X, Y, M, params



def plot_multiple_xor(X, Y, params, trial_indices, deltaT,
                      t_ctx_steps, t_stim_steps, t_total):
   
    n = len(trial_indices)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 1.5), sharex=True)
    t = np.arange(t_total) * deltaT

    phases = [
        ('Context', 0, t_ctx_steps, 'gray'),
        ('Stimulus', t_ctx_steps, t_ctx_steps + t_stim_steps, 'cyan'),
        ('Decision', t_ctx_steps + t_stim_steps, t_total, 'yellow')
    ]

    for row, idx in enumerate(trial_indices):
        ctx, coh, action = params[idx]
        ax_ctx, ax_coh, ax_act = axes[row]

        ax_ctx.plot(t, X[idx, :, 0].numpy(), color='tab:purple')
        ax_ctx.set_ylim(-1, 1)
        if row == 0:
            ax_ctx.set_title('Context')

        ax_coh.plot(t, X[idx, :, 1].numpy(), color='tab:green')
        ax_coh.set_ylim(-1, 1)
        if row == 0:
            ax_coh.set_title('Coherence')

        ax_act.step(t, Y[idx, :, 0].numpy(), where='post', color='tab:red')
        ax_act.set_ylim(-1.1, 1.1)
        if row == 0:
            ax_act.set_title('Choice')

        for ax in (ax_ctx, ax_coh, ax_act):
            for name, start, end, color in phases:
                ax.axvspan(start * deltaT, end * deltaT, color=color, alpha=0.2)

        axes[row, 0].set_ylabel(f'Trial {idx}')
        axes[row, 2].annotate(
            f'Ctx={ctx}, C={coh:.4f}, A={"R" if action else "L"}',
            xy=(1.01, 0.5), xycoords='axes fraction',
            va='center', fontsize=8
        )

    axes[-1, 0].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':

    X_sel = torch.zeros(n_sel, t_total, 3)
    Y_sel = torch.zeros(n_sel, t_total, 1)
    M_sel = torch.zeros_like(Y_sel)
    params_sel = []   
    idx = 0
    for coh in coherences:
        for ctx in [-1, 1]:
            x, y, m = make_single_xor_trial(ctx, float(coh))
            X_sel[idx] = x
            Y_sel[idx] = y
            M_sel[idx] = m
            params_sel.append((ctx, float(coh), int(y[0, -1, 0])))  
            idx += 1
    all_indices = list(range(n_sel))  # [0,1,2,...,15]
    plot_multiple_xor(X_sel, Y_sel, params_sel, all_indices,
        deltaT, t_ctx_steps, t_stim_steps, t_total)
    
import sys
sys.path.append(r'C:\Users\13911\Desktop') 

import torch
import numpy as np
import matplotlib.pyplot as plt
from LowRank import LowRankRNN, loss_mse, train, accuracy_general
from low_rank_rnns.helpers import map_device
from math import floor


deltaT = 20.
t_ctx     = 400
t_stim    = 800
t_dec     = 200
t_ctx_steps  = floor(t_ctx/deltaT)
t_stim_steps = floor(t_stim/deltaT)
t_dec_steps  = floor(t_dec/deltaT)
t_total      = t_ctx_steps + t_stim_steps + t_dec_steps



INPUT_SIZE  = 3      # context + coherence
HIDDEN_SIZE = 512    
OUTPUT_SIZE = 1      
NOISE_STD   = 0.02   
ALPHA       = 0.1    # dt/tau

net = LowRankRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    noise_std=NOISE_STD,
    alpha=ALPHA,
    rank=2,
    train_wi=False,
        train_wo=False,
        train_wrec=True,
        train_h0=False,
        train_si=True,
        train_so=True,
        wi_init=None,
        wo_init=None,
        m_init=None,
        n_init=None,
        si_init=None,
        so_init=None,
        h0_init=None,
)

X_all, Y_all, M_all, params_all = generate_xor_batch(n_trials=500, std=5e-2)
split = int(0.8 * X_all.shape[0])
X_tr, X_val = X_all[:split], X_all[split:]
Y_tr, Y_val = Y_all[:split], Y_all[split:]
M_tr, M_val = M_all[:split], M_all[split:]

X_tr, Y_tr, M_tr, X_val, Y_val, M_val = map_device(
    [X_tr, Y_tr, M_tr, X_val, Y_val, M_val], net
)

def run_training():
    train(
        net,
        X_tr, Y_tr, M_tr,
        n_epochs=100,
        lr=1e-2,#1e-3,
        batch_size=16,#128,
        plot_learning_curve=True,
        plot_gradient=False,
        clip_gradient=0.1,
        cuda=torch.cuda.is_available(),
        keep_best=True,
    )
   
    with torch.no_grad():
        out = net(X_val)
        val_loss = loss_mse(out, Y_val, M_val).item()
        val_acc  = accuracy_general(out, Y_val, M_val).item()
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

if __name__ == '__main__':
    run_training()
=
net.eval()
trial_idx = 1
ctx, coh, _ = params_all[split + trial_idx]
x, y_true, _ = make_single_xor_trial(ctx, 0.5)
with torch.no_grad():
    y_pred = net(x)[0, :, 0].cpu().numpy()
    y_true = y_true[0, :, 0].numpy()
t = np.arange(t_total) * deltaT

plt.figure(figsize=(8,3))
plt.step(t, y_true, where='post', label='Ground Truth', ls='--', c='k')
plt.plot(t, y_pred, label='Full-Rank RNN', c='r', lw=2)
for name, s, e, color in [
    ('Ctx', 0, t_ctx_steps, 'gray'),
    ('Stim', t_ctx_steps, t_ctx_steps+t_stim_steps, 'cyan'),
    ('Dec', t_ctx_steps+t_stim_steps, t_total, 'yellow'),
]:
    plt.axvspan(s*deltaT, e*deltaT, color=color, alpha=0.2)
plt.xlabel('Time (ms)')
plt.ylabel('Decision Signal')
plt.ylim(-1.1, 1.1)
plt.legend(loc='upper right')
plt.title(f'Trial {trial_idx}: Ctx={ctx}, Coh={coh:.2f}')
plt.tight_layout()
plt.show()


torch.save(net.state_dict(), "/Users/tinaguo/Desktop/XOR TASK CODE/Lowrank/LowRank.pth")
print("Model saved: /Users/tinaguo/Desktop/XOR TASK CODE/Lowrank/LowRank.pth")

