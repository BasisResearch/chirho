import torch


def initial_curve(x):
    condition = (x >= -1.0) & (x <= 1.0)
    values = (1.0 - x**2 + 0.3 * torch.sin(2. * torch.pi * x) ** 2.)
    return torch.where(condition, values, torch.zeros_like(x))


def plot_sols(xx, sols, ax, skip=5, color1='purple', color2='orange', thickness1=1.0, thickness2=0.1,
              final_sol_lbl: str = "End State"):

    ax.plot(xx, sols[0].detach(), color=color1, linestyle='--', label="Initial Condition", linewidth=thickness1)
    for t_sols in sols[1:-1:skip]:
        ax.plot(xx, t_sols.detach(), color=color2, alpha=0.5, linewidth=thickness2)
    ax.plot(xx, sols[-1].detach(), color=color1, label=final_sol_lbl, linewidth=thickness1)
