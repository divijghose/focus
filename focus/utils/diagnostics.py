"""Functions to diagnose the loss functional, optimization and weightings
"""
from ..functionals.loss import LossFunctional
from firedrake import *

def _plot_time_decay(window_size, lambda_t):
    """
    Plots the time decay function for a given window size and decay constant.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.linspace(0, window_size, 100)
    decay = np.exp(-lambda_t * t)

    plt.figure(figsize=(8, 5))
    plt.plot(t, decay, label=f"Decay (lambda_t={lambda_t})")
    plt.title("Time Decay Function")
    plt.xlabel("Time")
    plt.ylabel("Decay")
    plt.legend()
    plt.grid()
    plt.savefig(f"time_decay_lambda_{lambda_t}.png", dpi=300, bbox_inches='tight')

def _calculate_misfit_loss_magnitude(u_desired, u_current, lambda_t, t_window):
    pass

def _calculate_control_cost_magnitude(control):
    pass

def _calculate_total_loss_magnitude(control, u_desired, u_current, lambda_t, t_window):
    pass





def diagnose_loss_functional(window_size, lambda_t,):
    """
    Diagnoses the loss functional by plotting the time decay function.
    """
    print("Diagnosing Loss Functional...")
    _plot_time_decay(window_size, lambda_t)

if __name__ == "__main__":
    # Example usage
    window_size = 10
    lambda_t = 0.4
    diagnose_loss_functional(window_size=window_size, lambda_t=lambda_t)

def _ensemble_mean(fields, ensemble):
    ensemble_size = ensemble.ensemble_size
    mean_field = ensemble.allreduce(fields) / ensemble_size
    return mean_field




def diagnose_ensemble(field, ensemble):
    mean_field = _ensemble_mean(field, ensemble)
    if ensemble.ensemble_rank == ensemble.ensemble_size - 1:
        l2_norm = norm(abs(field - mean_field))

        print(f"Ensemble Mean L2 Norm: {l2_norm}")
