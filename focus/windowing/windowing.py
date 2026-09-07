# windowing/fixed_window.py

from firedrake import Constant
from firedrake.function import Function

from ..controls.distributed import DistributedControl
from ..utils.output_utils import get_logger
from .base import Windowing
from .tape_manager import TapeManager

logger = get_logger(__name__)


class FixedWindow(Windowing):
    """Receding horizon MPC with a fixed window size and stride.

    Manages the allocation of window controls, the forward pass recording,
    reduced functional construction, and the time-stepping loops for both
    the optimisation horizon (time-hop) and the execution horizon (time-step).

    Controls are always stored as ``list[list[Function]]`` of shape
    ``[num_controls][window_size]``, regardless of the number of controls.


    """

    def __init__(
        self,
        window_size: int,
        window_stride: int,
        pde_solver,
    ):
        """
        :param window_size: Number of time steps in each window.
        :type window_size: int
        :param window_stride: Number of time steps by which the window
            advances after each optimization cycle.
        :type window_stride: int
        :param pde_solver: The controlled PDE solver instance.
        :type pde_solver: ControlledSolver
        """
        super().__init__(window_size, window_stride)
        self.pde_solver = pde_solver
        self.tape_manager = TapeManager()
        self.window_controls: list[list[Function]] = []
        logger.debug(
            f"Initialized FixedWindow with size={window_size}, "
            f"stride={window_stride}."
        )
    # ------------------------------------------------------------------
    #                                   Window helper functions
    # ------------------------------------------------------------------
    def get_window_start_time(self):
        """
        Returns the start time of the current window.
        """
        return self.current_window_start * self.pde_solver.dt


    def get_window_end_time(self):
        """
        Returns the end time of the current window.
        """
        return self.current_window_end * self.pde_solver.dt
    # ------------------------------------------------------------------
    #                                   Window control allocation
    # ------------------------------------------------------------------

    def initialize_controls(
        self, initial_expression=None
    ) -> list[list[Function]]:
        """Allocate and initialise window controls for all attached controls.

        Creates a ``list[list[Function]]`` of shape
        ``[num_controls][window_size]``, where each function lives on the
        function space of its corresponding control. All functions are
        initialised from ``initial_expression``.

        :param initial_expression: A Firedrake expression or Constant used
            to initialise all control functions. Defaults to
            ``Constant(0.0)``.
        :returns: The allocated window controls.
        :rtype: list[list[Function]]
        :raises ValueError: If the solver has no controls attached.
        """
        if self.pde_solver.num_controls == 0:
            raise ValueError(
                "No controls are attached to the solver. "
                "Attach controls before calling initialize_controls()."
            )
        if initial_expression is None:
            initial_expression = Constant(0.0)

        self.window_controls = [
            [
                Function(
                    ctrl.function_space,
                    name=f"{ctrl.name}_window_hop{i}",
                )
                for i in range(self.window_size)
            ]
            for ctrl in self.pde_solver.controls
        ]

        for ctrl_window in self.window_controls:
            for m_i in ctrl_window:
                m_i.interpolate(initial_expression)

        logger.debug(
            f"Initialized window controls: {self.pde_solver.num_controls} "
            f"control(s) x {self.window_size} time hop(s)."
        )
        return self.window_controls

    # ------------------------------------------------------------------
    #                                   Control assignment
    # ------------------------------------------------------------------

    def _assign_controls(self, step: int) -> None:
        """Assign window control values at the given step to the solver.

        For :class:`DistributedControl` instances, :meth:`sync` is called
        after assignment to interpolate the control onto the state space.
        For :class:`DirichletControl` instances, the live reference in the
        BC is updated automatically.

        :param step: The time step index within the current window.
        :type step: int
        """
        for ctrl, ctrl_window in zip(
            self.pde_solver.controls, self.window_controls
        ):
            ctrl.assign(ctrl_window[step])
            if isinstance(ctrl, DistributedControl):
                ctrl.sync()

    # ------------------------------------------------------------------
    # Forward pass loops
    # ------------------------------------------------------------------

    def _forward_loop(
        self,
        loss_functional,
        n_steps: int,
        accumulate_loss: bool,
    ) -> float:
        """Run the forward time-stepping loop for a given number of steps.

        :param loss_functional: The loss functional callable.
        :param n_steps: Number of time steps to run.
        :type n_steps: int
        :param accumulate_loss: Whether to accumulate the loss functional.
        :type accumulate_loss: bool
        :returns: The accumulated loss value, or 0 if not accumulating.
        :rtype: float
        """
        J = 0
        for i in range(n_steps):
            self._assign_controls(i)
            self.pde_solver.update_forcing_function(self.global_hop_time)
            self.pde_solver.solve()

            if accumulate_loss:
                J += loss_functional(
                    self.window_controls[0][i]
                    if self.pde_solver.num_controls == 1
                    else [wc[i] for wc in self.window_controls],
                    t_current=self.global_hop_time,
                    t_window=self.window_hop_time,
                )

            self.global_hop_time += self.pde_solver.dt
            self.window_hop_time += self.pde_solver.dt

        return J

    def run_first_window(self, loss_functional) -> None:
        """Record the first window forward pass and build the reduced functional.

        Starts a new tape, runs the full window forward pass with loss
        accumulation, builds the :class:`ReducedFunctional`, and pauses
        annotation. Resets all time counters after completion.

        :param loss_functional: The loss functional callable with signature
            ``(control, t_current, t_window) -> float``.
        :raises RuntimeError: If window controls have not been initialised.
        """
        if not self.window_controls:
            raise RuntimeError(
                "Window controls have not been initialised. "
                "Call initialize_controls() before run_first_window()."
            )

        self.pde_solver.u_new.interpolate(self.pde_solver.p)

        self.tape_manager.start()
        J = self._forward_loop(
            loss_functional,
            n_steps=self.window_size,
            accumulate_loss=True,
        )

        flat_controls = [
            m_i
            for ctrl_window in self.window_controls
            for m_i in ctrl_window
        ]
        self.tape_manager.build_reduced_functional(
            J, flat_controls, self.pde_solver.p
        )
        self.tape_manager.pause()
        self._reset_times()
        logger.debug("First window completed and reduced functional built.")

    def time_hop_loop(self, loss_functional) -> None:
        """Run the forward pass over the full window for optimisation.

        Used during the optimisation cycle to evaluate the reduced
        functional over the full window horizon.

        :param loss_functional: The loss functional callable.
        """
        self._reset_local_window_times()
        self._forward_loop(
            loss_functional,
            n_steps=self.window_size,
            accumulate_loss=True,
        )

    def time_step_loop(self) -> None:
        """Advance the solution forward by one stride and update the window.

        Runs the forward pass over ``window_stride`` steps without loss
        accumulation, then advances the window, updates parameters, and
        resets time counters.
        """
        self._forward_loop(
            loss_functional=None,
            n_steps=self.window_stride,
            accumulate_loss=False,
        )
        self.advance_window()
        self._reset_global_window_times()
        self._reset_local_window_times()
        self._update_parameters()

    # ------------------------------------------------------------------
    # Parameter and window management
    # ------------------------------------------------------------------

    def _update_parameters(self) -> None:
        """Update solver and tape parameters for the next window."""
        self.pde_solver.set_parameters()
        self.tape_manager.update_parameters(self.pde_solver.p)
        logger.debug("Parameters updated for next window.")

    def reinitialize_window_controls(
        self, optimal_controls: list[list[Function]]
    ) -> None:
        """Warm-start the window controls from the previous optimal solution.

        Shifts the optimal controls forward by ``window_stride`` and fills
        the remaining steps with zeros.

        :param optimal_controls: The optimal controls from the previous
            optimisation cycle, of shape ``[num_controls][window_size]``.
        :raises NotImplementedError: If more than one control is attached.
        """
        if self.pde_solver.num_controls > 1:
            raise NotImplementedError(
                "Warm-starting is not yet implemented for more than one control."
            )
        ctrl_window = self.window_controls[0]
        # optimal = optimal_controls
        for i in range(self.window_size):
            shifted_index = i + self.window_stride
            logger.debug(f"shifted_index: {shifted_index}, window_size: {self.window_size}")
            if shifted_index < self.window_size:
                ctrl_window[i].interpolate(optimal_controls[shifted_index])
            else:
                ctrl_window[i].interpolate(Constant(0.0))

        logger.debug("Window controls reinitialized from optimal solution.")

    # ------------------------------------------------------------------
    # Time management (private)
    # ------------------------------------------------------------------

    def advance_window(self) -> None:
        """Advance the window indices by the stride."""
        self.current_window_start += self.window_stride
        self.current_window_end += self.window_stride
        self.window_number += 1
        logger.debug(
            f"Window advanced to [{self.current_window_start}, "
            f"{self.current_window_end}]. Window number: {self.window_number}."
        )
        self.global_step_time += self.window_stride * self.pde_solver.dt

    def _reset_times(self) -> None:
        """Reset all time counters to zero."""
        self.global_step_time = 0.0
        self.global_hop_time = 0.0
        self.window_hop_time = 0.0

    def _reset_local_window_times(self) -> None:
        """Reset the local window time counter."""
        self.window_hop_time = 0.0

    def _reset_global_window_times(self) -> None:
        """Advance the global hop time to match the global step time."""
        self.global_hop_time = self.global_step_time