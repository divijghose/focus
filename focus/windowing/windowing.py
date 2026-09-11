# windowing/fixed_window.py

from firedrake import Constant
from firedrake.adjoint import continue_annotation
from firedrake.function import Function
from pyadjoint import pause_annotation

from ..controls.distributed import DistributedControl
from ..utils.output_utils import get_logger
from .base import Windowing
from .tape_manager import TapeManager

logger = get_logger(__name__)


class FixedWindow(Windowing):
    """Receding horizon MPC with a fixed window size and stride.

    The tape is recorded exactly once in :meth:`run_first_window`. After
    that, the optimizer replays the tape internally by calling the reduced
    functional. Between windows, :meth:`time_step_loop` advances the
    solution outside the tape and updates the window initial condition via
    ``Jhat.update_parameters``.

    The time within each window is tracked by a single :class:`Constant`
    ``t_hop``, which is updated in place during the recorded forward pass
    so that the tape sees a single symbolic time node. The desired state
    expression must accept this ``Constant`` as its time argument.

    Window controls are always stored as ``list[list[Function]]`` of shape
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
            advances after each optimisation cycle.
        :type window_stride: int
        :param pde_solver: The controlled PDE solver instance.
        :type pde_solver: ControlledSolver
        """
        super().__init__(window_size, window_stride)
        self.pde_solver = pde_solver
        self.tape_manager = TapeManager()
        self.window_controls: list[list[Function]] = []
        self.t_hop = Constant(0.0)

        logger.debug(
            f"Initialized FixedWindow with size={window_size}, "
            f"stride={window_stride}."
        )


    @property
    def Jhat(self):
        """The reduced functional built from the recorded forward pass."""
        return self.tape_manager.Jhat

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
    # Tape recording
    # ------------------------------------------------------------------

    def run_first_window(self, loss_functional) -> None:
        """Record the forward pass once and build the reduced functional.

        This is the only place the tape is recorded. The sequence is:

        1. Anchor :attr:`u_old` and :attr:`u_new` to :attr:`p` so the
           tape registers ``p`` as the window initial condition.
        2. Resume annotation and start a fresh tape.
        3. For each hop in the window:
            a. Assign the window control to the solver control.
            b. Sync distributed controls onto the state space (once,
               inside the tape).
            c. Update the symbolic time constant :attr:`t_hop`.
            d. Update the forcing function.
            e. Solve the PDE.
            f. Accumulate the loss.
        4. Build the reduced functional from the accumulated loss.
        5. Pause annotation.

        :param loss_functional: The loss functional callable with signature
            ``(control, t_hop) -> float``.
        :raises RuntimeError: If window controls have not been initialised.
        """
        if not self.window_controls:
            raise RuntimeError(
                "Window controls have not been initialised. "
                "Call initialize_controls() before run_first_window()."
            )

        # Anchor u_old and u_new to p so the tape sees p as the
        # initial condition of this window.
        
        J = 0

        self.tape_manager.start()
        self.pde_solver.u_old.assign(self.pde_solver.p)
        self.pde_solver.u_new.assign(self.pde_solver.p)
        window_hop_time = 0.0

        for i in range(self.window_size):
            # Assign window controls and sync onto state space.
            # sync() is called here, inside the tape, so the interpolation
            # is registered as a symbolic operation.
            for ctrl, ctrl_window in zip(
                self.pde_solver.controls, self.window_controls
            ):
                ctrl.assign(ctrl_window[i])
                if isinstance(ctrl, DistributedControl):
                    ctrl.sync()

            # Update the symbolic time constant in place.
            # The tape sees a single Constant node, not a new expression
            # at each step.
            self.t_hop.assign(window_hop_time)
            self.pde_solver.update_forcing_function(self.global_hop_time)
            self.pde_solver.solve()

            J += loss_functional(
                self.window_controls[0][i]
                if self.pde_solver.num_controls == 1
                else [wc[i] for wc in self.window_controls],
                t_hop=self.t_hop,
            )

            self.global_hop_time += self.pde_solver.dt
            window_hop_time += self.pde_solver.dt
        self.tape_manager.pause()

        flat_controls = [
            m_i
            for ctrl_window in self.window_controls
            for m_i in ctrl_window
        ]
        self.tape_manager.build_reduced_functional(
            J, flat_controls, self.pde_solver.p
        )
        self._reset_times()
        logger.debug("First window recorded. Reduced functional built.")

    # ------------------------------------------------------------------
    #                                       Time step loop
    # ------------------------------------------------------------------

    def time_step_loop(self) -> None:
        """Advance the solution forward by one stride outside the tape.

        By this point the optimizer has computed the optimal controls.
        This method:

        1. Assigns the optimal control values for each stride step.
        2. Solves the PDE for each stride step.
        3. Advances the global step time.
        4. Updates :attr:`p` from :attr:`u_new`.
        5. Calls ``Jhat.update_parameters`` to re-anchor the tape to the
           new window initial condition.
        6. Advances the window indices.

        No sync is performed here since we are outside the tape and the
        interpolation does not need to be registered.
        """
        for i in range(self.window_stride):
            for ctrl, ctrl_window in zip(
                self.pde_solver.controls, self.window_controls
            ):
                ctrl.assign(ctrl_window[i])

            self.pde_solver.update_forcing_function(self.global_step_time)
            self.pde_solver.solve()
            self.global_step_time += self.pde_solver.dt

        # Update p to the new initial condition and re-anchor the tape.
        self.pde_solver.set_parameters()
        self.tape_manager.update_parameters(self.pde_solver.p)

        self.advance_window()
        self._reset_global_window_times()
        logger.debug(
            f"Time step loop complete. "
            f"Window advanced to [{self.current_window_start}, "
            f"{self.current_window_end}]. "
            f"Global step time: {self.global_step_time:.4f}."
        )

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def advance_window(self) -> None:
        """Advance the window indices by the stride."""
        self.current_window_start += self.window_stride
        self.current_window_end += self.window_stride
        self.window_number += 1

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
        optimal = optimal_controls[0]

        for i in range(self.window_size):
            shifted_index = i + self.window_stride
            if shifted_index < self.window_size:
                ctrl_window[i].interpolate(optimal[shifted_index])
            else:
                ctrl_window[i].interpolate(Constant(0.0))

        logger.debug("Window controls warm-started from optimal solution.")

    def get_window_start_time(self) -> float:
        """Return the start time of the current window."""
        return self.current_window_start * self.pde_solver.dt

    def get_window_end_time(self) -> float:
        """Return the end time of the current window."""
        return self.current_window_end * self.pde_solver.dt

    # ------------------------------------------------------------------
    # Time management (private)
    # ------------------------------------------------------------------

    def _reset_times(self) -> None:
        """Reset all time counters to zero."""
        self.global_step_time = 0.0
        self.global_hop_time = 0.0

    def _reset_global_window_times(self) -> None:
        """Advance the global hop time to match the global step time."""
        self.global_hop_time = self.global_step_time
