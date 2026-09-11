# windowing/tape_manager.py

from firedrake.function import Function
from pyadjoint import (
    Control,
    ReducedFunctional,
    Tape,
    continue_annotation,
    get_working_tape,
    pause_annotation,
)

from ..utils.output_utils import get_logger

logger = get_logger(__name__)


class TapeManager:
    """Manages the creation of a pyadjoint tape and ReducedFunctional
    """

    def __init__(self):
        self._tape: Tape | None = None
        self._Jhat: ReducedFunctional | None = None

    # ------------------------------------------------------------------
    #                                           Properties
    # ------------------------------------------------------------------

    @property
    def Jhat(self) -> ReducedFunctional:
        """The reduced functional built from the last recorded forward pass.

        :raises RuntimeError: If the reduced functional has not been built yet.
        """
        if self._Jhat is None:
            raise RuntimeError(
                "Reduced functional has not been built yet. "
                "Call build_reduced_functional() after a recorded forward pass."
            )
        return self._Jhat

    @property
    def tape(self) -> Tape:
        """The current pyadjoint tape.

        :raises RuntimeError: If no tape has been created yet.
        """
        if self._tape is None:
            raise RuntimeError(
                "No tape has been created yet. "
                "Call record() to start a new tape."
            )
        return self._tape



    def start(self) -> None:
        """Create a new tape and resume annotation.

        Replaces any existing tape. Called at the start of the first
        window forward pass.
        """
        continue_annotation()
        self._tape = get_working_tape()
        logger.debug("Tape started and annotation resumed.")

    def pause(self) -> None:
        """Pause annotation after the forward pass is complete."""
        pause_annotation()
        logger.debug("Annotation paused.")

    # ------------------------------------------------------------------
    #                                       Reduced functional
    # ------------------------------------------------------------------

    def build_reduced_functional(
        self,
        J: float,
        controls: list[Function],
        parameters: Function,
    ) -> ReducedFunctional:
        """Build the reduced functional from the recorded forward pass.

        :param J: The accumulated loss functional value.
        :type J: float
        :param controls: The list of control functions at each time hop,
            flattened across all control types and window steps.
        :type controls: list[Function]
        :param parameters: The parameter field (initial condition) for
            the current window.
        :type parameters: Function
        :returns: The assembled reduced functional.
        :rtype: ReducedFunctional
        :raises RuntimeError: If no tape has been started.
        """
        _ = self.tape  # raises if no tape exists
        self._Jhat = ReducedFunctional(
            J,
            controls=[Control(m) for m in controls],
            parameters=parameters,
        )
        logger.debug(
            f"Reduced functional built with {len(controls)} control(s)."
        )
        return self._Jhat

    def update_parameters(self, parameters: Function) -> None:
        """Update the parameter field in the reduced functional.

        Called by the windowing class after each leap to advance
        the initial condition for the next window.

        :param parameters: The updated parameter field.
        :type parameters: Function
        :raises RuntimeError: If the reduced functional has not been built.
        """
        self.Jhat.update_parameters(parameters)
        logger.debug("Reduced functional parameters updated.")

    def visualise_tape(self, filename: str = "tape_graph.pdf") -> None:
        """Visualize the current tape as a graph.

        :param filename: The base filename for the output graph (without extension).
        :type filename: str
        :raises RuntimeError: If no tape has been created yet.
        """
        _ = self.tape  # raises if no tape exists
        self._tape.visualise(filename)
        logger.debug(f"Tape visualized and saved to '{filename}.png'.")