FOCUS: Firedrake for Optimal Control and Uncertainty quantification with Stochastic PDEs
===================================================================

FOCUS is a Python library for optimal control and uncertainty quantification
for time-dependent partial differential equations (PDEs). It is built on
`Firedrake <https://www.firedrakeproject.org/>`_ and
`pyadjoint <https://pyadjoint.org/>`_, with a model predictive control approach to optimal control problems.

The project is under active development. Its source code and development
discussion are available on `GitHub <https://github.com/divijghose/focus>`_.

Why FOCUS?
----------

Many PDE-constrained optimization problems require repeatedly solving a
forward model while updating a control. FOCUS provides the building blocks for
that loop while keeping the PDE, objective functional, control, and optimizer
as separate components. This makes it possible to experiment with different
models and optimization strategies without rewriting the surrounding workflow.


Core concepts
-------------

The main components of a FOCUS application are:

``solvers``
   Firedrake-based PDE solvers that define the state evolution, boundary
   conditions, forcing, and diagnostics.

``controls``
   Control representations and updates, including additive distributed
   controls and Dirichlet controls.

``functionals``
   Objective functionals that measure the difference between a computed state,
   a desired state, and the applied control.

``windowing``
   Fixed or ensemble time windows that organize forward solves and control
   updates over a longer simulation horizon.

``optimizers``
   Optimization backends for reduced functionals, including TAO- and
   covariance-based approaches.

``utils``
   Configuration, diagnostics, error reporting, and output helpers.

Typical workflow
----------------

A typical application creates a mesh and function space, configures a PDE
solver, initializes a control, and then optimizes the objective one time
window at a time:

.. code-block:: python

   from firedrake import Constant, FunctionSpace, UnitIntervalMesh

   from focus.controls import additive_control
   from focus.functionals.loss import LossFunctional
   from focus.optimizers.tao import TAOOptimizer
   from focus.solvers.heat_equation import HeatEquationSolver
   from focus.windowing.fixed_window import FixedWindow

   mesh = UnitIntervalMesh(80)
   space = FunctionSpace(mesh, "CG", 2)
   solver = HeatEquationSolver(mesh, space, kappa=0.01, dt=0.01)
   solver.set_initial_condition(Constant(0.0))
   solver.set_bcs([Constant(0.0), Constant(0.0)])
   solver.build_solver()

   windowing = FixedWindow(
       window_size=5,
       window_stride=1,
       pde_solver=solver,
   )
   windowing.initialize_controls(initial_expression=Constant(1.0))

   # Define the desired state and objective for the application.
   desired_state = solver.set_desired_solution(...)
   objective = LossFunctional(
       desired_state,
       solver,
       {"lambda_t": 0.1, "control_weight": 1.0},
   )
   windowing.run_first_window(objective)
   optimizer = TAOOptimizer(windowing.Jhat, parameters={"method": "lbfgs"})

   optimal_control = optimizer.get_optimal_control()
   windowing.time_step_loop()
   windowing.reinitialize_window_controls(optimal_control)

A complete working example can be found in ``examples/heat_equation.py``.

Installation
------------

FOCUS currently targets Python 3.10 or newer and requires a working Firedrake
installation. Install Firedrake according to its
`official installation instructions <https://www.firedrakeproject.org/download.html>`_
before installing FOCUS and its Python dependencies.

From a checked-out copy of this repository, install the package with:

.. code-block:: console

   $ python -m pip install -e .

Configuration
-------------

Example programs read runtime settings from ``examples/input.yaml``. The
configuration includes the simulation horizon, time-window parameters,
objective weights, and output path:

.. code-block:: yaml

   T: 0.01
   window_size: 5
   window_stride: 1
   decay_constant: 0.1
   control_weight: 1.0

See the API reference for the supported classes and methods:

:doc:`API reference <api>`

.. toctree::
   :maxdepth: 1
   :hidden:

   api

:doc:`API reference <api>`
