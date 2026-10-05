from firedrake import *
from firedrake.adjoint import *
from pyadjoint import *
from firedrake import exp
from pyadjoint.optimization.tao_solver import TAOSolver, MinimizationProblem
import os
os.makedirs("results", exist_ok=True)
output_dir = "heat_rf_output"
output_dir = f"results/{output_dir}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
outfile = VTKFile(f"{output_dir}/heat_rf.pvd")
continue_annotation()

N = 64
mesh = UnitSquareMesh(N, N)
V = FunctionSpace(mesh, "CG", 2)

u = TrialFunction(V)
v = TestFunction(V)

u_init = Function(V, name="Initial condition")
u_old = Function(V, name="Old state")
u_new = Function(V, name="New state")
k = Constant(0.01)
lambda_t = Constant(0.1)
w_c = Constant(0.001)
bcs = [DirichletBC(V, 0.0, "on_boundary")]
t = Constant(0.0)
dt = Constant(1e-2)
t_max = 1.0

x, y = SpatialCoordinate(mesh)
u_init.interpolate(exp(-100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))
u_old.assign(u_init)
u_new.assign(u_init)

u_desired = Function(V, name="Desired state")
u_desired.interpolate(exp(-100 * ((x - 0.5) ** 2 + (y - 0.5) ** 2)))

def misfit(u,  t):
    return (exp(-lambda_t*t) * 0.5 * inner(u - u_desired, u - u_desired))

def control_loss(m):
    return (w_c * 0.5 * inner(m, m))

control_list = [Function(V, name=f"Control_{i}") for i in range(2)]
m = Function(V, name="Control")

a = inner(u, v) *dx + dt * k * inner(grad(u), grad(v)) * dx
L = inner(u_old, v) * dx + dt * inner(m, v) * dx
LinearProblem = LinearVariationalProblem(a, L, u_new, bcs=bcs)
HeatSolver = LinearVariationalSolver(LinearProblem)
J = 0.0
Jhat_sol_list = []
for i  in range(2):
    m.assign(control_list[i])
    u_old.assign(u_new)
    HeatSolver.solve()
    pause_annotation()
    # tape = get_working_tape()
    # tape.visualise(f"heat_equation_tape_{i}.pdf")
    Jhat_sol_list.append(ReducedFunctional(u_new, controls=Control(m), parameters=u_old))
    continue_annotation()
    t = t + dt
    J += assemble(misfit(u_new, t) *dx + control_loss(m) * dx)


        
pause_annotation()
Jhat = ReducedFunctional(J, controls=[Control(control_i) for control_i in control_list], parameters=u_init)
tape = get_working_tape()
tape.visualise("heat_equation_tape_final.pdf")
minimzation_problem = MinimizationProblem(Jhat)
tao_solver = TAOSolver(minimzation_problem, parameters= {
    "method": "lbfgs",
    "max_it": 20,
    "fatol": 0.0,
    "frtol": 0.0,
    "gatol": 1e-9,
    "grtol": 0.0,
})
optimal_controls = tao_solver.solve()
u_new.assign(Jhat_sol_list[0](optimal_controls[0]))
outfile.write(u_new, optimal_controls[0], time=0.0)








