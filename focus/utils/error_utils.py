from firedrake import norm
def point_wise_error(error, u, u_desired):
    error = error.interpolate(abs(u - u_desired))

def l2_error(u, u_desired):
    error = norm(u - u_desired)
    return error

def linf_error(point_wise_error):
    return point_wise_error.dat.data_ro.max()