class SolverDidNotConvergeOnce(RuntimeError):
    pass


def specify_if_solver_did_not_converge(solve):

    def wrapped_solve(*args, **kwargs):
        try:
            return solve(*args, **kwargs)
        except RuntimeError as e:
            if "solver did not converge" in str(e):
                raise SolverDidNotConvergeOnce(e)
            raise

    return wrapped_solve
