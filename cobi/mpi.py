try:
    from mpi4py import MPI
    mpi = MPI
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    barrier = com.Barrier
    finalize = mpi.Finalize
except:
    rank = 0
    size = 1
    barrier = lambda: -1
    finalize = lambda: -1