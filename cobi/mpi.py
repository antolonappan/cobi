"""
MPI Support Module
==================

This module provides MPI (Message Passing Interface) support for parallel
computation across multiple processors.

The module gracefully handles systems without MPI by providing fallback
implementations that work in single-processor mode.

Attributes
----------
mpi : module
    The mpi4py.MPI module if available, None otherwise
com : MPI.COMM_WORLD
    MPI communicator for all processes
rank : int
    Rank of the current process (0 if no MPI)
size : int
    Total number of processes (1 if no MPI)
barrier : function
    MPI barrier synchronization (no-op if no MPI)
finalize : function
    Finalizes MPI environment (no-op if no MPI)

Example
-------
    from cobi import mpi
    
    if mpi.rank == 0:
        print("This is the master process")
    
    # Wait for all processes to reach this point
    mpi.barrier()
    
    # Compute something on each process
    local_result = compute_chunk(mpi.rank, mpi.size)
    
    # Gather results on rank 0
    if mpi.rank == 0:
        results = mpi.com.gather(local_result, root=0)
"""

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