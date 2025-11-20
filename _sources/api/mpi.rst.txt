MPI Support Module
==================

.. automodule:: cobi.mpi
   :members:
   :undoc-members:

Module Attributes
-----------------

.. py:data:: mpi
   :type: module

   The mpi4py.MPI module if available, otherwise None.

.. py:data:: com
   :type: MPI.COMM_WORLD

   MPI communicator for all processes.

.. py:data:: rank
   :type: int

   Rank of the current process (0 if no MPI).

.. py:data:: size
   :type: int

   Total number of processes (1 if no MPI).

.. py:data:: barrier
   :type: function

   MPI barrier synchronization function.

.. py:data:: finalize
   :type: function

   Function to finalize the MPI environment.
