Simulation Subpackage
=====================

.. automodule:: cobi.simulation
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The simulation subpackage provides comprehensive tools for generating realistic
CMB observations including signal, foregrounds, and instrumental effects.

Main Classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   cobi.simulation.CMB
   cobi.simulation.Foreground
   cobi.simulation.Mask
   cobi.simulation.Noise
   cobi.simulation.LATsky
   cobi.simulation.SATsky
   cobi.simulation.LATskyC
   cobi.simulation.SATskyC

CMB Module
----------

.. automodule:: cobi.simulation.cmb
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Classes
^^^^^^^

.. autoclass:: cobi.simulation.cmb.CMB
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Functions
^^^^^^^^^

.. autofunction:: cobi.simulation.cmb.synfast_pol
.. autofunction:: cobi.simulation.cmb.hp_alm2map_spin
.. autofunction:: cobi.simulation.cmb.get_camb_cls

Foreground Module
-----------------

.. automodule:: cobi.simulation.fg
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Classes
^^^^^^^

.. autoclass:: cobi.simulation.fg.BandpassInt
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.fg.Foreground
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.fg.HILC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Mask Module
-----------

.. automodule:: cobi.simulation.mask
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Classes
^^^^^^^

.. autoclass:: cobi.simulation.mask.Mask
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Noise Module
------------

.. automodule:: cobi.simulation.noise
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Classes
^^^^^^^

.. autoclass:: cobi.simulation.noise.Noise
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Functions
^^^^^^^^^

.. autofunction:: cobi.simulation.noise.NoiseSpectra

Sky Module
----------

.. automodule:: cobi.simulation.sky
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Classes
^^^^^^^

.. autoclass:: cobi.simulation.sky.SkySimulation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.sky.LATsky
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.sky.SATsky
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.sky.LATskyC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: cobi.simulation.sky.SATskyC
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

