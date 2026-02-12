.. _SysId:

=========
System ID
=========

.. toctree::
    :hidden:

    API <sysid_api.rst>

``mujoco.sysid`` is a
`system identification <https://en.wikipedia.org/wiki/System_identification>`__
toolbox for MuJoCo. Given a model and recorded sensor data, it finds physical
parameters that make simulation match reality. It is included in the ``mujoco``
package as an optional extra and can be installed with:

.. code-block:: shell

   pip install mujoco[sysid]

.. _SysIdNotebook:

An interactive tutorial notebook is available on Colab: |sysid_colab|

.. |sysid_colab| image:: https://colab.research.google.com/assets/colab-badge.png
                 :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mujoco/sysid/sysid.ipynb

.. _SysIdBackground:

Background
==========

System identification is the process of estimating unknown physical parameters
of a dynamical system from measured input-output data. A common scenario in
robotics is starting with a model derived from CAD or a URDF, where the
kinematic structure (link geometry, joint topology, actuator layout) is known
but dynamic parameters are only approximate. Masses and inertias from CAD
assume uniform density, joint friction and damping are rarely modeled, and
actuator dynamics are often unknown. System identification closes this gap by
fitting these parameters to real measurements.

Gray-box approach
-----------------

This toolbox implements `gray-box
<https://en.wikipedia.org/wiki/Grey_box_model>`__ identification. You supply
the model structure (rigid-body topology, joint types, actuator layout) as a
MuJoCo XML via an :ref:`MjSpec <PyModelEdit>`, and designate which parameters
are unknown. The optimizer then adjusts only those parameters, leaving the
model structure fixed. This is in contrast to *black-box* approaches
(e.g. neural networks) that learn dynamics entirely from data with no assumed
structure.

The optimization problem
------------------------

Given :math:`K` parameters collected in a vector :math:`\theta` and :math:`N`
sensor measurements :math:`y`, the toolbox simulates the model to produce
predicted outputs :math:`\bar{y}(\theta)` and minimizes the weighted residual:

.. math::

   \min_\theta \; \tfrac{1}{2}\lVert W\bigl(\bar{y}(\theta) - y\bigr)\rVert^2
   \qquad \text{s.t.}\quad l \preccurlyeq \theta \preccurlyeq u

This is a box-constrained **nonlinear least-squares** problem. The optimizer
uses a Gauss-Newton / Levenberg-Marquardt algorithm with finite-difference
Jacobians. Each parameter perturbation requires an independent simulation
rollout, and all of them execute in a single batched call to
:ref:`mujoco.rollout <PyRollout>`, parallelized across CPU threads. For a
detailed treatment of the underlying optimizer, see the least-squares notebook:
|ls_colab|

.. |ls_colab| image:: https://colab.research.google.com/assets/colab-badge.png
              :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb

What can you identify?
----------------------

You can optimize any parameter that affects the simulated sensor outputs.
Common targets fall into two categories:

**Physics parameters** are properties of the model itself. Most can be set
directly on the :ref:`MjSpec <PyModelEdit>` via user-provided callbacks, for
example ``spec.joint("j1").damping = p.value[0]``. The toolbox also provides
convenience functions for parameterizations that are less straightforward:

.. list-table::
   :width: 90%
   :align: left
   :widths: 3 5
   :header-rows: 1

   * - Target
     - Approach
   * - Body mass
     - ``body_inertia_param(..., InertiaType.Mass)``
   * - Body mass + center of mass
     - ``body_inertia_param(..., InertiaType.MassIpos)``
   * - Full inertia (10-D pseudo-inertia)
     - ``body_inertia_param(..., InertiaType.Pseudo)``
   * - Actuator P/D gains
     - ``apply_pdgain(spec, "act1", p.value)``

**Measurement parameters** are properties of the sensing pipeline rather than
the physics, such as sensor delays, gains, and biases. These corrections are
applied to the residual *after* rollout via a user-provided ``modify_residual``
callback that post-processes the predicted and measured signals before the
optimizer sees them.

.. _SysIdIdentifiability:

Identifiability
---------------

Not all parameters can be uniquely determined from a given experiment, a
challenge known as `structural identifiability
<https://en.wikipedia.org/wiki/Structural_identifiability>`__. For example, if
a cart is driven by a motor with torque constant :math:`b` and mass :math:`m`,
the response depends only on the ratio :math:`b/m`. Doubling both produces
identical trajectories, so the individual values cannot be recovered regardless
of data quality. The toolbox flags such cases as infinite confidence intervals
after optimization.

In practice, identifiability depends on both the model structure and the
experimental data. Recording multiple trajectories under varied conditions
(e.g. with known mass perturbations) can resolve some ambiguities, and the
framework supports this natively through multiple sequences per
``ModelSequences``. However, some parameter combinations may remain
fundamentally unidentifiable and require reparameterization or additional prior
knowledge.

.. _SysIdWorkflow:

Workflow
========

An end-to-end identification has four stages: define parameters, package data,
build and run the optimizer, then inspect results.

Before using the toolbox, you need recorded trajectories from your system. This
means designing excitation trajectories that sufficiently exercise the dynamics
of interest, executing them on your robot (or test rig), and logging the control
inputs and sensor readings. Excitation design is problem-specific and outside the
scope of this toolbox, but case studies identifying models from
`MuJoCo Menagerie <https://github.com/google-deepmind/mujoco_menagerie>`__ will
be provided as practical examples.

The following walks through each stage of the toolbox itself.

Defining parameters
-------------------

Each unknown quantity is represented as a :ref:`Parameter <SysIdParameter>` with
a nominal value, bounds, and a *modifier* callback that writes the parameter
into an ``MjSpec`` during each optimization step. Parameters are collected into
a :ref:`ParameterDict <SysIdParameter>`:

.. code-block:: python

   import mujoco
   from mujoco import sysid

   spec = mujoco.MjSpec.from_file("robot.xml")
   model = spec.compile()

   params = sysid.ParameterDict()

   # A simple scalar parameter with a direct MjSpec setter.
   def set_link1_mass(spec, p):
       spec.body("link1").mass = p.value[0]

   params.add(sysid.Parameter(
       "link1_mass", nominal=2.0, min_value=0.5, max_value=5.0,
       modifier=set_link1_mass))

   # Or use a convenience factory for inertia parameters.
   params.add(sysid.body_inertia_param(
       spec, model, "link2", inertia_type=sysid.InertiaType.MassIpos))

Packaging measured data
-----------------------

Measured signals (controls sent to the robot, sensor readings recorded) are
wrapped in :ref:`TimeSeries <SysIdTimeSeries>` objects. A ``TimeSeries`` pairs a
data array with a *signal mapping* that records which columns correspond to
which named signals. This is necessary because the column ordering of your
logged data may differ from MuJoCo's internal sensor ordering, and because a
single recording may contain sensors of different types and dimensions. The
factory methods resolve this automatically from the model:

.. code-block:: python

   control = sysid.TimeSeries.from_control_names(times, ctrl_array, model)
   measured = sysid.TimeSeries.from_names(times, meas_array, model)

The optimizer also needs the initial state of the robot at the start of each
recording. This is an :ref:`mjSTATE_FULLPHYSICS <siFullPhysics>` vector that
captures the complete physics state (joint positions, velocities, and actuator
activations). For fixed-base robots this is usually straightforward, but
floating-base systems require estimating the freejoint pose, which is
non-trivial (todo: talk about this more).
A helper builds this vector from individual components:

.. code-block:: python

   initial_state = sysid.create_initial_state(model, qpos_0, qvel_0)

Everything is then bundled into a
:ref:`ModelSequences <SysIdModelSequences>`, one ``MjSpec`` paired with one
or more recorded trajectories:

.. code-block:: python

   ms = sysid.ModelSequences(
       "robot", spec, "traj_1", initial_state, control, measured)

Optimizing
----------

``build_residual_fn`` wires together the model, data, and any custom hooks into
a single callable that the optimizer can evaluate. ``optimize`` runs the
least-squares solver:

.. code-block:: python

   residual_fn = sysid.build_residual_fn(models_sequences=[ms])
   opt_params, opt_result = sysid.optimize(
       initial_params=params, residual_fn=residual_fn)

Inspecting results
------------------

``save_results`` writes parameter values (YAML), the identified model XMLs,
confidence intervals, and the raw optimization result to disk.
``default_report`` generates an interactive HTML report with rendered videos,
measurement-vs-simulation overlays, parameter comparison tables, and
covariance matrices. The :ref:`tutorial notebook <SysIdNotebook>` shows an
example of this report embedded directly in Colab.

.. code-block:: python

   sysid.save_results(
       "results/", [ms], params, opt_params, opt_result, residual_fn)
   sysid.default_report(
       [ms], params, opt_params, residual_fn, opt_result,
       save_path="results/report.html")

.. _SysIdCoreAPI:

Core API
========

This section explains the key classes and how they fit together. For complete
method signatures and docstrings, see the :doc:`API reference <sysid_api>`.

.. _SysIdParameter:

Parameter and ParameterDict
----------------------------

A ``Parameter`` represents a single unknown quantity to be identified. It holds
a current value, a nominal (initial) value, box bounds, and an optional
*modifier* callback. The modifier is how the parameter gets written into the
model: during each optimization step, the framework calls
``modifier(spec, param)`` so the user can set the appropriate ``MjSpec`` field.

.. note::

   Per-parameter modifiers work well when each parameter independently maps to
   a single ``MjSpec`` field. When parameters interact or you want all the
   application logic in one place, you can omit modifiers and instead write a
   custom ``build_model`` callback that reads values directly from the
   ``ParameterDict`` by name. See :ref:`Extension Points <SysIdExtensionPoints>`
   for details. The case studies all use this approach.

Parameters are collected into a ``ParameterDict``, which behaves like a
``dict[str, Parameter]`` but also provides vectorized access for the optimizer.
The optimizer works with flat arrays internally; ``as_vector()`` and
``update_from_vector()`` handle the conversion. Parameters marked as ``frozen``
are silently excluded from optimization but remain in the dictionary for
bookkeeping.

.. code-block:: python

   params = sysid.ParameterDict()

   # Scalar parameter with a direct MjSpec modifier.
   params.add(sysid.Parameter(
       "joint_damping", nominal=0.5, min_value=0.01, max_value=2.0,
       modifier=lambda spec, p: setattr(spec.joint("j1"), "damping", p.value[0]),
   ))

   # Multi-dimensional parameter (e.g. 5-element friction vector).
   params.add(sysid.Parameter(
       "friction", nominal=[0.5, 0, 0, 0, 0],
       min_value=[0.1, 0, 0, 0, 0], max_value=[1.0, 0, 0, 0, 0],
       modifier=lambda spec, p: setattr(spec.pair("cp"), "friction", p.value),
   ))

.. _SysIdTimeSeries:

TimeSeries
----------

``TimeSeries`` is an immutable container for timestamped signal data. It pairs
a 2-D array (time x channels) with a *signal mapping* that records which
columns correspond to which named signals and what type they are
(``SignalType.MjSensor``, ``MjCtrl``, ``MjStateQPos``, etc.).

There are three factory methods, depending on where your data comes from:

- ``from_names(times, data, model)``: for sensor data where columns are
  ordered the same as the model's sensors (the common case).
- ``from_control_names(times, data, model)``: same, for control inputs.
- ``from_custom_map(times, data, signals)``: when you need to specify the
  signal mapping explicitly, for example if your logged data has a different
  column layout than MuJoCo expects.

``TimeSeries`` also provides ``resample()`` and ``interpolate()`` for aligning
data recorded at different rates to the simulation timestep.

.. _SysIdModelSequences:

ModelSequences
--------------

``ModelSequences`` bundles a single ``MjSpec`` with one or more recorded
trajectories. Each trajectory consists of an initial state, a control
``TimeSeries``, and a measured sensor ``TimeSeries``. Providing multiple
trajectories for the same model improves identifiability by giving the
optimizer diverse excitation data.

.. code-block:: python

   ms = sysid.ModelSequences(
       name="robot",
       spec=spec,
       sequence_name=["walk_1", "walk_2"],
       initial_state=[state_1, state_2],
       control=[ctrl_ts_1, ctrl_ts_2],
       sensordata=[meas_ts_1, meas_ts_2],
   )

.. _SysIdSystemTrajectory:

SystemTrajectory
----------------

A ``SystemTrajectory`` encapsulates a single rolled-out trajectory: the
``MjModel`` used, the control and sensordata ``TimeSeries``, the initial state,
and optionally the full state trajectory. These are constructed by the framework
during optimization and passed to ``modify_residual`` callbacks. Users typically
do not create them directly, but may inspect them when writing custom residual
logic.

.. _SysIdResidualOptimization:

Residual and Optimization
=========================

.. _SysIdBuildResidualFn:

build_residual_fn
-----------------

``build_residual_fn`` wires together the model, data, and any custom hooks into
a single callable that the optimizer can evaluate. It returns a closure with
signature ``fn(x, params, **overrides)`` that internally applies parameters to
the model, rolls out trajectories, and computes residuals.

.. code-block:: python

   residual_fn = sysid.build_residual_fn(
       models_sequences=[ms],
       build_model=my_build_model,        # optional
       modify_residual=my_modify_residual, # optional
       custom_rollout=my_custom_rollout,   # optional
   )

The optional callbacks are the primary way to customize the residual pipeline;
see :ref:`Extension Points <SysIdExtensionPoints>` for details.

.. _SysIdOptimize:

optimize
--------

``optimize`` runs nonlinear least-squares optimization on the residual.

.. code-block:: python

   opt_params, opt_result = sysid.optimize(
       initial_params=params,
       residual_fn=residual_fn,
       optimizer="mujoco",  # or "scipy", "scipy_parallel_fd"
   )

Three backends are available:

- ``"mujoco"`` (default): uses :ref:`mujoco.minimize <PyMinimize>` with
  batched finite differences via :ref:`mujoco.rollout <PyRollout>`.
- ``"scipy"``: uses
  `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`__.
- ``"scipy_parallel_fd"``: uses scipy with parallelized finite differences.

Returns ``(opt_params, opt_result)``, an optimized ``ParameterDict`` and a
``scipy.optimize.OptimizeResult``.

.. _SysIdConfidenceIntervals:

Confidence intervals
--------------------

``calculate_intervals`` computes parameter covariance and confidence bounds
from the Jacobian at the optimum:

.. code-block:: python

   Sigma_X, intervals = sysid.calculate_intervals(
       residuals_star=opt_result.fun,
       J=opt_result.jac,
       alpha=0.05,
   )

The covariance matrix is derived from the Gauss-Newton Hessian approximation
:math:`J^\top J`. Confidence intervals use the t-distribution with
``n_residuals - n_params`` degrees of freedom.

.. note::

   Infinite confidence intervals are not a bug. They indicate that the
   parameter is unidentifiable from the available data, either because the
   residual is insensitive to that parameter or because of a structural
   ambiguity (see :ref:`Identifiability <SysIdIdentifiability>`). This is a
   useful diagnostic: consider freezing, removing, or reparameterizing such
   parameters.

.. _SysIdConvenienceFunctions:

Convenience Functions
=====================

.. _SysIdInertiaParam:

Inertia parameterization
------------------------

``body_inertia_param`` is a factory that creates a ``Parameter`` for a body's
inertial properties, with the appropriate modifier and bounds already
configured. The ``InertiaType`` enum controls the parameterization:

- ``Mass``: mass only (1-D).
- ``MassIpos``: mass and center-of-mass position (4-D).
- ``Pseudo``: full pseudo-inertia via log-Cholesky parameterization of the
  pseudo-inertia matrix (10-D), guaranteeing that the pseudo-inertia matrix is
  positive definite, ensuring full physical consistency without singularities
  (`Rucker & Wensing 2022 <https://ieeexplore.ieee.org/document/9690029>`__).

.. code-block:: python

   param = sysid.body_inertia_param(
       spec, model, "link1", inertia_type=sysid.InertiaType.Pseudo)

.. note::

   ``Pseudo`` is recommended when identifying the full rotational inertia. The
   log-Cholesky parameterization guarantees that the pseudo-inertia matrix is
   positive definite, ensuring full physical consistency, which ``Mass`` and
   ``MassIpos`` do not enforce.

.. _SysIdPDGainHelpers:

PD gain helpers
---------------

Convenience modifiers for actuator gains, commonly used inside a ``build_model``
callback:

.. code-block:: python

   def build_model(params, spec):
       sysid.apply_pdgain(spec, "shoulder", params["shoulder_gains"].value)
       sysid.apply_pgain(spec, "gripper", params["gripper_kp"].value[0])
       return spec.compile()

.. _SysIdModelUtilities:

Model utilities
---------------

``remove_visuals(spec)`` strips visual geoms from an ``MjSpec`` for faster
compilation during optimization loops.

.. _SysIdSignalCorrections:

Signal Corrections
==================

Real sensor data often requires corrections before it can be meaningfully
compared to simulation output. Sensors may have delays (the measured signal
lags behind the true state), multiplicative gain errors (a torque sensor reads
10% high), or additive biases (e.g. joint encoder offsets). The ``modify_residual`` callback is the place
to apply these corrections.

A ``modify_residual`` function receives the predicted and measured sensor
``TimeSeries`` after rollout and returns the final residual array. Inside it,
you can use the signal processing utilities provided by the toolbox:

- ``apply_delay(data, times, delay)``: shift a signal forward in time to
  compensate for sensor latency.
- ``apply_gain(data, gain)``: multiply signal columns by a gain vector.
- ``apply_bias(data, bias)``: add a bias vector to signal columns.
- ``weighted_diff(predicted, measured, weights)``: compute the weighted
  difference between predicted and measured signals.
- ``normalize_residual(residual)``: normalize a residual vector by its
  root-mean-square.
- ``apply_resample_and_delay(predicted, measured, delay, model)``: a
  convenience that resamples and applies delay in one step.

The following example shows a ``modify_residual`` that applies per-sensor
delays to position sensors and a gain correction to torque sensors:

.. code-block:: python

   def modify_residual(params, sensordata_predicted, sensordata_measured,
                        model, return_pred_all, **kwargs):
       pred = sensordata_predicted
       meas = sensordata_measured

       # Apply delay to position sensors.
       pos_idx = sysid.get_sensor_indices(model, ["joint1_pos", "joint2_pos"])
       pred_data = pred.data.copy()
       pred_data[:, pos_idx] = sysid.apply_delay(
           pred.data[:, pos_idx], pred.times, params["pos_delay"].value[0])

       # Apply gain correction to torque sensors.
       torque_idx = sysid.get_sensor_indices(model, ["joint1_torque"])
       pred_data[:, torque_idx] = sysid.apply_gain(
           pred_data[:, torque_idx], params["torque_scale"].value)

       residual = sysid.weighted_diff(pred_data, meas.data)
       residual = sysid.normalize_residual(residual)

       pred_out = sysid.TimeSeries.from_custom_map(
           pred.times, pred_data, pred.signal_mapping)
       return residual, pred_out, meas

   residual_fn = sysid.build_residual_fn(
       models_sequences=[ms],
       modify_residual=modify_residual,
   )

.. _SysIdReportingIO:

Reporting and I/O
=================

``save_results`` writes optimization artifacts to disk, including parameter
values (YAML), the identified model XMLs, confidence intervals, and the raw
``OptimizeResult`` (pickle).

``default_report`` generates an interactive HTML report with rendered videos,
measurement-vs-simulation overlays, parameter comparison tables, and covariance
matrices. The :ref:`tutorial notebook <SysIdNotebook>` shows an example of this
report embedded directly in Colab.

``render_rollout`` renders trajectories to RGB frames for visualization or
video export.

.. code-block:: python

   sysid.save_results(
       "results/", [ms], params, opt_params, opt_result, residual_fn)

   sysid.default_report(
       [ms], params, opt_params, residual_fn, opt_result,
       save_path="results/report.html")

   frames = sysid.render_rollout(
       model, data, state=state_array, framerate=30,
       camera="front", width=640, height=480)

.. _SysIdExtensionPoints:

Extension Points
================

The residual pipeline can be customized at three levels via callbacks passed to
``build_residual_fn``.

build_model
-----------

``BuildModelFn`` controls how parameters are applied to the model before each
rollout. The default applies each parameter's ``modifier`` callback and compiles
the spec. A custom ``build_model`` is useful when parameters interact (e.g. PD
gains that share structure across joints) or when you need to apply parameters
that don't map to a single ``MjSpec`` field:

.. code-block:: python

   def build_model(params, spec):
       # Apply PD gains with per-joint overrides.
       for name in ["hip", "knee", "ankle"]:
           sysid.apply_pdgain(spec, name, params[f"{name}_gains"].value)

       # Apply pseudo-inertia for multiple bodies.
       for body in ["thigh", "shin", "foot"]:
           sysid.apply_body_inertia(spec, body, params[f"{body}_inertia"])

       # Set armature and frictionloss per joint.
       for jnt in spec.joints:
           if jnt.name in params:
               jnt.armature = params[jnt.name].value[0]

       return spec.compile()

modify_residual
---------------

``ModifyResidualFn`` post-processes the residual after rollout. This is where
measurement corrections (delays, gains, biases) are applied, as described in
:ref:`Signal Corrections <SysIdSignalCorrections>`. It is also useful for
selecting a subset of sensors to include in the residual, or for applying
custom weighting.

custom_rollout
--------------

``CustomRolloutFn`` replaces the default ``sysid_rollout`` entirely. This is
the most powerful extension point, useful when the standard open-loop rollout
is not appropriate.

.. warning::

   A custom rollout bypasses the batched :ref:`mujoco.rollout <PyRollout>`
   parallelism. Optimization will be significantly slower because
   finite-difference perturbations are no longer evaluated in parallel.

For example, identifying the parameters of a quadrotor
with an inner-loop controller requires running the controller inside the
rollout loop rather than replaying recorded controls directly:

.. code-block:: python

   def custom_rollout(models, datas, control_signal, initial_states,
                       param_dicts, **kwargs):
       trajectories = []
       for model, data, ctrl, state in zip(
               models, datas, control_signal, initial_states):
           # Reset to initial state.
           mujoco.mj_resetData(model, data)
           data.qpos[:] = state[:model.nq]
           data.qvel[:] = state[model.nq:model.nq + model.nv]

           # Run with inner-loop controller.
           for t in range(ctrl.data.shape[0]):
               data.ctrl[:] = my_controller(model, data, ctrl.data[t])
               mujoco.mj_step(model, data)
               # ... record sensordata ...

           trajectories.append(sysid.SystemTrajectory(...))
       return trajectories

.. _SysIdDependencies:

Dependencies
============

The sysid toolbox builds on other ``mujoco`` sub-modules:

- :ref:`mujoco.rollout <PyRollout>`: batched trajectory rollouts used
  internally for computing residuals and finite-difference Jacobians.
- :ref:`mujoco.minimize <PyMinimize>`: nonlinear least-squares optimizer
  used as the default backend.
- :ref:`MjSpec <PyModelEdit>`: the model editing API used to modify model
  parameters during optimization.

Users do not need to invoke these directly.

.. _SysIdLimitations:

Limitations
===========

.. todo(kevin): flesh out limitations section

- The toolbox currently supports open-loop rollouts by default. Closed-loop
  identification (e.g. with a feedback controller in the loop) requires a
  ``custom_rollout``.
- Finite-difference Jacobians scale linearly with the number of parameters.
- The optimizer assumes smooth residuals. Contact-rich scenarios with
  discontinuous dynamics may cause convergence difficulties.

.. todo(kevin): add more limitations
