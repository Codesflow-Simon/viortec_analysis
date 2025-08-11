# Ligament solver

## Current capabilities

### Kinematics and statics solving
The folder `statics_solver` contains code for solving kinematic and staic equations, a *knee* is made up of a set of refernce frames, connected by joints or springs. I will cover the details of this library below. This library is all built in SymPy for symbolic computation.

#### Model
There is one model right now defined in `statics_model.py`, this is based on a two-ball joint with ligaments.

#### Reference frame
The class `ReferenceFrame` defined in `reference_frame.py`. A reference frame conisists of a (possibly empty) list of child frames, and an optional map to its parent called `rigid_body_mapping`, which is a `RigidBodyMap`. Map definitions are made in `mappings.py`. A key method for reference frames is `find_common_ancestor` which will search the family tree for the youngest common ancesetor of two maps (that may be one of the provided maps), this is useful for converting reference frames.

In `reference_frame.py` we also define the class `Point`, which is a reference frame combined with a 3D vector. This these classes have a few methods, one of the most important is `convert_to_frame` which converts the the vector into a different frame. This class impliments standard R^3 arithmatic, but only when frames between vectors are shared.

#### Rigid bodies 
A rigid body (defined in `rigid_body.py`) is a reference frame that also tracks a list of torques and vectors, which likely contain symbolic expressions. Calling `get_net_forces` will return the sum force-torque expressions needed to be solved at zero for statics. The force expression is just the sum of the forces while the torque expression sums torques, and adds the created torque from an applied force.

`rigid_body.py` also defines forces and torques. A `Force` consists of a `Point` tracking the force vector, and a `Point` tracking the application point. A `Torque` is just a torque wrapped `Point` vector since we do not track the application points.

#### Joints
Joints are defined in `joint_models.py` which is a systematic way of appling equal and opposite forces between rigid bodies. Construct the joint using the reference frames of the rigid bodies, then calling `get_contact_point` and `get_constraint_force` you can get the requred joint-forces.

A joint is parameterised by theta, though the method `set_theta`. Implimentations of `set_theta` should call `set_mapping` which defines the relative translation and rotation for a given theta. `PivotJoint` will define a rotation matrix corresponding to theta, and no transaltion (pivoting around 0) in the provided parent frame. `TwoBallJoint` constructs a more complex joint, for details about this see the code.

#### Springs
`Springs` are defined using two spring endpoints and some possible parameters. `AbstractSpring` handles the details and interfaces of the implimentiation. By calling `get_force_on_point1` (or `get_force_on_point2`) you can fetch the expression for the force that a spring applies on the points. Implimentations of the spring should impliment `get_force_magnitude` which should call `get_spring_length()` and define force as a function of that. Tension is positive.

#### Other details
`mappings.py` defines symbolic expressions of many key functions we require, and `visualiser.py` allows for plots to be made. An example model can be seen in `statics_model.py`.

### Ligament reconstruction
If force-elongation data is gathered, we may with to model it. We model in a Bayesian approach.