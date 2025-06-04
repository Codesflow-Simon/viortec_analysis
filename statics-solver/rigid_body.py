import numpy as np
from reference_frame import Point, ReferenceFrame
import sympy

class Force:
    def __init__(self, name: str, force: Point, application_point: Point):
        """
        A force is a vector that is applied to a point.
        Args:
            name: The name of the force.
            force: The force vector.
            application_point: The point at which the force is applied.
        """
        self.name = name
        self.force = force
        self.application_point = application_point

    def get_moment(self):
        return self.force.cross(self.application_point)

    def convert_to_frame(self, frame: ReferenceFrame):
        self.force = self.force.convert_to_frame(frame)
        self.application_point = self.application_point.convert_to_frame(frame)

    def get_force_in_frame(self, frame: ReferenceFrame):
        force = self.force.convert_to_frame(frame)
        application_point = self.application_point.convert_to_frame(frame)
        return Force(self.name, force, application_point)

    def __neg__(self):
        return Force(self.name, -self.force, self.application_point)

    def __str__(self):
        return f"Force: {self.name}, Vector: {self.force}, Application Point: {self.application_point}"
    
    def __repr__(self):
        return f"Force: {self.name}, Vector: {self.force}, Application Point: {self.application_point}"

class Torque:
    def __init__(self, name: str, torque: Point):
        self.name = name
        self.torque = torque

    def convert_to_frame(self, frame: ReferenceFrame):
        self.torque = self.torque.convert_to_frame(frame)

    def __neg__(self):
        return Torque(self.name, -self.torque)

class RigidBody:
    def __init__(self, name: str, reference_frame: ReferenceFrame):
        """
        A rigid body is a collection of points that are connected by rigid constraints.
        Assumes the center of mass is at the origin of the reference frame.
    
        Args:
            name: The name of the body.
            reference_frame: The reference frame of the body.
        """
        self.name = name
        self.body_frame = reference_frame
        self.forces = []
        self.torques = []

    def add_external_force(self, force: Force):
        if force is not None:
            self.forces.append(force)

    def add_external_torque(self, torque: Torque):
        if torque is not None:
            self.torques.append(torque)

    def add_force_pair(self, force: Force, other_body: "RigidBody"):
        """
        Add a force to the body and negative the force on the other body.
        """
        if force is not None:
            self.forces.append(force)
            other_body.forces.append(-force)

    def add_torque_pair(self, torque: Torque, other_body: "RigidBody"):
        """
        Add a torque to the body and negative the torque on the other body.
        """
        if torque is not None:
            self.torques.append(torque)
            other_body.torques.append(-torque)

    def get_force_by_name(self, name: str):
        for force in self.forces:
            if force.name == name:
                return force
        return None

    def get_net_forces(self):
        """
        Calculates the net force and net moment vectors acting on the rigid body,
        expressed in the body_frame. All component forces and torques are converted
        to the body_frame before summation.

        Returns:
            tuple: (net_force_coords, net_moment_coords)
                   net_force_coords (sympy.Matrix): 3x1 matrix of net force components.
                   net_moment_coords (sympy.Matrix): 3x1 matrix of net moment components.
        """
        net_force_coords = sympy.zeros(3, 1)
        net_moment_coords = sympy.zeros(3, 1)

        # Sum forces and moments from Force objects
        for force_obj in self.forces:
            # Ensure force vector is in body_frame
            if force_obj.force.reference_frame != self.body_frame:
                current_force_in_body_frame = force_obj.force.convert_to_frame(self.body_frame)
            else:
                current_force_in_body_frame = force_obj.force
            
            net_force_coords += current_force_in_body_frame.coordinates

            # Ensure application point is in body_frame for moment calculation
            if force_obj.application_point.reference_frame != self.body_frame:
                application_point_in_body_frame = force_obj.application_point.convert_to_frame(self.body_frame)
            else:
                application_point_in_body_frame = force_obj.application_point
            
            # Moment of this force about the body_frame origin: r x F
            # Both r and F must be in the body_frame
            moment_of_force = application_point_in_body_frame.coordinates.cross(current_force_in_body_frame.coordinates)
            net_moment_coords += moment_of_force

        # Sum torques from Torque objects
        for torque_obj in self.torques:
            print(f"Torque: {torque_obj}")
            # Ensure torque vector is in body_frame
            if torque_obj.torque.reference_frame != self.body_frame:
                current_torque_in_body_frame = torque_obj.torque.convert_to_frame(self.body_frame)
            else:
                current_torque_in_body_frame = torque_obj.torque
            
            net_moment_coords += current_torque_in_body_frame.coordinates

        return net_force_coords, net_moment_coords

        
        