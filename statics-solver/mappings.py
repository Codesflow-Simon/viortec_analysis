import numpy as np
import warnings
import sympy


class Mapping():
    def apply(self, vector):
        raise NotImplementedError("Subclasses must implement this method")

    def __matmul__(self, other):
        return self.apply(other)

class InvertibleMapping(Mapping):
    def inverse_apply(self, vector):
        if not isinstance(vector, sympy.Matrix) and hasattr(vector, 'shape') and vector.shape == (3,):
            vector = sympy.Matrix(vector)
        return self.get_inverse().apply(vector)

    def get_inverse(self):
        raise NotImplementedError("Subclasses must implement this method")

class IdentityMapping(InvertibleMapping):
    def apply(self, vector):
        return vector

    def inverse_apply(self, vector):
        return vector

class LinearMapping(InvertibleMapping):
    def __init__(self, matrix):
        self.update_matrix(matrix)
        self.inverse_matrix = None

    def realise_inverse(self):
        if self.inverse_matrix is None:
            if isinstance(self.matrix, np.ndarray):
                self.matrix = sympy.Matrix(self.matrix)
            self.inverse_matrix = self.matrix.inv()

    def update_matrix(self, matrix):
        if isinstance(matrix, np.ndarray):
            matrix = sympy.Matrix(matrix)
        if not isinstance(matrix, sympy.Matrix):
            raise TypeError("Matrix must be a sympy.Matrix or numpy array")
        
        self.matrix = matrix

    def apply(self, vector):
        if not isinstance(vector, sympy.Matrix) and hasattr(vector, 'shape'):
            if isinstance(vector, np.ndarray) and vector.shape == (3,):
                 vector = sympy.Matrix(vector).reshape(3,1)
            elif isinstance(vector, (list, tuple)) and len(vector) == 3:
                 vector = sympy.Matrix(vector).reshape(3,1)
        
        if isinstance(self.matrix, np.ndarray):
            self.matrix = sympy.Matrix(self.matrix)
            
        return self.matrix * vector

    def inverse_apply(self, vector):
        self.realise_inverse()
        if not isinstance(vector, sympy.Matrix) and hasattr(vector, 'shape'):
            if isinstance(vector, np.ndarray) and vector.shape == (3,):
                 vector = sympy.Matrix(vector).reshape(3,1)
            elif isinstance(vector, (list, tuple)) and len(vector) == 3:
                 vector = sympy.Matrix(vector).reshape(3,1)
        
        if isinstance(self.inverse_matrix, np.ndarray):
            self.inverse_matrix = sympy.Matrix(self.inverse_matrix)
            
        return self.inverse_matrix * vector

    def get_inverse(self):
        self.realise_inverse()
        if isinstance(self.inverse_matrix, np.ndarray):
            return LinearMapping(sympy.Matrix(self.inverse_matrix))
        return LinearMapping(self.inverse_matrix)

    def __str__(self):
        return f"LinearMapping: {self.matrix}"
    
    def __repr__(self):
        return f"LinearMapping: {self.matrix}"

class RotationalMapping(LinearMapping):
    def __init__(self, matrix=None):
        super().__init__(matrix)

        if matrix is None:
            matrix = sympy.eye(3)
        self.matrix = matrix

        identity_matrix = sympy.eye(3)
        try:
            is_orthogonal = sympy.simplify(self.matrix.T * self.matrix - identity_matrix).is_zero_matrix
            if not is_orthogonal:
                 # Fallback for numeric matrices with small float errors if symbolic simplify fails
                 if hasattr(self.matrix, 'is_symbolic') and not self.matrix.is_symbolic():
                     # Check if the norm of the difference is small for numeric matrices
                     diff_matrix = (self.matrix.T.evalf() * self.matrix.evalf()) - identity_matrix.evalf()
                     if not (diff_matrix.norm() < 1e-8):
                         raise ValueError("Matrix is not orthogonal - not a valid rotation matrix")
                 else: # If symbolic and simplify didn't prove it, raise
                    raise ValueError("Matrix is not orthogonal (symbolic check failed) - not a valid rotation matrix")

            det_matrix = self.matrix.det()
            is_det_one = sympy.simplify(det_matrix - 1) == 0
            if not is_det_one:
                if hasattr(det_matrix, 'is_symbolic') and not det_matrix.is_symbolic():
                    if not sympy.Abs(det_matrix.evalf() - 1) < 1e-8:
                        raise ValueError("Matrix determinant is not 1 - not a valid rotation matrix")
                else:
                    raise ValueError("Matrix determinant is not 1 (symbolic check failed) - not a valid rotation matrix")

        except AttributeError:
            pass

        for i in range(3):
            col_vec = self.matrix[:, i]
            norm_sq = col_vec.norm()**2
            is_unit_len = sympy.simplify(norm_sq - 1) == 0
            if not is_unit_len:
                 if hasattr(norm_sq, 'is_symbolic') and not norm_sq.is_symbolic():
                     if not sympy.Abs(norm_sq.evalf() - 1) < 1e-8:
                         raise ValueError(f"Column {i} is not a unit vector (numeric check failed)")
                 else:
                    pass

    @staticmethod
    def from_euler_angles(euler_angles: np.ndarray):
        if isinstance(euler_angles, np.ndarray):
            euler_angles = list(euler_angles)

        if not isinstance(euler_angles, (list, tuple)) or len(euler_angles) != 3 :
             if isinstance(euler_angles, sympy.Matrix) and euler_angles.shape in [(3,1), (1,3)]:
                 if euler_angles.shape == (1,3):
                     euler_angles = euler_angles.T
                 euler_angles = [euler_angles[i,0] for i in range(3)]
             else:
                raise TypeError("euler_angles must be a 3-element list/tuple or sympy Matrix")

        roll, pitch, yaw = euler_angles
        
        Rx = sympy.Matrix([[1, 0, 0],
                           [0, sympy.cos(roll), -sympy.sin(roll)],
                           [0, sympy.sin(roll), sympy.cos(roll)]])
                      
        Ry = sympy.Matrix([[sympy.cos(pitch), 0, sympy.sin(pitch)],
                           [0, 1, 0],
                           [-sympy.sin(pitch), 0, sympy.cos(pitch)]])
                      
        Rz = sympy.Matrix([[sympy.cos(yaw), -sympy.sin(yaw), 0],
                           [sympy.sin(yaw), sympy.cos(yaw), 0],
                           [0, 0, 1]])
                      
        R = Rz * Ry * Rx
        return RotationalMapping(matrix=R)

    @staticmethod
    def from_quaternion(q):
        if isinstance(q, np.ndarray):
            q = q.tolist()
            
        w, x, y, z = q
        R = sympy.Matrix([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return RotationalMapping(matrix=R)

    def apply(self, vector):
        if not isinstance(vector, sympy.Matrix):
            if isinstance(vector, (list, tuple)) and len(vector) == 3:
                vector = sympy.Matrix(vector).reshape(3,1)
            elif isinstance(vector, np.ndarray) and vector.shape == (3,):
                vector = sympy.Matrix(list(vector)).reshape(3,1)
        
        if vector.shape != (3,1):
            if vector.shape == (1,3):
                vector = vector.T
            elif vector.shape == (3,):
                 vector = sympy.Matrix(vector).reshape(3,1)
            else:
                 raise ValueError(f"Input vector must have shape (3,1) or (1,3) or (3,), got {vector.shape}")
            
        return super().apply(vector)

class TranslationMapping(InvertibleMapping):
    def __init__(self, vector: np.ndarray):
        if isinstance(vector, np.ndarray):
            vector = sympy.Matrix(list(vector))
        elif isinstance(vector, (list,tuple)):
            vector = sympy.Matrix(vector)

        if not isinstance(vector, sympy.Matrix):
            raise TypeError("Vector must be a sympy.Matrix, numpy array, list or tuple")
        
        if vector.shape != (3,1) and vector.shape != (1,3):
             if vector.shape == (3,):
                 vector = vector.reshape(3,1)
             else:
                raise ValueError(f"Vector must be a 3D vector (3x1 or 1x3 Matrix), got shape {vector.shape}")

        if vector.shape == (1,3):
            vector = vector.T
        
        self.vector = vector

    def apply(self, vector_in):
        if not isinstance(vector_in, sympy.Matrix):
            if isinstance(vector_in, (list, tuple)) and len(vector_in) == 3:
                vector_in = sympy.Matrix(vector_in).reshape(3,1)
            elif isinstance(vector_in, np.ndarray) and vector_in.shape == (3,):
                vector_in = sympy.Matrix(list(vector_in)).reshape(3,1)
        
        if vector_in.shape != (3,1):
             if vector_in.shape == (1,3):
                 vector_in = vector_in.T
             elif vector_in.shape == (3,):
                 vector_in = sympy.Matrix(vector_in).reshape(3,1)
             else:
                raise ValueError(f"Input vector must have shape (3,1) or (1,3) or (3,), got {vector_in.shape}")

        return vector_in + self.vector

    def inverse_apply(self, vector_in):
        if not isinstance(vector_in, sympy.Matrix):
            if isinstance(vector_in, (list, tuple)) and len(vector_in) == 3:
                vector_in = sympy.Matrix(vector_in).reshape(3,1)
            elif isinstance(vector_in, np.ndarray) and vector_in.shape == (3,):
                vector_in = sympy.Matrix(list(vector_in)).reshape(3,1)

        if vector_in.shape != (3,1):
             if vector_in.shape == (1,3):
                 vector_in = vector_in.T
             elif vector_in.shape == (3,):
                 vector_in = sympy.Matrix(vector_in).reshape(3,1)
             else:
                raise ValueError(f"Input vector must have shape (3,1) or (1,3) or (3,), got {vector_in.shape}")
        
        return vector_in - self.vector

    def get_inverse(self):
        return TranslationMapping(-self.vector)

    def __str__(self):
        return f"TranslationMapping: {self.vector}"
    
    def __repr__(self):
        return f"TranslationMapping: {self.vector}"

class RigidBodyMapping(InvertibleMapping):
    def __init__(self, rotation: RotationalMapping, translation: TranslationMapping):
        self.rotation = rotation
        self.translation = translation

    def apply(self, vector):
        rotated_vector = self.rotation.apply(vector)
        return self.translation.apply(rotated_vector)

    def inverse_apply(self, vector):
        translated_vector = self.translation.inverse_apply(vector)
        return self.rotation.inverse_apply(translated_vector)

    def get_inverse(self):
        return RigidBodyMapping(self.rotation.get_inverse(), self.translation.get_inverse())

    def __str__(self):
        return f"RigidBodyMapping: {self.rotation} + {self.translation}"
    
    def __repr__(self):
        return f"RigidBodyMapping: {self.rotation} + {self.translation}"

class TrilinearMapping(Mapping):
    def __init__(self, a, b, m1, b1, m2, b2, m3, b3):
        self.a = float(a)
        self.b = float(b)
        self.m1 = float(m1)
        self.b1 = float(b1)
        self.m2 = float(m2)
        self.b2 = float(b2)
        self.m3 = float(m3)
        self.b3 = float(b3)
        
        if self.a >= self.b:
            raise ValueError("Breakpoint a must be less than breakpoint b")

    def apply(self, vector):
        if isinstance(vector, sympy.Matrix):
            warnings.warn("TrilinearMapping received a sympy.Matrix, converting to numpy for processing. This may lose symbolic information.")
            vector_np = np.array(vector.evalf()).astype(float)
            if vector_np.shape == (3,1): vector_np = vector_np.flatten()
        elif not isinstance(vector, np.ndarray):
            vector_np = np.array(vector)
        else:
            vector_np = vector
            
        result_np = np.zeros_like(vector_np)
        
        mask1 = vector_np < self.a
        result_np[mask1] = self.m1 * vector_np[mask1] + self.b1
        
        mask2 = (vector_np >= self.a) & (vector_np < self.b)
        result_np[mask2] = self.m2 * vector_np[mask2] + self.b2
        
        mask3 = vector_np >= self.b
        result_np[mask3] = self.m3 * vector_np[mask3] + self.b3
        
        if isinstance(vector, sympy.Matrix):
            return sympy.Matrix(result_np).reshape(vector.shape[0], vector.shape[1])
            
        return result_np
        