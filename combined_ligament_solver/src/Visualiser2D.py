import matplotlib.pyplot as plt
import numpy as np
# It's good practice to explicitly import from your project modules
from .reference_frame import Point, ReferenceFrame 
from .rigid_body import Force as SimForce # Alias to avoid clash
import sympy
from sympy import Symbol, Expr # For checking if values are symbolic
import warnings
import copy
from .springs import AbstractSpring


class Visualiser2D:
    def __init__(self, world_frame: ReferenceFrame, projection_axes=(0, 1)):
        """
        Initialises the 2D visualiser.
        Args:
            world_frame: The designated world reference frame.
            projection_axes: Tuple indicating which 3D axes to use for 2D plot (e.g., (0,1) for XY).
        """
        if not isinstance(world_frame, ReferenceFrame):
            raise TypeError("world_frame must be an instance of ReferenceFrame.")
        self.world_frame = world_frame
        self.projection_axes = projection_axes 
        self.points_to_plot = [] 
        self.lines_to_plot = []  
        self.forces_to_plot = [] 
        self.circles_to_plot = []

        self.color_map = {
            "WorldFrame": "black", # Default for points in world frame
            "TibiaFrame": "blue",  # Example color for points originally in TibiaFrame
            "Femur": "red",        # Example for forces on Femur body
            "Tibia": "green",      # Example for forces on Tibia body
            "DefaultPointColor": "gray",
            "DefaultForceColor": "purple",
            "SpringForceOnFemur": "magenta", # Example specific force color
            "SpringForceOnTibia": "magenta",
            "KneeForce_Solved": "orange",
            "AppliedForce_Solved": "teal"
        }
        # Iterator for assigning new colors to frames/bodies not in the map
        self._color_iterator = iter(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
                                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])


    def _get_color_for_item(self, item_name):
        """Helper to get a color for a frame or body, cycling if not predefined."""
        if item_name in self.color_map:
            return self.color_map[item_name]
        else:
            try:
                new_color = next(self._color_iterator)
                self.color_map[item_name] = new_color # Cache for consistency
                return new_color
            except StopIteration:
                # Reset iterator or use a default if all have been used
                warnings.warn(f"Ran out of unique colors for items. Reusing or using default for '{item_name}'.")
                self._color_iterator = iter(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                                     "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]) # Reset
                return self.color_map["DefaultPointColor"]


    def _convert_point_to_world_numpy(self, point_obj: Point):
        """
        Converts a Point object to its coordinate representation in the world frame 
        as a NumPy array. Ensures coordinates are numerical.
        """
        if not isinstance(point_obj, Point):
            raise TypeError(f"Expected a Point object, got {type(point_obj)}.")

        # Ensure no symbolic values in coordinates
        for i, coord_val in enumerate(point_obj.coordinates):
            if isinstance(coord_val, Expr) and coord_val.has(Symbol):
                 raise ValueError(
                     f"Point coordinate {coord_val} (index {i}) for plotting contains symbolic values. "
                     "Substitute them with numerical values first."
                 )

        point_in_world = point_obj
        if point_obj.reference_frame != self.world_frame:
            point_in_world = point_obj.convert_to_frame(self.world_frame)
        
        # Convert sympy.Matrix to flat numpy array of floats
        try:
            # .evalf() is crucial for SymPy expressions/numbers
            return np.array(point_in_world.coordinates.evalf(chop=True)).astype(float).flatten()
        except Exception as e:
            raise ValueError(f"Failed to convert point coordinates {point_in_world.coordinates} to NumPy array: {e}")


    def add_point(self, point_obj: Point, label: str = None, color_basis_name: str = None):
        """
        Adds a point to be plotted.
        Args:
            point_obj: The Point object (must have numerical coordinates).
            label: Optional text label for the point.
            color_basis_name: Name to use for color mapping (e.g., original frame name or body name).
                              If None, uses the point's current reference frame name.
        """
        coords_np = self._convert_point_to_world_numpy(point_obj)
        
        name_for_color = color_basis_name if color_basis_name else point_obj.reference_frame.name
        actual_label = label if label else point_obj.reference_frame.name # Default label to frame name if not given

        self.points_to_plot.append((coords_np, name_for_color, actual_label))

    def add_line(self, point_obj_start: Point, point_obj_end: Point, label=None, color='gray', linestyle='-'):
        """
        Adds a line between two points.
        Args:
            point_obj_start: The starting Point object.
            point_obj_end: The ending Point object.
            color: Color of the line.
            linestyle: Linestyle for Matplotlib.
        """
        start_coords_np = self._convert_point_to_world_numpy(point_obj_start)
        end_coords_np = self._convert_point_to_world_numpy(point_obj_end)
        self.lines_to_plot.append((start_coords_np, end_coords_np, color, linestyle))

    def add_circle(self, center_point: Point, radius: float, label: str = None, color: str = 'gray', 
                   linestyle: str = '-', fill: bool = False, alpha: float = 0.3):
        """
        Adds a circle to be plotted.
        Args:
            center_point: The Point object representing the center of the circle.
            radius: The radius of the circle in world frame units.
            label: Optional text label for the circle.
            color: Color of the circle outline.
            linestyle: Linestyle for the circle outline.
            fill: Whether to fill the circle.
            alpha: Transparency of the fill (0=transparent, 1=opaque).
        """
        center_coords_np = self._convert_point_to_world_numpy(center_point)
        
        # Store circle parameters for plotting
        circle_data = {
            'center': center_coords_np,
            'radius': radius,
            'label': label,
            'color': color,
            'linestyle': linestyle,
            'fill': fill,
            'alpha': alpha
        }
        
        # Add to a new list for circles
        if not hasattr(self, 'circles_to_plot'):
            self.circles_to_plot = []
        self.circles_to_plot.append(circle_data)

    def add_force(self, force_obj: SimForce, associated_body_name: str = "DefaultBody", label=None):
        """
        Adds a force vector to be plotted.
        Args:
            force_obj: The statics_solver.rigid_body.Force object.
                       Its .force (Point) and .application_point (Point) must have numerical coordinates.
            associated_body_name: Name of the rigid body this force acts upon (for coloring).
        """
        if not isinstance(force_obj, SimForce):
            raise TypeError(f"Expected a statics_solver.rigid_body.Force object, got {type(force_obj)}")

        app_point_np_world = self._convert_point_to_world_numpy(force_obj.application_point)
        
        # The force_obj.force is a Point representing the force vector.
        # Its coordinates are components in its own reference_frame.
        # To plot it correctly in the world frame, we transform this vector.
        # A vector is transformed by applying the rotation matrix of the frame transformation.
        
        force_vec_point = force_obj.force # This is a Point object
        
        # Get the mapping from the force vector's frame to the world frame.
        # This requires traversing up the parent chain from force_vec_point.reference_frame to self.world_frame
        # and composing the rotation matrices.

        current_frame = force_vec_point.reference_frame
        accumulated_rotation_matrix = sympy.eye(3)

        # Use original force vector without frame transformation to avoid scaling issues
        original_force_vec = force_obj.force
        force_vec_np = self._convert_point_to_world_numpy(original_force_vec)
        
        magnitude = np.linalg.norm(force_vec_np)
        magnitude_str = f"{magnitude:.2f}N"
        force_label = force_obj.name
        
        # Debug: Print force information
        print(f"Force {force_label}: magnitude={magnitude:.6f}, vector={force_vec_np}")
        
        self.forces_to_plot.append((app_point_np_world, force_vec_np, associated_body_name, force_label, magnitude_str))


    def render(self, title="2D Static Visualisation", show_grid=True, equal_aspect=False, show_values=True, filename=None, force_scale_factor=None):
        """Renders the plot."""
        fig, ax = plt.subplots(figsize=(10, 8)) # Slightly larger figure
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"World Frame Axis {self.projection_axes[0]} (X)", fontsize=12)
        ax.set_ylabel(f"World Frame Axis {self.projection_axes[1]} (Y)", fontsize=12)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        if equal_aspect:
            ax.set_aspect('equal', adjustable='box')

        min_coord, max_coord = np.array([np.inf, np.inf]), np.array([-np.inf, -np.inf])

        def update_plot_bounds(coords_2d_arr):
            nonlocal min_coord, max_coord
            min_coord = np.minimum(min_coord, coords_2d_arr)
            max_coord = np.maximum(max_coord, coords_2d_arr)
            
        # Plot lines first
        for start_np, end_np, color, linestyle in self.lines_to_plot:
            p_start_2d = np.array([start_np[self.projection_axes[0]], start_np[self.projection_axes[1]]])
            p_end_2d = np.array([end_np[self.projection_axes[0]], end_np[self.projection_axes[1]]])
            ax.plot([p_start_2d[0], p_end_2d[0]], [p_start_2d[1], p_end_2d[1]], color=color, linestyle=linestyle, zorder=1, linewidth=1.5)
            update_plot_bounds(p_start_2d)
            update_plot_bounds(p_end_2d)
            
        # Plot circles
        for circle_data in self.circles_to_plot:
            center_2d = np.array([circle_data['center'][self.projection_axes[0]], circle_data['center'][self.projection_axes[1]]])
            radius = circle_data['radius']
            
            # Create circle patch
            circle = plt.Circle(center_2d, radius, 
                               color=circle_data['color'], 
                               linestyle=circle_data['linestyle'],
                               fill=circle_data['fill'], 
                               alpha=circle_data['alpha'],
                               linewidth=1.5,
                               zorder=1)
            ax.add_patch(circle)
            
            # Add label if provided
            if circle_data['label']:
                ax.text(center_2d[0], center_2d[1] + radius + 0.02, circle_data['label'], 
                       fontsize=9, ha='center', va='bottom', zorder=3)
            
            # Update plot bounds to include circle
            update_plot_bounds(center_2d + np.array([radius, radius]))
            update_plot_bounds(center_2d - np.array([radius, radius]))
            
        # Plot points
        plotted_labels_for_legend = {}
        for coords_np, item_name_for_color, label_text in self.points_to_plot:
            coords_2d = np.array([coords_np[self.projection_axes[0]], coords_np[self.projection_axes[1]]])
            color = self._get_color_for_item(item_name_for_color)
            
            # Add to legend only if label not already there for this color group
            legend_entry_key = f"{label_text}_point_{item_name_for_color}"
            plot_label_for_legend = None
            if legend_entry_key not in plotted_labels_for_legend:
                plot_label_for_legend = f"{label_text} ({item_name_for_color})"
                plotted_labels_for_legend[legend_entry_key] = True

            ax.plot(coords_2d[0], coords_2d[1], 'o', color=color, markersize=7, label=plot_label_for_legend, zorder=2, markeredgecolor='black', mew=0.5)
            if label_text: # Text next to point
                ax.text(coords_2d[0] + 0.015, coords_2d[1] + 0.015, label_text, fontsize=9, zorder=3, ha='left', va='bottom')
            update_plot_bounds(coords_2d)

        # Determine automatic force scale if not provided
        current_force_scale = force_scale_factor
        if current_force_scale is None:
            if not (min_coord[0] == np.inf): # if bounds are set
                plot_span = max(max_coord[0] - min_coord[0], max_coord[1] - min_coord[1])
                if plot_span > 1e-9: # Avoid division by zero for tiny plots
                    # Find a typical non-zero force magnitude
                    typical_mag = 1.0
                    non_zero_mags = [np.linalg.norm(f_vec) for _, f_vec, _, _, _ in self.forces_to_plot if np.linalg.norm(f_vec) > 1e-6]
                    if non_zero_mags:
                        typical_mag = np.median(non_zero_mags) if non_zero_mags else 1.0
                        if typical_mag < 1e-6: typical_mag = 1.0 # Handle all forces being tiny

                    current_force_scale = 0.1 * plot_span / typical_mag # Heuristic: make force vector 10% of plot span
                else: # Plot span is zero or tiny
                    current_force_scale = 0.1 # A small default scale
            else: # No points plotted yet to determine span
                 current_force_scale = 0.1 # Default if no points


        # Plot forces
        for app_point_np, force_vec_np, body_name, force_label, mag_str in self.forces_to_plot:
            app_2d = np.array([app_point_np[self.projection_axes[0]], app_point_np[self.projection_axes[1]]])
            force_2d_vec_component = np.array([force_vec_np[self.projection_axes[0]], force_vec_np[self.projection_axes[1]]])

            if np.linalg.norm(force_vec_np) < 1e-6: # Skip plotting zero/tiny magnitude forces
                continue

            # Debug: Print force plotting info
            print(f"Plotting force {force_label}: 2D_projection={force_2d_vec_component}, magnitude={np.linalg.norm(force_vec_np):.6f}")

            color = self._get_color_for_item(force_label if force_label in self.color_map else body_name) # Prefer force name for color if defined
            
            # Use the calculated or provided scale factor to make forces visible
            scaled_force_x = force_2d_vec_component[0] * current_force_scale
            scaled_force_y = force_2d_vec_component[1] * current_force_scale
            
            # Ensure arrow head is not too large for small vectors
            # Increased values to make arrows more prominent and obvious
            head_width_val = max(0.015, 0.12 * np.linalg.norm([scaled_force_x, scaled_force_y]))
            head_length_val = max(0.025, 0.18 * np.linalg.norm([scaled_force_x, scaled_force_y]))
            
            legend_entry_key = f"{force_label}_force_{body_name}"
            plot_label_for_legend = None
            if legend_entry_key not in plotted_labels_for_legend:
                plot_label_for_legend = f"{force_label} on {body_name}"
                plotted_labels_for_legend[legend_entry_key] = True

            ax.arrow(app_2d[0], app_2d[1], 
                     scaled_force_x, scaled_force_y,
                     head_width=head_width_val, 
                     head_length=head_length_val, 
                     fc=color, ec=color, label=plot_label_for_legend, zorder=3, length_includes_head=True, linewidth=2.5)
            
            # Position the force magnitude label.
            # Add a small offset from the arrow head, with a minimum distance to avoid overlap on small arrows.
            scaled_force_norm = np.linalg.norm([scaled_force_x, scaled_force_y])
            if scaled_force_norm > 1e-9:
                # The offset from the arrow head is larger for smaller arrows to ensure visibility.
                # It's a combination of a term proportional to the arrow's length and a minimum offset
                # derived from the arrow's head length.
                min_offset = head_length_val * 3.0 # Increase minimum offset for small arrows
                offset_from_head = max(scaled_force_norm * 0.1, min_offset)

                unit_vector_x = scaled_force_x / scaled_force_norm
                unit_vector_y = scaled_force_y / scaled_force_norm
                
                text_pos_x = app_2d[0] + scaled_force_x + unit_vector_x * offset_from_head
                text_pos_y = app_2d[1] + scaled_force_y + unit_vector_y * offset_from_head
            else:
                # For zero-magnitude forces, place the label slightly offset from the application point.
                # A small, somewhat arbitrary offset is chosen.
                text_pos_x = app_2d[0] + 0.02
                text_pos_y = app_2d[1] + 0.02

            if show_values:
                ax.text(text_pos_x, text_pos_y, mag_str, fontsize=8, color=color, zorder=4, ha='center', va='center')

            update_plot_bounds(app_2d)
            update_plot_bounds(app_2d + np.array([scaled_force_x, scaled_force_y]))


        # Auto-adjust plot limits
        if not (min_coord[0] == np.inf): 
            x_margin = (max_coord[0] - min_coord[0]) * 0.15 if (max_coord[0] - min_coord[0]) > 1e-9 else 0.5
            y_margin = (max_coord[1] - min_coord[1]) * 0.15 if (max_coord[1] - min_coord[1]) > 1e-9 else 0.5
            # Set limits with equal aspect ratio
            x_min = min_coord[0] - x_margin
            x_max = max_coord[0] + x_margin
            y_min = min_coord[1] - y_margin 
            y_max = max_coord[1] + y_margin
            
            # Make plot square by expanding the smaller dimension
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range > y_range:
                y_center = (y_max + y_min) / 2
                y_min = y_center - x_range/2
                y_max = y_center + x_range/2
            else:
                x_center = (x_max + x_min) / 2
                x_min = x_center - y_range/2
                x_max = x_center + y_range/2
                
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            
        handles, labels = ax.get_legend_handles_labels()
        # Filter out None labels which can happen if an item was already in legend by key
        filtered_handles_labels = [(h, l) for h, l in zip(handles, labels) if l is not None]
        if filtered_handles_labels:
            # Use a dictionary to remove duplicate labels from the legend explicitly
            by_label = dict(filtered_handles_labels)
            ax.legend(by_label.values(), by_label.keys(), loc='best', fontsize='small', framealpha=0.7)


        if filename:
            plt.savefig(filename, dpi=300) # Save with good resolution
            print(f"Visualisation saved to {filename}")
            plt.close(fig)
        else:
            return

class SpringVisualiser:
    def __init__(self, spring: AbstractSpring):
        self.spring = copy.deepcopy(spring)
        # The symbol 'spring_length' will represent the signed ELONGATION e = L_actual - L_rest
        self.elongation_symbol = Symbol('strain') 

        # The actual current physical length of the spring is L_actual = e + x0
        # This expression is used to define the coordinate of point_2.
        # The non-negativity of (self.elongation_symbol + self.spring.x0) will be handled
        # by clipping the numerical plot range in the render() method.
        actual_current_length_expr = self.elongation_symbol + self.spring.x0

        point_1  = Point([0, 0, 0], self.spring.point_1.reference_frame)
        point_2  = Point([actual_current_length_expr, 0, 0], self.spring.point_1.reference_frame)
        self.spring.set_points(point_1, point_2)

    def render(self, title="Spring Visualisation", bounds=(-0.01, 0.02)):
        fig, ax = plt.subplots(figsize=(10, 8))
        # Remove equal aspect ratio since force vs displacement plots typically don't need it
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Spring Strain (m)", fontsize=12)
        ax.set_ylabel("Spring Force (N)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_xlim(bounds[0], bounds[1])
        ax.set_aspect('auto')  # Remove equal aspect ratio to allow wider plot
        
        x_vals = np.linspace(bounds[0], bounds[1], 100)
        force = self.spring.get_force_on_point1()
        force_expr = force.force.coordinates[0]
        energy_func = sympy.lambdify(Symbol('strain'), force_expr)
        y_vals = energy_func(x_vals)
        
        # Add vertical lines at key points
        # Plot points at key spring transition points
        ax.plot(self.spring.a, force_expr.subs(Symbol('strain'), self.spring.a), 'ro', label='a', markersize=8)
        # Add text annotations showing coordinates
        ax.annotate(f'a={self.spring.a:.2f}', (self.spring.a, 0), 
                   xytext=(10, 10), textcoords='offset points', color='red')
        ax.plot(x_vals, y_vals, label="Spring Force", color="blue")
                
        