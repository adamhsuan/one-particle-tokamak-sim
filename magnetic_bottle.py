import numpy as np

num_tf_coils = 0
tf_coil_radii = 1
tokamak_outer_radius = 1
pf_coil_z_values = [0.9,-0.9]
pf_coil_currents = [10,10]
points_per_tf_coil_simulation = 10
points_per_pf_coil_simulation = 20
points_per_coil_visualization = 100
tf_coil_currents = []
proportional_feedback_pf_coils_constant = 5

dt = 0.00000001
total_sim_time = 0.0003
total_timesteps = int(total_sim_time/dt)
num_sim_frames = 1000
sim_frames_time_gap = total_sim_time/num_sim_frames
timesteps_per_sim_frame = int(sim_frames_time_gap/dt)
num_frame_points = 5000
sim_points_time_gap = total_sim_time/num_frame_points
timesteps_per_sim_point = int(sim_points_time_gap/dt)

mu_0 = 1.2566e-6
r0 = np.array([0, 0, -0.9])
r = r0
v0 = np.array([1000.0,1000.0,3300.0])
v = v0
a0 = np.array([0,0,0])
a = a0
g = 9.8
m = 9.109e-31
q = -1.602e-19
dl = 0.2

def calculate_B_at_r(r, num_tf_coils, tf_coils_points, tf_coils_dl_vectors, tf_coil_currents, pf_coils_points, pf_coils_dl_vectors, pf_coil_currents, mu_0):
    B = np.array([0.0,0.0,0.0])
    for coil_number in range(num_tf_coils):
        for point_index, r_coil in enumerate(tf_coils_points[coil_number]):
            dl = tf_coils_dl_vectors[coil_number][point_index]
            I = tf_coil_currents[coil_number]
            r_rel = r-r_coil
            B += mu_0 / ( 4 * np.pi ) * I * np.cross(dl, r_rel) / (np.linalg.norm(r_rel))**3
    num_pf_coils = len(pf_coil_currents)
    for coil_number in range(num_pf_coils):
        for point_index, r_coil in enumerate(pf_coils_points[coil_number]):
            dl = pf_coils_dl_vectors[coil_number][point_index]
            I = pf_coil_currents[coil_number]
            r_rel = r-r_coil
            B += mu_0 / ( 4 * np.pi ) * I * np.cross(dl, r_rel) / (np.linalg.norm(r_rel))**3
    return B


# returns a nested list of points on the tokamak coils, with structure: list -> list -> np array, where the np arrays are points 
# and the nested lists are coils
# also returns a nested list of current vectors with the same structure
def generate_tf_coils_points_and_dl_vectors(num_tf_coils, tf_coil_radii, tokamak_outer_radius, points_per_coil):
    tf_coils_points = []
    tf_coils_dl_vectors = []
    for tf_coil_number in range(num_tf_coils):
        tf_coil_points = []
        tf_coil_dl_vectors = []
        theta = 2*np.pi/num_tf_coils*tf_coil_number
        tf_coil_center = (tokamak_outer_radius - tf_coil_radii) * np.array([np.cos(theta), np.sin(theta),0])
        for point_on_tf_coil_number in range(points_per_coil):
            phi = -2*np.pi/points_per_coil*point_on_tf_coil_number
            point = tf_coil_center + tf_coil_radii * np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])
            tf_coil_points.append(point)
            dl = -tf_coil_radii * np.array([np.cos(theta) * (-1) * np.sin(phi), np.sin(theta) * (-1) * np.sin(phi), np.cos(phi)]) * 2 * np.pi / points_per_coil
            tf_coil_dl_vectors.append(dl)
        tf_coils_points.append(tf_coil_points)
        tf_coils_dl_vectors.append(tf_coil_dl_vectors)
    return [tf_coils_points, tf_coils_dl_vectors]

def generate_pf_coils_points_and_dl_vectors(pf_coil_z_values, tf_coil_radii, tokamak_outer_radius, points_per_coil):
    pf_coils_points = []
    pf_coils_dl_vectors = []
    for i, pf_coil_z in enumerate(pf_coil_z_values):
        pf_coil_points = []
        pf_coil_dl_vectors = []
        pf_coil_radius = tokamak_outer_radius - tf_coil_radii + np.sqrt(tf_coil_radii**2 - pf_coil_z**2)
        for point_on_pf_coil_number in range(points_per_coil):
            theta = 2*np.pi/points_per_coil*point_on_pf_coil_number
            point =  np.array([pf_coil_radius * np.cos(theta), pf_coil_radius * np.sin(theta), pf_coil_z])
            pf_coil_points.append(point)
            dl = np.array([pf_coil_radius * (-1) * np.sin(theta), pf_coil_radius * np.cos(theta), 0]) * 2 * np.pi / points_per_coil
            pf_coil_dl_vectors.append(dl)
        pf_coils_points.append(pf_coil_points)
        pf_coils_dl_vectors.append(pf_coil_dl_vectors)
    return [pf_coils_points, pf_coils_dl_vectors]

def point_inside_tokamak(point, tf_coil_radii, tokamak_outer_radius):
    x, y = point[0], point[1]
    return np.linalg.norm(point - (tokamak_outer_radius - tf_coil_radii) * np.array([x,y,0]/np.sqrt(x**2+y**2))) <= tf_coil_radii

# returns a list of cartesian points (np arrays) inside the tokamak, ordered for direct use in creating vtk file
def generate_points_inside_tokamak(tf_coil_radii, tokamak_outer_radius, dl):
    points_inside_tokamak = []
    for x in range(-tokamak_outer_radius, tokamak_outer_radius, dl):
        for y in range(-tokamak_outer_radius, tokamak_outer_radius, dl):
            for z in range(-tf_coil_radii, tf_coil_radii, dl):
                point = np.array([x,y,z])
                if point_inside_tokamak(point, tf_coil_radii, tokamak_outer_radius):
                    points_inside_tokamak.append(point)
    return points_inside_tokamak

# calculates the magnetic fields and returns a list for direct access in creating vtk file
def calculate_magnetic_fields_inside_tokakmak(tokamak_outer_radius, tf_coil_radii, tf_coils_points, tf_coils_dl_vectors, tf_coil_currents, pf_coils_points, pf_coils_dl_vectors, pf_coil_currents, mu_0, dl):
    magnetic_fields_inside_tokamak = []
    B_approx_center = mu_0*num_tf_coils*np.mean(tf_coil_currents)/(2*np.pi*(tokamak_outer_radius-tf_coil_radii))
    num_points = int(8*tokamak_outer_radius**2*tf_coil_radii/dl**3)
    print(num_points)
    k = 0
    magnetic_field_percentage = 0
    for z in [-1*tf_coil_radii+dl*i for i in range(int(2*tf_coil_radii/dl))]:
        for y in [-1*tokamak_outer_radius+dl*i for i in range(int(2*tokamak_outer_radius/dl))]:
            for x in [-1*tokamak_outer_radius+dl*i for i in range(int(2*tokamak_outer_radius/dl))]:
                point = np.array([x,y,z])
                if point_inside_tokamak(point, tf_coil_radii, tokamak_outer_radius):
                    B = calculate_B_at_r(point, num_tf_coils, tf_coils_points, tf_coils_dl_vectors, tf_coil_currents, pf_coils_points, pf_coils_dl_vectors, pf_coil_currents, mu_0)
                    magnetic_fields_inside_tokamak.append(B)
                else:
                    magnetic_fields_inside_tokamak.append(np.array([0,0,0]))
                if k/num_points*100 > magnetic_field_percentage:
                    print(f"{int(k/num_points*100)}% of magnetic fields calculated")
                    magnetic_field_percentage = int(k/num_points*100)+1
                k+=1
    return magnetic_fields_inside_tokamak

# create the tokamak shape vtk file

def create_tokamak_shape_vtk(num_tf_coils, tf_coil_radii, tf_coil_currents, tokamak_outer_radius, points_per_coil_visualization, pf_coil_z_values, pf_coil_currents):
    visualization_tf_coils_points, _ = generate_tf_coils_points_and_dl_vectors(num_tf_coils, tf_coil_radii, tokamak_outer_radius, points_per_coil_visualization)
    visualization_pf_coils_points, _ = generate_pf_coils_points_and_dl_vectors(pf_coil_z_values, tf_coil_radii, tokamak_outer_radius, points_per_coil_visualization)
    with open("bottle_shape.vtk","w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"tokamak shape\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        num_pf_coils = len(pf_coil_z_values)
        f.write(f"POINTS {(num_tf_coils+num_pf_coils)*points_per_coil_visualization} double\n")
        for coil_number in range(num_tf_coils):
            for r_vector in visualization_tf_coils_points[coil_number]:
                for value in r_vector:
                    f.write(f"{value} ")
                f.write("\n")
        for coil_number in range(num_pf_coils):
            for r_vector in visualization_pf_coils_points[coil_number]:
                for value in r_vector:
                    f.write(f"{value} ")
                f.write("\n")
        
        f.write(f"LINES {(num_tf_coils+num_pf_coils)*(points_per_coil_visualization)} {(num_tf_coils+num_pf_coils)*(points_per_coil_visualization)*3}\n")
        i = 0
        for coil_number in range(num_tf_coils):
            for j in range(points_per_coil_visualization-1):
                f.write(f"2 {i+j} {i+j+1}\n")
            f.write(f"2 {i+points_per_coil_visualization-1} {i}\n")
            i += points_per_coil_visualization
        for coil_number in range(num_pf_coils):
            for j in range(points_per_coil_visualization-1):
                f.write(f"2 {i+j} {i+j+1}\n")
            f.write(f"2 {i+points_per_coil_visualization-1} {i}\n")
            i += points_per_coil_visualization

        f.write(f"POINT_DATA {(num_tf_coils+num_pf_coils)*points_per_coil_visualization}\n")

        f.write(f"SCALARS currents double 1\n")
        f.write(f"LOOKUP_TABLE default\n")
        for coil_number in range(num_tf_coils):
            for i in range(points_per_coil_visualization):
                f.write(f"{tf_coil_currents[coil_number]}\n")
        for coil_number in range(num_pf_coils):
            for i in range(points_per_coil_visualization):
                f.write(f"{pf_coil_currents[coil_number]}\n")

simulation_tf_coils_points, simulation_tf_coils_dl_vectors = generate_tf_coils_points_and_dl_vectors(num_tf_coils, tf_coil_radii, tokamak_outer_radius, points_per_tf_coil_simulation)
simulation_pf_coils_points, simulation_pf_coils_dl_vectors = generate_pf_coils_points_and_dl_vectors(pf_coil_z_values, tf_coil_radii, tokamak_outer_radius, points_per_pf_coil_simulation)

# create the magnetic field vectors vtk file

def create_magnetic_field_vtk(tokamak_outer_radius, tf_coil_radii, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, dl):

    with open("magnetic_field_vectors.vtk","w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("magnetic field vectors\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {int(2*tokamak_outer_radius/dl)} {int(2*tokamak_outer_radius/dl)} {int(2*tf_coil_radii/dl)}\n")
        f.write(f"ORIGIN {-1*tokamak_outer_radius} {-1*tokamak_outer_radius} {-1*tf_coil_radii}\n")
        f.write(f"SPACING {dl} {dl} {dl}\n")
        magnetic_fields = calculate_magnetic_fields_inside_tokakmak(tokamak_outer_radius, tf_coil_radii, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, dl)
        f.write(f"POINT_DATA {len(magnetic_fields)}\n")
        f.write(f"VECTORS B double\n")
        for magnetic_field in magnetic_fields:
            for value in magnetic_field:
                f.write(f"{value} ")
            f.write("\n")


def calculate_a(q,m,v,B,g):
    a = q / m * np.cross(v, B) + np.array([0,0,-1*g])
    return a

# create the particle trajectory vtk files
def create_particle_trajectory_rk4_vtk(theta, r, v, a, q, m, g, dt, total_timesteps, timesteps_per_sim_frame, timesteps_per_sim_point, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, tf_coil_radii, tokamak_outer_radius, proportional_feedback_pf_coils_constant):

    rs_so_far = np.array([r])
    a_prev = np.array([0,0,0])
    for timestep in range(total_timesteps):
        if (timestep/total_timesteps*1000)%10 == 0:
            print(f"{timestep/total_timesteps*100}% of particle trajectory calculated")

        r1 = r
        v1 = v
        B1 = calculate_B_at_r(r1, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0)
        a1 = calculate_a(q,m,v1,B1,g)

        r2 = r + v1*dt/2
        v2 = v + a1*dt/2
        B2 = calculate_B_at_r(r2, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0)
        a2 = calculate_a(q,m,v2,B2,g)

        r3 = r + v2*dt/2
        v3 = v + a2*dt/2
        B3 = calculate_B_at_r(r3, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0)
        a3 = calculate_a(q,m,v3,B3,g)

        r4 = r + v3*dt
        v4 = v + a3*dt
        B4 = calculate_B_at_r(r4, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0)
        a4 = calculate_a(q,m,v4,B4,g)

        r += 1/6*(v1+2*v2+2*v3+v4)*dt
        v += 1/6*(a1+2*a2+2*a3+a4)*dt

        #parallel velocity is the projection of velocity path perpendiuclar to gyroscopic motion, which is calculated as the cross product
        #of accelerations in current and previous time 
        if np.array_equal(a_prev, np.array([0,0,0])):
            #ensures that we've covered at least one time step for extrapolation
            v_par = np.array([0,0,0])
        else:
            perp_path_vector = np.cross(a1, a_prev)
            v_par = np.dot(v,perp_path_vector) / np.linalg.norm(perp_path_vector)**2 * perp_path_vector
        a_prev = a1

        sim_step = int(timestep/timesteps_per_sim_frame)

        if timestep % timesteps_per_sim_point == 0:
            rs_so_far = np.append(rs_so_far, [r], axis=0)

        if timestep % timesteps_per_sim_frame == 0:
            print(f"created frame {timestep/timesteps_per_sim_frame}")

            with open(f"particle_trajectories/particle_trajectory_theta_{theta}_{sim_step:04d}.vtk","w") as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write(f"particle trajectory {sim_step:04d}\n")
                f.write("ASCII\n")
                f.write("DATASET POLYDATA\n")

                f.write(f"POINTS {len(rs_so_far)} double\n")
                for r_vector in rs_so_far:
                    for value in r_vector:
                        f.write(f"{value} ")
                    f.write("\n")
                f.write(f"LINES {len(rs_so_far)-1} {(len(rs_so_far)-1)*3}\n")
                for i in range(len(rs_so_far)-1):
                    f.write(f"2 {i} {i+1}\n")

                f.write("FIELD initial_angle 1\n")
                f.write(f"theta 1 1 double\n")
                for pf_coil_current in pf_coil_currents:
                    f.write(f"{theta} ")
                f.write("\n")

create_tokamak_shape_vtk(num_tf_coils, tf_coil_radii, tf_coil_currents, tokamak_outer_radius, points_per_coil_visualization, pf_coil_z_values, pf_coil_currents)

simulation_tf_coils_points, simulation_tf_coils_dl_vectors = generate_tf_coils_points_and_dl_vectors(num_tf_coils, tf_coil_radii, tokamak_outer_radius, points_per_tf_coil_simulation)
simulation_pf_coils_points, simulation_pf_coils_dl_vectors = generate_pf_coils_points_and_dl_vectors(pf_coil_z_values, tf_coil_radii, tokamak_outer_radius, points_per_pf_coil_simulation)

#create_magnetic_field_vtk(tokamak_outer_radius, tf_coil_radii, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, dl)

#create_particle_trajectory_euler_vtk(r, v, a, q, m, g, dt, total_timesteps, timesteps_per_sim_frame, timesteps_per_sim_point, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, tf_coil_radii, tokamak_outer_radius, proportinal_feedback_pf_coils_constant)

v_mag = 50000
theta = 0

for i in range(10):
    r0 = np.array([0.0, 0.0, 0.0])
    r = r0
    v_0 = v_mag*np.array([np.sin(theta),0,np.cos(theta)])
    v = v_0
    a0 = np.array([0,0,0])
    a = a0
    create_particle_trajectory_rk4_vtk(theta, r, v, a, q, m, g, dt, total_timesteps, timesteps_per_sim_frame, timesteps_per_sim_point, num_tf_coils, simulation_tf_coils_points, simulation_tf_coils_dl_vectors, tf_coil_currents, simulation_pf_coils_points, simulation_pf_coils_dl_vectors, pf_coil_currents, mu_0, tf_coil_radii, tokamak_outer_radius, proportional_feedback_pf_coils_constant)
    theta+= np.pi/2/10

                    
