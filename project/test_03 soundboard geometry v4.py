from math import sqrt, sin, cos, tan, atan
import numpy as np
from plot import plot, plot_itp, plot_scatter
from mathfunc import cubic_poly_one_real_root


# Create a txt file to save some information.
folder = 'D:/self_research/piano_project/data'
txt_path = f'{folder}/soundboard_01.txt'
txt_file = open(txt_path, 'w')
txt_file.write('Soundboard information\n')

# Input parameters.
Lx, Ly = 160.0, 220.0 # cm
Lx1, Ly1 = 70.0, 35.0 # cm

A1_ratio_x, A1_ratio_y = 0.15, 0.9
A2_ratio_x, A2_ratio_y = 0.55, 0.2

spring_distance = 1. #0.73 # cm

n_rib = 16
rib_angle = 40. # degree
first_rib_y = 0.85 * Ly
last_rib_x = 0.75 * Lx
plot_rib_mid = True
rib_beg_len = 15*np.ones(n_rib)
rib_end_len = 15*np.ones(n_rib)

rib_end_len[:2] *= 0.8
rib_beg_len[5] *= 1.1
rib_beg_len[6] *= 1.2
rib_beg_len[7] *= 1.45
rib_beg_len[8] *= 1.8
rib_beg_len[9] *= 2.45
rib_beg_len[10] *= 2.85
rib_beg_len[11] *= 2.1
rib_beg_len[12] *= 1.5
rib_beg_len[-2:] *= 0.8

n_bridge = 4
bridge_angle = np.array([70., 70., 55., 25.]) # degree
bridge_len = np.array([70., 100., 70., 55.])
B0_x, B0_y = 0.5*Lx1, 0.9*Ly
B1_x, B1_y = 0.25*Lx1, 0.85*Ly


# Check that parameters are valid.
assert Lx > Lx1
assert Ly > Ly1
assert 0.0 < A1_ratio_x < A2_ratio_x < 1.0
assert 0.0 < A2_ratio_y < A1_ratio_y < 1.0
assert n_bridge == bridge_angle.size == bridge_len.size


# Compute needed points to fit the soundboard curve.
A0_x, A0_y = Lx1, Ly
A3_x, A3_y = Lx, Ly1

A1_x, A1_y = A0_x+A1_ratio_x*(A3_x-A0_x), A3_y+A1_ratio_y*(A0_y-A3_y)
A2_x, A2_y = A0_x+A2_ratio_x*(A3_x-A0_x), A3_y+A2_ratio_y*(A0_y-A3_y)

p_x, p_y = [A0_x, A1_x, A2_x, A3_x], [A0_y, A1_y, A2_y, A3_y]

print(f'A0 = ({A0_x:.2f}, {A0_y:.2f})')
print(f'A1 = ({A1_x:.2f}, {A1_y:.2f})')
print(f'A2 = ({A2_x:.2f}, {A2_y:.2f})')
print(f'A3 = ({A3_x:.2f}, {A3_y:.2f})')


# Fit the soundboard shape curve shape using polynomial.
cubic_coef = np.polynomial.polynomial.polyfit(x=p_y, y=p_x, deg=3)
p0, p1, p2, p3 = cubic_coef


# Calculate the coordinates of springs.
linear_coef = np.polynomial.polynomial.polyfit(x=[A0_x, A3_x], y=[A0_y, A3_y], deg=1)
curve_alpha = atan(-linear_coef[1])

spring_0_y = np.arange(Ly1, Ly, spring_distance*np.sin(curve_alpha))
spring_0_x = p0 + p1*spring_0_y + p2*(spring_0_y**2) + p3*(spring_0_y**3)

spring_1_x = np.arange(0., Lx1, spring_distance)
spring_1_y = Ly * np.ones(spring_1_x.size)

spring_2_y = np.arange(0., Ly, spring_distance)
spring_2_x = np.zeros(spring_2_y.size)

spring_3_x = np.arange(0., Lx, spring_distance)
spring_3_y = np.zeros(spring_3_x.size)

spring_4_y = np.arange(0., Ly1, spring_distance)
spring_4_x = Lx * np.ones(spring_4_y.size)

spring_x = np.concatenate((spring_0_x, spring_1_x[::-1], spring_2_x[::-1], spring_3_x, spring_4_x))
spring_y = np.concatenate((spring_0_y, spring_1_y[::-1], spring_2_y[::-1], spring_3_y, spring_4_y))
n_spring = spring_x.size


# Calculate the coordinates of ribs
# Calculate the coordinates of the begin of each rib.
rib_alpha = np.deg2rad(rib_angle)
rib_sin, rib_cos, rib_tan = sin(rib_alpha), cos(rib_alpha), tan(rib_alpha)
rib_total_distance = abs(first_rib_y + rib_tan * last_rib_x) / sqrt(rib_tan**2 + 1)
rib_distance = rib_total_distance / (n_rib - 1)
print(f'rib_distance = {rib_distance:.2f} cm')

rib_distance_x, rib_distance_y = rib_distance / rib_sin, rib_distance / rib_cos
rib_beg_x, rib_beg_y = [0.], [first_rib_y]
rib_end_x, rib_end_y = [], []

rib_beg_is_on_y_axis = True
is_first_rib_beg_on_x_axis = True

for i in range(1, n_rib-1):
    if rib_beg_is_on_y_axis:
        y = first_rib_y - i * rib_distance_y
        if not y < 0.:
            rib_beg_x.append(0.)
            rib_beg_y.append(y)
        else:
            rib_beg_is_on_y_axis = False
            
    if not rib_beg_is_on_y_axis:
        if is_first_rib_beg_on_x_axis:
            y = rib_beg_y[-1]
            rib_distance_up = y * rib_cos
            rib_distance_lo = rib_distance - rib_distance_up
            assert rib_distance_lo > 0.
            x = rib_distance_lo / rib_sin
            rib_beg_x.append(x)
            rib_beg_y.append(0.)
            is_first_rib_beg_on_x_axis = False
        else:
            x = rib_beg_x[-1] + rib_distance_x
            if not x > last_rib_x:
                rib_beg_x.append(x)
                rib_beg_y.append(0.)
            else:
                raise ValueError('Incorrect rib coordinate: rib_beg_x < 0.')
        
rib_beg_x.append(last_rib_x)
rib_beg_y.append(0.)


# Calculate the coordinates of the end of each rib.
for i in range(0, n_rib):
    x0, y0 = rib_beg_x[i], rib_beg_y[i]
    x = x0 + (Ly - y0) / rib_tan
    if not x > Lx1:
        rib_end_x.append(x)
        rib_end_y.append(Ly)
    else:
        y = cubic_poly_one_real_root( rib_tan*p3, rib_tan*p2, rib_tan*p1-1, rib_tan*(p0-x0)+y0 )
        if not (y < Ly1):
            x = x0 + (y - y0) / rib_tan
            rib_end_x.append(x)
            rib_end_y.append(y)
        else:
            y = y0 + rib_tan * (Lx - x0)
            rib_end_x.append(Lx)
            rib_end_y.append(y)

rib_beg_x, rib_beg_y = np.asarray(rib_beg_x), np.asarray(rib_beg_y)
rib_end_x, rib_end_y = np.asarray(rib_end_x), np.asarray(rib_end_y)


# Calculate the coordinates of the two mid points of each rib.
rib_mid1_x = rib_beg_x + rib_beg_len * rib_cos
rib_mid1_y = rib_beg_y + rib_beg_len * rib_sin

rib_mid2_x = rib_end_x - rib_end_len * rib_cos
rib_mid2_y = rib_end_y - rib_end_len * rib_sin


# Prepare to plot the ribs
rib_x, rib_y = [], []
if plot_rib_mid:
    for i in range(0, n_rib):
        x0, y0 = rib_mid1_x[i], rib_mid1_y[i]
        rib_x.append( np.arange(rib_mid1_x[i], rib_mid2_x[i], rib_cos) )
        rib_y.append( y0 + rib_tan * (rib_x[i] - x0) )
else:
    for i in range(0, n_rib):
        x0, y0 = rib_beg_x[i], rib_beg_y[i]
        rib_x.append( np.arange(rib_beg_x[i], rib_end_x[i], rib_cos) )
        rib_y.append( y0 + rib_tan * (rib_x[i] - x0) )

rib_x, rib_y = np.concatenate(rib_x), np.concatenate(rib_y)


# Calculate coordinates of the bridges
bridge_alpha = np.deg2rad(bridge_angle)
bridge_sin, bridge_cos, bridge_tan = np.sin(bridge_alpha), np.cos(bridge_alpha), np.tan(bridge_alpha)
bridge_beg_x, bridge_beg_y = [B0_x, B1_x], [B0_y, B1_y]
bridge_end_x, bridge_end_y = [], []

for i in range(0, n_bridge):
    x = bridge_beg_x[i] + bridge_len[i] * bridge_cos[i]
    y = bridge_beg_y[i] - bridge_len[i] * bridge_sin[i]
    bridge_end_x.append(x)
    bridge_end_y.append(y)

    if 0 < i < n_bridge-1 :
        bridge_beg_x.append(x)
        bridge_beg_y.append(y)

bridge_beg_x, bridge_beg_y = np.asarray(bridge_beg_x), np.asarray(bridge_beg_y)
bridge_end_x, bridge_end_y = np.asarray(bridge_end_x), np.asarray(bridge_end_y)


# Prepare to plot the bridges
bridge_x, bridge_y = [], []
for i in range(0, n_bridge):
    x0, y0 = bridge_beg_x[i], bridge_beg_y[i]
    bridge_x.append( np.arange(bridge_beg_x[i], bridge_end_x[i], bridge_cos[i]) )
    bridge_y.append( y0 - bridge_tan[i] * (bridge_x[i] - x0) )

bridge_x, bridge_y = np.concatenate(bridge_x), np.concatenate(bridge_y)


# Save data
spring_y_path = f'{folder}/spring_y.npy'
spring_x_path = f'{folder}/spring_x.npy'
np.save(spring_y_path, spring_y)
np.save(spring_x_path, spring_x)
print(f'spring x coordinates saved to: "{spring_x_path}"')
print(f'spring y coordinates saved to: "{spring_y_path}"')

rib_beg_x_path, rib_beg_y_path = f'{folder}/rib_beg_x.npy', f'{folder}/rib_beg_y.npy'
rib_end_x_path, rib_end_y_path = f'{folder}/rib_end_x.npy', f'{folder}/rib_end_y.npy'
np.save(rib_beg_x_path, rib_beg_x)
np.save(rib_beg_y_path, rib_beg_y)
np.save(rib_end_x_path, rib_end_x)
np.save(rib_end_y_path, rib_end_y)
print(f'rib begin x coordinates saved to: "{rib_beg_x_path}"')
print(f'rib begin y coordinates saved to: "{rib_beg_y_path}"')
print(f'rib end x coordinates saved to: "{rib_end_x_path}"')
print(f'rib end y coordinates saved to: "{rib_end_y_path}"')

bridge_beg_x_path, bridge_beg_y_path = f'{folder}/bridge_beg_x.npy', f'{folder}/bridge_beg_y.npy'
bridge_end_x_path, bridge_end_y_path = f'{folder}/bridge_end_x.npy', f'{folder}/bridge_end_y.npy'
bridge_alpha_path = f'{folder}/bridge_alpha.npy'
np.save(bridge_beg_x_path, bridge_beg_x)
np.save(bridge_beg_y_path, bridge_beg_y)
np.save(bridge_end_x_path, bridge_end_x)
np.save(bridge_end_y_path, bridge_end_y)
np.save(bridge_alpha_path, bridge_alpha)
print(f'bridge begin x coordinates saved to: "{bridge_beg_x_path}"')
print(f'bridge begin y coordinates saved to: "{bridge_beg_y_path}"')
print(f'bridge end x coordinates saved to: "{bridge_end_x_path}"')
print(f'bridge end y coordinates saved to: "{bridge_end_y_path}"')
print(f'bridge alpha saved to: "{bridge_alpha_path}"')

txt_file.write(f"""
Lx = {Lx:.2f} cm, Ly = {Ly:.2f} cm\nLx1 = {Lx1:.2f} cm, Ly1 = {Ly1:.2f} cm\n
n_spring = {n_spring}, spring_distance={spring_distance:.6f}\n
n_rib = {n_rib}, rib_alpha = {rib_alpha:.6f}, rib_distance = {rib_distance:.6f} cm\n
n_bridge = {n_bridge}\n
""")
cubic_coef_str = 'cubic polynomial coefficients for the soundboard curve shape:\n' + \
    np.array2string(cubic_coef, max_line_width=10000, precision=16, separator=', ', floatmode='fixed', \
    formatter={'float_kind':lambda x: np.format_float_positional(x, precision=16)}) + '\n'
txt_file.write(cubic_coef_str)
txt_file.close()


# Plot the soundboard shape.
points = np.array([p_x, p_y]).T
points = np.append(points, np.array([[0,0], [0,Ly], [Lx,0]]), axis=0)

soundboard_path = f'{folder}/soundboard_01.png'
plot_scatter(np.concatenate((spring_y, rib_y, bridge_y)), np.concatenate((spring_x, rib_x, bridge_x)), \
             equal_axis=True, save_path=soundboard_path, title='Soundboard')
print(f'soundboard geometry saved to: "{soundboard_path}"')