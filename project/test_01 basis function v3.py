import numpy as np
import matplotlib.pyplot as plt

# Define the range for x and y
lb, ub = -np.pi*0, np.pi
x = np.linspace(lb, ub, 2000)
y = np.linspace(lb, ub, 2000)
dxy = ((ub-lb) / 2000)**2

# Create a grid of (x, y) points
x, y = np.meshgrid(x, y)

# Compute the corresponding values of z
z = np.sin(x)*np.sin(y) + np.cos(x)*np.sin(y)*1.0j
z2 = np.cos(x)*np.cos(y) + np.sin(x)*np.cos(y)*1.0j
#z2 = np.sin(2*x)*np.sin(3*y) + np.cos(2*x)*np.sin(3*y)*1.0j

"""sum = np.sum(z*z2.conj()*dxy)
print(f'sum.real = {sum.real}')
print(f'sum.imag = {sum.imag}')"""


# Plot the function
plt.imshow(z.real-0.35*z2.real, cmap='viridis', extent=[lb,ub,lb,ub], origin='lower')
plt.colorbar(label='z')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of z')
plt.show()
