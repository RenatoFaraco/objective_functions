import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from benchmarks import goemetric as geo

params = {
    'function': geo.cross_in_tray,
    'bounds': geo.limit_cross_in_tray,
    'n_dim': 2,
    'alpha': 0.01,
    'beta': 0.5,
    'delta': 0.1,
    'max_iter': 250,
    'pop_size': 10
}

# Create meshgrid over the search space
x_vals = np.linspace(params['bounds'][0, 0], params['bounds'][0, 1], 100)
y_vals = np.linspace(params['bounds'][1, 0], params['bounds'][1, 1], 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[params['function'](np.array([x, y])) for x in x_vals] for y in y_vals])

# 3D Plot
fig = plt.figure(figsize=(12, 6))
ax3d = fig.add_subplot(121, projection='3d')
ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax3d.contour(X, Y, Z, levels=50, colors='lightgrey', offset=np.min(Z))
ax3d.set_title('3D Surface Plot - Ackley Function')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

# 2D Contour Plot
ax2d = fig.add_subplot(122)
contour = ax2d.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, ax=ax2d)
ax2d.contour(X, Y, Z, colors='lightgrey', levels=10)
ax2d.set_title('2D Contour Plot - Ackley Function')
ax2d.set_xlabel('X')
ax2d.set_ylabel('Y')

plt.tight_layout()
plt.show()
