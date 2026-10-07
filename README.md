# ∇ Vector Calculus Laboratory

An interactive **Streamlit + Plotly** computational laboratory for visualizing the geometry of the three fundamental differential operators:

**Gradient · Divergence · Curl**

This is intended as a teaching and computational companion for Mathematical Physics, Vector Calculus, Electromagnetism, Fluid Mechanics and introductory computational physics.

## Laboratory modules

### Gradient

Start with a scalar field

\[
f(x,y)
\]

and investigate

\[
\nabla f =
\frac{\partial f}{\partial x}\hat{i}
+
\frac{\partial f}{\partial y}\hat{j}.
\]

The laboratory displays the scalar surface, gradient vectors, gradient magnitude and numerical derivative diagnostics.

Available scalar fields:

- Paraboloid
- Saddle
- Gaussian hill
- Sinusoidal
- Mexican hat

### Divergence

For a vector field

\[
\mathbf F=F_x\hat{i}+F_y\hat{j},
\]

the application calculates

\[
\nabla\cdot\mathbf F=
\frac{\partial F_x}{\partial x}
+
\frac{\partial F_y}{\partial y}.
\]

The visualization combines the vector field, streamlines, signed divergence map and a 3-D divergence landscape.

### Curl

For the two-dimensional fields used by the laboratory,

\[
(\nabla\times\mathbf F)_z=
\frac{\partial F_y}{\partial x}
-
\frac{\partial F_x}{\partial y}.
\]

The interface displays the vector field together with the signed curl distribution and a 3-D curl landscape.

Available vector fields:

- Uniform
- Radial source
- Radial sink
- Solid-body rotation
- Saddle flow
- Shear flow
- Vortex
- Spiral source

## Compare mode

The **Compare** laboratory puts gradient, divergence and curl into a common conceptual framework.

| Operator | Input | Output | Interpretation |
|---|---|---|---|
| Gradient \(\nabla f\) | Scalar field | Vector field | Steepest increase |
| Divergence \(\nabla\cdot\mathbf F\) | Vector field | Scalar field | Source / sink behaviour |
| Curl \(\nabla\times\mathbf F\) | Vector field | Vector field | Local rotational tendency |

## Numerical computation

The application uses a Cartesian numerical grid and evaluates spatial derivatives with NumPy finite differences.

The grid spacing is explicitly supplied to the differentiation routine:

\[
dx=x_{i+1}-x_i,\qquad dy=y_{j+1}-y_j.
\]

The user can vary:

- domain size;
- grid resolution;
- vector density;
- arrow scale;
- contours;
- streamlines.

This makes the application useful not only for visualization but also for discussing **discretization and numerical convergence**.

## Run locally

Install the dependencies:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Run:

\`\`\`bash
streamlit run app.py
\`\`\`

## Technology

- Python
- Streamlit
- NumPy
- Plotly

## Author

**Aman Kumar Patel**

Made with ❤️ by Aman Kumar Patel
