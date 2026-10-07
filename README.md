# ∇ Vector Calculus Visualizer

An interactive Streamlit application for visualizing the three central differential operators of vector calculus:

- **Gradient** — direction and magnitude of maximum increase of a scalar field.
- **Divergence** — local net outflow/inflow of a vector field.
- **Curl** — local rotational tendency of a vector field.

## Features

### Gradient
Choose among:
- Gaussian
- Paraboloid
- Saddle
- Sinusoidal
- Mixed scalar fields

The app displays the scalar surface together with gradient vectors.

### Divergence
Choose vector fields such as:
- Radial source
- Radial sink
- Solid-body rotation
- Irrotational saddle
- Shear
- Spiral

The divergence is calculated numerically using finite differences.

### Curl
For the same 2-D vector fields, the z-component of curl is calculated as

\[
(\\nabla \\times \\mathbf{F})_z =
\\frac{\\partial F_y}{\\partial x} -
\\frac{\\partial F_x}{\\partial y}.
\]

## Numerical method

The application uses NumPy's central-difference gradient operator through:

`numpy.gradient`

with the actual grid spacing supplied in both coordinate directions.

Plotly provides the interactive 2-D contour and 3-D surface/vector visualizations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Suggested teaching use

This project is designed as a visual companion for undergraduate and postgraduate Mathematical Physics / Vector Calculus. It allows students to connect the formal differential operators with their geometric meaning.

## Project

**Vector Calculus Visualizer**

Author: **Aman Kumar Patel**

Made with ❤️ by Aman Kumar Patel
