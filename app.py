import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Vector Calculus Visualizer",
    page_icon="∇",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size: 2.6rem;
    font-weight: 750;
    margin-bottom: 0.15rem;
}
.subtitle {
    font-size: 1.05rem;
    opacity: 0.78;
    margin-bottom: 1.2rem;
}
.formula {
    padding: 0.85rem 1rem;
    border-left: 4px solid #4b8bbe;
    background: rgba(128,128,128,0.08);
    border-radius: 0.35rem;
    font-size: 1.05rem;
}
.footer {
    text-align: center;
    opacity: 0.65;
    padding: 1.5rem 0 0.5rem;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">∇ Vector Calculus Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive visualization of gradient, divergence and curl — '
    'with numerical computation and vector-field interpretation.</div>',
    unsafe_allow_html=True,
)

# ----------------------------
# Mathematical fields
# ----------------------------
def scalar_field(x, y, kind):
    if kind == "Gaussian":
        return np.exp(-(x**2 + y**2) / 4.0)
    if kind == "Paraboloid":
        return x**2 + y**2
    if kind == "Saddle":
        return x**2 - y**2
    if kind == "Sinusoidal":
        return np.sin(x) * np.cos(y)
    if kind == "Mixed":
        return np.sin(x) * np.cos(y) + 0.12 * (x**2 + y**2)
    return x**2 + y**2

def vector_field(x, y, kind):
    if kind == "Radial source":
        return x, y
    if kind == "Radial sink":
        return -x, -y
    if kind == "Solid-body rotation":
        return -y, x
    if kind == "Irrotational saddle":
        return x, -y
    if kind == "Shear":
        return y, np.zeros_like(y)
    if kind == "Spiral":
        return x - y, x + y
    return x, y

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")

operation = st.sidebar.radio(
    "Choose operation",
    ["Gradient", "Divergence", "Curl"],
)

extent = st.sidebar.slider("Spatial extent", 2.0, 8.0, 4.0, 0.5)
resolution = st.sidebar.slider("Grid resolution", 25, 100, 55, 5)
show_vectors = st.sidebar.checkbox("Show vector arrows", True)
show_contours = st.sidebar.checkbox("Show contours", True)

x = np.linspace(-extent, extent, resolution)
y = np.linspace(-extent, extent, resolution)
X, Y = np.meshgrid(x, y)

dx = x[1] - x[0]
dy = y[1] - y[0]

if operation == "Gradient":
    field_name = st.sidebar.selectbox(
        "Scalar field f(x,y)",
        ["Gaussian", "Paraboloid", "Saddle", "Sinusoidal", "Mixed"],
    )
    F = scalar_field(X, Y, field_name)
    dFdy, dFdx = np.gradient(F, dy, dx)
    magnitude = np.sqrt(dFdx**2 + dFdy**2)

    st.subheader(f"Gradient of f(x,y) — {field_name}")
    st.markdown(
        '<div class="formula">∇f = (∂f/∂x) i + (∂f/∂y) j</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=x, y=y, z=F,
        colorscale="Viridis",
        colorbar=dict(title="f(x,y)"),
        contours=dict(z=dict(show=show_contours, usecolormap=True, project_z=True)),
        name="Scalar field",
    ))

    if show_vectors:
        step = max(1, resolution // 18)
        # Arrows are drawn slightly above the surface using cone traces.
        gx = dFdx[::step, ::step]
        gy = dFdy[::step, ::step]
        gz = np.full_like(gx, np.nanmax(F) * 0.025)
        scale = np.nanmax(np.sqrt(gx**2 + gy**2))
        if scale > 0:
            gx, gy = gx / scale, gy / scale
        fig.add_trace(go.Cone(
            x=X[::step, ::step].ravel(),
            y=Y[::step, ::step].ravel(),
            z=(F[::step, ::step] + gz).ravel(),
            u=gx.ravel(),
            v=gy.ravel(),
            w=np.zeros_like(gx).ravel(),
            sizemode="absolute",
            sizeref=0.45,
            showscale=False,
            anchor="tail",
            name="Gradient",
        ))

    fig.update_layout(
        height=650,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="f(x,y)",
            aspectmode="auto",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Max |∇f|", f"{np.max(magnitude):.4f}")
    c2.metric("Mean |∇f|", f"{np.mean(magnitude):.4f}")
    c3.metric("Grid spacing", f"{dx:.3f}")

    st.info(
        "Interpretation: the gradient points in the direction of the steepest "
        "increase of the scalar field. Its magnitude gives the maximum rate of change."
    )

else:
    field_name = st.sidebar.selectbox(
        "Vector field F(x,y)",
        [
            "Radial source",
            "Radial sink",
            "Solid-body rotation",
            "Irrotational saddle",
            "Shear",
            "Spiral",
        ],
    )

    U, V = vector_field(X, Y, field_name)

    dUdy, dUdx = np.gradient(U, dy, dx)
    dVdy, dVdx = np.gradient(V, dy, dx)

    divergence = dUdx + dVdy
    curl_z = dVdx - dUdy

    if operation == "Divergence":
        scalar = divergence
        title = f"Divergence of F — {field_name}"
        formula = "∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y"
        interpretation = (
            "Divergence measures the local net outflow of a vector field. "
            "Positive values behave like sources; negative values behave like sinks."
        )
        colorbar_title = "∇·F"
    else:
        scalar = curl_z
        title = f"Curl of F — {field_name}"
        formula = "∇×F = (∂Fᵧ/∂x − ∂Fₓ/∂y) k"
        interpretation = (
            "For a 2-D field, curl is represented by its z-component. "
            "It measures the local tendency of the field to rotate."
        )
        colorbar_title = "(∇×F)z"

    st.subheader(title)
    st.markdown(
        f'<div class="formula">{formula}</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    if show_contours:
        fig.add_trace(go.Contour(
            x=x,
            y=y,
            z=scalar,
            colorscale="RdBu_r",
            colorbar=dict(title=colorbar_title),
            contours=dict(showlabels=True),
            name=colorbar_title,
        ))

    if show_vectors:
        step = max(1, resolution // 18)
        fig.add_trace(go.Cone(
            x=X[::step, ::step].ravel(),
            y=Y[::step, ::step].ravel(),
            z=np.zeros_like(X[::step, ::step]).ravel(),
            u=U[::step, ::step].ravel(),
            v=V[::step, ::step].ravel(),
            w=np.zeros_like(U[::step, ::step]).ravel(),
            sizemode="absolute",
            sizeref=0.42,
            showscale=False,
            anchor="tail",
            name="Vector field",
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="x",
        yaxis_title="y",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Maximum", f"{np.max(scalar):.4f}")
    c2.metric("Minimum", f"{np.min(scalar):.4f}")
    c3.metric("Mean", f"{np.mean(scalar):.4f}")

    st.info(interpretation)

# ----------------------------
# Reference panel
# ----------------------------
with st.expander("Mathematical reference"):
    st.markdown("""
### Gradient

For a scalar field $f(x,y,z)$,

$$
\nabla f =
\frac{\partial f}{\partial x}\hat{i}
+\frac{\partial f}{\partial y}\hat{j}
+\frac{\partial f}{\partial z}\hat{k}.
$$

The gradient is a **vector field**.

### Divergence

For $\mathbf{F}=F_x\hat{i}+F_y\hat{j}+F_z\hat{k}$,

$$
\nabla\cdot\mathbf{F}
=
\frac{\partial F_x}{\partial x}
+\frac{\partial F_y}{\partial y}
+\frac{\partial F_z}{\partial z}.
$$

The divergence is a **scalar field**.

### Curl

$$
\nabla\times\mathbf{F}
=
\begin{vmatrix}
\hat{i}&\hat{j}&\hat{k}\\
\partial_x&\partial_y&\partial_z\\
F_x&F_y&F_z
\end{vmatrix}.
$$

For the 2-D field used here,

$$
(\nabla\times\mathbf{F})_z
=
\frac{\partial F_y}{\partial x}
-
\frac{\partial F_x}{\partial y}.
$$

The curl is a **vector field** in 3-D, represented by its z-component for this 2-D visualization.
""")

st.markdown(
    '<div class="footer">Made with ❤️ by Aman Kumar Patel</div>',
    unsafe_allow_html=True,
)
