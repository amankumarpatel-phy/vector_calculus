import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Vector Calculus Laboratory", page_icon="∇", layout="wide")

# ============================================================
# VISUAL SYSTEM
# ============================================================
st.markdown("""
<style>
.block-container{padding-top:1.2rem;max-width:1500px}
.hero{padding:1.45rem 1.6rem;border-radius:20px;background:linear-gradient(135deg,rgba(48,80,145,.18),rgba(108,60,145,.13));border:1px solid rgba(128,128,128,.22);margin-bottom:1rem}
.hero h1{margin:0;font-size:2.45rem;letter-spacing:-.02em}.hero p{margin:.45rem 0 0;opacity:.78}
.section{font-size:1.35rem;font-weight:700;margin:.5rem 0}
.formula{padding:.8rem 1rem;border-left:4px solid #4b8bbe;background:rgba(75,139,190,.08);border-radius:9px;margin:.65rem 0}
.small{opacity:.7;font-size:.9rem}
.footer{text-align:center;opacity:.62;padding:2rem 0 .5rem}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>∇ Vector Calculus Laboratory</h1>
<p>A high-clarity computational environment for understanding <b>gradient</b>,
<b>divergence</b> and <b>curl</b> geometrically and numerically.</p>
</div>
""",unsafe_allow_html=True)

# ============================================================
# FIELDS
# ============================================================
SCALARS={
"Paraboloid":(lambda x,y:x*x+y*y,r"f=x^2+y^2",r"\nabla f=2x\,\hat i+2y\,\hat j"),
"Saddle":(lambda x,y:x*x-y*y,r"f=x^2-y^2",r"\nabla f=2x\,\hat i-2y\,\hat j"),
"Gaussian hill":(lambda x,y:np.exp(-(x*x+y*y)/2),r"f=e^{-(x^2+y^2)/2}",r"\nabla f=-f(x\,\hat i+y\,\hat j)"),
"Sinusoidal":(lambda x,y:np.sin(x)*np.cos(y),r"f=\sin x\cos y",r"\nabla f=\cos x\cos y\,\hat i-\sin x\sin y\,\hat j"),
"Mexican hat":(lambda x,y:(1-x*x-y*y)*np.exp(-(x*x+y*y)/2),r"f=(1-r^2)e^{-r^2/2}",r"\text{Analytical derivatives evaluated symbolically}"),
}
VECTORS={
"Uniform":(lambda x,y:(np.ones_like(x),np.zeros_like(y)),r"\mathbf F=\hat i"),
"Radial source":(lambda x,y:(x,y),r"\mathbf F=x\hat i+y\hat j"),
"Radial sink":(lambda x,y:(-x,-y),r"\mathbf F=-x\hat i-y\hat j"),
"Solid-body rotation":(lambda x,y:(-y,x),r"\mathbf F=-y\hat i+x\hat j"),
"Saddle flow":(lambda x,y:(x,-y),r"\mathbf F=x\hat i-y\hat j"),
"Shear flow":(lambda x,y:(y,np.zeros_like(y)),r"\mathbf F=y\hat i"),
"Vortex":(lambda x,y:(-y/(x*x+y*y+.35),x/(x*x+y*y+.35)),r"\text{Regularized vortex}"),
"Spiral source":(lambda x,y:(x-y,x+y),r"\mathbf F=(x-y)\hat i+(x+y)\hat j"),
}

def field_grid(extent,n):
    x=np.linspace(-extent,extent,n);y=np.linspace(-extent,extent,n)
    return x,y,np.meshgrid(x,y)

def derivatives(U,V,dx,dy):
    Uy,Ux=np.gradient(U,dy,dx)
    Vy,Vx=np.gradient(V,dy,dx)
    return Ux+Vy,Vx-Uy

def clean_range(z):
    m=float(np.nanpercentile(np.abs(z),99))
    return max(m,1e-9)

def add_quiver(fig,X,Y,U,V,density,scale):
    step=max(1,int(len(x)/density))
    xx=X[::step,::step]; yy=Y[::step,::step]
    uu=U[::step,::step]; vv=V[::step,::step]
    mag=np.hypot(uu,vv); norm=np.nanpercentile(mag,92)
    if norm<1e-12:norm=1
    for xi,yi,ui,vi,mi in zip(xx.ravel(),yy.ravel(),uu.ravel(),vv.ravel(),mag.ravel()):
        ui,vi=ui/norm*scale,vi/norm*scale
        fig.add_trace(go.Scatter(
            x=[xi,xi+ui],y=[yi,yi+vi],mode="lines",
            line=dict(width=1.5),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[xi+ui],y=[yi+vi],mode="markers",
            marker=dict(size=4,symbol="triangle-up"),
            showlegend=False,
            hovertemplate=f"x={xi:.3f}<br>y={yi:.3f}<br>|F|={mi:.3f}<extra></extra>"))

def add_streamlines(fig,U,V,x,y):
    seeds=np.linspace(x[0]*.92,x[-1]*.92,11)
    for sx in seeds:
        for sy in (y[0]*.92,0,y[-1]*.92):
            px,py=sx,sy; xs=[px];ys=[py]
            for _ in range(100):
                ix=np.clip(np.searchsorted(x,px)-1,0,len(x)-1)
                iy=np.clip(np.searchsorted(y,py)-1,0,len(y)-1)
                ux,uy=U[iy,ix],V[iy,ix];m=np.hypot(ux,uy)
                if m<1e-8:break
                h=.075/max(m,.35);px+=h*ux;py+=h*uy
                if not(x[0]<=px<=x[-1] and y[0]<=py<=y[-1]):break
                xs.append(px);ys.append(py)
            if len(xs)>5:
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",
                    line=dict(width=1),opacity=.45,showlegend=False,hoverinfo="skip"))

def base2d(title):
    return dict(height=620,title=dict(text=title,x=.02,xanchor="left"),
        margin=dict(l=55,r=30,t=60,b=50),xaxis=dict(title="x",zeroline=True,showgrid=True),
        yaxis=dict(title="y",zeroline=True,showgrid=True,scaleanchor="x",scaleratio=1),
        hovermode="closest")

# ============================================================
# CONTROLS
# ============================================================
st.sidebar.header("Experiment")
mode=st.sidebar.radio("Differential operator",["Gradient","Divergence","Curl","Comparison"])
st.sidebar.divider()
extent=st.sidebar.slider("Domain half-width",1.5,8.0,4.0,.5)
N=st.sidebar.slider("Numerical resolution",31,121,61,5)
density=st.sidebar.slider("Vector density",6,18,11)
arrow_scale=st.sidebar.slider("Arrow length",.15,1.0,.45,.05)
show_grid=st.sidebar.checkbox("Grid / contours",True)
show_stream=st.sidebar.checkbox("Streamlines",True)

x,y,X,Y=field_grid(extent,N);dx=x[1]-x[0];dy=y[1]-y[0]

# ============================================================
# GRADIENT
# ============================================================
if mode=="Gradient":
    name=st.sidebar.selectbox("Scalar field",list(SCALARS))
    fun,formula,grad_formula=SCALARS[name]
    F=fun(X,Y);Fy,Fx=np.gradient(F,dy,dx);M=np.hypot(Fx,Fy)

    st.markdown('<div class="section">Gradient of a scalar field</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="formula"><b>Field:</b> {formula}<br><b>Operator:</b> {grad_formula}</div>',unsafe_allow_html=True)

    a,b,c,d=st.columns(4)
    a.metric("Maximum |∇f|",f"{M.max():.5g}")
    b.metric("Mean |∇f|",f"{M.mean():.5g}")
    b.metric("Grid Δx",f"{dx:.4g}")
    d.metric("Grid points",f"{N*N:,}")

    left,right=st.columns([1.12,1])
    with left:
        fig=go.Figure(go.Surface(x=x,y=y,z=F,colorscale="Viridis",
            contours=dict(z=dict(show=show_grid,usecolormap=True,project_z=True)),
            colorbar=dict(title="f"),hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>f=%{z:.4f}<extra></extra>"))
        step=max(1,N//density);gx=Fx[::step,::step];gy=Fy[::step,::step]
        norm=np.nanpercentile(np.hypot(gx,gy),92) or 1
        fig.add_trace(go.Cone(x=X[::step,::step].ravel(),y=Y[::step,::step].ravel(),
            z=(F[::step,::step]+.025*(np.ptp(F)+1e-9)).ravel(),
            u=(gx/norm*arrow_scale).ravel(),v=(gy/norm*arrow_scale).ravel(),
            w=np.zeros_like(gx).ravel(),anchor="tail",sizemode="absolute",sizeref=.3,
            showscale=False,name="∇f"))
        fig.update_layout(height=620,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="f(x,y)",
            camera=dict(eye=dict(x=1.55,y=1.55,z=1.15))),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("3-D surface: height represents f(x,y). Arrows show the local gradient direction.")

    with right:
        fig=go.Figure(go.Heatmap(x=x,y=y,z=M,colorscale="Turbo",colorbar=dict(title="|∇f|"),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>|∇f|=%{z:.4f}<extra></extra>"))
        add_quiver(fig,X,Y,Fx,Fy,density,arrow_scale)
        fig.update_layout(**base2d("Gradient magnitude and direction"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Colour = magnitude. Arrows = direction of steepest increase.")

    px=st.slider("Probe x",float(-extent),float(extent),0.0,.05,key="gx")
    py=st.slider("Probe y",float(-extent),float(extent),0.0,.05,key="gy")
    iy=np.argmin(np.abs(y-py));ix=np.argmin(np.abs(x-px))
    p1,p2,p3=st.columns(3)
    p1.metric("f(x₀,y₀)",f"{F[iy,ix]:.6g}")
    p2.metric("∂f/∂x",f"{Fx[iy,ix]:.6g}")
    p3.metric("∂f/∂y",f"{Fy[iy,ix]:.6g}")
    st.info("Geometric interpretation: ∇f is perpendicular to a level curve f=constant and points toward the direction of maximum increase.")

# ============================================================
# DIVERGENCE / CURL
# ============================================================
elif mode in ["Divergence","Curl"]:
    name=st.sidebar.selectbox("Vector field",list(VECTORS))
    fun,formula=VECTORS[name]
    U,V=fun(X,Y);D,C=derivatives(U,V,dx,dy)
    Z=D if mode=="Divergence" else C
    op_formula=r"\nabla\cdot\mathbf F=\partial_xF_x+\partial_yF_y" if mode=="Divergence" else r"(\nabla\times\mathbf F)_z=\partial_xF_y-\partial_yF_x"

    st.markdown(f'<div class="section">{mode} of a vector field</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="formula"><b>Field:</b> {formula}<br><b>Operator:</b> {op_formula}</div>',unsafe_allow_html=True)

    a,b,c,d=st.columns(4)
    a.metric("Maximum",f"{Z.max():.5g}");b.metric("Minimum",f"{Z.min():.5g}")
    c.metric("Mean",f"{Z.mean():.5g}");d.metric("RMS",f"{np.sqrt(np.mean(Z*Z)):.5g}")

    left,right=st.columns([1.12,1])
    with left:
        lim=clean_range(Z)
        fig=go.Figure(go.Contour(x=x,y=y,z=Z,colorscale="RdBu_r",zmin=-lim,zmax=lim,zmid=0,
            contours=dict(showlabels=show_grid),colorbar=dict(title=mode),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{z:.5g}<extra></extra>"))
        if show_stream:add_streamlines(fig,U,V,x,y)
        add_quiver(fig,X,Y,U,V,density,arrow_scale)
        fig.update_layout(**base2d(f"{mode}: signed scalar map + vector field"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("Blue/red sign structure shows negative/positive values; arrows show the underlying vector field.")

    with right:
        lim=clean_range(Z)
        fig=go.Figure(go.Surface(x=x,y=y,z=Z,colorscale="RdBu_r",cmin=-lim,cmax=lim,
            colorbar=dict(title=mode),contours=dict(z=dict(show=show_grid,project_z=True)),
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>value=%{z:.5g}<extra></extra>"))
        fig.update_layout(height=620,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title=mode,
            zaxis=dict(range=[-lim,lim])),margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("The 3-D height makes the sign and spatial variation immediately visible.")

    px=st.slider("Probe x",float(-extent),float(extent),0.0,.05,key=f"{mode}x")
    py=st.slider("Probe y",float(-extent),float(extent),0.0,.05,key=f"{mode}y")
    iy=np.argmin(np.abs(y-py));ix=np.argmin(np.abs(x-px))
    q1,q2,q3=st.columns(3)
    q1.metric(f"{mode}(x₀,y₀)",f"{Z[iy,ix]:.7g}")
    q2.metric("Fₓ(x₀,y₀)",f"{U[iy,ix]:.6g}")
    q3.metric("Fᵧ(x₀,y₀)",f"{V[iy,ix]:.6g}")

    if mode=="Divergence":
        st.info("Think of a tiny fluid element: positive divergence means more field leaves the element than enters; negative divergence means net inflow.")
    else:
        st.info("Curl measures local rotation. For a 2-D field the displayed quantity is the z-component; positive and negative signs correspond to opposite senses of rotation.")

# ============================================================
# COMPARISON
# ============================================================
else:
    st.markdown('<div class="section">Three operators — one visual language</div>',unsafe_allow_html=True)
    sf=st.sidebar.selectbox("Scalar field",list(SCALARS))
    vf=st.sidebar.selectbox("Vector field",list(VECTORS))
    F=SCALARS[sf][0](X,Y);Fy,Fx=np.gradient(F,dy,dx);G=np.hypot(Fx,Fy)
    U,V=VECTORS[vf][0](X,Y);D,C=derivatives(U,V,dx,dy)

    tabs=st.tabs(["Gradient","Divergence","Curl","Conceptual map"])
    with tabs[0]:
        fig=go.Figure(go.Heatmap(x=x,y=y,z=G,colorscale="Turbo",colorbar=dict(title="|∇f|")))
        add_quiver(fig,X,Y,Fx,Fy,density,arrow_scale)
        fig.update_layout(**base2d(f"Gradient of {sf}"))
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"\nabla f\;\rightarrow\;\text{vector field}")

    with tabs[1]:
        lim=clean_range(D)
        fig=go.Figure(go.Contour(x=x,y=y,z=D,colorscale="RdBu_r",zmid=0,zmin=-lim,zmax=lim,colorbar=dict(title="∇·F")))
        add_quiver(fig,X,Y,U,V,density,arrow_scale)
        fig.update_layout(**base2d(f"Divergence of {vf}"))
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"\nabla\cdot\mathbf F\;\rightarrow\;\text{scalar field}")

    with tabs[2]:
        lim=clean_range(C)
        fig=go.Figure(go.Contour(x=x,y=y,z=C,colorscale="RdBu_r",zmid=0,zmin=-lim,zmax=lim,colorbar=dict(title="curl z")))
        add_quiver(fig,X,Y,U,V,density,arrow_scale)
        fig.update_layout(**base2d(f"Curl of {vf}"))
        st.plotly_chart(fig,use_container_width=True)
        st.latex(r"\nabla\times\mathbf F\;\rightarrow\;\text{vector field}")

    with tabs[3]:
        st.markdown("""
| Operator | Acts on | Produces | Geometric question |
|---|---|---|---|
| **Gradient** ∇f | Scalar field | Vector | Where is the field increasing fastest? |
| **Divergence** ∇·F | Vector field | Scalar | Is there net local outflow or inflow? |
| **Curl** ∇×F | Vector field | Vector | How strongly does the field rotate locally? |
""")
        st.markdown("### Visual signatures")
        st.write("**Gradient:** arrows cross level curves normally and point uphill.")
        st.write("**Divergence:** arrows spread from source-like regions and converge into sink-like regions.")
        st.write("**Curl:** arrows circulate around regions of non-zero rotational tendency.")

# ============================================================
# NUMERICAL DIAGNOSTICS
# ============================================================
with st.expander("Numerical diagnostics"):
    st.write(f"Domain = [{-extent:.2f}, {extent:.2f}] × [{-extent:.2f}, {extent:.2f}]")
    st.write(f"Resolution = {N} × {N} = {N*N:,} sample points")
    st.write(f"Δx = {dx:.7f}, Δy = {dy:.7f}")
    st.write("Derivatives are evaluated by finite differences with NumPy's gradient operator.")
    st.write("Increase N and observe whether the displayed field stabilizes: this is a simple numerical-convergence experiment.")

st.markdown('<div class="footer">Made with ❤️ by Aman Kumar Patel</div>',unsafe_allow_html=True)
