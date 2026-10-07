import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Vector Calculus Laboratory", page_icon="∇", layout="wide")

st.markdown("""
<style>
.block-container{padding-top:1.5rem}
.hero{padding:1.5rem;border-radius:18px;background:linear-gradient(135deg,rgba(70,100,160,.18),rgba(130,70,170,.12));border:1px solid rgba(128,128,128,.2);margin-bottom:1rem}
.hero h1{margin:0;font-size:2.5rem}.hero p{opacity:.78;font-size:1.05rem}
.formula{padding:.9rem 1rem;border-left:4px solid #4b8bbe;background:rgba(75,139,190,.08);border-radius:8px;margin:.6rem 0}
.footer{text-align:center;opacity:.65;padding:2rem 0 .5rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>∇ Vector Calculus Laboratory</h1>
<p>Explore the geometry and physics of <b>gradient</b>, <b>divergence</b> and <b>curl</b> interactively.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Mathematical engine ----------------
def scalar(x,y,name):
    if name=="Paraboloid": return x*x+y*y
    if name=="Saddle": return x*x-y*y
    if name=="Gaussian hill": return np.exp(-(x*x+y*y)/2)
    if name=="Sinusoidal": return np.sin(x)*np.cos(y)
    r2=x*x+y*y
    return (1-r2)*np.exp(-r2/2)

def vector(x,y,name):
    if name=="Uniform": return np.ones_like(x),np.zeros_like(y)
    if name=="Radial source": return x,y
    if name=="Radial sink": return -x,-y
    if name=="Solid-body rotation": return -y,x
    if name=="Saddle flow": return x,-y
    if name=="Shear flow": return y,np.zeros_like(y)
    r2=x*x+y*y+0.35
    if name=="Vortex": return -y/r2,x/r2
    return x-y,x+y

def ops(U,V,dx,dy):
    dUdy,dUdx=np.gradient(U,dy,dx)
    dVdy,dVdx=np.gradient(V,dy,dx)
    return dUdx+dVdy,dVdx-dUdy

def arrows(fig,X,Y,U,V,spacing,scale):
    step=max(1,int(len(x)/spacing))
    xx=X[::step,::step]; yy=Y[::step,::step]
    uu=U[::step,::step]; vv=V[::step,::step]
    m=np.nanpercentile(np.hypot(uu,vv),90)
    if m==0:m=1
    for a,b,c,d in zip(xx.ravel(),yy.ravel(),uu.ravel(),vv.ravel()):
        c,d=c/m*scale,d/m*scale
        fig.add_trace(go.Scatter(x=[a,a+c],y=[b,b+d],mode="lines",
            line=dict(width=1.5),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[a+c],y=[b+d],mode="markers",
            marker=dict(size=4,symbol="triangle-up"),showlegend=False,
            hovertemplate=f"x={a:.2f}<br>y={b:.2f}<extra></extra>"))

def streamline(fig,U,V,x,y):
    seeds=np.linspace(x[0]*.9,x[-1]*.9,9)
    for sx in seeds:
        for sy in [y[0]*.9,0,y[-1]*.9]:
            px,py=sx,sy; xs=[px]; ys=[py]
            for _ in range(80):
                ix=np.clip(np.searchsorted(x,px)-1,0,len(x)-1)
                iy=np.clip(np.searchsorted(y,py)-1,0,len(y)-1)
                ux,uy=U[iy,ix],V[iy,ix]
                mag=np.hypot(ux,uy)
                if mag<1e-8: break
                px += 0.10*ux/max(mag,.4)
                py += 0.10*uy/max(mag,.4)
                if not (x[0]<=px<=x[-1] and y[0]<=py<=y[-1]): break
                xs.append(px);ys.append(py)
            if len(xs)>4:
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",
                    line=dict(width=1),opacity=.5,showlegend=False,hoverinfo="skip"))

# ---------------- Controls ----------------
st.sidebar.header("Laboratory")
mode=st.sidebar.radio("Operator",["Gradient","Divergence","Curl","Compare"])
extent=st.sidebar.slider("Domain",2.0,8.0,4.0,.5)
N=st.sidebar.slider("Grid",25,101,61,4)
density=st.sidebar.slider("Vector density",6,20,12)
scale=st.sidebar.slider("Arrow scale",.2,1.5,.65,.05)
contours=st.sidebar.checkbox("Contours",True)
streams=st.sidebar.checkbox("Streamlines",True)

x=np.linspace(-extent,extent,N); y=np.linspace(-extent,extent,N)
X,Y=np.meshgrid(x,y); dx=x[1]-x[0]; dy=y[1]-y[0]

# ---------------- Gradient ----------------
if mode=="Gradient":
    names=["Paraboloid","Saddle","Gaussian hill","Sinusoidal","Mexican hat"]
    name=st.sidebar.selectbox("Scalar field f(x,y)",names)
    F=scalar(X,Y,name)
    fy,fx=np.gradient(F,dy,dx)
    M=np.hypot(fx,fy)

    st.header("Gradient")
    st.markdown('<div class="formula">∇f = (∂f/∂x)i + (∂f/∂y)j</div>',unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("max |∇f|",f"{M.max():.5g}")
    c2.metric("mean |∇f|",f"{M.mean():.5g}")
    c3.metric("dx",f"{dx:.4f}")
    c4.metric("dy",f"{dy:.4f}")

    left,right=st.columns(2)
    with left:
        fig=go.Figure(go.Surface(x=x,y=y,z=F,colorscale="Viridis",
            contours=dict(z=dict(show=contours,usecolormap=True,project_z=True)),
            colorbar=dict(title="f")))
        step=max(1,N//density)
        gx,gy=fx[::step,::step],fy[::step,::step]
        norm=np.nanpercentile(np.hypot(gx,gy),90) or 1
        fig.add_trace(go.Cone(x=X[::step,::step].ravel(),y=Y[::step,::step].ravel(),
            z=(F[::step,::step]+.03*(F.max()-F.min()+1e-9)).ravel(),
            u=(gx/norm*scale).ravel(),v=(gy/norm*scale).ravel(),
            w=np.zeros_like(gx).ravel(),anchor="tail",sizemode="absolute",
            sizeref=.35,showscale=False,name="gradient"))
        fig.update_layout(height=600,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="f(x,y)"),
            margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure(go.Heatmap(x=x,y=y,z=M,colorscale="Turbo",colorbar=dict(title="|∇f|")))
        arrows(fig,X,Y,fx,fy,density,scale)
        fig.update_layout(height=600,title="Gradient direction + magnitude",
            xaxis_title="x",yaxis_title="y",yaxis=dict(scaleanchor="x"))
        st.plotly_chart(fig,use_container_width=True)

    st.success("The gradient points in the direction of steepest increase. Its magnitude is the maximum directional derivative.")

# ---------------- Divergence / Curl ----------------
elif mode in ["Divergence","Curl"]:
    names=["Uniform","Radial source","Radial sink","Solid-body rotation","Saddle flow","Shear flow","Vortex","Spiral source"]
    name=st.sidebar.selectbox("Vector field F(x,y)",names)
    U,V=vector(X,Y,name)
    D,C=ops(U,V,dx,dy)
    Z=D if mode=="Divergence" else C

    st.header(mode)
    formula="∇·F = ∂Fₓ/∂x + ∂Fᵧ/∂y" if mode=="Divergence" else "(∇×F)z = ∂Fᵧ/∂x − ∂Fₓ/∂y"
    st.markdown(f'<div class="formula">{formula}</div>',unsafe_allow_html=True)

    a,b,c,d=st.columns(4)
    a.metric("maximum",f"{Z.max():.5g}")
    b.metric("minimum",f"{Z.min():.5g}")
    c.metric("mean",f"{Z.mean():.5g}")
    d.metric("RMS",f"{np.sqrt(np.mean(Z*Z)):.5g}")

    left,right=st.columns(2)
    with left:
        fig=go.Figure(go.Contour(x=x,y=y,z=Z,colorscale="RdBu_r",zmid=0,
            contours=dict(showlabels=contours),colorbar=dict(title=mode)))
        if streams: streamline(fig,U,V,x,y)
        arrows(fig,X,Y,U,V,density,scale)
        fig.update_layout(height=600,title=f"{mode} map + vector field",
            xaxis_title="x",yaxis_title="y",yaxis=dict(scaleanchor="x"))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=go.Figure(go.Surface(x=x,y=y,z=Z,colorscale="RdBu_r",
            cmin=-np.max(np.abs(Z)),cmax=np.max(np.abs(Z)),
            colorbar=dict(title=mode),
            contours=dict(z=dict(show=contours,project_z=True))))
        fig.update_layout(height=600,title=f"3-D {mode} landscape",
            scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title=mode),
            margin=dict(l=0,r=0,t=50,b=0))
        st.plotly_chart(fig,use_container_width=True)

    if mode=="Divergence":
        st.info("Interpretation: positive divergence indicates local net outflow (source-like behaviour); negative divergence indicates net inflow (sink-like behaviour).")
    else:
        st.info("Interpretation: curl measures local rotational tendency. For a 2-D field, the plotted quantity is the z-component of the curl.")

# ---------------- Compare ----------------
else:
    st.header("Vector Calculus Comparison Laboratory")
    sf=st.sidebar.selectbox("Scalar field",["Paraboloid","Saddle","Gaussian hill","Sinusoidal","Mexican hat"])
    vf=st.sidebar.selectbox("Vector field",["Uniform","Radial source","Radial sink","Solid-body rotation","Saddle flow","Shear flow","Vortex","Spiral source"])

    F=scalar(X,Y,sf); fy,fx=np.gradient(F,dy,dx); G=np.hypot(fx,fy)
    U,V=vector(X,Y,vf); D,C=ops(U,V,dx,dy)

    tabs=st.tabs(["Gradient","Divergence","Curl","Concept Map"])
    for tab,title,Z,U0,V0,cs in [
        (tabs[0],"Gradient magnitude",G,fx,fy,"Turbo"),
        (tabs[1],"Divergence",D,U,V,"RdBu_r"),
        (tabs[2],"Curl z-component",C,U,V,"RdBu_r")]:
        with tab:
            fig=go.Figure(go.Heatmap(x=x,y=y,z=Z,colorscale=cs,
                zmid=0 if title!="Gradient magnitude" else None,colorbar=dict(title=title)))
            arrows(fig,X,Y,U0,V0,density,scale)
            fig.update_layout(height=620,xaxis_title="x",yaxis_title="y",yaxis=dict(scaleanchor="x"))
            st.plotly_chart(fig,use_container_width=True)

    with tabs[3]:
        st.markdown("""
| Operator | Input | Output | Geometric meaning |
|---|---|---|---|
| **Gradient ∇f** | Scalar field | Vector field | Direction of steepest increase |
| **Divergence ∇·F** | Vector field | Scalar field | Local source/sink strength |
| **Curl ∇×F** | Vector field | Vector field | Local rotational tendency |
""")
        st.latex(r"\nabla f,\qquad \nabla\cdot\mathbf F,\qquad \nabla\times\mathbf F")

# ---------------- Diagnostics ----------------
with st.expander("Numerical method and grid diagnostics"):
    st.write(f"Domain: [{-extent:.2f}, {extent:.2f}] × [{-extent:.2f}, {extent:.2f}]")
    st.write(f"Grid: {N} × {N}")
    st.write(f"dx = {dx:.6f},   dy = {dy:.6f}")
    st.write("Spatial derivatives are calculated with finite differences using NumPy's gradient operator.")
    st.write("Increase the grid resolution to investigate discretization and numerical-convergence effects.")

st.markdown('<div class="footer">Made with ❤️ by Aman Kumar Patel</div>',unsafe_allow_html=True)
