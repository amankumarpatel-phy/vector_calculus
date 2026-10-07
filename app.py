import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Vector Calculus Laboratory", page_icon="∇", layout="wide")

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
.block-container{padding-top:1.15rem;max-width:1550px}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(135deg,rgba(42,78,150,.18),rgba(116,61,150,.13));border:1px solid rgba(128,128,128,.2);margin-bottom:1rem}
.hero h1{margin:0;font-size:2.5rem;letter-spacing:-.025em}
.hero p{margin:.45rem 0 0;opacity:.78;font-size:1.05rem}
.formula{padding:.85rem 1rem;border-left:4px solid #4b8bbe;background:rgba(75,139,190,.08);border-radius:9px;margin:.65rem 0}
.small{opacity:.7;font-size:.9rem}
.footer{text-align:center;opacity:.62;padding:2rem 0 .5rem}
</style>
""",unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>∇ Vector Calculus Laboratory</h1>
<p>Interactive computational laboratory for the geometry, numerics and physical interpretation of
<b>gradient</b>, <b>divergence</b> and <b>curl</b>.</p>
</div>
""",unsafe_allow_html=True)

# ============================================================
# MATHEMATICAL MODELS
# ============================================================
SCALARS={
"Paraboloid":(lambda x,y:x*x+y*y,r"f=x^2+y^2",r"\nabla f=2x\,\hat i+2y\,\hat j"),
"Saddle":(lambda x,y:x*x-y*y,r"f=x^2-y^2",r"\nabla f=2x\,\hat i-2y\,\hat j"),
"Gaussian hill":(lambda x,y:np.exp(-(x*x+y*y)/2),r"f=e^{-(x^2+y^2)/2}",r"\nabla f=-f(x\,\hat i+y\,\hat j)"),
"Sinusoidal":(lambda x,y:np.sin(x)*np.cos(y),r"f=\sin x\cos y",r"\nabla f=\cos x\cos y\,\hat i-\sin x\sin y\,\hat j"),
"Mexican hat":(lambda x,y:(1-x*x-y*y)*np.exp(-(x*x+y*y)/2),r"f=(1-r^2)e^{-r^2/2}",r"\text{Numerical gradient used}"),
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

def grid2d(extent,n):
    x=np.linspace(-extent,extent,n); y=np.linspace(-extent,extent,n)
    return x,y,np.meshgrid(x,y)

def deriv(U,V,dx,dy):
    Uy,Ux=np.gradient(U,dy,dx); Vy,Vx=np.gradient(V,dy,dx)
    return Ux+Vy,Vx-Uy

def qrange(z):
    return max(float(np.nanpercentile(np.abs(z),99)),1e-10)

def quiver2d(fig,X,Y,U,V,x,density,scale):
    step=max(1,int(len(x)/density))
    xx=X[::step,::step]; yy=Y[::step,::step]
    uu=U[::step,::step]; vv=V[::step,::step]
    norm=np.nanpercentile(np.hypot(uu,vv),92) or 1
    for xi,yi,ui,vi in zip(xx.ravel(),yy.ravel(),uu.ravel(),vv.ravel()):
        ui,vi=ui/norm*scale,vi/norm*scale
        fig.add_trace(go.Scatter(x=[xi,xi+ui],y=[yi,yi+vi],mode="lines",
            line=dict(width=1.5),showlegend=False,hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[xi+ui],y=[yi+vi],mode="markers",
            marker=dict(size=4,symbol="triangle-up"),showlegend=False,hoverinfo="skip"))

def streamlines(fig,U,V,x,y):
    for sx in np.linspace(x[0]*.9,x[-1]*.9,9):
        for sy in (y[0]*.9,0,y[-1]*.9):
            px,py=sx,sy; xs=[px];ys=[py]
            for _ in range(100):
                ix=np.clip(np.searchsorted(x,px)-1,0,len(x)-1)
                iy=np.clip(np.searchsorted(y,py)-1,0,len(y)-1)
                ux,uy=U[iy,ix],V[iy,ix]; m=np.hypot(ux,uy)
                if m<1e-8:break
                h=.075/max(m,.4);px+=h*ux;py+=h*uy
                if not(x[0]<=px<=x[-1] and y[0]<=py<=y[-1]):break
                xs.append(px);ys.append(py)
            if len(xs)>5:
                fig.add_trace(go.Scatter(x=xs,y=ys,mode="lines",line=dict(width=1),
                    opacity=.45,showlegend=False,hoverinfo="skip"))

def layout2d(title):
    return dict(height=610,title=dict(text=title,x=.02),margin=dict(l=50,r=25,t=55,b=45),
                xaxis=dict(title="x",zeroline=True),yaxis=dict(title="y",scaleanchor="x",scaleratio=1))

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Laboratory")
mode=st.sidebar.radio("Module",["Gradient","Divergence","Curl","3D Vector Field","Flow Animation","Convergence","Comparison"])
st.sidebar.divider()
extent=st.sidebar.slider("Domain half-width",1.5,8.0,4.0,.5)
N=st.sidebar.slider("Resolution",31,121,61,5)
density=st.sidebar.slider("Vector density",6,18,11)
arrow_scale=st.sidebar.slider("Arrow scale",.15,1.0,.45,.05)
show_contours=st.sidebar.checkbox("Contours",True)
show_stream=st.sidebar.checkbox("Streamlines",True)

x,y,mesh=grid2d(extent,N); X,Y=mesh; dx=x[1]-x[0];dy=y[1]-y[0]

# ============================================================
# GRADIENT
# ============================================================
if mode=="Gradient":
    name=st.sidebar.selectbox("Scalar field",list(SCALARS))
    fun,formula,gformula=SCALARS[name]; F=fun(X,Y)
    Fy,Fx=np.gradient(F,dy,dx); M=np.hypot(Fx,Fy)

    st.header("Gradient — scalar field → vector field")
    st.markdown(f'<div class="formula"><b>Field:</b> {formula}<br><b>Operator:</b> {gformula}</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4)
    a.metric("max |∇f|",f"{M.max():.5g}");b.metric("mean |∇f|",f"{M.mean():.5g}")
    c.metric("Δx",f"{dx:.4g}");d.metric("samples",f"{N*N:,}")

    l,r=st.columns([1.12,1])
    with l:
        fig=go.Figure(go.Surface(x=x,y=y,z=F,colorscale="Viridis",
            contours=dict(z=dict(show=show_contours,usecolormap=True,project_z=True)),colorbar=dict(title="f")))
        step=max(1,N//density); gx=Fx[::step,::step];gy=Fy[::step,::step];norm=np.nanpercentile(np.hypot(gx,gy),92) or 1
        fig.add_trace(go.Cone(x=X[::step,::step].ravel(),y=Y[::step,::step].ravel(),
            z=(F[::step,::step]+.025*(np.ptp(F)+1e-9)).ravel(),
            u=(gx/norm*arrow_scale).ravel(),v=(gy/norm*arrow_scale).ravel(),w=np.zeros_like(gx).ravel(),
            anchor="tail",sizemode="absolute",sizeref=.3,showscale=False))
        fig.update_layout(height=610,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="f(x,y)",
            camera=dict(eye=dict(x=1.55,y=1.55,z=1.15))),margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig,use_container_width=True)
    with r:
        fig=go.Figure(go.Heatmap(x=x,y=y,z=M,colorscale="Turbo",colorbar=dict(title="|∇f|")))
        quiver2d(fig,X,Y,Fx,Fy,x,density,arrow_scale);fig.update_layout(**layout2d("Gradient magnitude + direction"))
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("#### Point probe")
    px=st.slider("x₀",float(-extent),float(extent),0.,.05,key="gradx")
    py=st.slider("y₀",float(-extent),float(extent),0.,.05,key="grady")
    iy=np.argmin(abs(y-py));ix=np.argmin(abs(x-px))
    a,b,c=st.columns(3);a.metric("f(x₀,y₀)",f"{F[iy,ix]:.6g}");b.metric("∂f/∂x",f"{Fx[iy,ix]:.6g}");c.metric("∂f/∂y",f"{Fy[iy,ix]:.6g}")
    st.info("∇f is normal to the level curve and points toward maximum local increase.")

# ============================================================
# DIVERGENCE / CURL
# ============================================================
elif mode in ["Divergence","Curl"]:
    name=st.sidebar.selectbox("Vector field",list(VECTORS));fun,formula=VECTORS[name]
    U,V=fun(X,Y);D,C=deriv(U,V,dx,dy);Z=D if mode=="Divergence" else C
    operator=r"\nabla\cdot\mathbf F=\partial_xF_x+\partial_yF_y" if mode=="Divergence" else r"(\nabla\times\mathbf F)_z=\partial_xF_y-\partial_yF_x"
    st.header(f"{mode} — vector field → scalar diagnostic")
    st.markdown(f'<div class="formula"><b>Field:</b> {formula}<br><b>Operator:</b> {operator}</div>',unsafe_allow_html=True)
    a,b,c,d=st.columns(4);a.metric("maximum",f"{Z.max():.5g}");b.metric("minimum",f"{Z.min():.5g}");c.metric("mean",f"{Z.mean():.5g}");d.metric("RMS",f"{np.sqrt(np.mean(Z*Z)):.5g}")
    lim=qrange(Z);l,r=st.columns([1.12,1])
    with l:
        fig=go.Figure(go.Contour(x=x,y=y,z=Z,colorscale="RdBu_r",zmin=-lim,zmax=lim,zmid=0,contours=dict(showlabels=show_contours),colorbar=dict(title=mode)))
        if show_stream:streamlines(fig,U,V,x,y)
        quiver2d(fig,X,Y,U,V,x,density,arrow_scale);fig.update_layout(**layout2d(f"{mode} map + vector field"));st.plotly_chart(fig,use_container_width=True)
    with r:
        fig=go.Figure(go.Surface(x=x,y=y,z=Z,colorscale="RdBu_r",cmin=-lim,cmax=lim,colorbar=dict(title=mode),contours=dict(z=dict(show=show_contours,project_z=True))))
        fig.update_layout(height=610,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title=mode,zaxis=dict(range=[-lim,lim])),margin=dict(l=0,r=0,t=35,b=0));st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Point probe")
    px=st.slider("x₀",float(-extent),float(extent),0.,.05,key=mode+"x");py=st.slider("y₀",float(-extent),float(extent),0.,.05,key=mode+"y")
    iy=np.argmin(abs(y-py));ix=np.argmin(abs(x-px))
    a,b,c=st.columns(3);a.metric(mode,f"{Z[iy,ix]:.7g}");b.metric("Fₓ",f"{U[iy,ix]:.6g}");c.metric("Fᵧ",f"{V[iy,ix]:.6g}")
    st.info("Positive/negative colour indicates the sign of the differential quantity; streamlines reveal the underlying field geometry.")

# ============================================================
# 3D VECTOR FIELD
# ============================================================
elif mode=="3D Vector Field":
    st.header("3D Vector Field Explorer")
    st.caption("A true three-dimensional field view using a sampled vector lattice.")
    vf=st.sidebar.selectbox("3D field",["Radial source","Radial sink","Solid-body rotation","Saddle field","Swirl + source"])
    extent3=st.sidebar.slider("3D extent",1.5,5.0,3.0,.5)
    n3=st.sidebar.slider("3D lattice",5,13,9,2)

    q=np.linspace(-extent3,extent3,n3);X3,Y3,Z3=np.meshgrid(q,q,q,indexing="ij")
    if vf=="Radial source": U3,V3,W3=X3,Y3,Z3
    elif vf=="Radial sink": U3,V3,W3=-X3,-Y3,-Z3
    elif vf=="Solid-body rotation": U3,V3,W3=-Y3,X3,np.zeros_like(Z3)
    elif vf=="Saddle field": U3,V3,W3=X3,-Y3,np.zeros_like(Z3)
    else:
        r2=X3**2+Y3**2+.7;U3,V3,W3=-Y3/r2,X3/r2,Z3*.15

    mag=np.sqrt(U3**2+V3**2+W3**2);norm=np.nanpercentile(mag,90) or 1
    fig=go.Figure(go.Cone(x=X3.ravel(),y=Y3.ravel(),z=Z3.ravel(),
        u=(U3/norm).ravel(),v=(V3/norm).ravel(),w=(W3/norm).ravel(),
        colorscale="Viridis",sizemode="absolute",sizeref=.45,showscale=True,
        colorbar=dict(title="|F|"),anchor="tail",
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>"))
    fig.update_layout(height=700,scene=dict(xaxis_title="x",yaxis_title="y",zaxis_title="z",
        aspectmode="cube",camera=dict(eye=dict(x=1.5,y=1.5,z=1.25))),margin=dict(l=0,r=0,t=35,b=0))
    st.plotly_chart(fig,use_container_width=True)
    a,b,c=st.columns(3);a.metric("max |F|",f"{mag.max():.5g}");b.metric("mean |F|",f"{mag.mean():.5g}");c.metric("vectors",f"{mag.size:,}")
    st.info("Rotate the scene to inspect the field from different directions. Colour encodes vector magnitude.")

# ============================================================
# FLOW ANIMATION
# ============================================================
elif mode=="Flow Animation":
    st.header("Animated Vector-Field Flow")
    vf=st.sidebar.selectbox("Flow field",["Radial source","Radial sink","Solid-body rotation","Vortex","Spiral source"])
    frames=st.sidebar.slider("Animation frames",10,50,25)
    particles=st.sidebar.slider("Particles",30,180,90)
    speed=st.sidebar.slider("Flow speed",.05,.5,.18,.01)

    U,V=VECTORS[vf][0](X,Y)
    rng=np.random.default_rng(42)
    x0=rng.uniform(-extent*.8,extent*.8,particles);y0=rng.uniform(-extent*.8,extent*.8,particles)
    allx=[x0.copy()];ally=[y0.copy()]
    px=x0.copy();py=y0.copy()
    for _ in range(frames-1):
        ix=np.clip(np.searchsorted(x,px)-1,0,N-1);iy=np.clip(np.searchsorted(y,py)-1,0,N-1)
        ux=U[iy,ix];uy=V[iy,ix];m=np.hypot(ux,uy)
        px=px+speed*ux/np.maximum(m,.4);py=py+speed*uy/np.maximum(m,.4)
        px=np.where((px<x[0])|(px>x[-1]),rng.uniform(x[0],x[-1],particles),px)
        py=np.where((py<y[0])|(py>y[-1]),rng.uniform(y[0],y[-1],particles),py)
        allx.append(px.copy());ally.append(py.copy())

    fig=go.Figure()
    fig.add_trace(go.Heatmap(x=x,y=y,z=np.hypot(U,V),colorscale="Viridis",opacity=.45,showscale=True,colorbar=dict(title="|F|")))
    quiver2d(fig,X,Y,U,V,x,density,arrow_scale)
    fig.add_trace(go.Scatter(x=allx[0],y=ally[0],mode="markers",
        marker=dict(size=7),name="particles"))
    fig.frames=[go.Frame(data=[go.Scatter(x=allx[k],y=ally[k],mode="markers",
        marker=dict(size=7),name="particles")],name=str(k)) for k in range(frames)]
    fig.update_layout(**layout2d("Particle advection through the vector field"),
        updatemenus=[dict(type="buttons",showactive=True,x=.02,y=1.12,
            buttons=[dict(label="▶ Play",method="animate",args=[None,{"frame":{"duration":80,"redraw":True},"fromcurrent":True}]),
                     dict(label="⏸ Pause",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0},"transition":{"duration":0}}])])])
    st.plotly_chart(fig,use_container_width=True)
    st.info("Particles are advected numerically through the field. Their motion provides an intuitive physical interpretation of vector-field direction and flow.")

# ============================================================
# CONVERGENCE LAB
# ============================================================
elif mode=="Convergence":
    st.header("Numerical Convergence Laboratory")
    st.caption("Compare finite-difference derivatives against known analytical results.")
    test=st.sidebar.selectbox("Analytical test",["Paraboloid gradient","Radial-source divergence","Solid-body curl"])
    point_x=st.sidebar.slider("Test x₀",-.9,.9,.55,.05);point_y=st.sidebar.slider("Test y₀",-.9,.9,.35,.05)
    resolutions=[11,21,31,41,61,81,101]
    errors=[];dxs=[]
    for n in resolutions:
        xx=np.linspace(-1,1,n);yy=np.linspace(-1,1,n);XX,YY=np.meshgrid(xx,yy);dd=xx[1]-xx[0]
        ix=np.argmin(abs(xx-point_x));iy=np.argmin(abs(yy-point_y))
        if test=="Paraboloid gradient":
            FF=XX**2+YY**2;fy,fx=np.gradient(FF,dd,dd); numerical=np.array([fx[iy,ix],fy[iy,ix]])
            exact=np.array([2*xx[ix],2*yy[iy]])
        elif test=="Radial-source divergence":
            UU,VV=XX,YY;DD,_=deriv(UU,VV,dd,dd);numerical=np.array([DD[iy,ix]]);exact=np.array([2.])
        else:
            UU,VV=-YY,XX;_,CC=deriv(UU,VV,dd,dd);numerical=np.array([CC[iy,ix]]);exact=np.array([2.])
        errors.append(float(np.linalg.norm(numerical-exact)));dxs.append(dd)

    fig=go.Figure(go.Scatter(x=dxs,y=errors,mode="lines+markers",name="absolute error"))
    fig.update_layout(height=520,xaxis_title="Δx",yaxis_title="absolute error",xaxis=dict(type="log"),yaxis=dict(type="log"))
    st.plotly_chart(fig,use_container_width=True)
    order=np.polyfit(np.log(dxs[-5:]),np.log(np.maximum(errors[-5:],1e-15)),1)[0]
    st.metric("Observed convergence order",f"{order:.3f}")
    st.write("For a centered finite-difference derivative on a smooth field, the expected asymptotic order is approximately 2.")
    st.dataframe({"Grid":resolutions,"Δx":dxs,"Absolute error":errors},use_container_width=True,hide_index=True)

# ============================================================
# COMPARISON
# ============================================================
elif mode=="Comparison":
    st.header("Three Operators — Unified Visual Comparison")
    sf=st.sidebar.selectbox("Scalar field",list(SCALARS));vf=st.sidebar.selectbox("Vector field",list(VECTORS))
    F=SCALARS[sf][0](X,Y);Fy,Fx=np.gradient(F,dy,dx);G=np.hypot(Fx,Fy)
    U,V=VECTORS[vf][0](X,Y);D,C=deriv(U,V,dx,dy)
    tabs=st.tabs(["Gradient","Divergence","Curl","Conceptual map"])
    with tabs[0]:
        fig=go.Figure(go.Heatmap(x=x,y=y,z=G,colorscale="Turbo",colorbar=dict(title="|∇f|")));quiver2d(fig,X,Y,Fx,Fy,x,density,arrow_scale);fig.update_layout(**layout2d("Gradient"));st.plotly_chart(fig,use_container_width=True)
    with tabs[1]:
        lim=qrange(D);fig=go.Figure(go.Heatmap(x=x,y=y,z=D,colorscale="RdBu_r",zmin=-lim,zmax=lim,zmid=0,colorbar=dict(title="∇·F")));quiver2d(fig,X,Y,U,V,x,density,arrow_scale);fig.update_layout(**layout2d("Divergence"));st.plotly_chart(fig,use_container_width=True)
    with tabs[2]:
        lim=qrange(C);fig=go.Figure(go.Heatmap(x=x,y=y,z=C,colorscale="RdBu_r",zmin=-lim,zmax=lim,zmid=0,colorbar=dict(title="curl z")));quiver2d(fig,X,Y,U,V,x,density,arrow_scale);fig.update_layout(**layout2d("Curl"));st.plotly_chart(fig,use_container_width=True)
    with tabs[3]:
        st.markdown("""
| Operator | Input | Output | Geometric question |
|---|---|---|---|
| **Gradient ∇f** | Scalar | Vector | Which direction increases fastest? |
| **Divergence ∇·F** | Vector | Scalar | Is there local net outflow/inflow? |
| **Curl ∇×F** | Vector | Vector | How strongly does the field rotate? |
""")
        st.latex(r"\boxed{\nabla f\qquad\nabla\cdot\mathbf F\qquad\nabla\times\mathbf F}")

# ============================================================
# FOOTER
# ============================================================
with st.expander("Numerical diagnostics"):
    st.write(f"2-D domain: [{-extent:.2f},{extent:.2f}] × [{-extent:.2f},{extent:.2f}]")
    st.write(f"Grid: {N} × {N} = {N*N:,} samples; Δx={dx:.7f}, Δy={dy:.7f}")
    st.write("Finite differences are evaluated with NumPy's gradient operator.")
    st.write("The Convergence module provides an explicit discretization-error experiment.")

st.markdown('<div class="footer">Made with ❤️ by Aman Kumar Patel</div>',unsafe_allow_html=True)
