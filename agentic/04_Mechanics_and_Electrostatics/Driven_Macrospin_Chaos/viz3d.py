"""Web-native 3D for the dynamics. Run: python viz3d.py   (needs dynamics.npz)

figures/m1_torque.html      M1, the torque decomposition, animated
figures/sphere_regimes.html the three driven orbits on the unit sphere
Standalone HTML with plotly from a CDN, so both embed live in the github.io portfolio.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

d = np.load('dynamics.npz')
U, V = np.mgrid[0:2 * np.pi:61j, 0:np.pi:31j]
SPHERE = go.Surface(x=np.cos(U) * np.sin(V), y=np.sin(U) * np.sin(V), z=np.cos(V),
                    colorscale=[[0, '#cccccc'], [1, '#cccccc']], opacity=0.10,
                    showscale=False, hoverinfo='skip', lighting=dict(ambient=1.0))
unit = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)

def arrow(base, vec, color, name, width=7):
    tip = base + vec
    return go.Scatter3d(x=[base[0], tip[0]], y=[base[1], tip[1]], z=[base[2], tip[2]],
                        mode='lines+markers', line=dict(color=color, width=width),
                        marker=dict(size=[0, 4], color=color), name=name)

def line(m, color, name, width=2):
    return go.Scatter3d(x=m[:, 0], y=m[:, 1], z=m[:, 2], mode='lines',
                        line=dict(color=color, width=width), name=name, hoverinfo='skip')

def layout(title, eye=(1.5, 1.3, 0.9), center=(0.0, 0.0, 0.0)):
    ax = dict(range=[-1.05, 1.05], showticklabels=False)
    xyz = lambda v: dict(zip('xyz', v))
    return dict(title=dict(text=title, font=dict(size=14)),
                scene=dict(xaxis=ax | dict(title='mx'), yaxis=ax | dict(title='my'),
                           zaxis=ax | dict(title='mz'), aspectmode='cube',
                           camera=dict(eye=xyz(eye), center=xyz(center))),
                margin=dict(l=0, r=0, t=64, b=0))

# ------------------------------------------- M1, the torque decomposition, animated
m, hhat = d['m1_m'], d['hhat']
tp, td = 0.35 * unit(d['m1_tp']), 0.35 * unit(d['m1_td'])
anim = lambda i: [arrow(np.zeros(3), m[i], '#1f77b4', 'm'),
                  arrow(m[i], tp[i], '#ff7f0e', 'precession torque'),
                  arrow(m[i], td[i], '#d62728', 'damping torque'),
                  line(m[:i + 1], '#1f77b4', 'trail', 4)]
fig = go.Figure(data=[SPHERE, line(m, '#bbbbbb', 'path of m'),
                      arrow(np.zeros(3), 1.15 * hhat, 'black', 'H_eff')] + anim(0),
                frames=[go.Frame(name=str(i), traces=[3, 4, 5, 6], data=anim(i))
                        for i in range(len(m))])
fig.update_layout(**layout(
    'M1 &mdash; the torque decomposition. Precession carries <b>m</b> around the cone, '
    'damping slides the cone shut.<br>'
    '<span style="font-size:12px">Arrows show direction only: the true ratio is '
    '|T<sub>damp</sub>|/|T<sub>prec</sub>| = &alpha; = 0.01. Flip the damping sign and '
    'the cone opens instead &mdash; that is what this picture is for.</span>'),
    updatemenus=[dict(type='buttons', showactive=False, x=0.02, y=0.05, buttons=[
        dict(label='play', method='animate', args=[None, dict(
            frame=dict(duration=50, redraw=True), fromcurrent=True,
            transition=dict(duration=0))]),
        dict(label='pause', method='animate', args=[[None], dict(
            frame=dict(duration=0, redraw=False), mode='immediate')])])])
fig.write_html('figures/m1_torque.html', include_plotlyjs='cdn', auto_play=False)

# ------------------------------------------------ the three driven orbits, true scale
m_eq = d['m_eq']
fig = go.Figure([SPHERE, go.Scatter3d(x=[m_eq[0]], y=[m_eq[1]], z=[m_eq[2]], mode='markers',
                                      marker=dict(size=4, color='black'), name='equilibrium')])
for tag, color in (('weak', '#1f77b4'), ('moderate', '#2ca02c'), ('strong', '#d62728')):
    mm = d[tag]
    cone = np.degrees(np.arccos(np.clip(mm[len(mm) // 2:] @ m_eq, -1.0, 1.0))).max()
    fig.add_trace(line(mm[::4], color, f'{tag} drive, {cone:.1f}&deg; cone'))
fig.update_layout(**layout(eye=1.75 * m_eq + np.array([0.0, -0.55, 0.0]),
                           center=0.55 * m_eq, title=
    f'Driven orbits on the unit sphere, true relative scale, f<sub>rf</sub> = '
    f'{d["f0"]:.3f} GHz on resonance.<br><span style="font-size:12px">Each starts at '
    'equilibrium and spirals out to its steady state. Which of these is chaotic is a '
    'question for the Lyapunov exponent, not the eye.</span>'))
fig.write_html('figures/sphere_regimes.html', include_plotlyjs='cdn')

# ----------- the same orbits in the tangent plane at equilibrium, each at its own scale
e1 = np.cross([0.0, 1.0, 0.0], m_eq)              # unit: m_eq has no y component
e2 = np.cross(m_eq, e1)
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=[
    f'{t} drive' for t in ('weak', 'moderate', 'strong')])
for col, (tag, color) in enumerate((('weak', '#1f77b4'), ('moderate', '#2ca02c'),
                                    ('strong', '#d62728')), start=1):
    mm = d[tag][len(d[tag]) // 2:]                # steady state only, transient dropped
    fig.add_trace(go.Scatter(x=mm @ e1, y=mm @ e2, mode='lines', name=tag,
                             line=dict(color=color, width=1)), row=1, col=col)
    fig.update_xaxes(title='m . e1', row=1, col=col)
    fig.update_yaxes(scaleanchor=f'x{col}', scaleratio=1, row=1, col=col)
fig.update_layout(height=360, showlegend=False, margin=dict(t=76), title=dict(font=dict(size=14),
    text='The same three orbits in the tangent plane at equilibrium, each at its own '
         'scale.<br><span style="font-size:12px">Transient dropped. All three are closed '
         'period-1 cycles locked to the drive, and they stay nearly circular (axis ratio '
         '1.06 to 1.02). Nothing here period-doubles: chaos, if any, lives above 5 mT.</span>'))
fig.write_html('figures/orbit_shapes.html', include_plotlyjs='cdn')
print('    wrote figures/m1_torque.html, sphere_regimes.html, orbit_shapes.html')
