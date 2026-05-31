import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os


def plot_3d_stress_field(beam, title='3D Stress Field'):
    n_x, n_y = 60, 30
    x_arr = np.linspace(0, beam.L, n_x)
    y_arr = np.linspace(-beam.cs.h/2, beam.cs.h/2, n_y)
    X, Y  = np.meshgrid(x_arr, y_arr)
    Z     = np.zeros_like(X)
    I     = beam.cs.second_moment_of_area
    for i, xi in enumerate(x_arr):
        idx = np.argmin(np.abs(beam.x - xi))
        M   = beam.moment[idx]
        for j, yj in enumerate(y_arr):
            Z[j, i] = abs(M * yj / I) / 1e6

    fig = go.Figure(data=[go.Surface(
        x=X, y=Y*1000, z=Z,
        colorscale='RdYlGn_r',
        colorbar=dict(
            title=dict(text='Stress (MPa)', font=dict(color='white')),
            tickfont=dict(color='white'),
        ),
        opacity=0.9,
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=16)),
       scene=dict(
            xaxis=dict(title='Span (m)', backgroundcolor='#1a1a2e',
                        gridcolor='#444466', color='white'),
            yaxis=dict(title='Height (mm)', backgroundcolor='#1a1a2e',
                        gridcolor='#444466', color='white'),
            zaxis=dict(title='Stress (MPa)', backgroundcolor='#1a1a2e',
                        gridcolor='#444466', color='white'),
            bgcolor='#1a1a2e',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        width=1000, height=700,
    )
    return fig


def plot_3d_monte_carlo_cloud(df, title='Monte Carlo - 3D View'):
    fig = go.Figure()

    safe_mask = df['any_failure'] == 0
    fail_mask = df['any_failure'] == 1

    fig.add_trace(go.Scatter3d(
        x=df.loc[safe_mask, 'peak_load_kN'],
        y=df.loc[safe_mask, 'turbulence_intensity'],
        z=df.loc[safe_mask, 'beam_safety_factor'],
        mode='markers',
        marker=dict(size=3, color='#00cc55', opacity=0.4),
        name='Safe (' + str(int(safe_mask.sum())) + ' flights)',
    ))

    if fail_mask.sum() > 0:
        fig.add_trace(go.Scatter3d(
            x=df.loc[fail_mask, 'peak_load_kN'],
            y=df.loc[fail_mask, 'turbulence_intensity'],
            z=df.loc[fail_mask, 'beam_safety_factor'],
            mode='markers',
            marker=dict(size=10, color='#ff4444', opacity=1.0, symbol='x'),
            name='FAILURE (' + str(int(fail_mask.sum())) + ' events)',
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color='white', size=15)),
        scene=dict(
            xaxis=dict(title='Peak Load (kN)', backgroundcolor='#0f0f23',
                        gridcolor='#333355', color='white'),
            yaxis=dict(title='Turbulence (m/s)', backgroundcolor='#0f0f23',
                        gridcolor='#333355', color='white'),
            zaxis=dict(title='Safety Factor', backgroundcolor='#0f0f23',
                        gridcolor='#333355', color='white'),
            bgcolor='#0f0f23',
        ),
        paper_bgcolor='#1a1a2e',
        font=dict(color='white'),
        width=1100, height=750,
    )
    return fig


def plot_interactive_scenario_dashboard(df):
    scenarios  = df['scenario_name'].unique()
    colors     = px.colors.qualitative.Set2[:len(scenarios)]
    color_map  = dict(zip(scenarios, colors))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Load by Scenario', 'SF vs Load',
                         'Flight Outcomes', '3D Safety Cloud'],
        specs=[[{'type': 'box'},     {'type': 'scatter'}],
               [{'type': 'pie'},     {'type': 'scatter3d'}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    for scen in scenarios:
        mask = df['scenario_name'] == scen
        fig.add_trace(
            go.Box(y=df.loc[mask, 'peak_load_kN'],
                    name=scen.replace('_', ' '),
                    marker_color=color_map[scen],
                    boxmean=True),
            row=1, col=1
        )

    for scen in scenarios:
        mask = df['scenario_name'] == scen
        fig.add_trace(
            go.Scatter(
                x=df.loc[mask, 'peak_load_kN'],
                y=df.loc[mask, 'beam_safety_factor'],
                mode='markers',
                marker=dict(size=4, color=color_map[scen], opacity=0.5),
                name=scen,
                showlegend=False,
            ),
            row=1, col=2
        )

    fig.add_hline(y=1.5, line_dash='dash', line_color='red', row=1, col=2)
    fig.add_hline(y=1.0, line_color='darkred', row=1, col=2)

    outcome_counts = df['outcome'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            marker=dict(colors=['#00cc55', '#ff4444']),
            hole=0.4,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter3d(
            x=df['peak_load_kN'],
            y=df['turbulence_intensity'],
            z=df['beam_safety_factor'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['beam_safety_factor'],
                colorscale='RdYlGn',
                opacity=0.5,
            ),
            showlegend=False,
        ),
        row=2, col=2
    )

    fig.update_layout(
        title=dict(
            text='Interactive Flight Safety Dashboard',
            font=dict(color='white', size=16),
            x=0.5,
        ),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#0f0f23',
        font=dict(color='white'),
        height=850,
    )
    return fig


def save_interactive_plots(beam, df_std, output_dir='reports'):
    paths = []

    print('  Generating interactive 3D stress field...')
    fig1 = plot_3d_stress_field(beam)
    p1   = os.path.join(output_dir, 'phase7_3d_stress_field.html')
    fig1.write_html(p1)
    paths.append(p1)
    print('  Saved: ' + p1)

    print('  Generating Monte Carlo 3D cloud...')
    fig2 = plot_3d_monte_carlo_cloud(df_std)
    p2   = os.path.join(output_dir, 'phase7_monte_carlo_3d.html')
    fig2.write_html(p2)
    paths.append(p2)
    print('  Saved: ' + p2)

    print('  Generating interactive dashboard...')
    fig3 = plot_interactive_scenario_dashboard(df_std)
    p3   = os.path.join(output_dir, 'phase7_interactive_dashboard.html')
    fig3.write_html(p3)
    paths.append(p3)
    print('  Saved: ' + p3)

    return paths