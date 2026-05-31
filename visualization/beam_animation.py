import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
import os


def _remove_collections(ax):
    for coll in list(ax.collections):
        try:
            coll.remove()
        except Exception:
            pass


def animate_beam_loading(beam_class, cross_section_class, material,
                          hydraulic_lines=None, max_load=100000,
                          n_frames=60, output_path='reports/beam_loading.gif', fps=15):
    cs   = cross_section_class(width=0.15, height=0.20)
    beam = beam_class(length=4.0, cross_section=cs, material=material, n_stations=100)
    loads = np.linspace(1000, max_load, n_frames)
    h_mm  = cs.h * 1000
    scale = 8.0
    yield_mpa = material.yield_strength / 1e6

    fig = plt.figure(figsize=(15, 8), facecolor='#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                            left=0.07, right=0.97, top=0.88, bottom=0.08)
    ax_beam   = fig.add_subplot(gs[0, :2])
    ax_stress = fig.add_subplot(gs[1, :2])
    ax_gauge  = fig.add_subplot(gs[:, 2])

    for ax in [ax_beam, ax_stress, ax_gauge]:
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444466')

    fig.suptitle('Aircraft Fin - Live Loading Simulation',
                  color='white', fontsize=13, fontweight='bold', y=0.96)

    beam.apply_tip_load(loads[0])
    x      = beam.x
    x_norm = x / beam.L
    defl   = beam.deflection * scale
    stress = np.abs(beam.stress_top) / 1e6

    ax_beam.plot([0, beam.L], [0, 0], color='#444466', linewidth=1, linestyle=':')
    beam_top,  = ax_beam.plot(x, defl + h_mm/2, color='#4ecdc4', lw=2.5)
    beam_bot,  = ax_beam.plot(x, defl - h_mm/2, color='#4ecdc4', lw=2.5)
    defl_line, = ax_beam.plot(x, defl, color='#ffd700', lw=2, linestyle='--')
    ax_beam.fill_between(x, defl - h_mm/2, defl + h_mm/2, alpha=0.2, color='#4ecdc4')
    wall = patches.Rectangle((-0.15, -h_mm), 0.15, 2*h_mm,
                               facecolor='#666688', edgecolor='white', lw=1.5)
    ax_beam.add_patch(wall)
    force_text = ax_beam.text(beam.L + 0.1, defl[-1] + h_mm/2 + 5,
                               'F = 1.0 kN', color='#ff6b6b', fontsize=10, fontweight='bold')
    ax_beam.set_xlim(-0.2, beam.L + 0.8)
    ax_beam.set_ylim(-h_mm*1.8, h_mm*1.8)
    ax_beam.set_xlabel('Spanwise Position (m)', color='white')
    ax_beam.set_ylabel('Position (mm)', color='white')
    ax_beam.set_title('Beam Deflection', color='white')
    ax_beam.grid(True, alpha=0.15, color='white')

    stress_line, = ax_stress.plot(x_norm, stress, color='#ff6b6b', lw=2.5)
    ax_stress.fill_between(x_norm, 0, stress, alpha=0.3, color='#ff6b6b')
    ax_stress.axhline(yield_mpa, color='white', lw=1.5, linestyle='--', label='Yield')
    ax_stress.axhline(yield_mpa/1.5, color='#ffd700', lw=1.2, linestyle=':', label='Design limit')
    stress_text = ax_stress.text(0.02, 0.92, 'Max stress = 0 MPa',
                                  transform=ax_stress.transAxes, color='white', fontsize=10,
                                  fontweight='bold',
                                  bbox=dict(facecolor='#0f0f23', alpha=0.7, boxstyle='round'))
    ax_stress.set_xlim(0, 1)
    ax_stress.set_ylim(0, yield_mpa * 1.2)
    ax_stress.set_xlabel('Span fraction', color='white')
    ax_stress.set_ylabel('Stress (MPa)', color='white')
    ax_stress.set_title('Stress Distribution', color='white')
    ax_stress.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white')
    ax_stress.grid(True, alpha=0.15, color='white')

    sf0 = material.safety_factor(beam.max_stress())
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(0, 6)
    ax_gauge.set_title('Safety Factor', color='white', fontsize=11)
    ax_gauge.set_xticks([])
    ax_gauge.set_ylabel('SF', color='white')
    ax_gauge.axhspan(0,   1.0, alpha=0.25, color='#cc0000')
    ax_gauge.axhspan(1.0, 1.5, alpha=0.25, color='#ff6600')
    ax_gauge.axhspan(1.5, 2.0, alpha=0.25, color='#ffcc00')
    ax_gauge.axhspan(2.0, 3.0, alpha=0.25, color='#88cc00')
    ax_gauge.axhspan(3.0, 6.0, alpha=0.25, color='#00aa44')
    ax_gauge.axhline(1.5, color='white', lw=2, linestyle='--')
    ax_gauge.axhline(1.0, color='red', lw=2.5)
    for y, lbl, col in [(0.5,'FAIL','#ff4444'),(1.25,'UNSAFE','#ff8800'),
                         (1.75,'MARGINAL','#ffcc00'),(2.5,'SAFE','#aadd00'),
                         (4.5,'OPTIMAL','#00cc55')]:
        ax_gauge.text(0.5, y, lbl, ha='center', va='center', color=col,
                       fontsize=9, fontweight='bold',
                       transform=ax_gauge.get_yaxis_transform())
    sf_color = '#00cc55' if sf0>=3 else '#aadd00' if sf0>=2 else '#ffcc00' if sf0>=1.5 else '#ff4444'
    sf_bar  = ax_gauge.bar(0.5, min(sf0,5.9), width=0.6, color=sf_color,
                             alpha=0.9, edgecolor='white', lw=1.5)
    sf_text = ax_gauge.text(0.5, min(sf0,5.9)+0.15, 'SF=0.00',
                             ha='center', color='white', fontsize=11, fontweight='bold')
    frame_text = fig.text(0.5, 0.01, 'Frame 1', ha='center', color='#888', fontsize=9)

    def update(fi):
        F  = loads[fi]
        beam.apply_tip_load(F)
        d  = beam.deflection * scale
        s  = np.abs(beam.stress_top) / 1e6
        sf = material.safety_factor(beam.max_stress())
        beam_top.set_ydata(d + h_mm/2)
        beam_bot.set_ydata(d - h_mm/2)
        defl_line.set_ydata(d)
        force_text.set_text('F = ' + str(round(F/1000,1)) + ' kN')
        force_text.set_y(d[-1] + h_mm/2 + 5)
        _remove_collections(ax_beam)
        ax_beam.fill_between(x, d - h_mm/2, d + h_mm/2, alpha=0.2, color='#4ecdc4')
        stress_line.set_ydata(s)
        _remove_collections(ax_stress)
        ax_stress.fill_between(x_norm, 0, s, alpha=0.3, color='#ff6b6b')
        ax_stress.axhline(yield_mpa, color='white', lw=1.5, linestyle='--')
        ax_stress.axhline(yield_mpa/1.5, color='#ffd700', lw=1.2, linestyle=':')
        stress_text.set_text('Max stress = ' + str(round(s[0],1)) + ' MPa')
        sf_clip = min(sf, 5.9)
        col = '#00cc55' if sf>=3 else '#aadd00' if sf>=2 else '#ffcc00' if sf>=1.5 else '#ff4444'
        sf_bar[0].set_height(sf_clip)
        sf_bar[0].set_facecolor(col)
        sf_text.set_text('SF=' + str(round(sf,2)))
        sf_text.set_y(sf_clip + 0.15)
        sf_text.set_color(col)
        frame_text.set_text('Frame ' + str(fi+1) + '/' + str(n_frames) +
                             '  Load: ' + str(round(F/1000,1)) + ' kN' +
                             '  SF: ' + str(round(sf,2)) +
                             '  Tip: ' + str(round(beam.deflection[-1]*1000,1)) + ' mm')
        return beam_top, beam_bot, defl_line, stress_line, stress_text, sf_text, frame_text

    print('  Creating animation: ' + str(n_frames) + ' frames @ ' + str(fps) + ' fps...')
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    print('  Saving to: ' + output_path)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=80)
    plt.close(fig)
    print('  Animation saved: ' + output_path)
    print('  Size: ' + str(round(os.path.getsize(output_path)/1024/1024,1)) + ' MB')
    return output_path


def animate_stress_wave(beam_class, cross_section_class, material,
                         output_path='reports/stress_wave.gif', n_frames=50, fps=12):
    cs        = cross_section_class(width=0.15, height=0.20)
    beam      = beam_class(4.0, cs, material, n_stations=100)
    loads     = np.linspace(5000, 150000, n_frames)
    h_mm      = cs.h * 1000
    yield_mpa = material.yield_strength / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='#1a1a2e')
    fig.suptitle('Cross-Section Stress Profile - Ramping Load',
                  color='white', fontsize=13, fontweight='bold')

    x_fracs   = [0.01, 0.5, 0.99]
    labels    = ['Root (x=0)', 'Mid-span (x=L/2)', 'Near Tip (x=L)']
    cols      = ['#ff6b6b', '#ffd700', '#4ecdc4']
    line_objs = []
    text_objs = []

    for ax, x_frac, label, col in zip(axes, x_fracs, labels, cols):
        ax.set_facecolor('#0f0f23')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#444466')
        beam.apply_tip_load(loads[0])
        y_arr, sig_arr = beam.get_stress_profile_at_x(x_frac*beam.L, n_points=80)
        y_mm    = y_arr * 1000
        sig_mpa = sig_arr / 1e6
        ln, = ax.plot(sig_mpa, y_mm, color=col, lw=2.5)
        ax.fill_betweenx(y_mm, 0, sig_mpa, alpha=0.3, color=col)
        ax.axvline( yield_mpa,     color='red',    lw=1.5, linestyle='--')
        ax.axvline(-yield_mpa,     color='red',    lw=1.5, linestyle='--')
        ax.axvline( yield_mpa/1.5, color='orange', lw=1,   linestyle=':')
        ax.axvline(-yield_mpa/1.5, color='orange', lw=1,   linestyle=':')
        ax.axhline(0, color='#888', lw=1, linestyle='-.')
        ax.axhspan(-h_mm*0.15, h_mm*0.15, alpha=0.1, color='green')
        ax.set_xlim(-yield_mpa*1.2, yield_mpa*1.2)
        ax.set_ylim(-h_mm*0.6, h_mm*0.6)
        ax.set_xlabel('Stress (MPa)', color='white')
        ax.set_title(label, color=col, fontsize=11, fontweight='bold')
        if ax == axes[0]:
            ax.set_ylabel('Distance from NA (mm)', color='white')
        txt = ax.text(0.05, 0.95, 's_max=0 MPa',
                       transform=ax.transAxes, color='white', fontsize=9, va='top',
                       bbox=dict(facecolor='#0f0f23', alpha=0.7, boxstyle='round'))
        line_objs.append((ln, ax, x_frac, col))
        text_objs.append(txt)

    load_text = fig.text(0.5, 0.02, 'Load: 0 kN',
                          ha='center', color='#ffd700', fontsize=12, fontweight='bold')

    def update(fi):
        F = loads[fi]
        beam.apply_tip_load(F)
        for (ln, ax, x_frac, col), txt in zip(line_objs, text_objs):
            y_arr, sig_arr = beam.get_stress_profile_at_x(x_frac*beam.L, n_points=80)
            y_mm    = y_arr * 1000
            sig_mpa = sig_arr / 1e6
            ln.set_xdata(sig_mpa)
            ln.set_ydata(y_mm)
            _remove_collections(ax)
            ax.fill_betweenx(y_mm, 0, sig_mpa, alpha=0.3, color=col)
            ax.axhspan(-h_mm*0.15, h_mm*0.15, alpha=0.1, color='green')
            ax.axvline( yield_mpa, color='red', lw=1.5, linestyle='--')
            ax.axvline(-yield_mpa, color='red', lw=1.5, linestyle='--')
            mx = abs(sig_mpa).max()
            txt.set_text('s_max=' + str(round(mx,1)) + ' MPa')
            txt.set_color('red' if mx > yield_mpa/1.5 else 'white')
        load_text.set_text('Load: ' + str(round(F/1000,1)) + ' kN')
        return []

    print('  Creating stress wave: ' + str(n_frames) + ' frames @ ' + str(fps) + ' fps...')
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000//fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=80)
    plt.close(fig)
    print('  Saved: ' + output_path)
    return output_path
