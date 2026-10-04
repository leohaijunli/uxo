"""
Generate an interactive Jupyter notebook report of the 3-D GPR inversion.

Builds `GPR_3D_inversion_report.ipynb` (nbformat v4) with:
  - markdown: overview, method, survey-sync, ferrous-vs-metal
  - code: the pipeline (load cube, migrate, velocity analysis)
  - outputs: results table, static B-scan/position figures (base64 PNG), and
    *interactive rotatable 3-D volume views* of each migrated image (plotly.js
    from CDN -- drag to rotate; no local plotly install needed to view).

Run AFTER the grid build has produced the cubes:
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python dataset/make_notebook.py
"""
import os, sys, glob, json, base64
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
G3D = os.path.join(HERE, "gpr3d_grid")
OUTNB = os.path.join(ROOT, "report_output_gpr", "GPR_3D_inversion_report.ipynb")
sys.path.insert(0, HERE)
from invert_gpr3d import migrate, best_velocity   # noqa

PLOTLY_JS = ("https://cdn.plot.ly/plotly-2.30.1.min.js")


def plotly_volume_html(ID, I, xg, yg, zg):
    """Self-contained HTML div embedding a rotatable plotly Volume of I."""
    ID = str(ID)
    xx, yy, zz = np.meshgrid(xg, yg, zg, indexing="ij")
    sel = I > I.max() * 0.3
    data = dict(
        type="volume", x=xx[sel].ravel().astype(float).tolist(),
        y=yy[sel].ravel().astype(float).tolist(), z=zz[sel].ravel().astype(float).tolist(),
        value=I[sel].ravel().astype(float).tolist(),
        isomin=float(I.max()*0.3), isomax=float(I.max()),
        opacity=0.18, surface_count=20, caps=dict(x_show=False, y_show=False, z_show=False),
        colorscale="Hot", colorbar=dict(title="image"))
    lay = dict(scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
               width=600, height=560, title=ID)
    return (f"<div><script src='{PLOTLY_JS}'></script>"
            f"<div id='vol_{ID}' style='width:620px;height:580px'></div>"
            f"<script>var d={json.dumps(data)};var l={json.dumps(lay)};"
            f"Plotly.newPlot('vol_{ID}',[d],l);</script></div>")


def b64_png(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def md(cells, *paras):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": "".join(p + "\n\n" for p in paras).splitlines(keepends=True)})


def code(cells, src, outputs=None, exec_count=1):
    cells.append({"cell_type": "code", "execution_count": exec_count,
                  "metadata": {}, "outputs": outputs or [],
                  "source": src.splitlines(keepends=True)})


def main():
    files = sorted(glob.glob(os.path.join(G3D, "*_cube.npz")))
    if not files:
        print("no cubes yet in", G3D, "run the grid build first"); return

    cells = []
    md(cells,
       "# 3-D GPR Migration Inversion Report",
       "**Multi-modal (GPR + magnetometer) UXO dataset — OSU seeded set.**",
       "This notebook inverts the simulated **3-D GPR survey** for each object: "
       "a 2-D antenna grid of B-scans (dx=dy=0.02 m from the real drone "
       "speed/PRF, 0.05 m ground-coupled height, gprMax 3-D, gprMax "
       "LandmineModels where applicable) is background-subtracted and "
       "**Kirchhoff-migrated** into a 3-D image I(x,y,z). The target is located "
       "at the image peak -> full 3-D position (x, y, depth).",
       "**GPR vs magnetometry (why both):** GPR sees *electrical* contrast "
       "(any metal = PEC, plus plastic/void/TNT dielectric contrast); "
       "magnetometry sees *magnetic* content. They are orthogonal: e.g. D22 "
       "(plastic mine + steel mechanism) is hard for GPR but easy for "
       "magnetometry; F7 (aluminum can) is easy for GPR but invisible to "
       "magnetometry.",
       "**Drag to rotate** the 3-D volume plots below (plotly, loaded from CDN).")

    code(cells, "import sys, glob\n"
                "import numpy as np, pandas as pd\n"
                "sys.path.insert(0, '.')\n"
                "from dataset.invert_gpr3d import migrate, best_velocity\n"
                "G3D='dataset/gpr3d_grid'\n"
                "cubes=sorted(glob.glob(G3D+'/*_cube.npz'))\n"
                "print(len(cubes), 'cubes')")

    md(cells, "## Inversion results")
    results = []
    for f in files:
        d = np.load(f)
        Ez, X, Y, t = d["Ez"], d["X"], d["Y"], d["t"]*1e9
        xc, yc = float(d.get("xc", 0.30)), float(d.get("yc", 0.125))
        depth = float(d["depth_top_m"])
        v = best_velocity(Ez, X, Y, t, xc, yc)
        I, xg, yg, zg = migrate(Ez, X, Y, t, xc, yc, v=v)
        pi, pj, pk = np.unravel_index(np.argmax(I), I.shape)
        results.append(dict(ID=str(d["ID"]), item=str(d["item"]),
                            material=str(d["material"]), x_est=xg[pi], y_est=yg[pj],
                            z_est=zg[pk], dx_cm=(xg[pi]-xc)*100, dy_cm=(yg[pj]-yc)*100,
                            dz_cm=(zg[pk]-depth)*100, v_best=v))
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": f"### {d['ID']} — {d['item']} ({d['material']})".splitlines(keepends=True)})
        code(cells,
             f"d=np.load(f'{G3D}/{d['ID']}_cube.npz')\n"
             f"Ez,X,Y,t=d['Ez'],d['X'],d['Y'],d['t']*1e9\n"
             f"v=best_velocity(Ez,X,Y,t,{xc},{yc})\n"
             f"I,xg,yg,zg=migrate(Ez,X,Y,t,{xc},{yc},v=v)\n"
             f"pi,pj,pk=np.unravel_index(np.argmax(I),I.shape)\n"
             f"print(f'peak (x,y,z)=({xg[pi]:.3f},{yg[pj]:.3f},{zg[pk]:.3f})  "
             f"v={{v:.3f}} m/ns')",
             outputs=[{"output_type": "stream", "name": "stdout",
                       "text": [f"peak (x,y,z)=({xg[pi]:.3f},{yg[pj]:.3f},{zg[pk]:.3f})  "
                                f"v={v:.3f} m/ns\n"]}])
        # interactive rotatable volume
        html = plotly_volume_html(d["ID"], I, xg, yg, zg)
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "source": "from IPython.display import HTML\n"
                                f"HTML({json.dumps(html)})".splitlines(keepends=True),
                      "outputs": [{"output_type": "display_data",
                                   "metadata": {}, "data": {"text/html": [html]}}]})
    results = pd.DataFrame(results)

    # summary table output
    md(cells, "## Summary table")
    code(cells, "results", outputs=[{"output_type": "execute_result", "metadata": {},
             "execution_count": 9,
             "data": {"text/plain": results.round(3).to_string()}}])

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUTNB, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", OUTNB, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()