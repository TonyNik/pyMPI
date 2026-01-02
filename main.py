#!/usr/bin/env python3
"""

A command-line version of the Magnetic Particle Imaging (MPI) simulation tool.
This script uses the core simulation modules (Control.py, Phantom.py, Scanner.py,
XRecon.py, and Interference.py) to generate simulation data and display results using
VTK (for volume rendering) and matplotlib (for 2D projections).

"""

import sys
import argparse
import ast
import json
import napari
import matplotlib.pyplot as plt
from render import *
from Model.Control import get_ReconImg, get_ReconImgItf, get_OriImgData, JsonDefaultEnconding

from MPIRF.ReconClass.BaseClass.OutputSaver import save_simulation_message, save_vtk_screenshot, save_projection_slice


import numpy as np



def main():
    parser = argparse.ArgumentParser(
        description="Magnetic Particle Imaging Simulation (CLI Version)")

    # Particle
    parser.add_argument("--particle_tem", type=str, default="19.85",
                        help="Particle temperature in ℃ (default: 19.85)")
    parser.add_argument("--particle_dia", type=float, default=27,
                        help="Particle diameter in nm (default: 27)")
    parser.add_argument("--particle_sat", type=float, default=1,
                        help="Particle saturation in T (default: 1)")
    parser.add_argument("--particle_con", type=str, default="100",
                        help="Particle concentration in mmol/L (default: 100)")

    # Gradients
    parser.add_argument("--hard_sel_gra_x", type=str, default="2.75",
                        help="Selection gradient X in T/m (default: 2.75)")
    parser.add_argument("--hard_sel_gra_y", type=str, default="2.75",
                        help="Selection gradient Y in T/m (default: 2.75)")
    parser.add_argument("--hard_sel_gra_z", type=str, default="5.5",
                        help="Selection gradient Z in T/m (default: 5.5)")

    # Drive fields (used by Bruker)
    parser.add_argument("--hard_dri_fre_x", type=str, default="24.51",
                        help="Drive frequency X in KHz (default: 24.51)")
    parser.add_argument("--hard_dri_fre_y", type=str, default="26.04",
                        help="Drive frequency Y in KHz (default: 26.04)")
    parser.add_argument("--hard_dri_fre_z", type=str, default="25.25",
                        help="Drive frequency Z in KHz (default: 25.25)")

    parser.add_argument("--hard_dri_amp_x", type=str, default="6",
                        help="Drive amplitude X in mT (default: 6)")
    parser.add_argument("--hard_dri_amp_y", type=str, default="6",
                        help="Drive amplitude Y in mT (default: 6)")
    parser.add_argument("--hard_dri_amp_z", type=str, default="6",
                        help="Drive amplitude Z in mT (default: 6)")

    # Timing
    parser.add_argument("--hard_rep_time", type=str, default="21.54",
                        help="Repetition time in ms (default: 21.54)")
    parser.add_argument("--hard_sam_fre", type=str, default="1.25",
                        help="Sampling frequency in MHz (default: 0.25)")

    # Interference / noise
    parser.add_argument("--particle_rel", type=str, default="False",
                        help="Particle relaxation flag (True/False) (default: False)")
    parser.add_argument("--particle_rel_val", type=str, default="3",
                        help="Particle relaxation value in µs (default: 3)")
    parser.add_argument("--noise", type=str, default="False",
                        help="Noise flag (True/False) (default: False)")
    parser.add_argument("--noise_val", type=str, default="15",
                        help="Noise value in dB (default: 25)")
    parser.add_argument("--background", type=str, default="False",
                        help="Background flag (True/False) (default: False)")
    parser.add_argument("--background_val", type=str, default="5",
                        help="Background value in dB (default: 5)")

    # Phantom & outputs
    parser.add_argument("--combo_theme", type=int, default=9,
                        help="Phantom shape index (0: E Shape, 1: P Shape, 2: Point) (default: 0)")
    parser.add_argument("--combo_theme_plane", type=int, default=0,
                        help="Projection view index (0: XY Max, 1: XY Slice, etc.) (default: 0)")
    parser.add_argument("--slice_index", type=int, default=10,
                        help="Slice index for single-slice projection (default: 0)")
    parser.add_argument("--output_json", type=str, default="temp/output.json",
                        help="Path to save simulation JSON output")
    parser.add_argument("--mode", type=str, default="interference", choices=["generate", "interference"],
                        help="Simulation mode: 'generate' or 'interference'")



    parser.add_argument("--trajectory_type", type=str, default="FFL-STACK",
                        help="Trajectory type. Options include: "
                             "'FFL-STACK' (Sinogram FFL reco according to Gapyak), "
                             "'FFP-3D-L' (FFP 3D Lissajous), "
                             "'FFP-3D-RAD-L' (FFP 3D Radial Lissajous trajectory), "
                             "'FFP-3D-SP' (FFP 3D Spiral trajectory trajectory), "
                             "'FFP-3D-CART' (FFP 3D Cartesian trajectory).")

    parser.add_argument("--output_img", type=str, default="temp/3Doutput.png",
                        help="Path to save the VTK render window screenshot (PNG format)")
    parser.add_argument("--output_slice", type=str, default="temp/2Doutput.png",
                        help="Path to save the 2D projection/slice image (PNG format)")

    args = parser.parse_args()
    params = vars(args)

    # Unpack & convert
    tt = ast.literal_eval(params["particle_tem"])
    dt = float(params["particle_dia"]) * 1e-9
    ms = float(params["particle_sat"])
    cn_val = ast.literal_eval(params["particle_con"])
    if isinstance(cn_val, (list, tuple)): cn_val = cn_val[0]
    cn = cn_val * 1e-3

    gx = ast.literal_eval(params["hard_sel_gra_x"])
    gy = ast.literal_eval(params["hard_sel_gra_y"])
    gz = ast.literal_eval(params["hard_sel_gra_z"])

    fx = ast.literal_eval(params["hard_dri_fre_x"]) * 1e3
    fy = ast.literal_eval(params["hard_dri_fre_y"]) * 1e3
    fz = ast.literal_eval(params["hard_dri_fre_z"]) * 1e3

    ax = ast.literal_eval(params["hard_dri_amp_x"]) * 1e-3
    ay = ast.literal_eval(params["hard_dri_amp_y"]) * 1e-3
    az = ast.literal_eval(params["hard_dri_amp_z"]) * 1e-3

    rt = ast.literal_eval(params["hard_rep_time"]) * 1e-3
    sf = ast.literal_eval(params["hard_sam_fre"]) * 1e6

    ref = params["particle_rel"].lower() in ["true", "1", "yes"]
    re_val = ast.literal_eval(params["particle_rel_val"]) * 1e-6 if ref else 0
    nsf = params["noise"].lower() in ["true", "1", "yes"]
    ns_val = ast.literal_eval(params["noise_val"]) if nsf else 0
    bgf = params["background"].lower() in ["true", "1", "yes"]
    bg_val = ast.literal_eval(params["background_val"]) if bgf else 0

    index = params["combo_theme"]
    trajectory_type = params["trajectory_type"]

    # Run simulation
    ImgStru, OrgImgData, Message = get_ReconImg(
        tt, dt, ms, cn, gx, gy, gz, fx, fy, fz,
        ax, ay, az, rt, sf, ref, re_val, nsf, ns_val, bgf, bg_val, index, trajectory_type=trajectory_type
    )

    #OrgImgData = OrgImgData / np.max(OrgImgData)
    print(np.shape(OrgImgData))

    ImgData = ImgStru.get_ImagSiganl()[1][0]
    print("Simulation data generated.")

    # 2D projection
    plot_projection(ImgData, plane_index=params["combo_theme_plane"], slice_index=params["slice_index"])
    if params["output_slice"]:
        save_projection_slice(ImgData, plane_index=params["combo_theme_plane"],
                              slice_index=params["slice_index"], filename=params["output_slice"])

    # 3D render (napari)
    viewer = render_napari_volumes(OrgImgData, ImgData, overlay_text=None, title="MPI Phantom + Recon")

    if params["mode"] == "interference":
        params["particle_rel"] = "False"
        params["noise"] = "True"
        params["background"] = "False"

        ref = params["particle_rel"].lower() in ["true", "1", "yes"]
        re_val = ast.literal_eval(params["particle_rel_val"]) * 1e-6 if ref else 0
        nsf = params["noise"].lower() in ["true", "1", "yes"]
        ns_val = ast.literal_eval(params["noise_val"]) if nsf else 0
        bgf = params["background"].lower() in ["true", "1", "yes"]
        bg_val = ast.literal_eval(params["background_val"]) if bgf else 0

        ImgStru = get_ReconImgItf(ref, re_val, nsf, ns_val, bgf, bg_val, Message)
        ImgData = ImgStru.get_ImagSiganl()[1][0]
        print("Interference simulation applied.")

        plot_projection(ImgData, plane_index=params["combo_theme_plane"], slice_index=params["slice_index"])
        if params["output_slice"]:
            save_projection_slice(ImgData, plane_index=params["combo_theme_plane"],
                                  slice_index=params["slice_index"], filename="temp/2Dint.png")

        if params["output_json"]:
            save_simulation_message(Message, "temp/output_int.json")

        viewer = render_napari_volumes(
            OrgImgData,
            ImgData,
            axis_order="xyz",
            rendering="mip",  # try "attenuated_mip" too
            #overlay_text=str(Message),  # or any text you want
            overlay_position="top_left",
            output_img=params["output_img"] if params["output_img"] else None,
        )
        napari.run()


if __name__ == "__main__":
    main()
