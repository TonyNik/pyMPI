import numpy as np, math
from MPIRF.Config.ConstantList import *
import vtkmodules.all as vtk
import os
from itertools import permutations
from collections import deque

class PhantomClass(object):
    def __init__(self, Temperature=25, Diameter=50e-9, MagSaturation=0.6, Concentration=5e9, index=0):
        self._Tt = Temperature + TDT                    # Temperature in Celsius
        self._Diameter = Diameter                       # Diameter in m
        self._Volume = self.__get_ParticleVolume()      # Volume in m^3
        self._MCore = MagSaturation / U0                # Core magnetization in A/m (T/μ0)
        self._Mm = self.__get_MagMomentSaturation()     # Magnetic moment saturation: A/m * m^3
        self._Bb = self.__get_ParticleProperty()        # μ0 * _Mm / (kB * T)
        self._Concentration = Concentration             # in mol/L (?)
        self._index = index

    def __get_ParticleVolume(self):
        return (self._Diameter ** 3) * PI / 6.0

    def __get_MagMomentSaturation(self):
        return self._MCore * self._Volume

    def __get_ParticleProperty(self):
        return (U0 * self._Mm) / (KB * self._Tt)

    def get_Bcoeff(self):
        return self._Bcoeff

    def getShape(self, Xn, Yn, Zn):
        if self._index == 0:
            return self.getEShape(Xn, Yn, Zn)
        elif self._index == 1:
            return self.getPShape(Xn, Yn, Zn)
        elif self._index == 2:
            return self.getPointShape(Xn, Yn, Zn)
        elif self._index == 3:
            return self.getMPIShape(Xn, Yn, Zn)  # 'MPI' word
        elif self._index == 4:
            # “Shape phantom”: truncated cone (axis - x), r_tip=1 mm, half-angle=10°, height=22 mm
            return self.getShapePhantomShape(
                Xn, Yn, Zn,
                tip_radius_mm=1.0,
                half_angle_deg=10.0,  # half-apex angle (NOT full)
                height_mm=22.0,
                axis='x',
                voxel_mm_hint=0.1,  # matches Scanner step of 1e-4 m (=0.1 mm) by default
                fit_to_grid=True
            )
        elif self._index == 5:
            # NEW: STL-calibrated resolution phantom (manifold + 5 tubes with measured angles)
            return self.getResolutionPhantom_STLCalibrated(
                Xn, Yn, Zn,
                voxel_mm_hint=0.1
            )
        elif self._index == 6:
            # Concentration phantom with exact values from the table (mmol/L)
            return self.getConcentrationPhantom(
                Xn, Yn, Zn,
                cube_edge_mm=2.0,
                center_spacing_xy_mm=12.0,
                center_spacing_z_mm=6.0,
                voxel_mm_hint=0.1,
                absolute_concs_mmol=[100.0, 66.6, 44.4, 29.6, 19.7, 13.1, 8.77, 5.85]
            )
        elif self._index == 7:
            p = getattr(self, "_shape_params", {}) or {}
            return self.getFourCircles(
                Xn, Yn, Zn,
                radii_mm=p.get("radii_mm", (4, 3, 2, 1)),
                voxel_size_mm=float(p.get("voxel_size_mm", 1.0)),
                outer_margin_mm=float(p.get("outer_margin_mm", 1.0)),
                z_thickness_mm=p.get("z_thickness_mm", 4),
                cluster_gap_mm=float(p.get("cluster_gap_mm", 10)),
                push_frac=float(p.get("push_frac", 0.0)),
                concentration_mmol_L = p.get("concentration_mmol_L", 5e18)
            )
        elif self._index == 8:
            p = getattr(self, "_shape_params", {}) or {}
            return self.getSingleCircle(
                Xn, Yn, Zn,
                radius_mm=float(p.get("radius_mm", 17)),
                voxel_size_mm=float(p.get("voxel_size_mm", 1)),
                outer_margin_mm=float(p.get("outer_margin_mm", 1.0)),
                z_thickness_mm=p.get("z_thickness_mm", 10),
                concentration_mmol_L=p.get("concentration_mmol_L", None)
            )
        elif self._index == 9:
            p = getattr(self, "_shape_params", {}) or {}
            return self.getFourCubes(
                Xn, Yn, Zn,
                side_mm=p.get("side_mm", (5, 4, 3, 2)),
                voxel_size_mm=float(p.get("voxel_size_mm", 1)),
                outer_margin_mm=float(p.get("outer_margin_mm", 1.0)),
                z_thickness_mm=p.get("z_thickness_mm", 7),
                cluster_gap_mm=float(p.get("cluster_gap_mm", 10)),
                push_frac=float(p.get("push_frac", 0.0)),
                concentration_mmol_L=p.get("concentration_mmol_L", None)
            )
        elif self._index == 10:
            return self.getSTLShape(Xn, Yn, Zn)  # STL file
        else:
            return self.getEShape(Xn, Yn, Zn)

    def getEShape(self, Xn, Yn, Zn):
        print("E-shape phantom selected")
        C = np.zeros((Xn, Yn, Zn))
        C[int(Xn * (30 / 201)):int(Xn * (170 / 201)), 10:Yn - 10, int(Zn * (50 / 201)):int(Zn * (150 / 201))] = np.ones(
            (int(Xn * (170 / 201)) - int(Xn * (30 / 201)), Yn - 20, int(Zn * (150 / 201)) - int(Zn * (50 / 201))))
        C[int(Xn * (53 / 201)):int(Xn * (88 / 201)), 10:Yn - 10, int(Zn * (76 / 201)):int(Zn * (150 / 201))] = 0
        C[int(Xn * (112 / 201)):int(Xn * (147 / 201)), 10:Yn - 10, int(Zn * (76 / 201)):int(Zn * (150 / 201))] = 0
        return C * self._Concentration * 1e-12

    def getPShape(self, Xn, Yn, Zn):
        print("P-shape phantom selected")
        C = np.zeros((Xn, Yn, Zn))
        C[int(Xn * (50 / 201)):int(Xn * (150 / 201)),
          int(Yn * (25 / 201)):int(Yn * (175 / 201)),
          10:Zn - 10] = 1
        C[int(Xn * (75 / 201)):int(Xn * (125 / 201)),
          int(Yn * (100 / 201)):int(Yn * (150 / 201)),
          10:Zn - 10] = 0
        C[int(Xn * (75 / 201)):int(Xn * (150 / 201)),
          int(Yn * (25 / 201)):int(Yn * (75 / 201)),
          10:Zn - 10] = 0
        return C * self._Concentration * 1e-12

    def getPointShape(self, Xn, Yn, Zn, pt=None):
        """
        Point-like phantom: a single voxel set to 1 at 'pt' or at the volume center.
        """
        print("Point-like phantom selected")
        C = np.zeros((Xn, Yn, Zn))

        if pt is None:
            ix, iy, iz = Xn // 2, Yn // 2, Zn // 2
        else:
            ix, iy, iz = pt
            ix = max(0, min(Xn - 1, int(ix)))
            iy = max(0, min(Yn - 1, int(iy)))
            iz = max(0, min(Zn - 1, int(iz)))

        C[ix, iy, iz] = 1.0
        return C * self._Concentration * 1e-12

    def getMPIShape(self, Xn: int, Yn: int, Zn: int, text: str = "MPI"):
        """
        Create a block-letter phantom spelling `text` (default: 'MPI'),
        constrained to the SAME (x,y,z) extents used by getEShape:
          x: [30/201 .. 170/201] * Xn
          y: [10 .. Yn-10]
          z: [50/201 .. 150/201] * Zn

        Updates:
          - Letters are 25% shorter in y (75% of box height), centered vertically.
          - Inter-letter spacing increased further.
          - 'M' keeps a clear top notch via diagonal thickness control + V-notch carving.
          - z thickness unchanged.

        Returns a (Xn, Yn, Zn) float array scaled by self._Concentration * 1e-12.
        Axes: [x, y, z]
        """

        # ---- Match E-shape bounding box exactly ----
        x0 = int(Xn * (30 / 201))
        x1 = int(Xn * (170 / 201))
        y0 = 10
        y1 = Yn - 10
        z0 = int(Zn * (50 / 201))
        z1 = int(Zn * (150 / 201))

        W = max(1, x1 - x0)  # drawable width inside the box
        H = max(1, y1 - y0)  # drawable height inside the box

        C = np.zeros((Xn, Yn, Zn), dtype=float)
        if W < 5 or H < 5 or z1 <= z0:
            print("Word phantom selected but grid too small for bounding box; returning zeros.")
            return C

        text = (text or "MPI").upper()
        n_letters = len(text)

        # ---- Layout inside the (x0:x1, y0:y1) slab ----
        spacing_frac = 0.22  # increased separation to reduce cross-talk
        stroke_frac = 0.14  # bold-ish strokes in x,y
        height_frac = 0.75  # letters are 25% shorter in y
        min_spacing = 2
        min_stroke = 2

        spacing = max(min_spacing, int(round(spacing_frac * W)))
        letter_w = max(5, (W - spacing * (n_letters - 1)) // max(1, n_letters))
        letter_h = max(5, int(round(height_frac * H)))

        # stroke thickness (clamped so letters don't collapse)
        stroke = max(min_stroke, int(round(stroke_frac * min(letter_w, letter_h))))
        stroke = min(stroke, letter_w // 3, letter_h // 3)

        # recompute total width and center horizontally
        total_w = letter_w * n_letters + spacing * (n_letters - 1)
        if total_w > W:
            overflow = total_w - W
            letter_w = max(5, letter_w - (overflow // max(1, n_letters)))
            total_w = letter_w * n_letters + spacing * (n_letters - 1)
        x_left_pad = max(0, (W - total_w) // 2)  # center the word

        # vertical centering for the shorter letters
        y_pad = max(0, (H - letter_h) // 2)

        # ---- 2D drawing helpers (on the local plane of size W×H) ----
        def _draw_rect(mask, x0_, x1_, y0_, y1_):
            x0c, x1c = max(0, x0_), min(mask.shape[0], x1_)
            y0c, y1c = max(0, y0_), min(mask.shape[1], y1_)
            if x1c > x0c and y1c > y0c:
                mask[x0c:x1c, y0c:y1c] = True

        def _draw_line_thick(mask, x0_, y0_, x1_, y1_, rad):
            # distance-based thick line
            W_, H_ = mask.shape
            xs = np.arange(W_)[:, None]
            ys = np.arange(H_)[None, :]
            dx = x1_ - x0_
            dy = y1_ - y0_
            if dx == 0 and dy == 0:
                mask[(xs - x0_) ** 2 + (ys - y0_) ** 2 <= rad * rad] = True
                return
            denom = float(dx * dx + dy * dy)
            t = ((xs - x0_) * dx + (ys - y0_) * dy) / denom
            t = np.clip(t, 0.0, 1.0)
            projx = x0_ + t * dx
            projy = y0_ + t * dy
            dist2 = (xs - projx) ** 2 + (ys - projy) ** 2
            mask[dist2 <= rad * rad] = True

        def _fill_triangle(mask, ax, ay, bx, by, cx, cy):
            """Rasterize a filled triangle into `mask` using half-planes; True means 'inside'."""
            W_, H_ = mask.shape
            xs = np.arange(W_)[:, None]
            ys = np.arange(H_)[None, :]

            def side(x1, y1, x2, y2, x, y):
                return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

            s1 = side(ax, ay, bx, by, xs, ys)
            s2 = side(bx, by, cx, cy, xs, ys)
            s3 = side(cx, cy, ax, ay, xs, ys)

            # point is inside if all cross products have the same sign (or zero)
            inside = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
            mask[inside] = True

        def _carve_triangle(mask, ax, ay, bx, by, cx, cy):
            tri = np.zeros_like(mask, dtype=bool)
            _fill_triangle(tri, ax, ay, bx, by, cx, cy)
            mask[tri] = False  # subtract

        # ---- Letter glyphs ----
        def _letter_M(w, h, t):
            m = np.zeros((w, h), dtype=bool)

            # Vertical stems
            _draw_rect(m, 0, t, 0, h)
            _draw_rect(m, w - t, w, 0, h)

            # Diagonals: slightly slimmer than verticals to avoid overfill on short letters
            td = max(1, int(round(0.7 * t)))
            cx = w // 2
            _draw_line_thick(m, 0, h - 1, cx, 0, td)  # left diagonal
            _draw_line_thick(m, w - 1, h - 1, cx, 0, td)  # right diagonal

            # Carve a V-notch near the top to preserve the 'M' valley even with thick strokes.
            # Apex slightly below top to keep a clean point; base stays inside the stems.
            apex_y = max(0, int(0.02 * h))
            base_y = min(h, int(0.45 * h))
            left_base_x = max(t, int(0.22 * w))
            right_base_x = min(w - t, int(0.78 * w))
            _carve_triangle(m, cx, apex_y, left_base_x, base_y, right_base_x, base_y)

            return m

        def _letter_P(w, h, t):
            m = np.zeros((w, h), dtype=bool)
            _draw_rect(m, 0, t, 0, h)  # backbone
            bowl_bot = max(0, int(0.45 * h))  # top ~55%
            _draw_rect(m, 0, w, bowl_bot, h)  # outer bowl
            _draw_rect(m, t, w - t, bowl_bot + t, h - t)  # hollow
            return m

        def _letter_I(w, h, t):
            m = np.zeros((w, h), dtype=bool)
            _draw_rect(m, 0, w, 0, t)  # top serif
            _draw_rect(m, 0, w, h - t, h)  # bottom serif
            cx0 = max(0, w // 2 - (t // 2 + t // 4))
            cx1 = min(w, cx0 + t + (t // 2))
            _draw_rect(m, cx0, cx1, 0, h)  # central vertical
            return m

        def _make_letter(ch, w, h, t):
            ch = ch.upper()
            if ch == 'M':
                return _letter_M(w, h, t)
            elif ch == 'P':
                return _letter_P(w, h, t)
            elif ch == 'I':
                return _letter_I(w, h, t)
            return np.zeros((w, h), dtype=bool)  # unknowns -> blank

        # ---- Compose local plane (W×H) and paste into global [x0:x1, y0:y1] ----
        plane_local = np.zeros((W, H), dtype=bool)
        x_cursor = x_left_pad
        for idx, ch in enumerate(text):
            letter = _make_letter(ch, w=letter_w, h=letter_h, t=stroke)  # (letter_w × letter_h)
            xs0 = x_cursor
            xs1 = min(W, xs0 + letter_w)
            ys0 = y_pad
            ys1 = min(H, ys0 + letter_h)
            # paste with bounds checks
            w_slice = xs1 - xs0
            h_slice = ys1 - ys0
            if w_slice > 0 and h_slice > 0:
                plane_local[xs0:xs1, ys0:ys1] |= letter[:w_slice, :h_slice]
            x_cursor += letter_w + (spacing if idx < n_letters - 1 else 0)
            if x_cursor >= W:
                break

        # Paste into the global volume within the E-shape box (z thickness unchanged)
        C[x0:x1, y0:y1, z0:z1] = plane_local[:, :, None].astype(float)

        print(f"Word phantom selected (shorter-y, wider spacing, fixed 'M'): '{text}'")
        return C * self._Concentration * 1e-12

    def getShapePhantomShape(self,
                     Xn, Yn, Zn,
                     tip_radius_mm=1.0,
                     half_angle_deg=10.0,
                     height_mm=22.0,
                     axis='x',
                     voxel_mm_hint=0.1,
                     fit_to_grid=True,
                     return_meta=False):
        """
        Truncated cone phantom aligned with the 'axis' (default: x-axis).
        • Tip radius r_tip = tip_radius_mm
        • Half-apex angle α = half_angle_deg  (so full apex angle = 2α)
        • Height h = height_mm
        • By default, we assume 1 voxel ≈ voxel_mm_hint mm (0.1 mm matches Scanner step=1e-4 m).
          If the height wouldn't fit into Xn, we adapt the effective voxel size to fit ('fit_to_grid=True').

        Geometry (continuous):
            r(s) = r_tip + s * tan(α),  0 ≤ s ≤ h
            R_base = r_tip + h * tan(α)

        Returns:
            3D array C[Xn,Yn,Zn] with concentration scaling (same scaling as other shapes):
                C = 1 inside cone, 0 outside, finally scaled by self._Concentration * 1e-12
            If return_meta=True, returns (C_scaled, meta_dict) with effective voxel mm, base radius, etc.
        """


        # --- orientation (currently implemented for axis='x') ---
        if axis.lower() != 'x':
            raise NotImplementedError("getConeShape currently supports axis='x' (so YZ slices are circles).")

        # --- effective voxel size in mm; optionally adapt to ensure the full height fits ---
        eff_voxel_mm = float(voxel_mm_hint)
        h_vox = int(round(height_mm / eff_voxel_mm))
        if fit_to_grid and h_vox > (Xn - 2):
            # squeeze to fit the specified height into available Xn while preserving the tip & angle
            eff_voxel_mm = float(height_mm / max(1, (Xn - 2)))
            h_vox = int(round(height_mm / eff_voxel_mm))

        # --- cone parameters in voxel units ---
        alpha = math.radians(half_angle_deg)  # half-angle
        tan_a = math.tan(alpha)
        r_tip_vox = tip_radius_mm / eff_voxel_mm
        r_base_vox = r_tip_vox + h_vox * tan_a

        # --- place apex near the low-x side; center the axis in YZ ---
        i0 = 1  # apex slice index
        i1 = min(i0 + h_vox, Xn - 1)  # last slice belonging to the frustum
        yc = (Yn - 1) / 2.0
        zc = (Zn - 1) / 2.0

        # --- precompute per-slice radius (in voxels) ---
        i_arr = np.arange(Xn)
        s_vox = (i_arr - i0).astype(float)  # axial distance from apex in voxels
        valid = (i_arr >= i0) & (i_arr <= i1)
        r_i = r_tip_vox + s_vox * tan_a
        r_i[~valid] = -1.0  # mark out-of-range slices

        # --- radial grid in YZ (distance to axis centerline) ---
        Yg, Zg = np.meshgrid(np.arange(Yn), np.arange(Zn), indexing='ij')
        rho = np.sqrt((Yg - yc) ** 2 + (Zg - zc) ** 2)

        # --- fill cone volume ---
        C = np.zeros((Xn, Yn, Zn), dtype=float)
        for i in range(Xn):
            ri = r_i[i]
            if ri <= 0:
                continue
            C[i, :, :] = (rho <= ri).astype(float)

        # --- scale by concentration (match other phantoms’ convention) ---
        C_scaled = C * self._Concentration * 1e-12

        if not return_meta:
            return C_scaled

        # Metadata (handy for debugging/verification)
        # Physical volume of the discretized object (mm^3 == µL)
        vox_vol_mm3 = eff_voxel_mm ** 3
        vol_mm3 = float(np.sum(C)) * vox_vol_mm3
        meta = {
            "effective_voxel_mm": eff_voxel_mm,
            "height_vox": h_vox,
            "r_tip_vox": r_tip_vox,
            "r_base_vox": r_base_vox,
            "r_base_mm": r_base_vox * eff_voxel_mm,
            "discrete_volume_uL": vol_mm3,  # 1 mm^3 = 1 µL
            "target_volume_uL": (math.pi * height_mm * ((r_tip_vox * eff_voxel_mm + h_vox * eff_voxel_mm * tan_a) ** 2
                                                        + (r_tip_vox * eff_voxel_mm) * (
                                                                    r_tip_vox * eff_voxel_mm + h_vox * eff_voxel_mm * tan_a)
                                                        + (r_tip_vox * eff_voxel_mm) ** 2) / 3.0)
        }
        return C_scaled, meta

    def getResolutionPhantom_STLCalibrated(self,
                                           Xn, Yn, Zn,
                                           voxel_mm_hint=0.1,
                                           manifold_radius_mm=2.40,
                                           manifold_length_mm=3.0,
                                           tube_radius_mm=0.50,
                                           use_mm_anchor=False,
                                           anchor_y_mm=8.50,
                                           anchor_z_mm=15.87,
                                           return_meta=False):
        """
        STL-calibrated resolution phantom (robust auto-scaling):
          • One entrance manifold on the min-x face (short cylinder along +x).
          • Five tubes starting just after the manifold:
              XY plane:  -10°, +15°
              X   axis:   0°
              XZ plane:  +20°, -30°
          • Auto-calibrates the effective voxel size so geometry fits the grid:
              - entrance radius ≤ 0.20 * min(Yn, Zn)
              - tube radius ≥ 2.0 vox (anti-aliasing) but not oversized
          • Optional mm-anchoring for entrance center; defaults to YZ center for safety.

        Returns
        -------
        C_scaled : (Xn,Yn,Zn) float array
            1 inside union of manifold+tubes (scaled by self._Concentration * 1e-12), 0 outside.
        meta : dict (optional)
        """


        # ------------------ AUTO CALIBRATE VOXEL SIZE ------------------
        # Start from hint, then adapt to grid to avoid "fill the FOV" or "vanish" issues.
        # Target constraints (voxels):
        minYZ = float(max(1, min(Yn, Zn)))
        target_tube_r_vox_min = 2.0  # keep tubes visibly ≥ ~2 vox
        target_man_r_vox_max = 0.20 * minYZ  # entrance radius bounded to 20% of YZ span
        # Candidate voxel sizes from each constraint:
        #   tube radius >= 2 vox  -> eff ≤ tube_radius_mm / 2
        #   manifold radius ≤ 20% -> eff ≥ manifold_radius_mm / (0.20 * minYZ)
        eff_from_tube = tube_radius_mm / target_tube_r_vox_min
        eff_from_man = manifold_radius_mm / max(1.0, target_man_r_vox_max)
        # Choose an eff that satisfies BOTH:
        eff_voxel_mm = max(float(voxel_mm_hint), eff_from_man)
        eff_voxel_mm = min(eff_voxel_mm, eff_from_tube) if eff_from_tube > 0 else eff_voxel_mm
        # If that inverted the inequalities (small grids, extreme params), prioritize not overfilling:
        if manifold_radius_mm / eff_voxel_mm > target_man_r_vox_max:
            eff_voxel_mm = manifold_radius_mm / max(1.0, target_man_r_vox_max)

        # Final radii/lengths in voxels (with gentle clamps)
        r_man_vox = max(1.5, manifold_radius_mm / eff_voxel_mm)
        r_tube_vox = max(2.0, min(0.08 * minYZ, tube_radius_mm / eff_voxel_mm))  # cap tube radius to 8% of minYZ
        L_man_vox = manifold_length_mm / eff_voxel_mm
        L_man_vox = max(1.0, min(L_man_vox, 0.15 * Xn))  # manifold length ≤ 15% of x-span

        # ------------------ ORIGIN (min-x face) ------------------
        if use_mm_anchor:
            yc = anchor_y_mm / eff_voxel_mm
            zc = anchor_z_mm / eff_voxel_mm
            # If the anchor lands outside, fall back to center
            if not (0 <= yc < Yn and 0 <= zc < Zn):
                yc, zc = (Yn - 1) / 2.0, (Zn - 1) / 2.0
        else:
            yc, zc = (Yn - 1) / 2.0, (Zn - 1) / 2.0
        p0 = np.array([1.0, yc, zc], dtype=float)  # one voxel inside min-x plane

        # ------------------ DIRECTIONS ------------------
        def uv_from_angles(xy_deg=None, xz_deg=None):
            if xy_deg is not None:
                th = math.radians(xy_deg)
                v = np.array([math.cos(th), math.sin(th), 0.0], dtype=float)
            elif xz_deg is not None:
                ph = math.radians(xz_deg)
                v = np.array([math.cos(ph), 0.0, math.sin(ph)], dtype=float)
            else:
                v = np.array([1.0, 0.0, 0.0], dtype=float)
            n = np.linalg.norm(v)
            return v / n if n > 0 else np.array([1.0, 0.0, 0.0], dtype=float)

        dirs = [
            uv_from_angles(xy_deg=-10.0),
            uv_from_angles(xy_deg=+15.0),
            np.array([1.0, 0.0, 0.0], dtype=float),
            uv_from_angles(xz_deg=+20.0),
            uv_from_angles(xz_deg=-30.0),
        ]

        # ------------------ HELPERS ------------------
        def max_length_inside(p0, d, shape):
            """Forward extent t >= 0 until leaving [0..N-1]^3, with 1-voxel safety margin."""
            bounds_min = np.array([0.0, 0.0, 0.0])
            bounds_max = np.array([shape[0] - 1.0, shape[1] - 1.0, shape[2] - 1.0])
            t_exit = float('inf')
            for ax in range(3):
                if abs(d[ax]) < 1e-12:
                    if p0[ax] < bounds_min[ax] or p0[ax] > bounds_max[ax]:
                        return 0.0
                    continue
                t1 = (bounds_min[ax] - p0[ax]) / d[ax]
                t2 = (bounds_max[ax] - p0[ax]) / d[ax]
                t_pos = [t for t in (t1, t2) if t > 0]
                if not t_pos:
                    return 0.0
                t_exit = min(t_exit, min(t_pos))
            return max(0.0, t_exit - 1.0)

        def paint_cylinder(C, p0, d, L_vox, r_vox):
            """Rasterize a finite cylinder via closest-point distance to the axis segment."""
            if L_vox <= 0.0 or r_vox <= 0.0:
                return 0
            p1 = p0 + L_vox * d
            pad = r_vox + 2.0
            xyz_min = np.floor(np.minimum(p0, p1) - pad).astype(int)
            xyz_max = np.ceil(np.maximum(p0, p1) + pad).astype(int)

            i0 = max(0, xyz_min[0]);
            i1 = min(C.shape[0] - 1, xyz_max[0])
            j0 = max(0, xyz_min[1]);
            j1 = min(C.shape[1] - 1, xyz_max[1])
            k0 = max(0, xyz_min[2]);
            k1 = min(C.shape[2] - 1, xyz_max[2])
            if (i0 > i1) or (j0 > j1) or (k0 > k1):
                return 0

            ii = np.arange(i0, i1 + 1, dtype=float)
            jj = np.arange(j0, j1 + 1, dtype=float)
            kk = np.arange(k0, k1 + 1, dtype=float)
            I, J, K = np.meshgrid(ii, jj, kk, indexing='ij')

            Vx = I - p0[0];
            Vy = J - p0[1];
            Vz = K - p0[2]
            t = Vx * d[0] + Vy * d[1] + Vz * d[2]
            t = np.clip(t, 0.0, L_vox)
            Cx = p0[0] + t * d[0];
            Cy = p0[1] + t * d[1];
            Cz = p0[2] + t * d[2]
            dx = I - Cx;
            dy = J - Cy;
            dz = K - Cz
            mask = (dx * dx + dy * dy + dz * dz) <= (r_vox * r_vox)

            npaint = int(np.count_nonzero(mask))
            if npaint:
                C[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1][mask] = 1.0
            return npaint

        # ------------------ RASTERIZE ------------------
        C = np.zeros((Xn, Yn, Zn), dtype=float)

        # 1) Entrance manifold (short +x cylinder)
        d_man = np.array([1.0, 0.0, 0.0], dtype=float)
        Lm = min(L_man_vox, max_length_inside(p0, d_man, C.shape))
        if Lm > 0:
            paint_cylinder(C, p0, d_man, Lm, r_man_vox)

        # 2) Start point for branches: just after manifold (plus 1*radius padding)
        start = p0 + (Lm + max(1.0, r_tube_vox)) * d_man

        # 3) Tubes
        for d in dirs:
            L_vox = max_length_inside(start, d, C.shape)
            # trim a bit to avoid touching far boundaries due to rounding
            L_vox = max(0.0, min(L_vox, Xn + Yn + Zn))  # harmless clamp
            paint_cylinder(C, start, d, L_vox, r_tube_vox)

        # Safety: if somehow nothing painted (shouldn't happen), paint a slim +x tube at center
        if not np.any(C):
            p0_fb = np.array([1.0, (Yn - 1) / 2.0, (Zn - 1) / 2.0], dtype=float)
            L_fb = max_length_inside(p0_fb, d_man, C.shape)
            paint_cylinder(C, p0_fb, d_man, L_fb, max(2.0, r_tube_vox))

        # Scale by concentration (convention used elsewhere)
        C_scaled = C * float(self._Concentration) * 1e-12

        if not return_meta:
            return C_scaled

        meta = {
            "effective_voxel_mm": float(eff_voxel_mm),
            "r_man_vox": float(r_man_vox),
            "r_tube_vox": float(r_tube_vox),
            "L_man_vox": float(Lm),
            "origin_vox": (float(p0[0]), float(p0[1]), float(p0[2])),
            "angles": {"xy_deg": [-10.0, +15.0], "xz_deg": [+20.0, -30.0], "x_ref": 0.0},
            "note": "Auto-calibrated voxel size to fit grid: entrance ≤ 20% of min(Y,Z); tubes ≥ 2 vox.",
        }
        return C_scaled, meta

    def getSTLShape(self, Xn, Yn, Zn, fill_fraction=0.90,
                    seal_radius_vox=2, bbox_pad=2):
        """
        Build a phantom where nanoparticles occupy ONLY the printable vacancies
        (channels/voids) of a 3D-printed part. These vacancies are open to the
        exterior in the raw STL but would be sealed after filling.

        Algorithm (voxel domain inside a tight ROI):
          1) Voxelize the printed SOLID: S (True=plastic).
          2) Virtually seal ports via morphological closing: S_sealed = close(S, r).
          3) Outside (under sealed condition): flood-fill(~S_sealed) from ROI boundary.
          4) Vacancies = (~S) & (~Outside).

        Parameters
        ----------
        Xn, Yn, Zn : int
            Grid dimensions.
        fill_fraction : float in (0,1]
            Uniform scale-to-fit of the part in the grid box.
        seal_radius_vox : int
            Spherical closing radius (voxels). Increase to seal larger ports.
        bbox_pad : int
            Padding (voxels) around the part's bounds for the voxelization ROI.

        Returns
        -------
        np.ndarray
            (Xn, Yn, Zn) float array with 1 in vacancies only, scaled by concentration.
        """
        try:
            import scipy.ndimage as ndi
        except Exception as e:
            raise RuntimeError(
                "getSTLShape requires SciPy (scipy.ndimage) for morphological closing. "
                "Please install SciPy or let me provide a VTK-based fallback."
            ) from e

        stl_path = "./STLs/Shape.STL"
        if not stl_path:
            raise ValueError("getSTLShape: 'stl_path' must be provided when index==4.")
        print(f"[STL vacancies] file='{stl_path}', fill_fraction={fill_fraction}, "
              f"seal_radius_vox={seal_radius_vox}, bbox_pad={bbox_pad}")

        # --- 1) Read STL and fit to grid ----------------------------------------
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_path))
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError(f"getSTLShape: Failed to load STL or STL is empty: {stl_path}")

        xmin, xmax, ymin, ymax, zmin, zmax = poly.GetBounds()
        dx = max(xmax - xmin, 1e-9)
        dy = max(ymax - ymin, 1e-9)
        dz = max(zmax - zmin, 1e-9)

        targx = max(1.0, (Xn - 1) * float(fill_fraction))
        targy = max(1.0, (Yn - 1) * float(fill_fraction))
        targz = max(1.0, (Zn - 1) * float(fill_fraction))
        s = min(targx / dx, targy / dy, targz / dz)

        cx_src, cy_src, cz_src = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        cx_tgt, cy_tgt, cz_tgt = 0.5 * (Xn - 1), 0.5 * (Yn - 1), 0.5 * (Zn - 1)

        tf = vtk.vtkTransform()
        tf.PostMultiply()
        tf.Scale(s, s, s)
        tf.Translate(cx_tgt - s * cx_src, cy_tgt - s * cy_src, cz_tgt - s * cz_src)

        tpf = vtk.vtkTransformPolyDataFilter()
        tpf.SetInputData(poly)
        tpf.SetTransform(tf)
        tpf.Update()
        solid_poly = tpf.GetOutput()

        # Triangulate & orient normals (robust signed distance)
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(solid_poly)
        tri.Update()
        solid_poly = tri.GetOutput()

        nrm = vtk.vtkPolyDataNormals()
        nrm.SetInputData(solid_poly)
        nrm.ConsistencyOn()
        nrm.AutoOrientNormalsOn()
        nrm.SplittingOff()
        nrm.Update()
        solid_poly = nrm.GetOutput()

        ipd_solid = vtk.vtkImplicitPolyDataDistance()
        ipd_solid.SetInput(solid_poly)

        # --- Tight ROI (keeps compute small but includes ports) ------------------
        bxmin, bxmax, bymin, bymax, bzmin, bzmax = solid_poly.GetBounds()
        i0 = max(0, int(math.floor(bxmin)) - int(bbox_pad))
        i1 = min(Xn - 1, int(math.ceil(bxmax)) + int(bbox_pad))
        j0 = max(0, int(math.floor(bymin)) - int(bbox_pad))
        j1 = min(Yn - 1, int(math.ceil(bymax)) + int(bbox_pad))
        k0 = max(0, int(math.floor(bzmin)) - int(bbox_pad))
        k1 = min(Zn - 1, int(math.ceil(bzmax)) + int(bbox_pad))

        # --- 2) Voxelize SOLID (S) in ROI ---------------------------------------
        sx, sy, sz = (i1 - i0 + 1), (j1 - j0 + 1), (k1 - k0 + 1)
        S = np.zeros((sx, sy, sz), dtype=bool)
        for ii, i in enumerate(range(i0, i1 + 1)):
            for jj, j in enumerate(range(j0, j1 + 1)):
                for kk, k in enumerate(range(k0, k1 + 1)):
                    S[ii, jj, kk] = (ipd_solid.EvaluateFunction(i, j, k) <= 0.0)

        solid_vox = int(S.sum())
        if solid_vox == 0:
            print("[STL vacancies] Warning: solid voxelization is empty inside ROI.")

        # --- 3) Virtually seal ports: morphological closing of S -----------------
        r = int(max(1, seal_radius_vox))
        # Create a spherical structuring element of radius r
        grid = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
        selem = (grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) <= (r * r)

        S_sealed = ndi.binary_closing(S, structure=selem)

        # --- 4) Outside under sealed condition: flood-fill in ~S_sealed ----------
        free_sealed = ~S_sealed  # places reachable by fluid when ports are sealed
        visited = np.zeros_like(free_sealed, dtype=bool)


        q = deque()

        def push(ii, jj, kk):
            if 0 <= ii < sx and 0 <= jj < sy and 0 <= kk < sz:
                if free_sealed[ii, jj, kk] and not visited[ii, jj, kk]:
                    visited[ii, jj, kk] = True
                    q.append((ii, jj, kk))

        # Seed all 6 ROI faces
        for ii in range(sx):
            for jj in range(sy):
                push(ii, jj, 0)
                push(ii, jj, sz - 1)
        for ii in range(sx):
            for kk in range(sz):
                push(ii, 0, kk)
                push(ii, sy - 1, kk)
        for jj in range(sy):
            for kk in range(sz):
                push(0, jj, kk)
                push(sx - 1, jj, kk)

        # BFS 6-connected
        neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
        while q:
            ii, jj, kk = q.popleft()
            for di, dj, dk in neigh:
                ni, nj, nk = ii + di, jj + dj, kk + dk
                if 0 <= ni < sx and 0 <= nj < sy and 0 <= nk < sz:
                    if free_sealed[ni, nj, nk] and not visited[ni, nj, nk]:
                        visited[ni, nj, nk] = True
                        q.append((ni, nj, nk))

        outside_blocked = visited  # what remains connected to ROI boundary after sealing

        # --- 5) Vacancies = (~S) minus outside_blocked ---------------------------
        vacancies_roi = (~S) & (~outside_blocked)
        vac_vox = int(vacancies_roi.sum())

        if vac_vox == 0:
            print("[STL vacancies] No vacancy voxels found. "
                  "Try increasing 'seal_radius_vox' (e.g., 3–6) or 'bbox_pad'.")

        # --- 6) Write back to full grid and scale by concentration ---------------
        C = np.zeros((Xn, Yn, Zn), dtype=float)
        C[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1] = vacancies_roi.astype(float)

        return C * self._Concentration * 1e-12

    def getConcentrationPhantom(self,
                                Xn, Yn, Zn,
                                cube_edge_mm=2.0,
                                center_spacing_xy_mm=12.0,
                                center_spacing_z_mm=6.0,
                                voxel_mm_hint=0.1,
                                base_conc_mmol=None,
                                dilution=1.5,
                                absolute_concs_mmol=None,
                                return_meta=False):
        """
        8-cube concentration phantom.

        Geometry (mm):
          • 8 cubes, edge = 2.0 mm → 8 µL each.
          • XY spacing (center-to-center) = 12.0 mm  → offsets ±6 mm in x and y.
          • Z spacing (center-to-center)  = 6.0  mm  → offsets ±3 mm in z.
          • Top layer (z=+3 mm): samples 1..4 starting at front-left (+x,+y), clockwise.
          • Bottom layer (z=-3 mm): samples 5..8 same order.

        Concentration (mmol/L):
          • If absolute_concs_mmol (len=8) is provided, use it exactly (independent of CLI).
          • Else, use a dilution series c_k = C0 / (dilution**(k-1)),
            with C0 = base_conc_mmol or self._Concentration.

          Voxels inside cube k get value (c_k * 1e-12) to match the project’s scaling.

        Robustness:
          • Auto-adjusts effective voxel size if needed so each cube spans ≥ ~2 vox/edge,
            while ensuring the layout fits the grid.

        Returns
        -------
        C : ndarray (Xn,Yn,Zn) – concentration map (scaled by 1e-12)
        meta : dict (optional)
        """
        # ------------------ choose effective voxel size (mm/voxel) ------------------
        edge = float(cube_edge_mm)
        half_edge = edge / 2.0
        off_xy = float(center_spacing_xy_mm) / 2.0    # ±6 mm
        off_z  = float(center_spacing_z_mm)  / 2.0    # ±3 mm

        # need ≥2 vox/edge → eff ≤ edge/2
        res_upper = edge / 2.0

        # Fit constraint from center to far cube face (mm)
        need_x_mm = off_xy + half_edge   # 7 mm
        need_y_mm = off_xy + half_edge   # 7 mm
        need_z_mm = off_z  + half_edge   # 4 mm

        # Available half-voxel spans (margin keeps a voxel or two off the border)
        margin_vox = 2.0
        hx = max(1.0, (Xn - 1) / 2.0 - margin_vox)
        hy = max(1.0, (Yn - 1) / 2.0 - margin_vox)
        hz = max(1.0, (Zn - 1) / 2.0 - margin_vox)

        # Fit: eff ≥ need_mm / halfspan_vox
        fit_lower = max(need_x_mm / hx, need_y_mm / hy, need_z_mm / hz)

        eff = float(voxel_mm_hint)
        # Clamp into [fit_lower, res_upper] if possible; otherwise prefer fit
        if fit_lower <= res_upper:
            eff = min(max(eff, fit_lower), res_upper)
        else:
            eff = max(eff, fit_lower)

        # Final safety to avoid vanishing cubes
        vox_per_edge = edge / eff
        if vox_per_edge < 2.0:
            eff = edge / 2.0
            vox_per_edge = 2.0

        # ------------------ concentrations (mmol/L) ------------------
        if absolute_concs_mmol is not None and len(absolute_concs_mmol) == 8:
            concs = [float(v) for v in absolute_concs_mmol]
        else:
            C0 = float(self._Concentration) if base_conc_mmol is None else float(base_conc_mmol)
            concs = [C0 / (float(dilution) ** i) for i in range(8)]  # 0..7

        # ------------------ centers in voxel coordinates ------------------
        cx = (Xn - 1) / 2.0
        cy = (Yn - 1) / 2.0
        cz = (Zn - 1) / 2.0

        offx_vox = off_xy / eff
        offy_vox = off_xy / eff
        offz_vox = off_z  / eff
        h_vox    = half_edge / eff

        # Order per spec (clockwise on top layer starting at +x,+y):
        centers_vox = [
            (cx + offx_vox, cy + offy_vox, cz + offz_vox),  # 1
            (cx + offx_vox, cy - offy_vox, cz + offz_vox),  # 2
            (cx - offx_vox, cy - offy_vox, cz + offz_vox),  # 3
            (cx - offx_vox, cy + offy_vox, cz + offz_vox),  # 4
            (cx + offx_vox, cy + offy_vox, cz - offz_vox),  # 5
            (cx + offx_vox, cy - offy_vox, cz - offz_vox),  # 6
            (cx - offx_vox, cy - offy_vox, cz - offz_vox),  # 7
            (cx - offx_vox, cy + offy_vox, cz - offz_vox),  # 8
        ]

        # ------------------ rasterize axis-aligned cubes ------------------
        C = np.zeros((Xn, Yn, Zn), dtype=float)

        def paint_cube(C, center, h):
            x0 = max(0, int(math.floor(center[0] - h)))
            x1 = min(C.shape[0]-1, int(math.ceil (center[0] + h)))
            y0 = max(0, int(math.floor(center[1] - h)))
            y1 = min(C.shape[1]-1, int(math.ceil (center[1] + h)))
            z0 = max(0, int(math.floor(center[2] - h)))
            z1 = min(C.shape[2]-1, int(math.ceil (center[2] + h)))
            if x0 > x1 or y0 > y1 or z0 > z1:
                return None
            ii = np.arange(x0, x1+1, dtype=float)
            jj = np.arange(y0, y1+1, dtype=float)
            kk = np.arange(z0, z1+1, dtype=float)
            I, J, K = np.meshgrid(ii, jj, kk, indexing='ij')
            mask = (np.abs(I - center[0]) <= h) & (np.abs(J - center[1]) <= h) & (np.abs(K - center[2]) <= h)
            return (slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)), mask

        filled_vox = 0
        for k, ctr in enumerate(centers_vox):
            painted = paint_cube(C, ctr, h_vox)
            if painted is None:
                continue
            slc, mask = painted
            val = concs[k] * 1e-12  # mmol/L → project scaling
            sub = C[slc]
            sub[mask] = np.maximum(sub[mask], val)
            C[slc] = sub
            filled_vox += int(np.count_nonzero(mask))

        # Safety: if nothing painted (very tiny grids), drop a central voxel with the highest conc
        if filled_vox == 0:
            C[int(cx), int(cy), int(cz)] = concs[0] * 1e-12

        if not return_meta:
            return C

        meta = {
            "effective_voxel_mm": float(eff),
            "cube_edge_mm": float(edge),
            "center_spacing_xy_mm": float(center_spacing_xy_mm),
            "center_spacing_z_mm": float(center_spacing_z_mm),
            "centers_vox": [tuple(map(float, c)) for c in centers_vox],
            "concentrations_mmol_per_L": [float(v) for v in concs],
            "voxels_filled": int(filled_vox),
            "volume_uL_each": edge**3,  # mm^3 == µL
        }
        return C, meta

    def getFourCircles(self,
                       Xn, Yn, Zn,
                       radii_mm=(4, 3, 2, 1),
                       voxel_size_mm=0.1,
                       outer_margin_mm=1.0,
                       z_thickness_mm=None,
                       cluster_gap_mm=0.5,
                       push_frac=0.0,
                       concentration_mmol_L=None):
        """
        Four circular objects (z-cylinders) clustered around the FOV center.

        New:
          concentration_mmol_L : float or None
              If provided (mmol/L), overrides self._Concentration ONLY for this phantom.
              Internally converted to mol/L via *1e-3.

        Other controls:
          cluster_gap_mm : minimum desired gap between neighboring circles (mm).
          push_frac      : 0..1; fraction of remaining room used to push circles outward.
        """


        # -- concentration override (mol/L) --
        if concentration_mmol_L is None:
            conc_mol_L = float(getattr(self, "_Concentration", 0.0))
        else:
            conc_mol_L = float(concentration_mmol_L) * 1e-3  # mmol/L -> mol/L

        # -- units mm -> vox --
        mm_per_vox = max(float(voxel_size_mm), 1e-9)
        radii_vox = np.array([max(1, int(round(float(r) / mm_per_vox))) for r in radii_mm], dtype=int)
        gap_vox = max(0, int(round(float(cluster_gap_mm) / mm_per_vox)))
        margin_vox = max(1, int(round(float(outer_margin_mm) / mm_per_vox)))
        push_frac = float(np.clip(push_frac, 0.0, 1.0))

        # -- z slab --
        if z_thickness_mm is None:
            kz0 = max(1, min(10, Zn // 10))
            kz1 = max(kz0 + 1, Zn - max(10, Zn // 10))
        else:
            half_thick_vox = max(1, int(round(0.5 * float(z_thickness_mm) / mm_per_vox)))
            cz = Zn // 2
            kz0 = max(0, cz - half_thick_vox)
            kz1 = min(Zn, cz + half_thick_vox)
            if kz1 <= kz0:
                kz0, kz1 = max(1, min(10, Zn // 10)), max(kz0 + 1, Zn - max(10, Zn // 10))

        # -- center --
        cx0, cy0 = int(Xn // 2), int(Yn // 2)

        # -- choose corner assignment minimizing needed offsets --
        best = None
        for perm in permutations(radii_vox.tolist(), 4):
            TLp, TRp, BLp, BRp = perm
            dx_req = int(np.ceil((max(TLp + TRp, BLp + BRp) + gap_vox) / 2.0))
            dy_req = int(np.ceil((max(TLp + BLp, TRp + BRp) + gap_vox) / 2.0))
            cost = max(dx_req, dy_req)
            if (best is None) or (cost < best[0]):
                best = (cost, (int(TLp), int(TRp), int(BLp), int(BRp)), int(dx_req), int(dy_req))

        _, (TL, TR, BL, BR), dx_req, dy_req = best  # ints

        # -- edge capacity --
        r_left_max = max(TL, BL)
        r_right_max = max(TR, BR)
        r_top_max = max(TL, TR)
        r_bottom_max = max(BL, BR)

        dx_cap_left = cx0 - (margin_vox + r_left_max)
        dx_cap_right = (Xn - 1 - (margin_vox + r_right_max)) - cx0
        dy_cap_top = cy0 - (margin_vox + r_top_max)
        dy_cap_bot = (Yn - 1 - (margin_vox + r_bottom_max)) - cy0

        dx_max = int(max(0, min(dx_cap_left, dx_cap_right)))
        dy_max = int(max(0, min(dy_cap_top, dy_cap_bot)))

        # -- start at minimal non-overlap; then push outward by push_frac --
        extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
        extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
        dx = min(dx_max, dx_req + extra_dx)
        dy = min(dy_max, dy_req + extra_dy)

        # helpers
        def place(dx_i, dy_i):
            cxL, cxR = cx0 - dx_i, cx0 + dx_i
            cyT, cyB = cy0 - dy_i, cy0 + dy_i
            return [(cxL, cyT, TL), (cxR, cyT, TR), (cxL, cyB, BL), (cxR, cyB, BR)]

        def edges_ok(centers, margin):
            for (cx, cy, rv) in centers:
                if cx - rv < margin or cx + rv > Xn - 1 - margin:
                    return False
                if cy - rv < margin or cy + rv > Yn - 1 - margin:
                    return False
            return True

        def nonoverlap_ok(centers, gap):
            for i in range(4):
                x1, y1, r1 = centers[i]
                for j in range(i + 1, 4):
                    x2, y2, r2 = centers[j]
                    if (x1 - x2) ** 2 + (y1 - y2) ** 2 < (r1 + r2 + gap) ** 2:
                        return False
            return True

        def shrink_uniform_to_fit(centers, margin):
            alpha = 1.0
            for (cx, cy, rv) in centers:
                allow = min(cx - margin, Xn - 1 - margin - cx, cy - margin, Yn - 1 - margin - cy)
                if allow < rv:
                    if allow <= 0:
                        return 0.0
                    alpha = min(alpha, float(allow) / float(rv))
            return alpha

        # finalize placement
        for _ in range(8):
            centers = place(dx, dy)
            if not edges_ok(centers, margin_vox):
                alpha = shrink_uniform_to_fit(centers, margin_vox)
                if alpha <= 0.0:
                    TL = TR = BL = BR = 1
                else:
                    TL = max(1, int(np.floor(alpha * TL)))
                    TR = max(1, int(np.floor(alpha * TR)))
                    BL = max(1, int(np.floor(alpha * BL)))
                    BR = max(1, int(np.floor(alpha * BR)))
                dx_req = int(np.ceil((max(TL + TR, BL + BR) + gap_vox) / 2.0))
                dy_req = int(np.ceil((max(TL + BL, TR + BR) + gap_vox) / 2.0))
                extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
                extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
                dx = min(dx_max, dx_req + extra_dx)
                dy = min(dy_max, dy_req + extra_dy)
                continue

            if not nonoverlap_ok(centers, gap_vox):
                grew = False
                if dx < dx_max:
                    dx += 1;
                    grew = True
                if dy < dy_max:
                    dy += 1;
                    grew = True
                if not grew:
                    TL = max(1, int(np.floor(0.95 * TL)))
                    TR = max(1, int(np.floor(0.95 * TR)))
                    BL = max(1, int(np.floor(0.95 * BL)))
                    BR = max(1, int(np.floor(0.95 * BR)))
                    dx_req = int(np.ceil((max(TL + TR, BL + BR) + gap_vox) / 2.0))
                    dy_req = int(np.ceil((max(TL + BL, TR + BR) + gap_vox) / 2.0))
                    extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
                    extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
                    dx = min(dx_max, dx_req + extra_dx)
                    dy = min(dy_max, dy_req + extra_dy)
                continue

            break

        # build volume
        C = np.zeros((Xn, Yn, Zn), dtype=float)
        x = np.arange(Xn)[:, None]
        y = np.arange(Yn)[None, :]

        for (cx, cy, rv) in place(dx, dy):
            mask2d = (x - cx) ** 2 + (y - cy) ** 2 <= (rv ** 2)
            for kz in range(kz0, kz1):
                sl = C[:, :, kz]
                sl[mask2d] = 1.0
                C[:, :, kz] = sl

        # scale with per-phantom concentration (mol/L) and existing global factor
        return C * conc_mol_L * 1e-12

    def getSingleCircle(self,
                        Xn, Yn, Zn,
                        radius_mm=4.0,
                        voxel_size_mm=0.1,
                        outer_margin_mm=1.0,
                        z_thickness_mm=None,
                        concentration_mmol_L=None):
        """
        Single circular object (a z-extruded cylinder) centered in the FOV.

        Parameters
        ----------
        Xn, Yn, Zn : int
            Grid size (voxels).
        radius_mm : float
            Desired circle radius in millimeters.
        voxel_size_mm : float
            Voxel pitch in mm/voxel. Set to match your Scanner step (e.g., 0.1 mm).
        outer_margin_mm : float
            Minimum clearance from any FOV edge (mm). The radius will auto-shrink
            if necessary to respect this margin.
        z_thickness_mm : float or None
            If None, fills an interior z slab leaving ~10 voxels top/bottom margin.
            If a float (mm), extrudes a centered slab of that physical thickness.
        concentration_mmol_L : float or None
            Per-phantom concentration override (mmol/L). If None, uses self._Concentration
            (which is already in mol/L internally). When provided, it's converted to mol/L.

        Returns
        -------
        C : ndarray (Xn, Yn, Zn)
            Phantom volume where the circle voxels are scaled by concentration.
        """
        import numpy as np

        # --- concentration selection (mol/L) ---
        if concentration_mmol_L is None:
            conc_mol_L = float(getattr(self, "_Concentration", 0.0))  # already mol/L
        else:
            conc_mol_L = float(concentration_mmol_L) * 1e-3  # mmol/L -> mol/L

        # --- units: mm -> voxels ---
        mm_per_vox = max(float(voxel_size_mm), 1e-9)
        r_vox = int(round(float(radius_mm) / mm_per_vox))
        margin_vox = max(1, int(round(float(outer_margin_mm) / mm_per_vox)))

        # --- center of FOV ---
        cx0, cy0 = int(Xn // 2), int(Yn // 2)

        # --- max allowed radius to respect margins (stay fully inside FOV) ---
        r_allow = min(
            cx0 - margin_vox,
            (Xn - 1 - margin_vox) - cx0,
            cy0 - margin_vox,
            (Yn - 1 - margin_vox) - cy0
        )
        r_allow = int(max(1, r_allow))
        if r_vox > r_allow:
            # shrink to fit, keep user informed
            print(f"[SingleCircle] Requested radius {r_vox} vox > allowed {r_allow} vox; shrinking to fit.")
            r_vox = r_allow

        # --- z slab selection ---
        if z_thickness_mm is None:
            kz0 = max(1, min(10, Zn // 10))
            kz1 = max(kz0 + 1, Zn - max(10, Zn // 10))
        else:
            half_thick_vox = max(1, int(round(0.5 * float(z_thickness_mm) / mm_per_vox)))
            cz = int(Zn // 2)
            kz0 = max(0, cz - half_thick_vox)
            kz1 = min(Zn, cz + half_thick_vox)
            if kz1 <= kz0:
                kz0 = max(1, min(10, Zn // 10))
                kz1 = max(kz0 + 1, Zn - max(10, Zn // 10))

        # --- build volume ---
        C = np.zeros((Xn, Yn, Zn), dtype=float)
        x = np.arange(Xn)[:, None]
        y = np.arange(Yn)[None, :]

        mask2d = (x - cx0) ** 2 + (y - cy0) ** 2 <= (r_vox ** 2)
        for kz in range(kz0, kz1):
            sl = C[:, :, kz]
            sl[mask2d] = 1.0
            C[:, :, kz] = sl

        # scale like other phantoms: concentration (mol/L) * small global factor
        return C * conc_mol_L * 1e-12

    def getFourCubes(self,
                     Xn, Yn, Zn,
                     side_mm=(4, 3, 2, 1),
                     voxel_size_mm=1.0,
                     outer_margin_mm=1.0,
                     z_thickness_mm=None,
                     cluster_gap_mm=0.5,
                     push_frac=0.0,
                     concentration_mmol_L=None):
        """
        Four cuboids in a 2×2 layout. Uses voxel-accurate widths so that
        (4,3,2,1) mm remain four different sizes even at 1 mm/voxel.
        """
        import numpy as np
        from itertools import permutations

        # --- concentration override (mol/L) ---
        if concentration_mmol_L is None:
            conc_mol_L = float(getattr(self, "_Concentration", 0.0))
        else:
            conc_mol_L = float(concentration_mmol_L) * 1e-3  # mmol/L -> mol/L

        # --- units ---
        mm_per_vox = max(float(voxel_size_mm), 1e-12)
        margin_vox = max(1, int(np.ceil(float(outer_margin_mm) / mm_per_vox)))
        gap_vox = max(0, int(np.ceil(float(cluster_gap_mm) / mm_per_vox)))
        push_frac = float(np.clip(push_frac, 0.0, 1.0))

        # --- voxel-accurate XY widths (in voxels) ---
        side_mm = tuple(map(float, side_mm))
        W_all = [max(1, int(np.ceil(s / mm_per_vox))) for s in side_mm]  # e.g., 4mm@1mm→4, 3mm→3, etc.

        # Common Z thickness (in voxels), either user-given or auto (use smallest XY width)
        if z_thickness_mm is None:
            Wz_req = min(W_all)
        else:
            Wz_req = max(1, int(np.ceil(float(z_thickness_mm) / mm_per_vox)))

        cz = int(Zn // 2)
        # available half-range in Z considering margins
        z_allow_half = int(max(1, min(cz - margin_vox, (Zn - 1 - margin_vox) - cz)))
        Wz = min(Wz_req, 2 * z_allow_half + 1)  # ensure it fits; keep odd/even as needed

        # helper: split a width into left/right half-extents (handles even widths)
        def split_halves(W):
            left = (W - 1) // 2
            right = W - 1 - left
            return int(left), int(right)

        # --- choose corner assignment (TL,TR,BL,BR) minimizing required offsets ---
        best = None
        for perm in permutations(W_all, 4):
            W_TL, W_TR, W_BL, W_BR = perm
            # center-to-center min sep along x/y to avoid overlap + enforce gap
            dx_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_TR,
                                           W_BL + gap_vox + W_BR)))
            dy_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_BL,
                                           W_TR + gap_vox + W_BR)))
            cost = max(dx_req, dy_req)
            if (best is None) or (cost < best[0]):
                best = (cost, (W_TL, W_TR, W_BL, W_BR), dx_req, dy_req)

        _, (W_TL, W_TR, W_BL, W_BR), dx_req, dy_req = best

        # --- margin capacities from center ---
        cx0, cy0 = int(Xn // 2), int(Yn // 2)

        # maximum half-extent toward each wall (use ceiling to be safe with even widths)
        rL = int(np.ceil(max((W_TL - 1) / 2, (W_BL - 1) / 2)))
        rR = int(np.ceil(max((W_TR - 1) / 2, (W_BR - 1) / 2)))
        rT = int(np.ceil(max((W_TL - 1) / 2, (W_TR - 1) / 2)))
        rB = int(np.ceil(max((W_BL - 1) / 2, (W_BR - 1) / 2)))

        dx_cap_left = cx0 - (margin_vox + rL)
        dx_cap_right = (Xn - 1 - (margin_vox + rR)) - cx0
        dy_cap_top = cy0 - (margin_vox + rT)
        dy_cap_bot = (Yn - 1 - (margin_vox + rB)) - cy0

        dx_max = int(max(0, min(dx_cap_left, dx_cap_right)))
        dy_max = int(max(0, min(dy_cap_top, dy_cap_bot)))

        # start from minimal non-overlap, then push outward
        extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
        extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
        dx = min(dx_max, dx_req + extra_dx)
        dy = min(dy_max, dy_req + extra_dy)

        # build rectangles given a (dx,dy)
        def rects(dx_i, dy_i):
            # centers
            cxL, cxR = cx0 - dx_i, cx0 + dx_i
            cyT, cyB = cy0 - dy_i, cy0 + dy_i
            # split each width to left/right, top/bottom
            TLxL, TLxR = split_halves(W_TL);
            TLyT, TLyB = split_halves(W_TL)
            TRxL, TRxR = split_halves(W_TR);
            TRyT, TRyB = split_halves(W_TR)
            BLxL, BLxR = split_halves(W_BL);
            BLyT, BLyB = split_halves(W_BL)
            BRxL, BRxR = split_halves(W_BR);
            BRyT, BRyB = split_halves(W_BR)
            # rectangles [x0,x1] & [y0,y1]
            R_TL = (cxL - TLxL, cxL + TLxR, cyT - TLyT, cyT + TLyB)
            R_TR = (cxR - TRxL, cxR + TRxR, cyT - TRyT, cyT + TRyB)
            R_BL = (cxL - BLxL, cxL + BLxR, cyB - BLyT, cyB + BLyB)
            R_BR = (cxR - BRxL, cxR + BRxR, cyB - BRyT, cyB + BRyB)
            return [R_TL, R_TR, R_BL, R_BR]

        def edges_ok(rect_list, margin):
            for (x0, x1, y0, y1) in rect_list:
                if x0 < margin or x1 > Xn - 1 - margin: return False
                if y0 < margin or y1 > Yn - 1 - margin: return False
            return True

        def nonoverlap_ok(rect_list, gap):
            # Require at least 'gap' voxels empty between rectangles along x OR y
            for i in range(4):
                x0i, x1i, y0i, y1i = rect_list[i]
                for j in range(i + 1, 4):
                    x0j, x1j, y0j, y1j = rect_list[j]
                    # Separation conditions (strictly more than 'gap-1' voxels apart)
                    sep_x = (x0i - x1j - 1) >= gap or (x0j - x1i - 1) >= gap
                    sep_y = (y0i - y1j - 1) >= gap or (y0j - y1i - 1) >= gap
                    if not (sep_x or sep_y):
                        return False
            return True

        # finalize placement
        for _ in range(10):
            R = rects(dx, dy)
            if not edges_ok(R, margin_vox):
                # shrink largest widths a bit if we absolutely can't fit (rare)
                shrink = [W_TL, W_TR, W_BL, W_BR]
                idx = int(np.argmax(shrink))
                if idx == 0: W_TL = max(1, W_TL - 1)
                if idx == 1: W_TR = max(1, W_TR - 1)
                if idx == 2: W_BL = max(1, W_BL - 1)
                if idx == 3: W_BR = max(1, W_BR - 1)
                # recompute requirements
                dx_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_TR,
                                               W_BL + gap_vox + W_BR)))
                dy_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_BL,
                                               W_TR + gap_vox + W_BR)))
                extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
                extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
                dx = min(dx_max, dx_req + extra_dx)
                dy = min(dy_max, dy_req + extra_dy)
                continue

            if not nonoverlap_ok(R, gap_vox):
                grew = False
                if dx < dx_max: dx += 1; grew = True
                if dy < dy_max: dy += 1; grew = True
                if grew: continue
                # last resort: reduce all widths by one voxel
                W_TL = max(1, W_TL - 1)
                W_TR = max(1, W_TR - 1)
                W_BL = max(1, W_BL - 1)
                W_BR = max(1, W_BR - 1)
                dx_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_TR,
                                               W_BL + gap_vox + W_BR)))
                dy_req = int(np.ceil(0.5 * max(W_TL + gap_vox + W_BL,
                                               W_TR + gap_vox + W_BR)))
                extra_dx = int(np.floor(push_frac * max(0, dx_max - dx_req)))
                extra_dy = int(np.floor(push_frac * max(0, dy_max - dy_req)))
                dx = min(dx_max, dx_req + extra_dx)
                dy = min(dy_max, dy_req + extra_dy)
                continue
            break

        # --- build volume with exact widths (half-open indexing) ---
        C = np.zeros((Xn, Yn, Zn), dtype=float)

        # z extents (half-open)
        z_left, z_right = split_halves(Wz)
        z0 = max(0, cz - z_left)
        z1 = min(Zn, cz + z_right + 1)  # half-open

        # final rectangles and paint
        R = rects(dx, dy)
        for (x0, x1, y0, y1) in R:
            xx0 = max(0, x0)
            xx1 = min(Xn, x1 + 1)  # half-open
            yy0 = max(0, y0)
            yy1 = min(Yn, y1 + 1)  # half-open
            C[xx0:xx1, yy0:yy1, z0:z1] = 1.0

        return C * conc_mol_L * 1e-12


'''
    def getMPIShape(self, Xn: int, Yn: int, Zn: int, text: str = "MPI"):
        """
        Create a block-letter phantom spelling `text` (default: 'MPI'),
        constrained to the SAME (x,y,z) extents used by getEShape:
          x: [30/201 .. 170/201] * Xn
          y: [10 .. Yn-10]
          z: [50/201 .. 150/201] * Zn

        Updates vs prior version:
          - Letters are 25% shorter in y (use 75% of box height), centered vertically.
          - Inter-letter spacing increased.
          - z thickness unchanged.

        Returns a (Xn, Yn, Zn) float array scaled by self._Concentration * 1e-12.
        Axes: [x, y, z]
        """

        # ---- Match E-shape bounding box exactly ----
        x0 = int(Xn * (30 / 201))
        x1 = int(Xn * (170 / 201))
        y0 = 10
        y1 = Yn - 10
        z0 = int(Zn * (50 / 201))
        z1 = int(Zn * (150 / 201))

        W = max(1, x1 - x0)  # drawable width inside the box
        H = max(1, y1 - y0)  # drawable height inside the box

        C = np.zeros((Xn, Yn, Zn), dtype=float)
        if W < 5 or H < 5 or z1 <= z0:
            print("Word phantom selected but grid too small for bounding box; returning zeros.")
            return C

        text = (text or "MPI").upper()
        n_letters = len(text)

        # ---- Layout inside the (x0:x1, y0:y1) slab ----
        # Further increase spacing; use 75% height for letters.
        spacing_frac = 0.23  # was 0.12 → more separation to reduce cross-talk
        stroke_frac = 0.1  # keep thicker strokes in x,y (relative to letter size)
        height_frac = 0.7  # letters are 25% shorter in y
        min_spacing = 2
        min_stroke = 2

        spacing = max(min_spacing, int(round(spacing_frac * W)))
        letter_w = max(5, (W - spacing * (n_letters - 1)) // max(1, n_letters))
        letter_h = max(5, int(round(height_frac * H)))  # reduced height

        # stroke thickness (clamped so letters don't collapse)
        stroke = max(min_stroke, int(round(stroke_frac * min(letter_w, letter_h))))
        stroke = min(stroke, letter_w // 3, letter_h // 3)

        # recompute total width and center horizontally
        total_w = letter_w * n_letters + spacing * (n_letters - 1)
        if total_w > W:
            overflow = total_w - W
            letter_w = max(5, letter_w - (overflow // max(1, n_letters)))
            total_w = letter_w * n_letters + spacing * (n_letters - 1)
        x_left_pad = max(0, (W - total_w) // 2)  # center the word in the box

        # vertical centering for the shorter letters
        y_pad = max(0, (H - letter_h) // 2)

        # ---- 2D drawing helpers (on the local plane of size W×H) ----
        def _draw_rect(mask, x0_, x1_, y0_, y1_):
            x0c, x1c = max(0, x0_), min(mask.shape[0], x1_)
            y0c, y1c = max(0, y0_), min(mask.shape[1], y1_)
            if x1c > x0c and y1c > y0c:
                mask[x0c:x1c, y0c:y1c] = True

        def _draw_line_thick(mask, x0_, y0_, x1_, y1_, rad):
            # distance-based thick line
            W_, H_ = mask.shape
            xs = np.arange(W_)[:, None]
            ys = np.arange(H_)[None, :]
            dx = x1_ - x0_
            dy = y1_ - y0_
            if dx == 0 and dy == 0:
                mask[(xs - x0_) ** 2 + (ys - y0_) ** 2 <= rad * rad] = True
                return
            denom = float(dx * dx + dy * dy)
            t = ((xs - x0_) * dx + (ys - y0_) * dy) / denom
            t = np.clip(t, 0.0, 1.0)
            projx = x0_ + t * dx
            projy = y0_ + t * dy
            dist2 = (xs - projx) ** 2 + (ys - projy) ** 2
            mask[dist2 <= rad * rad] = True

        def _letter_M(w, h, t):
            m = np.zeros((w, h), dtype=bool)
            _draw_rect(m, 0, t, 0, h)  # left vertical
            _draw_rect(m, w - t, w, 0, h)  # right vertical
            cx = w // 2  # diagonals to top center
            _draw_line_thick(m, 0, h - 1, cx, 0, max(1, t))
            _draw_line_thick(m, w - 1, h - 1, cx, 0, max(1, t))
            return m

        def _letter_P(w, h, t):
            m = np.zeros((w, h), dtype=bool)
            _draw_rect(m, 0, t, 0, h)  # backbone
            bowl_bot = max(0, int(0.45 * h))  # top ~55%
            _draw_rect(m, 0, w, bowl_bot, h)  # outer bowl
            _draw_rect(m, t, w - t, bowl_bot + t, h - t)  # hollow
            return m

        def _letter_I(w, h, t):
            m = np.zeros((w, h), dtype=bool)
            _draw_rect(m, 0, w, 0, t)  # top serif
            _draw_rect(m, 0, w, h - t, h)  # bottom serif
            cx0 = max(0, w // 2 - (t // 2 + t // 4))
            cx1 = min(w, cx0 + t + (t // 2))
            _draw_rect(m, cx0, cx1, 0, h)  # central vertical
            return m

        def _make_letter(ch, w, h, t):
            ch = ch.upper()
            if ch == 'M':
                return _letter_M(w, h, t)
            elif ch == 'P':
                return _letter_P(w, h, t)
            elif ch == 'I':
                return _letter_I(w, h, t)
            return np.zeros((w, h), dtype=bool)  # unknowns -> blank

        # ---- Compose local plane (W×H) and paste into global [x0:x1, y0:y1] ----
        plane_local = np.zeros((W, H), dtype=bool)
        x_cursor = x_left_pad
        for idx, ch in enumerate(text):
            letter = _make_letter(ch, w=letter_w, h=letter_h, t=stroke)  # (letter_w × letter_h)
            xs0 = x_cursor
            xs1 = min(W, xs0 + letter_w)
            ys0 = y_pad
            ys1 = min(H, ys0 + letter_h)
            # paste with bounds checks
            w_slice = xs1 - xs0
            h_slice = ys1 - ys0
            if w_slice > 0 and h_slice > 0:
                plane_local[xs0:xs1, ys0:ys1] |= letter[:w_slice, :h_slice]
            x_cursor += letter_w + (spacing if idx < n_letters - 1 else 0)
            if x_cursor >= W:
                break

        # Paste into the global volume within the E-shape box (z thickness unchanged)
        C[x0:x1, y0:y1, z0:z1] = plane_local[:, :, None].astype(float)

        print(f"Word phantom selected (shorter-y & wider spacing): '{text}'")
        return C * self._Concentration * 1e-12

'''




