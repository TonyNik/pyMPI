# coding=UTF-8
import numpy as np

from MPIRF.ReconClass.BaseClass.ReconBase import *
from MPIRF.Config.ConstantList import *
from Model.Interference import *
from scipy.interpolate import griddata
import csv

_EPS = 1e-12

def _langevin(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    m = np.abs(x) < 1e-6
    if np.any(~m):
        xx = x[~m]
        out[~m] = 1.0/np.tanh(xx) - 1.0/xx
    if np.any(m):
        xx = x[m]
        out[m] = xx/3.0 - (xx**3)/45.0
    return out

def _langevin_prime(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    m = np.abs(x) < 1e-8
    if np.any(~m):
        xx = x[~m]
        out[~m] = 1.0/(xx*xx) - 1.0/(np.sinh(xx)**2)
    if np.any(m):
        out[m] = 1.0/3.0
    return out

def _ramp_filter_1d_psfaware(proj_row, ds, beta, lam_rel=0.02, hann_rel=1.0):
    """
    Ram–Lak × Hann with gentle Tikhonov on the PSF deconvolution.
      lam_rel : relative Tikhonov (0.01–0.05 is usually safe)
      hann_rel: 1.0 = current Hann, <1.0 = milder, >1.0 = stronger
    """
    Ns = int(proj_row.shape[0])
    if Ns <= 1:
        return proj_row

    F     = np.fft.rfft(proj_row, n=Ns)
    freqs = np.fft.rfftfreq(Ns, d=ds)

    # Ramp
    H_ramp = np.abs(freqs)
    H_ramp /= (np.max(H_ramp) + 1e-12)

    # Hann (mild apodization)
    H_hann = 0.5 * (1.0 + np.cos(np.pi * freqs / (np.max(freqs) + 1e-12)))
    H_hann = H_hann ** hann_rel

    # PSF deconvolution term: 1 / (fspace + λ)
    s = (np.arange(Ns) - Ns//2) * ds
    z = beta * np.abs(s)
    Lp = _langevin_prime(z)
    L  = _langevin(z)
    zsafe = np.where(z < 1e-8, 1e-8, z)
    fspace = Lp + (L / zsafe)
    lam = lam_rel * float(np.max(fspace))
    H_psf = 1.0 / (fspace + lam)

    H = H_ramp * H_hann
    Ff = F * H
    out = np.fft.irfft(Ff, n=Ns)

    # optional DC re-baseline
    out -= np.mean(out)
    return out



class XReconClass(ReconBaseClass):

    def __init__(self, Message):
        super().__init__()

        # Keep the message around (handy for debugging / parity with originals)
        self._Message = Message

        # ---- Interference / flags (robust defaults) ----
        ext = Message.get(EXTENDED, {})

        self.Rel=Message[EXTENDED]["Relaxation"]
        self.RelT = Message[EXTENDED]["RelaxationTime"]
        #self.Rel = bool(ext.get("Relaxation", False))
        #self.RelT = float(ext.get("RelaxationTime", 0.0))

        self.Noise=Message[EXTENDED]["NoiseFlag"]
        self.Background=Message[EXTENDED]["BackgroundFlag"]
        #self.Noise = bool(ext.get("NoiseFlag", False))
        #self.Background = bool(ext.get("BackgroundFlag", False))

        # Shape-safe defaults for noise/background (Fn x 3 like the raw voltage)
        U_meas = np.asarray(Message[MEASUREMENT][MEASIGNAL])
        zeros = np.zeros_like(U_meas)

        if self.Noise:
            self.NoiseValue=Message[EXTENDED]["NoiseValue"]
        if self.Background:
            self.BackgroundValue=Message[EXTENDED]["BackgroundValue"]
        #self.NoiseValue = np.asarray(ext.get("NoiseValue", zeros))
        #self.BackgroundValue = np.asarray(ext.get("BackgroundValue", zeros))

        # ---- Sampling / timing ----
        self.Fn = int(Message[SAMPLE][SAMNUMBER])
        self.Fs = float(Message[SAMPLE][FREQUENCY])
        self.Rt = float(Message[DRIVEFIELD][REPEATTIME])

        # ---- Mode detection (FFP vs FFL) from declared metadata only ----
        topo = str(Message[SAMPLE].get(TOPOLOGY, "")).upper()
        mtype = str(Message[MEASUREMENT].get(TYPE, "")).upper()
        self._is_ffl = ("FFL" in topo) or ("FFL" in mtype)

        # ---- Ensure container exists even if base class changes ----
        if not hasattr(self, "_ImagSignal"):
            self._ImagSignal = []

        # Perform reconstruction immediately (matches current call sites)
        self._ImageRecon(Message)

    def _ImageRecon(self, Message):
        topo = str(Message[SAMPLE].get(TOPOLOGY, "")).upper()


        if ("FFL" in topo) or ("FFL" in str(Message[MEASUREMENT].get(TYPE, "")).upper()):
            vol = self.__FFL_recon(Message)
            self._ImagSignal.append(vol)
            self._ImagSignal.append([vol])
            return True

        else:
            self._ImagSignal.append(self.__XSpace(Message[MEASUREMENT][MEASIGNAL], Message[MEASUREMENT][AUXSIGNAL]))
            self._ImagSignal.append(self._ImageReshape(Message[EXTENDED][RFFP], Message[EXTENDED][STEP]))

            return True

    # ---------- FFL recon ----------
    def __FFL_recon(self, Message):
        step = float(Message[EXTENDED]["STEP"])
        nx, ny, nz = [int(x) for x in Message[MEASUREMENT][MEANUMBER]]


        U_meas = np.asarray(Message[MEASUREMENT][MEASIGNAL])  # (Fn,3)
        vperp = np.asarray(Message[EXTENDED]["FFL_VPERP"])  # (Fn,)
        validm = np.asarray(Message[EXTENDED].get("FFL_VALID", np.ones(U_meas.shape[0], bool)))

        # scalarize: ||U|| (magnitude). With analytic dDH, ||U|| = const * (G * v_perp) * spat
        meas = np.linalg.norm(U_meas, axis=1)

        # --- optional relax/noise/bg (preserve your flags) ---
        # --- Background ---
        if self.Background and self.BackgroundValue is not None:
            bg = np.asarray(self.BackgroundValue, dtype=float)
            if bg.ndim == 2 and bg.shape[1] == 3:  # (Fn,3) -> reduce per-sample magnitude
                bg = np.linalg.norm(bg, axis=1)  # (Fn,)
            elif bg.ndim == 1 and bg.shape[0] == meas.shape[0]:
                pass  # (Fn,) already OK
            elif bg.ndim == 0:
                bg = float(bg)  # scalar
            else:
                raise ValueError(f"Unsupported BackgroundValue shape {bg.shape}; expected scalar, (Fn,), or (Fn,3).")
            meas = meas + bg

        # --- Noise ---
        if self.Noise and self.NoiseValue is not None:
            noise_add = np.asarray(self.NoiseValue, dtype=float)  # renamed from 'nz'
            if noise_add.ndim == 2 and noise_add.shape[1] == 3:  # (Fn,3) -> (Fn,)
                noise_add = np.linalg.norm(noise_add, axis=1)
            elif noise_add.ndim == 1 and noise_add.shape[0] == meas.shape[0]:
                pass
            elif noise_add.ndim == 0:
                noise_add = float(noise_add)
            else:
                raise ValueError(f"Unsupported NoiseValue shape {noise_add.shape}; expected scalar, (Fn,), or (Fn,3).")
            meas = meas + noise_add

        if self.Rel:
            meas = InterferenceClass().RelaxationCPU(meas, self.RelT, self.Fn, self.Fs, self.Rt)

        meta3d = Message[EXTENDED].get("FFL_SINOGRAM_3D", None)
        assert meta3d is not None, "Missing FFL_SINOGRAM_3D metadata"

        th_cent = np.asarray(meta3d["theta"])
        s_cent = np.asarray(meta3d["s_cent"])
        kz_idx = np.asarray(meta3d["kz_idx"], dtype=np.int64)
        iz = np.asarray(meta3d["iz"], dtype=np.int64)
        ith = np.asarray(meta3d["ith"], dtype=np.int64)
        isb = np.asarray(meta3d["is"], dtype=np.int64)
        valid_count = int(meta3d.get("valid_count", meas.shape[0]))

        NTH, Ns, Nz_slices = th_cent.size, s_cent.size, kz_idx.size
        ds = (s_cent[1] - s_cent[0]) if Ns > 1 else step
        s_edges = np.linspace(s_cent[0] - 0.5 * ds, s_cent[-1] + 0.5 * ds, Ns + 1)

        # === correct normalization: divide by (v_perp * G_perp) ===
        Gperp = float(Message[EXTENDED].get("FFL_BOOK_G", 0.0))
        if Gperp <= 0:
            # fallback to legacy: estimate from gradients (kept for compatibility)
            Gx = float(Message[SELECTIONFIELD][XGRADIENT])
            Gy = float(Message[SELECTIONFIELD][YGRADIENT])
            # use RMS across angles
            Gperp = float(np.sqrt((Gx ** 2 + Gy ** 2) / 2.0))

        denom = (vperp * Gperp + _EPS)
        meas = meas / denom

        # === bin into a (θ, s | z) stack (ignore invalid or trailing samples) ===
        use = (np.arange(meas.shape[0]) < valid_count) & (iz >= 0) & validm
        vol = np.zeros((nx, ny, nz), dtype=float)

        xs = np.arange(nx) * step - (nx - 1) * step * 0.5
        ys = np.arange(ny) * step - (ny - 1) * step * 0.5
        X, Y = np.meshgrid(xs, ys, indexing='ij')

        # IMPORTANT: FBP uses the **projection normal** nθ.
        # Here θ parameterizes the **line direction** pθ = (cosθ, sinθ),
        # so the normal is qθ = (-sinθ, cosθ). Use that in backprojection.
        c_n = -np.sin(th_cent)
        s_n = np.cos(th_cent)

        # β for PSF (constant across θ for the book-true tensor):
        Tt = float(Message[MAGNETICPARTICL][TEMPERATURE])
        Mm = float(Message[MAGNETICPARTICL][SATURATIONMAG])
        Bb = (U0 * Mm) / (KB * Tt + 1e-12) if 'KB' in globals() else (U0 * Mm) / (
                    KB * Tt + 1e-12)  # adapt to your ConstantList

        for sidx in range(Nz_slices):
            sino = np.zeros((NTH, Ns), dtype=float)
            wts = np.zeros((NTH, Ns), dtype=float)

            msk = use & (iz == sidx)
            if not np.any(msk):
                continue

            np.add.at(sino, (ith[msk], isb[msk]), meas[msk])
            np.add.at(wts, (ith[msk], isb[msk]), 1.0)
            ok = wts > 0
            sino[ok] /= wts[ok]

            # PSF-aware filtering per angle
            for th in range(NTH):
                beta = Bb * Gperp
                sino[th, :] = _ramp_filter_1d_psfaware(sino[th, :], ds, beta)

            # Standard parallel-beam backprojection with normal nθ = qθ
            img2d = np.zeros((nx, ny), dtype=float)
            for th in range(NTH):
                s_plane = X * c_n[th] + Y * s_n[th]
                tpos = (s_plane - s_edges[0]) / (s_edges[1] - s_edges[0])
                i0 = np.floor(tpos).astype(int)
                a = tpos - i0
                i0 = np.clip(i0, 0, Ns - 1)
                i1 = np.clip(i0 + 1, 0, Ns - 1)
                proj = sino[th, :]
                val = (1.0 - a) * proj[i0] + a * proj[i1]
                img2d += val

            img2d /= max(NTH, 1)  # normalize by number of views
            kz = int(np.clip(kz_idx[sidx], 0, nz - 1))
            vol[:, :, kz] = img2d

        if np.max(vol) > 0:
            vol /= np.max(vol)
        return vol

    # ---------- FFP helper (unchanged) ----------
    def __XSpace(self,U, Vffp):

        temp = Vffp ** 2
        VffpLen = np.sqrt(temp[0] + temp[1] + temp[2])
        VffpDir = np.divide(Vffp, np.tile(VffpLen, (3, 1)))

        temp = np.transpose(U) * VffpDir
        SigTan = temp[0] + temp[1] + temp[2]

        if self.Rel:
            SigTan=InterferenceClass().RelaxationCPU(SigTan,self.RelT, self.Fn, self.Fs, self.Rt)

        if self.Noise:
            tempn=np.transpose(self.NoiseValue) * VffpDir
            noise=tempn[0] + tempn[1] + tempn[2]
            SigTan=SigTan+noise
        if self.Background:
            tempb = np.transpose(self.BackgroundValue) * VffpDir
            background = tempb[0] + tempb[1] + tempb[2]
            SigTan = SigTan + background

        #########################################################################################
        # temp = self._ffv2 ** 2
        # VffpLen = np.sqrt(temp[0] + temp[1] + temp[2])
        #########################################################################################

        ImgTan = SigTan / VffpLen
        #ImgTan = ImgTan / np.max(ImgTan)
        return ImgTan

    def _ImageReshape(self, Rffp, Step):
        pointx = np.arange(min(Rffp[0][:]), max(Rffp[0][:]) + Step, Step)
        pointy = np.arange(min(Rffp[1][:]), max(Rffp[1][:]) + Step, Step)
        pointz = np.arange(min(Rffp[2][:]), max(Rffp[2][:]) + Step, Step)
        xpos, ypos , zpos= np.meshgrid(pointy, pointx, pointz)
        ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='linear')

        temp=np.isnan(ImgTan[1:-1, 1:-1, 1:-1])
        if True in temp:
            ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='nearest')

        ImgTan = ImgTan[1:-1, 1:-1, 1:-1]
        ImgTan = ImgTan / np.max(ImgTan)
        return [ImgTan]
'''
    def _ImageReshape(self, Rffp, Step):
        pointx = np.arange(min(Rffp[0][:]), max(Rffp[0][:]) + Step, Step)
        pointy = np.arange(min(Rffp[1][:]), max(Rffp[1][:]) + Step, Step)
        pointz = np.arange(min(Rffp[2][:]), max(Rffp[2][:]) + Step, Step)
        xpos, ypos , zpos= np.meshgrid(pointy, pointx, pointz)
        ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='linear')

        temp=np.isnan(ImgTan[1:-1, 1:-1, 1:-1])
        if True in temp:
            ImgTan = griddata((Rffp[1], Rffp[0], Rffp[2]), self._ImagSignal[0], (xpos, ypos, zpos), method='nearest')

        ImgTan = ImgTan[1:-1, 1:-1, 1:-1]
        ImgTan = ImgTan / np.max(ImgTan)
        return [ImgTan]
'''