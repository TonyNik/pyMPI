
# Scanner.py
from pylab import *
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
import numpy as np

from Model.Phantom import *
from MPIRF.Config.UIConstantListEn import SIMUVOL
from MPIRF.Config.ConstantList import (
    GET_VALUE, EXTENDED, STEP, RFFP,
    MAGNETICPARTICL, TEMPERATURE, DIAMETER, SATURATIONMAG,
    SELECTIONFIELD, XGRADIENT, YGRADIENT, ZGRADIENT,
    DRIVEFIELD, XDIRECTIOND, YDIRECTIOND, ZDIRECTIOND, REPEATTIME, WAVEFORMD, SINE,
    SAMPLE, TOPOLOGY, FFP, FREQUENCY, SAMNUMBER, BEGINTIME, SENSITIVITY,
    MEASUREMENT, TYPE, BGFLAG, MEASIGNAL, AUXSIGNAL, MEANUMBER,
    U0, PI
)
from MPIRF.DataClass.BassClass.DataBase import *


# ============================== Helpers ==============================

def _langevin_prime(x, eps=1e-12):
    """Derivative of Langevin: L'(x) = 1/x^2 - 1/sinh^2(x); limit x→0 is 1/3."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    m = np.abs(x) < eps
    if np.any(~m):
        xx = x[~m]
        out[~m] = 1.0/(xx*xx) - 1.0/(np.sinh(xx)**2)
    if np.any(m):
        out[m] = 1.0/3.0
    return out


def _center_out_indices(n, k):
    """Return k indices centered in [0..n-1], expanding center-out."""
    if k <= 0:
        return np.array([], dtype=int)
    c = n // 2
    if k == 1:
        return np.array([c], dtype=int)
    order = [c]
    step = 1
    while len(order) < k:
        if c - step >= 0:
            order.append(c - step)
        if len(order) < k and c + step < n:
            order.append(c + step)
        step += 1
    return np.array(sorted(order), dtype=int)

def ThdFunc(DH,l,r,o,GSc,B,C):
    DHt=np.tile(DH, (l, r, o, 1))
    test=np.subtract(DHt,GSc)
    H=np.sqrt(test[:,:,:,2]**2+test[:,:,:,1]**2+test[:,:,:,0]**2)
    DLF = (1 / ((B * H) ** 2)) - (1 / ((np.sinh(B * H)) ** 2))

    s = C * DLF
    signal=np.sum(s[:,:,:])
    #print(f"ThdFunc call for {signal}")

    return signal

# ============================== Scanner ==============================

class ScannerClass(DataBaseClass):

    def __init__(self,
                 VirtualPhantom,
                 SelectGradietX=2.4,
                 SelectGradietY=4.8,
                 SelectGradietZ=2.4,
                 DriveFrequencyX=26.04e3,
                 DriveFrequencyY=24.51e3,
                 DriveFrequencyZ=21.54e3,
                 DriveAmplitudeX=12e-3,
                 DriveAmplitudeY=12e-3,
                 DriveAmplitudeZ=12e-3,
                 RepetitionTime=21.54e-3,
                 SampleFrequency=2.5e5,
                 trajectory_type="3D"):

        super().__init__()

        # -------------------- Phantom & constants --------------------
        self._VirtualPhantom = VirtualPhantom
        self._CoilSensitivity = 1.0

        # Selection gradients (A/m^2)
        self._Gx = SelectGradietX / U0
        self._Gy = SelectGradietY / U0
        self._Gz = SelectGradietZ / U0
        self._Gg = np.array([[self._Gx], [self._Gy], [self._Gz]])

        # Drive amplitudes (A/m) and frequencies (Hz)
        self._Ax = DriveAmplitudeX / U0;
        self._Fx = DriveFrequencyX
        self._Ay = DriveAmplitudeY / U0;
        self._Fy = DriveFrequencyY
        self._Az = DriveAmplitudeZ / U0;
        self._Fz = DriveFrequencyZ

        self._Rt = RepetitionTime
        self._Sf = SampleFrequency

        # Time axis
        self._T = np.arange((1.0 / self._Sf), self._Rt, (1.0 / self._Sf))
        self._Fn = self._T.shape[0]

        # Grid step and FOV extents
        self._Step = 1e-4

        def _safe_div(a, b):
            return a / (b if abs(b) > 1e-12 else 1e-12)

        #self._Maxx = abs(_safe_div(self._Ax, self._Gx))
        #self._Maxy = abs(_safe_div(self._Ay, self._Gy))
        #self._Maxz = abs(_safe_div(self._Az, self._Gz))

        self._Maxx = self._Ax / self._Gx
        self._Maxy = self._Ay / self._Gy
        self._Maxz = self._Az / self._Gz


        # Grid sizes
        self._Nx = len(np.arange(-self._Maxx, self._Maxx + self._Step, self._Step))
        self._Ny = len(np.arange(-self._Maxy, self._Maxy + self._Step, self._Step))
        self._Nz = len(np.arange(-self._Maxz, self._Maxz + self._Step, self._Step))

        # Phantom volume
        self._PhantomMatrix = self._VirtualPhantom.getShape(self._Nx, self._Ny, self._Nz)

        # Coordinate arrays used in triple loops (FFP and FFL)
        self._xs = np.arange(self._Nx) * self._Step - self._Maxx
        self._ys = np.arange(self._Ny) * self._Step - self._Maxy
        self._zs = np.arange(self._Nz) * self._Step - self._Maxz

        # --------------- Trajectory switch ---------------
        tr = str(trajectory_type).upper()
        self._is_ffl_true = False

        # -------- FFL implementation according to Gapyak https://doi.org/10.1137/23M1600529 --------
        if tr in {"FFL-TRUE-STACK-DRIVE", "FFL-TRUE-STACK", "FFL-STACK"}:
            print(
                "FFL-TRUE-STACK-DRIVE: GSc via triple loop (like FFP); rotation by gradient-drive; translation by uniform drive.")
            self._is_ffl_true = True
            self._build_ffl_true_stack_drive()
            print("Calculating Signal (FFL true fields)...")
            self._get_Voltage_CPU_FFL_true()
            print("Signal Calculation completed.")
            self._init_Message()
            return

        # -------- FFP branches--------
        if tr == "FFP-3D-L":
            print("3D Lissajous trajectory selected")
            self._DHx, self._DeriDHx = self.__DriveStrength(self._Ax, self._Fx, self._T)
            self._DHy, self._DeriDHy = self.__DriveStrength(self._Ay, self._Fy, self._T)
            self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fz, self._T)
        elif tr == "FFP-3D-RAD-L":
            print("3D Radial Lissajous trajectory selected")
            self._DHx, self._DeriDHx, self._DHy, self._DeriDHy = self.__DriveStrengthRadLissajous(
                self._Ax, self._Fx, self._Fy, self._T)
            self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fz, self._T)
        elif trajectory_type.upper() == "FFP-3D-SP":
            print("3D Spiral trajectory selected")
            self._DHx, self._DeriDHx, self._DHy, self._DeriDHy = self.__DriveStrengthSpiral(self._Ax, self._Fx, self._Fy, self._T)
            self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fx/(12^2), self._T)
        elif tr == "FFP-3D-CART":
            print("3D Cartesian trajectory selected")
            self._DHx, self._DeriDHx = self.__DriveStrength(self._Ax, self._Fx, self._T)
            self._DHy, self._DeriDHy = self.__DriveStrength(self._Ay, self._Fx / 12.0, self._T)
            self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fx / (12.0 ** 2), self._T)
        else:
            print("Unknown trajectory_type; defaulting to 3D Lissajous")
            self._DHx, self._DeriDHx = self.__DriveStrength(self._Ax, self._Fx, self._T)
            self._DHy, self._DeriDHy = self.__DriveStrength(self._Ay, self._Fy, self._T)
            self._DHz, self._DeriDHz = self.__DriveStrength(self._Az, self._Fz, self._T)

        # FFP: positions / velocities from drive vs gradients (for EXTENDED)
        self._DH = np.array([self._DHx, self._DHy, self._DHz])
        self._DeriDH = np.array([self._DeriDHx, self._DeriDHy, self._DeriDHz])
        self._Ffpr = np.divide(self._DH, np.tile(self._Gg, (1, np.shape(self._DH)[1])))
        self._Ffpx = self._Ffpr[0]
        self._Ffpy = self._Ffpr[1]
        self._Ffpz = self._Ffpr[2]
        self._Ffpv = np.divide(self._DeriDH, np.tile(self._Gg, (1, np.shape(self._DeriDH)[1])))

        print("Calculating Signal...")
        self._get_Voltage_CPU_FFP()
        print("Signal Calculation completed.")
        self._init_Message()
        return

    # ======================= Drive waveforms (used by FFP) =======================

    def __DriveStrength(self, A, F, t):
        DH  = A * np.cos(2.0 * PI * F * t + PI/2.0) * (-1.0)
        dDH = A * np.sin(2.0 * PI * F * t + PI/2.0) * 2.0 * PI * F
        return DH, dDH

    def __DriveStrengthRadLissajous(self, A, F1, F2, t):
        print("Radial Lissajous Trajectory used.")
        DHx  = A * np.sin(2 * PI * F1 * t) * np.sin(2.0 * PI * F2 * t)
        dDHx = A * (2 * PI * F1 * np.cos(2 * PI * F1 * t) * np.sin(2.0 * PI * F2 * t)
                    + 2 * PI * F2 * np.sin(2 * PI * F1 * t) * np.cos(2.0 * PI * F2 * t))
        DHy  = A * np.cos(2 * PI * F1 * t) * np.sin(2.0 * PI * F2 * t)
        dDHy = (-A * 2 * PI * F1 * np.sin(2 * PI * F1 * t) * np.sin(2.0 * PI * F2 * t)
                + A * 2 * PI * F2 * np.cos(2.0 * PI * F1 * t) * np.cos(2.0 * PI * F2 * t))
        return DHx, dDHx, DHy, dDHy

    def __DriveStrengthSpiral(self, DriveAmplitude, DriveFrequency1, DriveFrequency2, TSquence):
        print("3D-Spiral Trajectory used.")
        DHx = DriveAmplitude * np.sin(2 * PI * DriveFrequency2 * TSquence) * np.sin(2.0 * PI * DriveFrequency1 * TSquence)
        DeriDHx = DriveAmplitude * 2 * PI * DriveFrequency2 * np.cos(2 * PI * DriveFrequency2 * TSquence) * np.sin(2.0 * PI * DriveFrequency1 * TSquence) + DriveAmplitude * 2 * PI * DriveFrequency1 * np.sin(2 * PI * DriveFrequency2 * TSquence) * np.cos(2.0 * PI * DriveFrequency1 * TSquence)
        DHy = DriveAmplitude * np.cos(2 * PI * DriveFrequency2 * TSquence) * np.sin(2.0 * PI * DriveFrequency1 * TSquence)
        DeriDHy = - DriveAmplitude * 2 * PI * DriveFrequency2 * np.sin(2 * PI * DriveFrequency2 * TSquence) * np.sin(2.0 * PI * DriveFrequency1 * TSquence) + DriveAmplitude * 2 * PI * DriveFrequency1 * np.cos(2 * PI * DriveFrequency2 * TSquence) * np.cos(2.0 * PI * DriveFrequency1 * TSquence)
        return DHx, DeriDHx, DHy, DeriDHy


    # ======================= FFP forward model=======================
    def _get_Voltage_CPU_FFP(self):
        tn = self._Fn + self._Nx*self._Ny*self._Nz
        num=0
        self._Voltage = np.zeros((self._Fn, 3))
        GSc = np.zeros((self._Nx,self._Ny,self._Nz, 3))
        print("Looping for GSc...")
        for i in range(self._Nx):
            x = (i) * self._Step - self._Maxx
            for j in range(self._Ny):
                y = (j) * self._Step - self._Maxy
                for k in range(self._Nz):
                    #x=(i)*(1e-4) - self._Maxx
                    #y=(j)*(1e-4) - self._Maxy
                    z=(k) * self._Step - self._Maxz
                    temp=np.multiply(self._Gg,[[x],[y],[z]])
                    GSc[i, j, k, 0]=temp[0]
                    GSc[i, j, k, 1]=temp[1]
                    GSc[i, j, k, 2]=temp[2]
                    num = num + 1
                    #bar.setValue(num, tn, SIMUVOL)

        def ThdFunc(DH, l, r, o, GSc, B, C):
            DHt = np.tile(DH, (l, r, o, 1))
            test = np.subtract(DHt, GSc)
            H = np.sqrt(test[:, :, :, 2] ** 2 + test[:, :, :, 1] ** 2 + test[:, :, :, 0] ** 2)
            DLF = (1 / ((B * H) ** 2)) - (1 / ((np.sinh(B * H)) ** 2))

            s = C * DLF
            signal = np.sum(s[:, :, :])
            # print(f"ThdFunc call for {signal}")

            return signal

        print("Entering multi-thread calculation.")
        #executor = ProcessPoolExecutor(max_workers=4)
        executor = ThreadPoolExecutor(max_workers=GET_VALUE('PROTHDS'))
        all_task = [executor.submit(lambda p: ThdFunc(*p), [self._DH[:,i],self._Nx,self._Ny,self._Nz,GSc,self._VirtualPhantom._Bb,self._PhantomMatrix]) for i in range(int(self._Fn))]
        wait(all_task, return_when=ALL_COMPLETED)
        print("Exiting multi-thread calculation.")
        Coeff=self._CoilSensitivity*self._VirtualPhantom._Bb* self._VirtualPhantom._Mm
        i=0
        for value in all_task:
            temp=value.result()
            self._Voltage[i, 0] = Coeff * self._DeriDH[0,i] * temp
            self._Voltage[i, 1] = Coeff * self._DeriDH[1,i] * temp
            self._Voltage[i, 2] = Coeff * self._DeriDH[2,i] * temp
            i=i+1
            num = num + 1
            #bar.setValue(num, tn, SIMUVOL)
        executor.shutdown()

    # ======================= FFL TRUE: build & forward =======================
    def _Mtheta(self, theta, G):
        c, s = float(np.cos(theta)), float(np.sin(theta))
        R = np.array([[ c, -s, 0.0],
                      [ s,  c, 0.0],
                      [0.0, 0.0, 1.0]], dtype=float)
        S = np.diag([0.0, -G, +G])   # eigenvalues: 0 (line dir), -G (in-plane normal), +G (z)
        return R @ S @ R.T

    # ======== (REPLACED) build per-angle GSc by the paper-true tensor ========
    def _build_GSc_FFL_theta(self, theta, G):
        """
        Paper-true selection field:
          H_S^θ(r) = Mθ r,  with  Mθ = Rθ diag(0,-G,+G) Rθ^T  (eq. (8))
        """
        M = self._Mtheta(theta, G)
        GScθ = np.zeros((self._Nx, self._Ny, self._Nz, 3), dtype=float)

        # coordinates on our voxel grid
        xs = self._xs; ys = self._ys; zs = self._zs
        for i in range(self._Nx):
            x = xs[i]
            for j in range(self._Ny):
                y = ys[j]
                for k in range(self._Nz):
                    z = zs[k]
                    r = np.array([x, y, z], dtype=float)
                    GScθ[i, j, k, :] = M @ r
        return GScθ

    # ======== stack budgeting + uniform drive that moves the FFL ========
    def _build_ffl_true_stack_drive(self):
        """
        Paper-true 3D FFL stack:
          Mθ = Rθ diag(0,-G,+G) Rθ^T
          r0(θ,s,z) = s qθ + z e_z
          DH(t) = Mθ r0  (so B = DH - Mθ r has FFL at r0 + α pθ)

        Fixes in this version:
          • Analytic dDH/dt = Mθ (dr0/dt) = -G * v_perp * qθ  within each (z,θ) segment
          • No finite differences across segment boundaries
          • Valid-mask for samples (used by recon)
        """
        FN = self._Fn

        # single magnitude G for the traceless tensor (eq.(8) style)
        Gbook = float(min(abs(self._Gy), abs(self._Gz)))
        Gbook = max(Gbook, 1e-12)
        self._FFL_BOOK_G = Gbook

        Nz_goal = int(self._Nz)

        # Requested (keep defaults but allow optional override via attributes)
        NTH_req, Ns_req =  180 , 64             #180, 64
        NTH_req = int(getattr(self, "_ffl_num_angles", NTH_req) or NTH_req)
        Ns_req = int(getattr(self, "_ffl_num_sbins", Ns_req) or Ns_req)

        # Lower bounds (tune these if you want to prioritize angles even more)
        MIN_NS, MIN_NTH =  16, 2                     #16, 8

        # Start from a sensible s-resolution, then exact-fit the angles to the budget
        Ns = max(MIN_NS, Ns_req)
        max_angles = int(FN // max(1, Nz_goal * Ns))
        NTH = max(MIN_NTH, min(NTH_req, max_angles))

        s_max = float(np.sqrt(self._Maxx ** 2 + self._Maxy ** 2))

        # If still over budget (due to integer floors), drop z as a last resort
        while Nz_goal * NTH * Ns > FN and Nz_goal > 1:
            Nz_goal -= 1

        kz_idx = _center_out_indices(self._Nz, Nz_goal)
        z_levels = self._zs[kz_idx]
        th_cent = np.linspace(0.0, np.pi, NTH, endpoint=False)
        s_cent = np.linspace(-s_max, s_max, Ns)
        ds = (s_cent[1] - s_cent[0]) if Ns > 1 else self._Step

        # Precompute field bank per θ
        self._GSc_bank = np.zeros((NTH, self._Nx, self._Ny, self._Nz, 3), dtype=float)
        for k, theta in enumerate(th_cent):
            self._GSc_bank[k] = self._build_GSc_FFL_theta(theta, Gbook)

        DH = np.zeros((3, self._Fn), dtype=float)
        dDH = np.zeros_like(DH)
        iz = np.full(self._Fn, -1, dtype=np.int32)
        ith = np.zeros(self._Fn, dtype=np.int32)
        isb = np.zeros(self._Fn, dtype=np.int32)
        vperp = np.zeros(self._Fn, dtype=float)
        v0 = abs(ds) * self._Sf if Ns > 1 else 1.0

        self._LineDir = np.zeros((self._Fn, 3), dtype=float)
        self._r0 = np.zeros((self._Fn, 3), dtype=float)
        valid = np.zeros(self._Fn, dtype=bool)

        t = 0
        for kz_i in range(Nz_goal):
            zlev = float(z_levels[kz_i])
            for k, theta in enumerate(th_cent):
                c, s = float(np.cos(theta)), float(np.sin(theta))
                pθ = np.array([c, s, 0.0], dtype=float)  # line direction
                qθ = np.array([-s, c, 0.0], dtype=float)  # sweep normal
                Mθ = self._Mtheta(theta, Gbook)

                for m in range(Ns):
                    if t >= self._Fn: break
                    sb = float(s_cent[m])
                    r0 = sb * qθ + np.array([0.0, 0.0, zlev], dtype=float)

                    # place the FFL
                    DH[:, t] = Mθ @ r0
                    # analytic derivative within the segment
                    dDH[:, t] = -Gbook * v0 * qθ

                    self._LineDir[t, :] = pθ
                    self._r0[t, :] = r0
                    iz[t], ith[t], isb[t] = kz_i, k, m
                    vperp[t] = v0
                    valid[t] = True
                    t += 1
                if t >= self._Fn: break
            if t >= self._Fn: break

        # Tail hold if budget < Fn (mark as invalid so recon ignores)
        if t < self._Fn:
            DH[:, t:] = DH[:, [max(0, t - 1)]]
            dDH[:, t:] = 0.0
            valid[t:] = False
            iz[t:] = -1

        self._DH, self._DeriDH = DH, dDH
        self._iz_map, self._ith_map, self._is_map = iz, ith, isb
        self._th_cent, self._s_cent, self._z_levels = th_cent, s_cent, z_levels
        self._kz_idx = kz_idx
        self._vperp = vperp
        self._NTH, self._Ns, self._Nz_sino = NTH, Ns, Nz_goal
        self._valid_mask = valid
        self._samples_used = int(np.count_nonzero(valid))
        self._valid_count = self._samples_used

        print(f"[FFL-BOOK] z_slices={Nz_goal}/{self._Nz} angles/slice={NTH} s-bins/slice={Ns} "
              f"samples_used={self._samples_used} Fn={self._Fn}")

    # ======== (UNCHANGED numerically, but now uses the paper-true GSc_bank) ========
    def _get_Voltage_CPU_FFL_true(self):
        """
        s̃θ(t) ≈ trace(A_h[X_{eθ}[ρ]])(r(t)) · v(t)
        We compute a scalar 'spatial factor' (sum C·L'(β||·||)) and scale per-channel with dDH/dt,
        matching the x-space Core-Operator form; see Theorem 5 & eq. (30).
        """
        self._Voltage = np.zeros((self._Fn, 3), dtype=float)
        C  = self._PhantomMatrix
        Bb = self._VirtualPhantom._Bb
        Mm = self._VirtualPhantom._Mm
        coeff = self._CoilSensitivity * Bb * Mm

        GSc_bank = self._GSc_bank

        def ThdFunc_FFL(i):
            k = self._ith_map[i]
            GScθ = GSc_bank[k]  # (Nx,Ny,Nz,3)
            Bx = self._DH[0, i] - GScθ[..., 0]
            By = self._DH[1, i] - GScθ[..., 1]
            Bz = self._DH[2, i] - GScθ[..., 2]
            H  = np.sqrt(Bx*Bx + By*By + Bz*Bz) + 1e-20
            DLF = _langevin_prime(Bb * H)
            spat = np.sum(C * DLF)
            return coeff * np.array([self._DeriDH[0, i] * spat,
                                     self._DeriDH[1, i] * spat,
                                     self._DeriDH[2, i] * spat], dtype=float)

        executor = ThreadPoolExecutor(max_workers=GET_VALUE('PROTHDS'))
        futures = [executor.submit(ThdFunc_FFL, i) for i in range(self._Fn)]
        wait(futures, return_when=ALL_COMPLETED)
        for i, f in enumerate(futures):
            self._Voltage[i, :] = f.result()
        executor.shutdown()

    # ======================= Message packing & ABC shim =======================

    def _init_Message(self):
        if getattr(self, "_is_ffl_knopp", False):
            return self._init_Message_FFL_knopp()
        if getattr(self, "_is_ffl_true", False):
            return self._init_Message_FFL_true()
        else:
            return self._init_Message_FFP()

    def _init_Message_FFP(self):
        # Magnetic particle parameters
        self._set_MessageValue(MAGNETICPARTICL, TEMPERATURE,   self._VirtualPhantom._Tt)
        self._set_MessageValue(MAGNETICPARTICL, DIAMETER,      self._VirtualPhantom._Diameter)
        self._set_MessageValue(MAGNETICPARTICL, SATURATIONMAG, self._VirtualPhantom._Mm)

        # Selection field (diagonal gradients)
        self._set_MessageValue(SELECTIONFIELD, XGRADIENT, self._Gx)
        self._set_MessageValue(SELECTIONFIELD, YGRADIENT, self._Gy)
        self._set_MessageValue(SELECTIONFIELD, ZGRADIENT, self._Gz)

        # Drive field setup (amplitude, frequency)
        self._set_MessageValue(DRIVEFIELD, XDIRECTIOND, np.array([self._Ax, self._Fx, 0]))
        self._set_MessageValue(DRIVEFIELD, YDIRECTIOND, np.array([self._Ay, self._Fy, 0]))
        self._set_MessageValue(DRIVEFIELD, ZDIRECTIOND, np.array([self._Az, self._Fz, 0]))
        self._set_MessageValue(DRIVEFIELD, REPEATTIME,  self._Rt)
        self._set_MessageValue(DRIVEFIELD, WAVEFORMD,   SINE)

        # Sampling
        self._set_MessageValue(SAMPLE, TOPOLOGY,  FFP)
        self._set_MessageValue(SAMPLE, FREQUENCY, self._Sf)
        self._set_MessageValue(SAMPLE, SAMNUMBER, self._Fn)
        self._set_MessageValue(SAMPLE, BEGINTIME, None)
        self._set_MessageValue(SAMPLE, SENSITIVITY, self._CoilSensitivity)

        # Measurement
        self._set_MessageValue(MEASUREMENT, TYPE,    2)
        self._set_MessageValue(MEASUREMENT, BGFLAG,  np.ones(np.shape(self._Voltage), dtype='bool'))
        self._set_MessageValue(MEASUREMENT, MEASIGNAL, self._Voltage)
        self._set_MessageValue(MEASUREMENT, AUXSIGNAL, self._Ffpv)
        self._set_MessageValue(MEASUREMENT, MEANUMBER, np.array([self._Nx, self._Ny, self._Nz], dtype='int64'))

        self.Message[EXTENDED] = {STEP: self._Step, RFFP: self._Ffpr}

        # Extended
        ext = {STEP: self._Step}
        if hasattr(self, "_Ffpr"):
            ext[RFFP] = self._Ffpr
        self.Message[EXTENDED] = ext
        return True

    def _init_Message_FFL_true(self):
        # Magnetic particle parameters
        self._set_MessageValue(MAGNETICPARTICL, TEMPERATURE,   self._VirtualPhantom._Tt)
        self._set_MessageValue(MAGNETICPARTICL, DIAMETER,      self._VirtualPhantom._Diameter)
        self._set_MessageValue(MAGNETICPARTICL, SATURATIONMAG, self._VirtualPhantom._Mm)

        # Selection field (gradients in lab frame)
        self._set_MessageValue(SELECTIONFIELD, XGRADIENT, self._Gx)
        self._set_MessageValue(SELECTIONFIELD, YGRADIENT, self._Gy)
        self._set_MessageValue(SELECTIONFIELD, ZGRADIENT, self._Gz)

        # Drive field setup (amplitude, frequency)
        self._set_MessageValue(DRIVEFIELD, XDIRECTIOND, np.array([self._Ax, self._Fx, 0]))
        self._set_MessageValue(DRIVEFIELD, YDIRECTIOND, np.array([self._Ay, self._Fy, 0]))
        self._set_MessageValue(DRIVEFIELD, ZDIRECTIOND, np.array([self._Az, self._Fz, 0]))
        self._set_MessageValue(DRIVEFIELD, REPEATTIME,  self._Rt)
        self._set_MessageValue(DRIVEFIELD, WAVEFORMD,   SINE)

        # Sampling
        self._set_MessageValue(SAMPLE, TOPOLOGY,  "FFL")
        self._set_MessageValue(SAMPLE, FREQUENCY, self._Sf)
        self._set_MessageValue(SAMPLE, SAMNUMBER, self._Fn)
        self._set_MessageValue(SAMPLE, BEGINTIME, None)
        self._set_MessageValue(SAMPLE, SENSITIVITY, self._CoilSensitivity)

        # Measurement (include line geometry in AUX for debugging/compat)
        self._set_MessageValue(MEASUREMENT, TYPE,    2)
        self._set_MessageValue(MEASUREMENT, BGFLAG,  np.ones(np.shape(self._Voltage), dtype='bool'))
        self._set_MessageValue(MEASUREMENT, MEASIGNAL, self._Voltage)
        self._set_MessageValue(MEASUREMENT, AUXSIGNAL,
                               np.stack([self._LineDir, self._r0], axis=0))
        self._set_MessageValue(MEASUREMENT, MEANUMBER,
                               np.array([self._Nx, self._Ny, self._Nz], dtype='int64'))

        # Extended metadata for 3D stacking reconstruction (+ vperp & line geo)
        self.Message[EXTENDED] = {
            "STEP": self._Step,
            "FFL_VPERP":    self._vperp,          # (Fn,)
            "FFL_LINE_DIR": self._LineDir,        # (Fn,3)
            "FFL_LINE_ORG": self._r0,             # (Fn,3)
            "FFL_SINOGRAM_3D": {
                "theta":     self._th_cent.astype(np.float64),
                "s_cent":    self._s_cent.astype(np.float64),
                "z_levels":  self._z_levels.astype(np.float64),
                "kz_idx":    self._kz_idx.astype(np.int32),
                "iz":        self._iz_map.astype(np.int32),
                "ith":       self._ith_map.astype(np.int32),
                "is":        self._is_map.astype(np.int32),
                "valid_count": int(self._valid_count),
            },
            "FFL_BOOK_G": float(getattr(self, "_FFL_BOOK_G", 0.0)),  # <- new
        }
        return True

