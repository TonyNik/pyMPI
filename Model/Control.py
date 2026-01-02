from Model.Scanner import *
from Model.Phantom import *
from Model.Interference import *
import json
import numpy as np

from MPIRF.ReconClass.XRecon import *
from MPIRF.Config.UIConstantListEn import IMGRECONSTR
from MPIRF.Config.ConstantList import *

def get_ReconImg(tt, dt, ms, cn,
                 gx, gy, gz,
                 fx, fy, fz,
                 ax, ay, az,
                 rt, sf,
                 ref, re,        # relaxation flag & time constant
                 nsf, ns,        # noise flag & level
                 bgf, bg,        # background flag & params
                 index=0,
                 trajectory_type="3D"):
    """
    Original pipeline entry.
    Builds Phantom & Scanner, (optionally) applies interference,
    runs reconstruction, and returns (ImgStru, OrgImgData, Message).
    """
    print("get_ReconImg call")
    # 1) Build phantom & scanner
    P1 = PhantomClass(tt, dt, ms, cn, index)
    print("Phantom instance created - PhantomClass call.")
    S1 = ScannerClass(P1, gx, gy, gz, fx, fy, fz, ax, ay, az, rt, sf,
                      trajectory_type=trajectory_type)
    print("Scanner instance created - ScannerClass call.")

    # Keep original measurement
    Message = S1.Message
    Message[MEASUREMENT]['OriMeaSignal'] = Message[MEASUREMENT][MEASIGNAL]

    # 2) Interference (noise / background) — optional
    noise = None
    bgsignal = None
    if nsf:
        noise = InterferenceClass().GaussianNoise(Message, ns)
    if bgf:
        bgsignal = InterferenceClass().BackgroundHarmonic(Message, bg)

    # 3) EXTENDED bookkeeping
    if EXTENDED not in Message:
        Message[EXTENDED] = {}
    Message[EXTENDED]["NoiseFlag"] = bool(nsf)
    Message[EXTENDED]["BackgroundFlag"] = bool(bgf)
    if noise is not None:
        Message[EXTENDED]["NoiseValue"] = noise
    if bgsignal is not None:
        Message[EXTENDED]["BackgroundValue"] = bgsignal
    Message[EXTENDED]["Relaxation"] = bool(ref)
    Message[EXTENDED]["RelaxationTime"] = re

    # 4) Reconstruction
    print("Creating XReconClass instance... - XReconClass call")
    ImgStru = XReconClass(Message)
    print("XReconClass instance created.")

    # 5) Original phantom image (same grid size as scanner)
    N = Message[MEASUREMENT][MEANUMBER]   # [Nx, Ny, Nz]
    OrgImgData = P1.getShape(int(N[0]), int(N[1]), int(N[2]))
    OrgMax = np.max(OrgImgData)
    if OrgMax > 0:
        OrgImgData = OrgImgData / OrgMax

    return ImgStru, OrgImgData, Message


def get_ReconImgItf(ref, re, nsf, ns, bgf, bg, Message):
    print("get_ReconImgItf call.")
    Message[MEASUREMENT][MEASIGNAL] = Message[MEASUREMENT]['OriMeaSignal']

    noise = None
    bgsignal = None
    if nsf:
        noise = InterferenceClass().GaussianNoise(Message, ns)
    if bgf:
        bgsignal = InterferenceClass().BackgroundHarmonic(Message, bg)

    Message[EXTENDED]["NoiseFlag"] = nsf
    Message[EXTENDED]["BackgroundFlag"] = bgf

    if noise is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+noise
        Message[EXTENDED]["NoiseValue"] = noise

    if bgsignal is not None:
        # S1.Message[MEASUREMENT][MEASIGNAL]=S1.Message[MEASUREMENT][MEASIGNAL]+bgsignal
        Message[EXTENDED]["BackgroundValue"] = bgsignal

    #bar = myProgressBar()
    #bar.setValue(0, 0, IMGRECONSTR)

    Message[EXTENDED]["Relaxation"] = ref
    Message[EXTENDED]["RelaxationTime"] = re

    ImgStru = XReconClass(Message)
    #bar.setValue(100, 0, IMGRECONSTR)
    #bar.close()

    return ImgStru


def get_OriImgData(index=0):
    P1 = PhantomClass(index=index)
    OriImgData = P1.getShape(100, 100, 50)
    return OriImgData / np.max(OriImgData)


class JsonDefaultEnconding(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, complex):
            return "{real}+{image}i".format(real=o.real, image=o.real)
        if isinstance(o, np.ndarray):
            return o.tolist()
