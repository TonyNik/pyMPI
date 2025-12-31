
import napari
import matplotlib.pyplot as plt

import numpy as np



"""

Key features:
- One napari window, Phantom + Reconstruction side-by-side (grid mode).
- Overlay handling that WON'T spam the canvas:
  - If overlay_text is a dict (e.g. your "Message"), it auto-summarizes.
  - If overlay_text is a long string, it auto-truncates.
  - NEEDS FURTHER OPTIMIZATION
"""


from typing import Optional, Tuple, Union, Any, Mapping


ArrayLike3D = Union[np.ndarray]


def _to_zyx(vol: ArrayLike3D, axis_order: str) -> np.ndarray:
    v = np.asarray(vol)
    if v.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {v.shape}")

    axis_order = axis_order.lower().strip()
    if axis_order == "xyz":
        v = np.transpose(v, (2, 1, 0))  # (Z,Y,X) for napari
    elif axis_order == "zyx":
        pass
    else:
        raise ValueError("axis_order must be 'xyz' or 'zyx'")

    return v


def _downsample_zyx(v: np.ndarray, factor: int) -> np.ndarray:
    if factor is None or factor <= 1:
        return v
    return v[::factor, ::factor, ::factor]


def _as_float32_contiguous(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    return np.ascontiguousarray(v)


def _robust_contrast_limits(
    v: np.ndarray,
    lo_pct: float = 1.0,
    hi_pct: float = 99.5,
    *,
    clip_negative: bool = True,
) -> Tuple[float, float]:
    x = v
    if clip_negative:
        x = np.clip(x, 0, None)

    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)

    lo = float(np.percentile(x, lo_pct))
    hi = float(np.percentile(x, hi_pct))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(x))
        hi = float(np.max(x) + 1e-6)

    return lo, hi


def _shorten_text(s: str, max_len: int = 220) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _summarize_message_dict(msg: Mapping[str, Any], max_len: int = 220) -> str:
    """
    Create a small, readable summary from your big parameter dict.
    """
    lines = []

    # Magnetic particle summary
    mp = msg.get("MagneticParticle") if isinstance(msg, Mapping) else None
    if isinstance(mp, Mapping):
        d = _safe_float(mp.get("Diameter"))
        t = _safe_float(mp.get("Temperature"))
        if d is not None or t is not None:
            parts = []
            if d is not None:
                parts.append(f"d={d:g} m")
            if t is not None:
                parts.append(f"T={t:g} K")
            lines.append("Particle: " + ", ".join(parts))

    # Selection field summary
    sf = msg.get("SelectionField") if isinstance(msg, Mapping) else None
    if isinstance(sf, Mapping):
        gx = _safe_float(sf.get("XGradient") or sf.get("xGradient") or sf.get("Gx"))
        gy = _safe_float(sf.get("YGradient") or sf.get("yGradient") or sf.get("Gy"))
        gz = _safe_float(sf.get("ZGradient") or sf.get("zGradient") or sf.get("Gz"))
        grads = [g for g in (gx, gy, gz) if g is not None]
        if grads:
            # show only what exists
            chunks = []
            if gx is not None:
                chunks.append(f"Gx={gx:g}")
            if gy is not None:
                chunks.append(f"Gy={gy:g}")
            if gz is not None:
                chunks.append(f"Gz={gz:g}")
            lines.append("SelField: " + ", ".join(chunks))

    # Flags
    for key in ("NoiseFlag", "BackgroundFlag", "Relaxation"):
        if key in msg:
            lines.append(f"{key}: {msg.get(key)}")

    # If nothing useful found, just show keys count
    if not lines:
        try:
            lines = [f"Message keys: {len(list(msg.keys()))}"]
        except Exception:
            lines = ["Message"]

    return _shorten_text("\n".join(lines), max_len=max_len)


def _normalize_overlay_text(overlay_text: Any, max_len: int = 220) -> Optional[str]:
    """
    Accepts:
      - None -> no overlay
      - dict-like -> summarized overlay
      - str -> truncated if too long
      - anything else -> str(...) then truncated
    """
    if overlay_text is None:
        return None

    if isinstance(overlay_text, Mapping):
        return _summarize_message_dict(overlay_text, max_len=max_len)

    if isinstance(overlay_text, str):
        return _shorten_text(overlay_text, max_len=max_len)

    return _shorten_text(str(overlay_text), max_len=max_len)


def render_napari_volumes(
    phantom_xyz: ArrayLike3D,
    recon_xyz: ArrayLike3D,
    *,
    axis_order: str = "xyz",
    title: str = "MPI Phantom + Recon",
    overlay_text: Any = None,                 # can be dict/string/None
    overlay_position: str = "top_left",        # top_left, top_center, ...
    overlay_max_len: int = 220,
    rendering: str = "mip",                    # mip, attenuated_mip, translucent, iso, minip
    attenuation: float = 0.02,                 # for attenuated_mip
    iso_percentile: float = 99.0,              # for iso
    interpolation3d: Optional[str] = None,     # "linear" or "nearest" (if supported)
    downsample: int = 1,
    voxel_size_zyx: Optional[Tuple[float, float, float]] = None,
    shared_contrast: bool = True,
    lo_pct: float = 1.0,
    hi_pct: float = 99.5,
    clip_negative_for_display: bool = True,
    output_img: Optional[str] = None,
    screenshot_delay_ms: int = 600,
):
    """
    Returns:
        viewer (napari.Viewer)
    """

    phantom = _to_zyx(phantom_xyz, axis_order=axis_order)
    recon = _to_zyx(recon_xyz, axis_order=axis_order)

    phantom = _downsample_zyx(phantom, downsample)
    recon = _downsample_zyx(recon, downsample)

    phantom = _as_float32_contiguous(phantom)
    recon = _as_float32_contiguous(recon)

    # Contrast limits
    if shared_contrast:
        stacked = np.concatenate([phantom.ravel(), recon.ravel()])
        lo, hi = _robust_contrast_limits(
            stacked, lo_pct=lo_pct, hi_pct=hi_pct, clip_negative=clip_negative_for_display
        )
        cl_ph = (lo, hi)
        cl_rc = (lo, hi)
    else:
        cl_ph = _robust_contrast_limits(
            phantom, lo_pct=lo_pct, hi_pct=hi_pct, clip_negative=clip_negative_for_display
        )
        cl_rc = _robust_contrast_limits(
            recon, lo_pct=lo_pct, hi_pct=hi_pct, clip_negative=clip_negative_for_display
        )

    viewer = napari.Viewer(ndisplay=3, title=title)

    l1 = viewer.add_image(
        phantom,
        name="Phantom",
        colormap="gray",
        rendering=rendering,
        contrast_limits=cl_ph,
    )
    l2 = viewer.add_image(
        recon,
        name="Reconstruction",
        colormap="gray",
        rendering=rendering,
        contrast_limits=cl_rc,
    )

    # Optional tuning (guarded for non compatible versions) - needs further optimization
    if interpolation3d:
        for layer in (l1, l2):
            try:
                layer.interpolation3d = interpolation3d
            except Exception:
                pass

    if rendering == "attenuated_mip":
        for layer in (l1, l2):
            try:
                layer.attenuation = float(attenuation)
            except Exception:
                pass

    if rendering == "iso":
        stacked = np.concatenate([phantom.ravel(), recon.ravel()])
        stacked = stacked[np.isfinite(stacked)]
        if stacked.size:
            thr = float(np.percentile(stacked, iso_percentile))
            for layer in (l1, l2):
                try:
                    layer.iso_threshold = thr
                except Exception:
                    pass

    if voxel_size_zyx is not None:
        for layer in (l1, l2):
            try:
                layer.scale = tuple(map(float, voxel_size_zyx))
            except Exception:
                pass

    # Grid side-by-side
    try:
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 2)
        viewer.grid.stride = 1
    except Exception:
        pass

    # Overlay
    text = _normalize_overlay_text(overlay_text, max_len=overlay_max_len)
    if text:
        try:
            viewer.text_overlay.visible = True
            viewer.text_overlay.text = text
            try:
                # napari expects "top_left" not "top left"
                viewer.text_overlay.position = overlay_position
            except Exception:
                pass
        except Exception:
            pass

    # Screenshot
    if output_img:
        try:
            from qtpy.QtCore import QTimer

            def _snap():
                try:
                    viewer.reset_view()
                except Exception:
                    pass
                try:
                    viewer.screenshot(path=output_img, canvas_only=True, flash=False)
                except Exception:
                    try:
                        viewer.screenshot(path=output_img)
                    except Exception:
                        pass

            QTimer.singleShot(int(screenshot_delay_ms), _snap)
        except Exception:
            pass

    return viewer








def plot_projection(ImgData, plane_index=0, slice_index=0):
    """
    Generate a 2D projection view (using matplotlib) of the reconstructed image.
    plane_index=0: Maximum intensity projection (X-Y) over the entire volume.
    plane_index=1: Single-slice view (X-Y).
    """
    l, r, o = np.shape(ImgData)
    plt.figure()
    if plane_index == 0:
        Ixy = np.zeros((l, r))
        for i in range(l):
            for j in range(r):
                Ixy[i, j] = np.max(ImgData[i, j, :])
        Ixy = Ixy / np.max(Ixy)
        plt.imshow(Ixy, cmap=plt.get_cmap('binary'))
        plt.title("X-Y Max Projection")
        plt.axis("off")
        plt.show()
    elif plane_index == 1:
        Ixy = ImgData[:, :, slice_index]
        Ixy = Ixy / np.max(Ixy)
        plt.imshow(Ixy, cmap=plt.get_cmap('binary'))
        plt.title("X-Y Slice")
        plt.axis("off")
        plt.show()
    else:
        print("Projection for plane index {} not implemented.".format(plane_index))
