"""UI utility classes for the N64 DLSS control panel."""

import tkinter as tk
from tkinter import ttk


class ToolTip:
    """Hover tooltip for any tkinter widget. Shows after 500ms delay."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, event=None):
        self._hide()
        self._after_id = self.widget.after(500, self._show)

    def _show(self):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#ffffe0", relief="solid", borderwidth=1,
            font=("TkDefaultFont", 9), padx=6, pady=4,
        )
        label.pack()

    def _hide(self, event=None):
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


def set_widget_state(widget, enabled):
    """Recursively enable or disable a widget and all its children."""
    state_flag = "!disabled" if enabled else "disabled"
    try:
        if isinstance(widget, (ttk.Widget,)):
            widget.state([state_flag])
        else:
            widget.configure(state="normal" if enabled else "disabled")
    except (tk.TclError, AttributeError):
        pass  # frames, plain labels, etc.
    for child in widget.winfo_children():
        set_widget_state(child, enabled)


TOOLTIPS = {
    "enable_diffusion": "Toggle AI upscaling on/off. Hotkey: F2",
    "prompt": "Text prompt that guides the AI enhancement style",
    "denoise_strength": "How much the AI changes the image.\nLow = faithful, High = creative",
    "delta": "Virtual residual noise weight. Controls prompt adherence vs image fidelity",
    "temporal_blend": "Blend with previous frame (naive mode only).\nHigher = more stable but less responsive",
    "apply_settings": "Apply prompt, strength, and delta changes",
    "cn_enable": "Enable structural guidance from ControlNet (requires model reload)",
    "cn_canny": "Edge detection mode \u2014 preserves outlines",
    "cn_depth_midas": "Neural depth estimation via MiDaS",
    "cn_depth_zbuffer": "Native N64 Z-buffer depth (fastest, most accurate)",
    "cn_scale": "ControlNet conditioning strength. Higher = stronger structural guidance",
    "cn_canny_low": "Canny lower threshold (canny mode only)",
    "cn_canny_high": "Canny upper threshold (canny mode only)",
    "cn_zbuf_debug": "Show Z-buffer depth overlay in corner",
    "cn_zbuf_snapshot": "Save Z-buffer visualizations to disk",
    "cn_reload": "Reload model with current ControlNet settings",
    "tc_mode": "Temporal consistency method:\n\u2022 Off: no blending\n\u2022 Naive: simple alpha blend\n\u2022 Flow-Guided: optical flow warping\n\u2022 Feature Bank: attention-based coherence\n\u2022 V2V + Flow: combined (best quality)",
    "tc_flow_blend": "How much to favor warped history over new frame",
    "tc_occ_thresh": "Depth difference threshold for occlusion detection.\nLower = more aggressive masking",
    "tc_fi_strength": "Feature bank influence on generation.\nHigher = more temporal coherence",
    "tc_fi_thresh": "Cosine similarity threshold for feature matching.\nHigher = only inject very similar features",
    "tc_warped_noise": "Warp noise with optical flow for coherent denoising.\nReduces texture shimmer in static regions",
    "tc_motion_optical": "Compute motion from consecutive frames (works with any game)",
    "tc_motion_gamestate": "Read motion from N64 memory (SM64 US only, most accurate)",
    "tc_flow_debug": "Show optical flow visualization overlay",
}
