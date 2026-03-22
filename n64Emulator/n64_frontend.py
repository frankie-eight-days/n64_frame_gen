"""
N64 Libretro Frontend for StreamDiffusion Integration

Loads the mupen64plus-next libretro core and captures frames via the
retro_video_refresh_t callback. Supports both software (angrylion) and
hardware-accelerated (gliden64/parallel-rdp via OpenGL) rendering.

Usage:
    python n64_frontend.py <rom_path> [core_path]

Controls:
    Arrow Keys   D-Pad
    WASD         Analog Stick (N64 Joystick)
    X            A Button
    Z            B Button
    Left Shift   Z Trigger
    Q / E        L / R Shoulder
    Enter        Start
    I/J/K/L      C-Buttons (Up/Left/Down/Right)
    ESC          Quit
    F1           Reset
"""

import ctypes
import ctypes.util
import sys
import os
import time
import struct
import numpy as np
import pygame
from pathlib import Path

# =============================================================================
# Libretro Constants
# =============================================================================

RETRO_API_VERSION = 1

RETRO_ENVIRONMENT_GET_OVERSCAN = 2
RETRO_ENVIRONMENT_GET_CAN_DUPE = 3
RETRO_ENVIRONMENT_SET_MESSAGE = 6
RETRO_ENVIRONMENT_SHUTDOWN = 7
RETRO_ENVIRONMENT_SET_PERFORMANCE_LEVEL = 8
RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY = 9
RETRO_ENVIRONMENT_SET_PIXEL_FORMAT = 10
RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS = 11
RETRO_ENVIRONMENT_SET_HW_RENDER = 14
RETRO_ENVIRONMENT_GET_VARIABLE = 15
RETRO_ENVIRONMENT_SET_VARIABLES = 16
RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE = 17
RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME = 18
RETRO_ENVIRONMENT_GET_LOG_INTERFACE = 27
RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY = 31
RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO = 32
RETRO_ENVIRONMENT_SET_CONTROLLER_INFO = 35
RETRO_ENVIRONMENT_SET_GEOMETRY = 37
RETRO_ENVIRONMENT_GET_LANGUAGE = 39
RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION = 52
RETRO_ENVIRONMENT_SET_CORE_OPTIONS = 53
RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2 = 67
RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL = 68
RETRO_ENVIRONMENT_SET_CONTENT_INFO_OVERRIDE = 65
RETRO_ENVIRONMENT_GET_INPUT_MAX_USERS = 61
RETRO_ENVIRONMENT_EXPERIMENTAL = 0x10000
RETRO_ENVIRONMENT_GET_INPUT_BITMASKS = 51 | RETRO_ENVIRONMENT_EXPERIMENTAL
RETRO_ENVIRONMENT_SET_SERIALIZATION_QUIRKS = 44
RETRO_ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE = 47 | RETRO_ENVIRONMENT_EXPERIMENTAL
RETRO_ENVIRONMENT_SET_MINIMUM_AUDIO_LATENCY = 63
RETRO_ENVIRONMENT_SET_FASTFORWARDING_OVERRIDE = 64
RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER = 56

RETRO_PIXEL_FORMAT_0RGB1555 = 0
RETRO_PIXEL_FORMAT_XRGB8888 = 1
RETRO_PIXEL_FORMAT_RGB565 = 2

RETRO_DEVICE_NONE = 0
RETRO_DEVICE_JOYPAD = 1
RETRO_DEVICE_ANALOG = 5

RETRO_DEVICE_ID_JOYPAD_B = 0
RETRO_DEVICE_ID_JOYPAD_Y = 1
RETRO_DEVICE_ID_JOYPAD_SELECT = 2
RETRO_DEVICE_ID_JOYPAD_START = 3
RETRO_DEVICE_ID_JOYPAD_UP = 4
RETRO_DEVICE_ID_JOYPAD_DOWN = 5
RETRO_DEVICE_ID_JOYPAD_LEFT = 6
RETRO_DEVICE_ID_JOYPAD_RIGHT = 7
RETRO_DEVICE_ID_JOYPAD_A = 8
RETRO_DEVICE_ID_JOYPAD_X = 9
RETRO_DEVICE_ID_JOYPAD_L = 10
RETRO_DEVICE_ID_JOYPAD_R = 11
RETRO_DEVICE_ID_JOYPAD_L2 = 12

RETRO_DEVICE_INDEX_ANALOG_LEFT = 0
RETRO_DEVICE_INDEX_ANALOG_RIGHT = 1
RETRO_DEVICE_ID_ANALOG_X = 0
RETRO_DEVICE_ID_ANALOG_Y = 1

# HW context types
RETRO_HW_CONTEXT_NONE = 0
RETRO_HW_CONTEXT_OPENGL = 1
RETRO_HW_CONTEXT_OPENGLES2 = 2
RETRO_HW_CONTEXT_OPENGL_CORE = 3
RETRO_HW_CONTEXT_OPENGLES3 = 4
RETRO_HW_CONTEXT_VULKAN = 6

# Special data pointer for HW-rendered frames
RETRO_HW_FRAME_BUFFER_VALID = ctypes.c_void_p(-1).value  # (void*)-1

# =============================================================================
# Libretro Structs
# =============================================================================

class RetroSystemInfo(ctypes.Structure):
    _fields_ = [
        ("library_name", ctypes.c_char_p),
        ("library_version", ctypes.c_char_p),
        ("valid_extensions", ctypes.c_char_p),
        ("need_fullpath", ctypes.c_bool),
        ("block_extract", ctypes.c_bool),
    ]

class RetroGameGeometry(ctypes.Structure):
    _fields_ = [
        ("base_width", ctypes.c_uint),
        ("base_height", ctypes.c_uint),
        ("max_width", ctypes.c_uint),
        ("max_height", ctypes.c_uint),
        ("aspect_ratio", ctypes.c_float),
    ]

class RetroSystemTiming(ctypes.Structure):
    _fields_ = [
        ("fps", ctypes.c_double),
        ("sample_rate", ctypes.c_double),
    ]

class RetroSystemAVInfo(ctypes.Structure):
    _fields_ = [
        ("geometry", RetroGameGeometry),
        ("timing", RetroSystemTiming),
    ]

class RetroGameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]

class RetroVariable(ctypes.Structure):
    _fields_ = [
        ("key", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]

# HW render callback types
HW_CONTEXT_RESET_T = ctypes.CFUNCTYPE(None)
HW_GET_CURRENT_FB_T = ctypes.CFUNCTYPE(ctypes.c_size_t)
HW_GET_PROC_ADDRESS_T = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p)

class RetroHWRenderCallback(ctypes.Structure):
    _fields_ = [
        ("context_type", ctypes.c_uint),
        ("context_reset", HW_CONTEXT_RESET_T),
        ("get_current_framebuffer", HW_GET_CURRENT_FB_T),
        ("get_proc_address", HW_GET_PROC_ADDRESS_T),
        ("depth", ctypes.c_bool),
        ("stencil", ctypes.c_bool),
        ("bottom_left_origin", ctypes.c_bool),
        ("version_major", ctypes.c_uint),
        ("version_minor", ctypes.c_uint),
        ("cache_context", ctypes.c_bool),
        ("context_destroy", HW_CONTEXT_RESET_T),
        ("debug_context", ctypes.c_bool),
    ]

# =============================================================================
# Callback Types
# =============================================================================

ENVIRONMENT_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)
VIDEO_REFRESH_CB = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t
)
AUDIO_SAMPLE_CB = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
AUDIO_SAMPLE_BATCH_CB = ctypes.CFUNCTYPE(
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t
)
INPUT_POLL_CB = ctypes.CFUNCTYPE(None)
INPUT_STATE_CB = ctypes.CFUNCTYPE(
    ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint
)

# =============================================================================
# OpenGL helpers
# =============================================================================

def _load_sdl2():
    """Find and load SDL2 DLL (bundled with pygame)."""
    sdl_path = os.path.join(os.path.dirname(pygame.__file__), "SDL2.dll")
    if os.path.exists(sdl_path):
        return ctypes.CDLL(sdl_path)
    # fallback
    return ctypes.CDLL("SDL2.dll")

# =============================================================================
# N64 Frontend
# =============================================================================

class N64Frontend:
    WINDOW_SCALE = 2

    KEY_MAP = {
        pygame.K_x:      RETRO_DEVICE_ID_JOYPAD_A,
        pygame.K_z:      RETRO_DEVICE_ID_JOYPAD_B,
        pygame.K_LSHIFT: RETRO_DEVICE_ID_JOYPAD_L2,
        pygame.K_q:      RETRO_DEVICE_ID_JOYPAD_L,
        pygame.K_e:      RETRO_DEVICE_ID_JOYPAD_R,
        pygame.K_RETURN: RETRO_DEVICE_ID_JOYPAD_START,
        pygame.K_UP:     RETRO_DEVICE_ID_JOYPAD_UP,
        pygame.K_DOWN:   RETRO_DEVICE_ID_JOYPAD_DOWN,
        pygame.K_LEFT:   RETRO_DEVICE_ID_JOYPAD_LEFT,
        pygame.K_RIGHT:  RETRO_DEVICE_ID_JOYPAD_RIGHT,
    }

    ANALOG_KEYS = {
        pygame.K_w: (RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, -32767),
        pygame.K_s: (RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_Y, 32767),
        pygame.K_a: (RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, -32767),
        pygame.K_d: (RETRO_DEVICE_INDEX_ANALOG_LEFT, RETRO_DEVICE_ID_ANALOG_X, 32767),
        pygame.K_i: (RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, -32767),
        pygame.K_k: (RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_Y, 32767),
        pygame.K_j: (RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, -32767),
        pygame.K_l: (RETRO_DEVICE_INDEX_ANALOG_RIGHT, RETRO_DEVICE_ID_ANALOG_X, 32767),
    }

    # Core options -- supports both mupen64plus-next and parallel_n64 key prefixes
    CORE_OPTIONS = {
        # mupen64plus-next keys
        "mupen64plus-rdp-plugin": "gliden64",
        "mupen64plus-rsp-plugin": "hle",
        "mupen64plus-cpucore": "dynamic_recompiler",
        "mupen64plus-aspect": "4:3",
        "mupen64plus-43screensize": "320x240",
        "mupen64plus-EnableFBEmulation": "True",
        # parallel_n64 keys -- force angrylion software renderer
        "parallel-n64-gfxplugin": "angrylion",
        "parallel-n64-audio-buffer-size": "2048",
        "parallel-n64-screensize": "320x240",
        "parallel-n64-aspectratiohint": "normal",
        "parallel-n64-cpucore": "dynamic_recompiler",
    }

    def __init__(self, core_path, system_dir=None, save_dir=None):
        self.base_dir = Path(core_path).parent.parent
        self.system_dir = system_dir or str(self.base_dir / "system")
        self.save_dir = save_dir or str(self.base_dir / "saves")
        os.makedirs(self.system_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        self._system_dir_b = self.system_dir.encode("utf-8")
        self._save_dir_b = self.save_dir.encode("utf-8")

        self.core = ctypes.CDLL(core_path)
        self._setup_core_api()

        # State
        self.pixel_format = RETRO_PIXEL_FORMAT_0RGB1555
        self.frame_data = None
        self.frame_width = 320
        self.frame_height = 240
        self.target_fps = 60.0
        self.sample_rate = 44100.0
        self.running = False
        self.need_fullpath = True
        self.variables_updated = False
        self.keys_pressed = set()

        # HW rendering state
        self.hw_render = False
        self.hw_context_reset = None
        self.hw_context_destroy = None
        self.hw_bottom_left_origin = True
        self.sdl2 = None
        self.opengl32 = None

        # StreamDiffusion hook
        self.on_frame = None

        # Prevent GC
        self._env_cb = ENVIRONMENT_CB(self._environment)
        self._video_cb = VIDEO_REFRESH_CB(self._video_refresh)
        self._audio_cb = AUDIO_SAMPLE_CB(self._audio_sample)
        self._audio_batch_cb = AUDIO_SAMPLE_BATCH_CB(self._audio_sample_batch)
        self._input_poll_cb = INPUT_POLL_CB(self._input_poll)
        self._input_state_cb = INPUT_STATE_CB(self._input_state)
        self._hw_get_fb_cb = HW_GET_CURRENT_FB_T(self._hw_get_current_framebuffer)
        self._hw_get_proc_cb = HW_GET_PROC_ADDRESS_T(self._hw_get_proc_address)
        self._var_bufs = {}

    def _setup_core_api(self):
        c = self.core
        c.retro_api_version.restype = ctypes.c_uint
        c.retro_api_version.argtypes = []
        c.retro_init.restype = None
        c.retro_init.argtypes = []
        c.retro_deinit.restype = None
        c.retro_deinit.argtypes = []
        c.retro_set_environment.restype = None
        c.retro_set_environment.argtypes = [ENVIRONMENT_CB]
        c.retro_set_video_refresh.restype = None
        c.retro_set_video_refresh.argtypes = [VIDEO_REFRESH_CB]
        c.retro_set_audio_sample.restype = None
        c.retro_set_audio_sample.argtypes = [AUDIO_SAMPLE_CB]
        c.retro_set_audio_sample_batch.restype = None
        c.retro_set_audio_sample_batch.argtypes = [AUDIO_SAMPLE_BATCH_CB]
        c.retro_set_input_poll.restype = None
        c.retro_set_input_poll.argtypes = [INPUT_POLL_CB]
        c.retro_set_input_state.restype = None
        c.retro_set_input_state.argtypes = [INPUT_STATE_CB]
        c.retro_get_system_info.restype = None
        c.retro_get_system_info.argtypes = [ctypes.POINTER(RetroSystemInfo)]
        c.retro_get_system_av_info.restype = None
        c.retro_get_system_av_info.argtypes = [ctypes.POINTER(RetroSystemAVInfo)]
        c.retro_load_game.restype = ctypes.c_bool
        c.retro_load_game.argtypes = [ctypes.POINTER(RetroGameInfo)]
        c.retro_unload_game.restype = None
        c.retro_unload_game.argtypes = []
        c.retro_run.restype = None
        c.retro_run.argtypes = []
        c.retro_reset.restype = None
        c.retro_reset.argtypes = []
        c.retro_set_controller_port_device.restype = None
        c.retro_set_controller_port_device.argtypes = [ctypes.c_uint, ctypes.c_uint]
        c.retro_get_memory_data.restype = ctypes.c_void_p
        c.retro_get_memory_data.argtypes = [ctypes.c_uint]
        c.retro_get_memory_size.restype = ctypes.c_size_t
        c.retro_get_memory_size.argtypes = [ctypes.c_uint]

    # -----------------------------------------------------------------
    # HW rendering callbacks
    # -----------------------------------------------------------------

    def _hw_get_current_framebuffer(self):
        return 0  # default framebuffer

    def _hw_get_proc_address(self, name):
        if not name:
            return 0
        # Try SDL_GL_GetProcAddress first (handles both core and extension funcs)
        if self.sdl2:
            addr = self.sdl2.SDL_GL_GetProcAddress(name)
            if addr:
                return addr
        # Fallback: wglGetProcAddress for extension functions
        if self.opengl32:
            addr = self.opengl32.wglGetProcAddress(name)
            if addr:
                return addr
        name_str = name.decode("utf-8", errors="replace") if isinstance(name, bytes) else str(name)
        print(f"  WARNING: get_proc_address({name_str}) -> NULL")
        return 0

    # -----------------------------------------------------------------
    # Environment callback
    # -----------------------------------------------------------------

    def _environment(self, cmd, data):
        if cmd == RETRO_ENVIRONMENT_GET_CAN_DUPE:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = True
            return True

        if cmd == RETRO_ENVIRONMENT_GET_SYSTEM_DIRECTORY:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._system_dir_b
            return True

        if cmd == RETRO_ENVIRONMENT_GET_SAVE_DIRECTORY:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_char_p))[0] = self._save_dir_b
            return True

        if cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT:
            if data:
                self.pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0]
            return True

        if cmd == RETRO_ENVIRONMENT_GET_VARIABLE:
            if data:
                var_ptr = ctypes.cast(data, ctypes.POINTER(RetroVariable))
                key = var_ptr.contents.key
                if key:
                    key_str = key.decode("utf-8")
                    if key_str in self.CORE_OPTIONS:
                        val = self.CORE_OPTIONS[key_str]
                        self._var_bufs[key_str] = ctypes.c_char_p(val.encode("utf-8"))
                        val_off = RetroVariable.value.offset
                        ctypes.memmove(
                            data + val_off,
                            ctypes.byref(self._var_bufs[key_str]),
                            ctypes.sizeof(ctypes.c_char_p),
                        )
                        return True
            return False

        if cmd == RETRO_ENVIRONMENT_GET_VARIABLE_UPDATE:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = self.variables_updated
                self.variables_updated = False
            return True

        if cmd in (RETRO_ENVIRONMENT_SET_VARIABLES, RETRO_ENVIRONMENT_SET_CORE_OPTIONS):
            self.variables_updated = True
            return True

        if cmd in (RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2,
                   RETRO_ENVIRONMENT_SET_CORE_OPTIONS_V2_INTL):
            self.variables_updated = True
            return True

        if cmd == RETRO_ENVIRONMENT_GET_CORE_OPTIONS_VERSION:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = 0
            return True

        if cmd == RETRO_ENVIRONMENT_SET_HW_RENDER:
            if data:
                hw = RetroHWRenderCallback.from_address(data)
                ctx_type = hw.context_type
                ctx_names = {
                    0: "none", 1: "opengl", 2: "opengles2",
                    3: "opengl_core", 4: "opengles3", 6: "vulkan",
                }
                print(f"  HW render request: {ctx_names.get(ctx_type, ctx_type)}"
                      f" {hw.version_major}.{hw.version_minor}")

                # Only accept OpenGL contexts (not Vulkan)
                if ctx_type in (RETRO_HW_CONTEXT_OPENGL,
                                RETRO_HW_CONTEXT_OPENGL_CORE,
                                RETRO_HW_CONTEXT_OPENGLES2,
                                RETRO_HW_CONTEXT_OPENGLES3):
                    self.hw_render = True
                    self.hw_bottom_left_origin = hw.bottom_left_origin
                    self.hw_context_reset = hw.context_reset
                    self.hw_context_destroy = hw.context_destroy
                    # Fill in our callbacks
                    hw.get_current_framebuffer = self._hw_get_fb_cb
                    hw.get_proc_address = self._hw_get_proc_cb
                    print("  -> Accepted (OpenGL)")
                    return True
                else:
                    print(f"  -> Refused (unsupported context type)")
                    return False
            return False

        if cmd == RETRO_ENVIRONMENT_SHUTDOWN:
            self.running = False
            return True

        if cmd == RETRO_ENVIRONMENT_SET_GEOMETRY:
            if data:
                geo = RetroGameGeometry.from_address(data)
                self.frame_width = geo.base_width
                self.frame_height = geo.base_height
            return True

        if cmd == RETRO_ENVIRONMENT_SET_SYSTEM_AV_INFO:
            if data:
                av = RetroSystemAVInfo.from_address(data)
                self.frame_width = av.geometry.base_width
                self.frame_height = av.geometry.base_height
                self.target_fps = av.timing.fps
            return True

        if cmd == RETRO_ENVIRONMENT_GET_LANGUAGE:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = 0
            return True

        if cmd == RETRO_ENVIRONMENT_GET_OVERSCAN:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_bool))[0] = False
            return True

        if cmd == RETRO_ENVIRONMENT_GET_INPUT_MAX_USERS:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = 4
            return True

        if cmd == RETRO_ENVIRONMENT_GET_AUDIO_VIDEO_ENABLE:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_int))[0] = 3
            return True

        if cmd == RETRO_ENVIRONMENT_GET_PREFERRED_HW_RENDER:
            if data:
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint))[0] = RETRO_HW_CONTEXT_OPENGL_CORE
            return True

        if cmd == RETRO_ENVIRONMENT_GET_LOG_INTERFACE:
            return False

        # Acknowledge silently
        if cmd in (RETRO_ENVIRONMENT_SET_INPUT_DESCRIPTORS,
                   RETRO_ENVIRONMENT_SET_CONTROLLER_INFO,
                   RETRO_ENVIRONMENT_SET_SUPPORT_NO_GAME,
                   RETRO_ENVIRONMENT_SET_PERFORMANCE_LEVEL,
                   RETRO_ENVIRONMENT_SET_MESSAGE,
                   RETRO_ENVIRONMENT_SET_SERIALIZATION_QUIRKS,
                   RETRO_ENVIRONMENT_SET_CONTENT_INFO_OVERRIDE,
                   RETRO_ENVIRONMENT_SET_MINIMUM_AUDIO_LATENCY,
                   RETRO_ENVIRONMENT_SET_FASTFORWARDING_OVERRIDE):
            return True

        return False

    # -----------------------------------------------------------------
    # Video callback
    # -----------------------------------------------------------------

    def _video_refresh(self, data, width, height, pitch):
        if not data:
            return  # duped frame

        self.frame_width = width
        self.frame_height = height

        if self.hw_render and data == RETRO_HW_FRAME_BUFFER_VALID:
            # HW rendered frame: read pixels from OpenGL framebuffer
            self._read_gl_frame(width, height)
        elif data and data != RETRO_HW_FRAME_BUFFER_VALID:
            # Software rendered frame: direct pixel data
            self._read_sw_frame(data, width, height, pitch)

        if self.on_frame is not None and self.frame_data is not None:
            self.on_frame(self.frame_data)

    def _read_gl_frame(self, width, height):
        """Read frame from OpenGL framebuffer via glReadPixels."""
        from OpenGL import GL
        buf = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        # OpenGL origin is bottom-left, flip vertically
        if self.hw_bottom_left_origin:
            frame = frame[::-1].copy()
        self.frame_data = frame

    def _read_sw_frame(self, data, width, height, pitch):
        """Read frame from software-rendered pixel buffer."""
        buf = ctypes.string_at(data, pitch * height)

        if self.pixel_format == RETRO_PIXEL_FORMAT_XRGB8888:
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(height, pitch)
            pixels = raw[:, : width * 4].reshape(height, width, 4)
            self.frame_data = pixels[:, :, [2, 1, 0]].copy()

        elif self.pixel_format == RETRO_PIXEL_FORMAT_RGB565:
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(height, pitch)
            px = raw[:, : width * 2].view(np.uint16).reshape(height, width)
            r = ((px >> 11) & 0x1F).astype(np.uint8) * 255 // 31
            g = ((px >> 5) & 0x3F).astype(np.uint8) * 255 // 63
            b = (px & 0x1F).astype(np.uint8) * 255 // 31
            self.frame_data = np.stack([r, g, b], axis=2)

        else:  # 0RGB1555
            raw = np.frombuffer(buf, dtype=np.uint8).reshape(height, pitch)
            px = raw[:, : width * 2].view(np.uint16).reshape(height, width)
            r = ((px >> 10) & 0x1F).astype(np.uint8) * 255 // 31
            g = ((px >> 5) & 0x1F).astype(np.uint8) * 255 // 31
            b = (px & 0x1F).astype(np.uint8) * 255 // 31
            self.frame_data = np.stack([r, g, b], axis=2)

    # -----------------------------------------------------------------
    # Audio & Input callbacks
    # -----------------------------------------------------------------

    def _audio_sample(self, left, right):
        pass

    def _audio_sample_batch(self, data, frames):
        return frames

    def _input_poll(self):
        pass

    def _input_state(self, port, device, index, btn_id):
        if port != 0:
            return 0
        if device == RETRO_DEVICE_JOYPAD:
            for key, mapped_id in self.KEY_MAP.items():
                if mapped_id == btn_id and key in self.keys_pressed:
                    return 1
            return 0
        if device == RETRO_DEVICE_ANALOG:
            for key, (a_idx, a_id, value) in self.ANALOG_KEYS.items():
                if a_idx == index and a_id == btn_id and key in self.keys_pressed:
                    return value
            return 0
        return 0

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def init(self):
        version = self.core.retro_api_version()
        if version != RETRO_API_VERSION:
            raise RuntimeError(f"API version mismatch: {version}")

        # Create OpenGL window BEFORE core init -- the core may call
        # SET_HW_RENDER during init/load_game and needs get_proc_address
        # to resolve GL functions immediately.
        pygame.init()
        # Use compatibility profile -- the core may request RETRO_HW_CONTEXT_OPENGL
        # (not core profile), and gliden64 may use legacy GL functions.
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_COMPATIBILITY,
        )
        pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        self._screen = pygame.display.set_mode(
            (640, 480),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE,
        )
        print(f"OpenGL: {pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION)}."
              f"{pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION)}")
        pygame.display.set_caption("N64 Frontend - Initializing...")

        # Load SDL2 and OpenGL32 for GL proc address resolution
        self.sdl2 = _load_sdl2()
        self.sdl2.SDL_GL_GetProcAddress.restype = ctypes.c_void_p
        self.sdl2.SDL_GL_GetProcAddress.argtypes = [ctypes.c_char_p]
        self.opengl32 = ctypes.windll.opengl32
        self.opengl32.wglGetProcAddress.restype = ctypes.c_void_p
        self.opengl32.wglGetProcAddress.argtypes = [ctypes.c_char_p]

        self.core.retro_set_environment(self._env_cb)
        self.core.retro_init()
        self.core.retro_set_video_refresh(self._video_cb)
        self.core.retro_set_audio_sample(self._audio_cb)
        self.core.retro_set_audio_sample_batch(self._audio_batch_cb)
        self.core.retro_set_input_poll(self._input_poll_cb)
        self.core.retro_set_input_state(self._input_state_cb)

        info = RetroSystemInfo()
        self.core.retro_get_system_info(ctypes.byref(info))
        print(f"Core: {info.library_name.decode()} v{info.library_version.decode()}")
        self.need_fullpath = info.need_fullpath

    def load_game(self, rom_path):
        rom_path = os.path.abspath(rom_path)
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM not found: {rom_path}")

        game_info = RetroGameInfo()
        game_info.path = rom_path.encode("utf-8")
        game_info.meta = None

        with open(rom_path, "rb") as f:
            self._rom_data = f.read()
        self._rom_buf = ctypes.create_string_buffer(self._rom_data, len(self._rom_data))
        game_info.data = ctypes.cast(self._rom_buf, ctypes.c_void_p)
        game_info.size = len(self._rom_data)

        print(f"Loading ROM ({len(self._rom_data) / 1024 / 1024:.1f} MB)...")
        if not self.core.retro_load_game(ctypes.byref(game_info)):
            raise RuntimeError("Core failed to load ROM")

        av = RetroSystemAVInfo()
        self.core.retro_get_system_av_info(ctypes.byref(av))
        self.frame_width = av.geometry.base_width
        self.frame_height = av.geometry.base_height
        self.target_fps = av.timing.fps
        self.sample_rate = av.timing.sample_rate
        print(f"Video: {self.frame_width}x{self.frame_height} @ {self.target_fps:.1f} fps")

        self.core.retro_set_controller_port_device(0, RETRO_DEVICE_JOYPAD)

        # Notify core that OpenGL context is ready
        if self.hw_render and self.hw_context_reset:
            print("Signaling OpenGL context ready...")
            self.hw_context_reset()

    def _init_gl_display(self):
        """Set up OpenGL state for rendering captured frames as a textured quad."""
        from OpenGL import GL
        self._gl = GL
        self._gl_texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._gl_texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)

    def _draw_frame_gl(self, width, height):
        """Draw the current frame_data to the OpenGL window using legacy GL."""
        GL = self._gl
        if self.frame_data is None:
            return
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._gl_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB,
            self.frame_data.shape[1], self.frame_data.shape[0],
            0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self.frame_data.tobytes(),
        )
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, 1, 0, 1, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glViewport(0, 0, width, height)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glBegin(GL.GL_QUADS)
        GL.glTexCoord2f(0, 1); GL.glVertex2f(0, 0)
        GL.glTexCoord2f(1, 1); GL.glVertex2f(1, 0)
        GL.glTexCoord2f(1, 0); GL.glVertex2f(1, 1)
        GL.glTexCoord2f(0, 0); GL.glVertex2f(0, 1)
        GL.glEnd()

    def run(self):
        screen = self._screen
        disp_w, disp_h = screen.get_size()

        self._init_gl_display()
        pygame.display.set_caption("N64 Frontend")
        clock = pygame.time.Clock()
        self.running = True
        frame_count = 0
        fps_timer = time.time()
        actual_fps = 0.0

        print(f"\nRunning! Controls: Arrows=D-pad, WASD=Stick, X=A, Z=B")
        print(f"LShift=Z, Q/E=L/R, Enter=Start, IJKL=C-btns, ESC=Quit\n")

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_F1:
                        self.core.retro_reset()
                    else:
                        self.keys_pressed.add(event.key)
                elif event.type == pygame.KEYUP:
                    self.keys_pressed.discard(event.key)
                elif event.type == pygame.VIDEORESIZE:
                    disp_w, disp_h = event.w, event.h

            self.core.retro_run()

            if self.hw_render:
                # HW path: core rendered to GL context, also captured via glReadPixels
                pass
            else:
                # SW path: draw captured frame_data as GL texture
                self._draw_frame_gl(disp_w, disp_h)

            pygame.display.flip()

            frame_count += 1
            now = time.time()
            if now - fps_timer >= 1.0:
                actual_fps = frame_count / (now - fps_timer)
                frame_count = 0
                fps_timer = now
                mode = "GL" if self.hw_render else "SW"
                pygame.display.set_caption(
                    f"N64 Frontend [{mode}] - {actual_fps:.1f} fps | "
                    f"{self.frame_width}x{self.frame_height}"
                )

            clock.tick(self.target_fps)

        if self.hw_render and self.hw_context_destroy:
            self.hw_context_destroy()
        self.core.retro_unload_game()
        self.core.retro_deinit()
        pygame.quit()

    def get_frame(self):
        return self.frame_data

    RETRO_MEMORY_SYSTEM_RAM = 2

    def get_rdram(self):
        ptr = self.core.retro_get_memory_data(self.RETRO_MEMORY_SYSTEM_RAM)
        size = self.core.retro_get_memory_size(self.RETRO_MEMORY_SYSTEM_RAM)
        if ptr and size:
            return (ctypes.c_uint8 * size).from_address(ptr), size
        return None, 0


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python n64_frontend.py <rom_path> [core_path]")
        print()
        print("  rom_path    Path to N64 ROM (.z64, .n64, .v64)")
        print("  core_path   Path to mupen64plus_next_libretro.dll (optional)")
        sys.exit(1)

    rom_path = sys.argv[1]
    script_dir = Path(__file__).parent
    default_core = script_dir / "cores" / "parallel_n64_libretro.dll"
    core_path = sys.argv[2] if len(sys.argv) > 2 else str(default_core)

    if not os.path.exists(core_path):
        print(f"Error: Core not found at {core_path}")
        sys.exit(1)

    frontend = N64Frontend(core_path)

    # ---- StreamDiffusion integration point ----
    # frontend.on_frame = lambda rgb: your_pipeline(rgb)
    # rgb is numpy (H, W, 3) uint8 array
    # -------------------------------------------

    frontend.init()
    frontend.load_game(rom_path)
    frontend.run()


if __name__ == "__main__":
    main()
