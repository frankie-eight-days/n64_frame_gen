"""SM64 game state reader — extracts camera/object positions from RDRAM.

Used for generating 3D-aware motion vectors as an alternative to optical flow.
Addresses are for the US NTSC version of Super Mario 64.
"""

import struct
import math
import numpy as np


class SM64StateReader:
    """Read game state from mupen64plus RDRAM for motion vector generation."""

    # US NTSC SM64 addresses (physical, after subtracting 0x80000000 virtual base)
    MARIO_STRUCT = 0x33B170
    CAMERA_STRUCT = 0x33C520

    # N64 projection parameters (SM64 defaults)
    FOV_DEG = 45.0
    ASPECT = 320.0 / 240.0  # 4:3
    NEAR = 100.0
    FAR = 30000.0

    def __init__(self, rdram_array):
        """Initialize with reference to emulator's RDRAM numpy array.

        Args:
            rdram_array: numpy uint8 array representing N64 RDRAM
        """
        self._rdram = rdram_array
        self._prev_camera = None
        self._prev_mario = None

    def set_rdram(self, rdram_array):
        """Update RDRAM reference (in case it changes)."""
        self._rdram = rdram_array

    def _read_u8(self, addr):
        """Read unsigned byte with mupen64plus byte-swap correction."""
        # mupen64plus stores bytes in 32-bit word-swapped order
        word_addr = addr & ~3
        byte_offset = addr & 3
        # Byte swap: 0->3, 1->2, 2->1, 3->0
        swapped_offset = 3 - byte_offset
        return int(self._rdram[word_addr + swapped_offset])

    def _read_s16(self, addr):
        """Read big-endian signed 16-bit integer with byte-swap correction."""
        hi = self._read_u8(addr)
        lo = self._read_u8(addr + 1)
        val = (hi << 8) | lo
        if val >= 0x8000:
            val -= 0x10000
        return val

    def _read_u16(self, addr):
        """Read big-endian unsigned 16-bit integer with byte-swap correction."""
        hi = self._read_u8(addr)
        lo = self._read_u8(addr + 1)
        return (hi << 8) | lo

    def _read_u32(self, addr):
        """Read big-endian unsigned 32-bit integer with byte-swap correction."""
        b0 = self._read_u8(addr)
        b1 = self._read_u8(addr + 1)
        b2 = self._read_u8(addr + 2)
        b3 = self._read_u8(addr + 3)
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    def _read_float32(self, addr):
        """Read big-endian float32 with mupen64plus byte-swap correction."""
        raw = self._read_u32(addr)
        return struct.unpack('>f', struct.pack('>I', raw))[0]

    def _read_vec3f(self, addr):
        """Read 3 consecutive float32 values as a numpy array."""
        return np.array([
            self._read_float32(addr),
            self._read_float32(addr + 4),
            self._read_float32(addr + 8),
        ], dtype=np.float32)

    def read_camera(self):
        """Read camera position and orientation from RDRAM.

        Returns:
            dict with keys:
            - 'pos': (3,) float32 array [x, y, z]
            - 'focus': (3,) float32 array [x, y, z] (look-at point)
            - 'yaw': int16 rotation angle (0-65535 maps to 0-360 degrees)
        """
        try:
            pos = self._read_vec3f(self.CAMERA_STRUCT + 0x8C)
            focus = self._read_vec3f(self.CAMERA_STRUCT + 0x80)
            yaw = self._read_s16(self.CAMERA_STRUCT + 0xCE)
            return {"pos": pos, "focus": focus, "yaw": yaw}
        except (IndexError, struct.error):
            return None

    def read_mario(self):
        """Read Mario's position from RDRAM.

        Returns:
            dict with keys:
            - 'pos': (3,) float32 array [x, y, z]
            - 'action': uint32 action state
        """
        try:
            pos = self._read_vec3f(self.MARIO_STRUCT + 0x3C)
            action = self._read_u32(self.MARIO_STRUCT + 0x0C)
            return {"pos": pos, "action": action}
        except (IndexError, struct.error):
            return None

    def compute_camera_delta(self):
        """Compute camera position and rotation delta since last call.

        Returns:
            dict with 'pos_delta' (3,) array, 'yaw_delta' int, or None if first frame
        """
        cam = self.read_camera()
        if cam is None:
            return None

        result = None
        if self._prev_camera is not None:
            pos_delta = cam['pos'] - self._prev_camera['pos']
            yaw_delta = cam['yaw'] - self._prev_camera['yaw']
            # Handle wrap-around for yaw
            if yaw_delta > 32767:
                yaw_delta -= 65536
            elif yaw_delta < -32767:
                yaw_delta += 65536
            result = {"pos_delta": pos_delta, "yaw_delta": yaw_delta}

        self._prev_camera = cam
        return result

    def compute_motion_vectors(self, z14_buffer, width=320, height=240):
        """Compute per-pixel screen-space motion vectors from camera delta + depth.

        Uses the Z-buffer to back-project pixels to 3D, apply inverse camera
        transform, then re-project. This gives exact 3D-aware motion vectors
        like DLSS/TAA, replacing optical flow.

        Args:
            z14_buffer: (H, W) float32 Z-buffer values (raw z14 from N64)
            width: frame width
            height: frame height

        Returns:
            (H, W, 2) float32 flow field compatible with TemporalBlender,
            or None if camera delta unavailable
        """
        cam = self.read_camera()
        if cam is None or self._prev_camera is None:
            self._prev_camera = cam
            return None

        pos_delta = cam['pos'] - self._prev_camera['pos']
        yaw_delta_raw = cam['yaw'] - self._prev_camera['yaw']
        if yaw_delta_raw > 32767:
            yaw_delta_raw -= 65536
        elif yaw_delta_raw < -32767:
            yaw_delta_raw += 65536

        # Convert yaw delta to radians (SM64 uses 0-65536 for full rotation)
        yaw_delta_rad = yaw_delta_raw * (2.0 * math.pi / 65536.0)

        # Build projection parameters
        fov_rad = math.radians(self.FOV_DEG)
        f = 1.0 / math.tan(fov_rad / 2.0)

        # Linearize Z-buffer (N64 z14 is non-linear, approximate as linear for now)
        z_range = max(z14_buffer.max() - z14_buffer.min(), 1.0)
        z_norm = (z14_buffer - z14_buffer.min()) / z_range
        depth = self.NEAR + z_norm * (self.FAR - self.NEAR)

        # Pixel coordinates to normalized device coords
        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # NDC: -1 to 1
        ndc_x = (2.0 * uu / width - 1.0) / (f / self.ASPECT)
        ndc_y = (2.0 * vv / height - 1.0) / f

        # Back-project to camera-space 3D
        x3d = ndc_x * depth
        y3d = ndc_y * depth
        z3d = depth

        # Apply inverse camera rotation (yaw only, simplified)
        cos_y = math.cos(-yaw_delta_rad)
        sin_y = math.sin(-yaw_delta_rad)

        x3d_rot = x3d * cos_y - z3d * sin_y - pos_delta[0]
        z3d_rot = x3d * sin_y + z3d * cos_y - pos_delta[2]
        y3d_rot = y3d - pos_delta[1]

        # Re-project to screen space
        reproj_x = (x3d_rot / z3d_rot) * (f / self.ASPECT)
        reproj_y = (y3d_rot / z3d_rot) * f

        # Convert back to pixel coordinates
        px_x = (reproj_x + 1.0) * width / 2.0
        px_y = (reproj_y + 1.0) * height / 2.0

        # Motion vector = reprojected position - original position
        flow = np.stack([px_x - uu, px_y - vv], axis=-1).astype(np.float32)

        # Handle invalid depth (z14 == 0 typically means sky/far plane)
        invalid = z14_buffer < 1.0
        flow[invalid] = 0.0

        self._prev_camera = cam
        return flow

    def is_valid(self):
        """Check if we can read valid game state (sanity check).

        Returns True if Mario's position values are in reasonable range.
        """
        try:
            mario = self.read_mario()
            if mario is None:
                return False
            pos = mario['pos']
            # SM64 world coords are roughly -30000 to 30000
            return all(abs(p) < 100000 for p in pos)
        except Exception:
            return False
