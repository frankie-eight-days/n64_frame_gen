"""Automated training data capture for SM64.

Injects random controller inputs and warps between levels to generate
diverse training frames. Works by manipulating the frontend's keys_pressed
set and writing level warp values to RDRAM.

SM64 US NTSC addresses (physical = virtual - 0x80000000):
  gCurrLevelNum:  0x33B249
  gCurrAreaIndex: 0x33B24A
  gMarioState->action: 0x33B17C (u32)

Usage:
    auto = AutoCapture(frontend, processor)
    # In main loop:
    auto.tick()
"""

import time
import random
import pygame


# SM64 US v1.0 level IDs (physical RDRAM addresses)
ADDR_LEVEL = 0x33B249
ADDR_AREA = 0x33B24A

# SM64 levels with good visual diversity
SM64_LEVELS = [
    (9, 1, "Bob-omb Battlefield"),
    (24, 1, "Whomp's Fortress"),
    (12, 1, "Jolly Roger Bay"),
    (5, 1, "Cool Cool Mountain"),
    (4, 1, "Big Boo's Haunt"),
    (7, 1, "Hazy Maze Cave"),
    (22, 1, "Lethal Lava Land"),
    (8, 1, "Shifting Sand Land"),
    (23, 1, "Dire Dire Docks"),
    (10, 1, "Snowman's Land"),
    (11, 1, "Wet-Dry World"),
    (36, 1, "Tall Tall Mountain"),
    (13, 1, "Tiny-Huge Island"),
    (14, 1, "Tick Tock Clock"),
    (15, 1, "Rainbow Ride"),
    (6, 1, "Castle Inside"),
    (16, 1, "Castle Grounds"),
    (26, 1, "Castle Courtyard"),
    (17, 1, "Bowser Dark World"),
    (19, 1, "Bowser Fire Sea"),
    (21, 1, "Bowser Sky"),
    (27, 1, "Peach's Slide"),
    (20, 1, "Wing Cap"),
    (18, 1, "Metal Cap"),
    (28, 1, "Secret Aquarium"),
]

# Pygame keys we can inject for movement
MOVE_KEYS = [pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d]
ACTION_KEYS = [pygame.K_x, pygame.K_z]  # A, B buttons
CAMERA_KEYS = [pygame.K_i, pygame.K_j, pygame.K_k, pygame.K_l]  # C-buttons


class AutoCapture:
    """Automated controller input + level warping for diverse frame capture."""

    def __init__(self, frontend, processor, warp_interval=20.0, input_change_interval=1.5):
        """
        Args:
            frontend: N64Frontend instance (for keys_pressed injection)
            processor: DiffusionProcessor instance (for RDRAM access + capture control)
            warp_interval: seconds between level warps
            input_change_interval: seconds between random input changes
        """
        self.frontend = frontend
        self.processor = processor
        self.warp_interval = warp_interval
        self.input_change_interval = input_change_interval

        self._active = False
        self._last_warp = 0
        self._last_input_change = 0
        self._level_index = 0
        self._injected_keys = set()

        # Shuffle level order for variety
        self._level_order = list(range(len(SM64_LEVELS)))
        random.shuffle(self._level_order)

    @property
    def active(self):
        return self._active

    def start(self):
        """Start auto-capture. Enables frame capture and begins input injection."""
        self._active = True
        self.processor.capture_enabled = True
        self._last_warp = time.time()
        self._last_input_change = time.time()
        self._level_index = 0
        random.shuffle(self._level_order)
        print(f"Auto-capture started — {len(SM64_LEVELS)} levels, "
              f"warp every {self.warp_interval}s")

    def stop(self):
        """Stop auto-capture."""
        self._active = False
        self.processor.capture_enabled = False
        self._clear_injected_keys()
        print(f"Auto-capture stopped — {self.processor._capture_saved} frames saved")

    def toggle(self):
        if self._active:
            self.stop()
        else:
            self.start()

    def tick(self):
        """Call once per main loop iteration. Handles input injection and level warping."""
        if not self._active:
            return

        now = time.time()

        # Warp to next level periodically
        if now - self._last_warp >= self.warp_interval:
            self._warp_next_level()
            self._last_warp = now

        # Change random inputs periodically
        if now - self._last_input_change >= self.input_change_interval:
            self._randomize_inputs()
            self._last_input_change = now

    def _write_u8(self, addr, value):
        """Write a byte to RDRAM with mupen64plus byte-swap correction."""
        rdram = self.processor._rdram
        if rdram is None:
            return
        word_addr = addr & ~3
        byte_offset = addr & 3
        swapped_offset = 3 - byte_offset
        rdram[word_addr + swapped_offset] = value & 0xFF

    def _warp_next_level(self):
        """Warp to the next level in the shuffled order."""
        if self.processor._rdram is None:
            return

        idx = self._level_order[self._level_index % len(self._level_order)]
        level_id, area_id, name = SM64_LEVELS[idx]
        self._level_index += 1

        # Reshuffle when we've gone through all levels
        if self._level_index >= len(self._level_order):
            self._level_index = 0
            random.shuffle(self._level_order)

        self._write_u8(ADDR_LEVEL, level_id)
        self._write_u8(ADDR_AREA, area_id)
        print(f"  Warped to: {name} (level {level_id})")

    def _randomize_inputs(self):
        """Inject random controller inputs for diverse camera angles and movement."""
        self._clear_injected_keys()

        # Random movement (0-2 directions)
        n_move = random.randint(0, 2)
        for key in random.sample(MOVE_KEYS, min(n_move, len(MOVE_KEYS))):
            self._inject_key(key)

        # Random actions (occasional jump/attack)
        if random.random() < 0.3:
            self._inject_key(random.choice(ACTION_KEYS))

        # Random camera rotation
        if random.random() < 0.5:
            self._inject_key(random.choice(CAMERA_KEYS))

    def _inject_key(self, key):
        """Add a key to the frontend's pressed set."""
        self.frontend.keys_pressed.add(key)
        self._injected_keys.add(key)

    def _clear_injected_keys(self):
        """Remove all auto-injected keys."""
        for key in self._injected_keys:
            self.frontend.keys_pressed.discard(key)
        self._injected_keys.clear()
