
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler v3
===================
Adds:
  1) Graph JSON IR -> lowering to tiled schedule
  2) Liveness-based SRAM reuse allocator (linear-scan style)

Outputs:
  ./toy_out_v3/
    - sram_map.txt
    - prologue_packB.bin/.trace.txt
    - path_compute.bin/.trace.txt  + sim_path_compute.txt
    - path_bypass.bin/.trace.txt   + sim_path_bypass.txt
    - dispatch.json
    - normalized_graph.json

Graph idea:
  entry:
    if (c):
      y = MatMul(A,B) -> RequantReLU(per-channel) -> (optional) AddResidual(Skip) -> Store(Y)
    else:
      bypass: Store(Y, A)

Φ is implemented by firmware selecting one of two command buffers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import struct
import json
import os


# ============================================================
# 1) Command ABI (32B) + opcodes
# ============================================================

OP_DMA_LOAD_2D       = 0x01
OP_DMA_STORE_2D      = 0x02
OP_DMA_LOAD_LINEAR   = 0x03
OP_MAC_ACC32         = 0x10
OP_REQUANT_RELU_PC   = 0x21
OP_PACK_B            = 0x30
OP_ADD_I8_CLAMP      = 0x40   # OUT = OUT + SKIP (int8 sat add)

FLAG_RELU_ENABLE     = 0x01
FLAG_IS_2D           = 0x02
FLAG_ADD_RELU_MIN0   = 0x04   # (optional) enforce >= zC when zC==0


@dataclass
class Cmd32:
    opcode: int
    flags: int = 0
    wait0: int = 0
    wait1: int = 0
    sig0: int = 0
    sig1: int = 0
    src_addr: int = 0
    dst_addr: int = 0
    size_or_M: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0

    def pack(self) -> bytes:
        header = struct.pack(
            "<BBBBBBH",
            self.opcode & 0xFF,
            self.flags & 0xFF,
            self.wait0 & 0xFF,
            self.wait1 & 0xFF,
            self.sig0 & 0xFF,
            self.sig1 & 0xFF,
            0
        )
        payload = struct.pack(
            "<IIIIII",
            self.src_addr & 0xFFFFFFFF,
            self.dst_addr & 0xFFFFFFFF,
            self.size_or_M & 0xFFFFFFFF,
            self.arg0 & 0xFFFFFFFF,
            self.arg1 & 0xFFFFFFFF,
            self.arg2 & 0xFFFFFFFF
        )
        blob = header + payload
        assert len(blob) == 32
        return blob

    def to_trace(self) -> str:
        return (f"Cmd(op=0x{self.opcode:02X}, flags=0x{self.flags:02X}, "
                f"w0={self.wait0}, w1={self.wait1}, s0={self.sig0}, s1={self.sig1}, "
                f"src=0x{self.src_addr:08X}, dst=0x{self.dst_addr:08X}, "
                f"sz/M=0x{self.size_or_M:08X}, a0=0x{self.arg0:08X}, "
                f"a1=0x{self.arg1:08X}, a2=0x{self.arg2:08X})")


# ============================================================
# 2) Model spec / events
# ============================================================

@dataclass
class ModelSpec:
    A_base: int
    B_base: int
    Y_base: int
    Skip_base: int
    Bpack_base: int
    MS_base: int

    M: int = 64
    N: int = 64
    K: int = 64
    tileM: int = 32
    tileN: int = 32
    tileK: int = 64

    zA: int = 0
    zB: int = 0
    zC: int = 0
    relu: bool = True

    ms_entry_size: int = 8

    sram_size_bytes: int = 256 * 1024
    sram_align: int = 64


@dataclass
class Events:
    # We'll allocate events dynamically per tile, but keep a base
    base: int = 1


# ============================================================
# 3) Address helpers (row-major)
# ============================================================

def addr_A(m: ModelSpec, row: int, col: int) -> int:
    return m.A_base + (row * m.K + col) * 1

def addr_B(m: ModelSpec, row: int, col: int) -> int:
    return m.B_base + (row * m.N + col) * 1

def addr_Y(m: ModelSpec, row: int, col: int) -> int:
    return m.Y_base + (row * m.N + col) * 1

def addr_Skip(m: ModelSpec, row: int, col: int) -> int:
    return m.Skip_base + (row * m.N + col) * 1

def bpack_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    block_bytes = m.K * m.tileN
    return m.Bpack_base + n_tile_index * block_bytes

def ms_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.MS_base + n_tile_index * m.tileN * m.ms_entry_size


# ============================================================
# 4) Graph JSON IR
# ============================================================

DEFAULT_GRAPH = {
  "name": "if_matmul_requant_relu_residual",
  "inputs": {
    "A": {"dtype":"i8", "shape":[64,64]},
    "B": {"dtype":"i8", "shape":[64,64]},
    "Skip": {"dtype":"i8", "shape":[64,64]},
    "cond": {"dtype":"bool", "shape":[]}
  },
  "outputs": {
    "Y": {"dtype":"i8", "shape":[64,64]}
  },
  "if": {
    "cond": "cond",
    "then": [
      {"op":"PACK_B", "inputs":["B"], "outputs":["Bpack"], "attrs":{"tileN":32}},
      {"op":"MATMUL", "inputs":["A","Bpack"], "outputs":["ACC"], "attrs":{"acc_dtype":"i32"}},
      {"op":"REQUANT_RELU_PC", "inputs":["ACC"], "outputs":["Oq"], "attrs":{"per_channel_axis":"N"}},
      {"op":"ADD_RESIDUAL", "inputs":["Oq","Skip"], "outputs":["Oadd"], "attrs":{"clamp_min":-128,"clamp_max":127,"relu_min0":True}},
      {"op":"STORE", "inputs":["Oadd"], "outputs":["Y"]}
    ],
    "else": [
      {"op":"STORE", "inputs":["A"], "outputs":["Y"]}
    ]
  }
}


# ============================================================
# 5) Liveness-based SRAM allocator (linear scan)
# ============================================================

@dataclass
class LiveInterval:
    name: str
    start: int
    end: int
    size: int
    align: int
    addr: Optional[int] = None


class LinearScanSRAMAllocator:
    """
    Allocate SRAM addresses for values based on lifetimes.
    Reuse freed blocks when possible.
    Simple free-list by (addr,size).
    """

    def __init__(self, base: int, size: int, align: int):
        self.base = base
        self.size = size
        self.default_align = align
        self.free: List[Tuple[int,int]] = []   # list of (addr,size)
        self.bump = base
        self.alloc_map: Dict[str, Tuple[int,int]] = {}

    @staticmethod
    def _align_up(x: int, a: int) -> int:
        return (x + a - 1) // a * a

    def _try_alloc_from_free(self, nbytes: int, align: int) -> Optional[int]:
        # first-fit
        for i, (addr, sz) in enumerate(self.free):
            aaddr = self._align_up(addr, align)
            pad = aaddr - addr
            if sz >= pad + nbytes:
                # allocate from this block
                alloc_addr = aaddr
                # remaining before padding
                before = pad
                after = sz - (pad + nbytes)
                new_blocks = []
                if before > 0:
                    new_blocks.append((addr, before))
                if after > 0:
                    new_blocks.append((aaddr + nbytes, after))
                # replace this entry
                self.free.pop(i)
                self.free.extend(new_blocks)
                self.free.sort()
                return alloc_addr
        return None

    def alloc(self, name: str, nbytes: int, align: Optional[int] = None) -> int:
        if name in self.alloc_map:
            return self.alloc_map[name][0]
        a = align if align is not None else self.default_align

        addr = self._try_alloc_from_free(nbytes, a)
        if addr is None:
            addr = self._align_up(self.bump, a)
            end = addr + nbytes
            if end > self.base + self.size:
                raise MemoryError(f"SRAM overflow allocating {name} {nbytes} bytes")
            self.bump = end

        self.alloc_map[name] = (addr, nbytes)
        return addr

    def free_block(self, addr: int, nbytes: int) -> None:
        self.free.append((addr, nbytes))
        self.free.sort()
        # optional: coalesce adjacent blocks
        merged: List[Tuple[int,int]] = []
        for a, s in self.free:
            if not merged:
                merged.append((a,s))
                continue
            pa, ps = merged[-1]
            if pa + ps == a:
                merged[-1] = (pa, ps + s)
            else:
                merged.append((a,s))
        self.free = merged

    def dump(self) -> str:
        lines = ["SRAM Allocation Map (final addresses):"]
        for k, (addr, sz) in sorted(self.alloc_map.items(), key=lambda kv: kv[1][0]):
            lines.append(f"  {k:16s} @ 0x{addr:08X}  size={sz}")
        lines.append(f"  BUMP used: {self.bump - self.base} / {self.size}")
        lines.append(f"  Free blocks: {self.free}")
        return "\n".join(lines)


# ============================================================
# 6) Scheduler + Lowering (graph -> tiled op list)
# ============================================================

@dataclass
class TileTask:
    # A single tile work item for compute path
    tm: int   # tile index in M
    tn: int   # tile index in N
    # steps to run for this tile in order
    steps: List[str]


class GraphLowerer:
    """
    Convert graph JSON into a canonical plan:
      - Identify compute path ops: PACK_B(optional), MATMUL, REQUANT, ADD_RESIDUAL(optional), STORE
      - Build per-tile tasks for tile loop
      - else-path becomes bypass store (tile copy)
    """

    def __init__(self, graph: Dict[str, Any], model: ModelSpec):
        self.g = graph
        self.m = model

    def normalize(self) -> Dict[str, Any]:
        # minimal normalize: ensure tile sizes match model, etc.
        g = json.loads(json.dumps(self.g))
        # enforce tileN
        for n in g["if"]["then"]:
            if n["op"] == "PACK_B":
                n.setdefault("attrs", {})
                n["attrs"]["tileN"] = self.m.tileN
        return g

    def build_plan(self) -> Dict[str, Any]:
        g = self.normalize()
        then_ops = [n["op"] for n in g["if"]["then"]]
        else_ops = [n["op"] for n in g["if"]["else"]]

        # detect optional components
        use_pack = "PACK_B" in then_ops
        use_residual = "ADD_RESIDUAL" in then_ops

        plan = {
            "use_pack": use_pack,
            "use_residual": use_residual,
            "then_ops": g["if"]["then"],
            "else_ops": g["if"]["else"],
            "tiles": []
        }

        tm_count = self.m.M // self.m.tileM
        tn_count = self.m.N // self.m.tileN
        for tm in range(tm_count):
            for tn in range(tn_count):
                steps = ["LOAD_A", "LOAD_B", "MAC", "REQUANT"]
                if use_residual:
                    steps += ["LOAD_SKIP", "ADD"]
                steps += ["STORE_Y"]
                plan["tiles"].append(TileTask(tm=tm, tn=tn, steps=steps))
        return plan


# ============================================================
# 7) Command emission from plan (with overlap + liveness alloc)
# ============================================================

class CommandEmitter:
    """
    Build commands for:
      - prologue PACK_B (once)
      - compute path: tiled schedule with double buffers + optional skip + add
      - bypass path: tiled copy

    Also:
      - assigns SRAM buffers using liveness-based reuse allocator.
    """

    def __init__(self, model: ModelSpec):
        self.m = model

    def _bytes_A_tile(self) -> int:  return self.m.tileM * self.m.K * 1
    def _bytes_B_tile(self) -> int:  return self.m.K * self.m.tileN * 1
    def _bytes_OUT(self) -> int:     return self.m.tileM * self.m.tileN * 1
    def _bytes_ACC(self) -> int:     return self.m.tileM * self.m.tileN * 4
    def _bytes_SKIP(self) -> int:    return self.m.tileM * self.m.tileN * 1

    # ----- Build liveness for buffers we need in compute path -----
    def build_liveness(self, plan: Dict[str, Any]) -> List[LiveInterval]:
        """
        We model lifetimes over a flattened "instruction index" timeline in the tile loop.
        For a toy:
          - A buffer needed from LOAD_A until MAC consumes it
          - B buffer needed from LOAD_B until MAC consumes it
          - ACC needed from MAC until REQUANT consumes it
          - OUT needed from REQUANT until STORE (and ADD if residual)
          - SKIP needed from LOAD_SKIP until ADD consumes it
        We'll allocate:
          - 2x A buffers (ping-pong)
          - 2x B buffers
          - 2x SK buffers (only if residual)
          - 1x ACC
          - 1x OUT
        """
        use_residual = plan["use_residual"]

        # timeline slots: each tile has a fixed number of abstract steps
        # We'll treat each step as 1 "time unit" for liveness purposes.
        steps_per_tile = 5 + (2 if use_residual else 0)  # LOAD_A,LOAD_B,MAC,REQUANT,(LOAD_SKIP,ADD),STORE
        total = len(plan["tiles"]) * steps_per_tile

        def t_index(tile_i: int, step_i: int) -> int:
            return tile_i * steps_per_tile + step_i

        # Ping-pong buffers:
        # We use A0/A1 alternating tiles; same for B and SK.
        intervals: List[LiveInterval] = []

        # Define lifetimes for one tile instance (then repeated with offset)
        # We'll conservatively allocate fixed buffers with worst-case overlap:
        # - A0 and A1 each live roughly from LOAD_A to MAC of their tiles; overlap with next tile prefetch is possible.
        # But since we explicitly ping-pong, we can just allocate 2 A buffers permanently (start=0..end=total)
        # However to demonstrate reuse, we mark them as long-lived and allow allocator to pack others around.
        intervals.append(LiveInterval("A0", 0, total, self._bytes_A_tile(), self.m.sram_align))
        intervals.append(LiveInterval("A1", 0, total, self._bytes_A_tile(), self.m.sram_align))
        intervals.append(LiveInterval("B0", 0, total, self._bytes_B_tile(), self.m.sram_align))
        intervals.append(LiveInterval("B1", 0, total, self._bytes_B_tile(), self.m.sram_align))
        if use_residual:
            intervals.append(LiveInterval("SK0", 0, total, self._bytes_SKIP(), self.m.sram_align))
            intervals.append(LiveInterval("SK1", 0, total, self._bytes_SKIP(), self.m.sram_align))

        # ACC and OUT can be reused (single instance) across tiles sequentially
        intervals.append(LiveInterval("ACC", 0, total, self._bytes_ACC(), self.m.sram_align))
        intervals.append(LiveInterval("OUT", 0, total, self._bytes_OUT(), self.m.sram_align))

        return intervals

    def allocate_sram(self, intervals: List[LiveInterval]) -> Tuple[Dict[str,int], str]:
        alloc = LinearScanSRAMAllocator(base=0x0000, size=self.m.sram_size_bytes, align=self.m.sram_align)

        # Sort by start time; if equal, allocate larger first (better packing)
        intervals_sorted = sorted(intervals, key=lambda it: (it.start, -it.size))

        active: List[LiveInterval] = []

        def expire(old_t: int):
            nonlocal active
            still = []
            for it in active:
                if it.end <= old_t:
                    # free its block
                    assert it.addr is not None
                    alloc.free_block(it.addr, it.size)
                else:
                    still.append(it)
            active = still

        for it in intervals_sorted:
            expire(it.start)
            addr = alloc.alloc(it.name, it.size, it.align)
            it.addr = addr
            active.append(it)

        addrs = {it.name: it.addr for it in intervals_sorted if it.addr is not None}
        return addrs, alloc.dump()

    # ----- emit helpers -----
    def emit_pack_b(self) -> List[Cmd32]:
        m = self.m
        return [Cmd32(
            opcode=OP_PACK_B,
            src_addr=m.B_base,
            dst_addr=m.Bpack_base,
            size_or_M=m.K,
            arg0=m.N,
            arg1=m.tileN,
            arg2=0
        )]

    def emit_compute(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        use_residual = plan["use_residual"]

        A0, A1 = sram["A0"], sram["A1"]
        B0, B1 = sram["B0"], sram["B1"]
        ACC = sram["ACC"]
        OUT = sram["OUT"]
        SK0 = sram.get("SK0", 0)
        SK1 = sram.get("SK1", 0)

        cmds: List[Cmd32] = []
        ev = Events()
        next_event = ev.base

        def new_ev() -> int:
            nonlocal next_event
            x = next_event
            next_event += 1
            return x

        # Helper commands
        def dma_load_A(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.K
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.K, arg2=m.K)

        def dma_load_B(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.K * m.tileN
            return Cmd32(OP_DMA_LOAD_LINEAR, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_)

        def dma_load_SKIP(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def mac(A_sram: int, B_sram: int, waitA: int, waitB: int, sig: int) -> Cmd32:
            return Cmd32(OP_MAC_ACC32, wait0=waitA, wait1=waitB, sig0=sig,
                         src_addr=A_sram, dst_addr=ACC,
                         size_or_M=m.tileM, arg0=B_sram, arg1=m.tileN, arg2=m.tileK)

        def requant(ms_addr: int, wait: int, sig: int) -> Cmd32:
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(OP_REQUANT_RELU_PC, flags=flags, wait0=wait, sig0=sig,
                         src_addr=ACC, dst_addr=OUT,
                         size_or_M=m.tileM, arg0=m.tileN, arg1=ms_addr, arg2=0)

        def add_residual(skip_sram: int, wait_out: int, wait_sk: int, sig: int) -> Cmd32:
            clamp_min = (-128) & 0xFFFF
            clamp_max = (127) & 0xFFFF
            clamp_pack = (clamp_max << 16) | clamp_min
            flags = FLAG_ADD_RELU_MIN0 if (m.relu and m.zC == 0) else 0
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_ADD_I8_CLAMP, flags=flags, wait0=wait_out, wait1=wait_sk, sig0=sig,
                         src_addr=OUT, dst_addr=OUT,
                         size_or_M=bytes_, arg0=skip_sram, arg1=clamp_pack, arg2=0)

        def store(dst_dram: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=wait,
                         src_addr=OUT, dst_addr=dst_dram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        # Build tile list in same order as plan
        tiles: List[TileTask] = plan["tiles"]

        # Prefetch tile0
        t0 = tiles[0]
        EA0 = new_ev(); EB0 = new_ev()
        cmds.append(dma_load_A(A0, addr_A(m, t0.tm*m.tileM, 0), EA0))
        cmds.append(dma_load_B(B0, bpack_tile_addr(m, t0.tn), EB0))
        if use_residual:
            ESK0 = new_ev()
            cmds.append(dma_load_SKIP(SK0, addr_Skip(m, t0.tm*m.tileM, t0.tn*m.tileN), ESK0))
        else:
            ESK0 = 0

        # Tile loop with overlap
        for i, t in enumerate(tiles):
            use_A = A0 if (i % 2 == 0) else A1
            use_B = B0 if (i % 2 == 0) else B1
            waitA = EA0 if (i % 2 == 0) else EA1
            waitB = EB0 if (i % 2 == 0) else EB1

            if use_residual:
                use_SK = SK0 if (i % 2 == 0) else SK1
                waitSK = ESK0 if (i % 2 == 0) else ESK1
            else:
                use_SK = 0
                waitSK = 0

            EACC = new_ev()
            cmds.append(mac(use_A, use_B, waitA, waitB, EACC))

            # Prefetch next tile into alternate buffers
            if i + 1 < len(tiles):
                tnext = tiles[i+1]
                alt_A = A1 if (i % 2 == 0) else A0
                alt_B = B1 if (i % 2 == 0) else B0
                EA1 = new_ev() if (i % 2 == 0) else EA0  # placeholders overwritten below
                EB1 = new_ev() if (i % 2 == 0) else EB0

                # careful: set the correct event variables for next iteration
                if i % 2 == 0:
                    # next will use A1,B1
                    EA1 = EA1
                    EB1 = EB1
                else:
                    # next will use A0,B0
                    EA0 = EA0
                    EB0 = EB0

                # emit loads
                newAev = EA1 if (i % 2 == 0) else EA0
                newBev = EB1 if (i % 2 == 0) else EB0
                cmds.append(dma_load_A(alt_A, addr_A(m, tnext.tm*m.tileM, 0), newAev))
                cmds.append(dma_load_B(alt_B, bpack_tile_addr(m, tnext.tn), newBev))

                if use_residual:
                    alt_SK = SK1 if (i % 2 == 0) else SK0
                    newSkEv = new_ev()
                    if i % 2 == 0:
                        ESK1 = newSkEv
                    else:
                        ESK0 = newSkEv
                    cmds.append(dma_load_SKIP(alt_SK, addr_Skip(m, tnext.tm*m.tileM, tnext.tn*m.tileN), newSkEv))

            # Requant
            ms_addr = ms_tile_addr(m, t.tn)
            EOUT = new_ev()
            cmds.append(requant(ms_addr, EACC, EOUT))

            # Optional residual add
            if use_residual:
                EADD = new_ev()
                cmds.append(add_residual(use_SK, EOUT, waitSK, EADD))
                wait_store = EADD
            else:
                wait_store = EOUT

            # Store
            ydst = addr_Y(m, t.tm*m.tileM, t.tn*m.tileN)
            cmds.append(store(ydst, wait_store))

        return cmds

    def emit_bypass(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        """
        Else path STORE(A)->Y, tiled + overlap.
        Use A0/A1 buffers (already in sram map).
        """
        m = self.m
        A0, A1 = sram["A0"], sram["A1"]

        cmds: List[Cmd32] = []
        next_event = 1

        def new_ev() -> int:
            nonlocal next_event
            x = next_event
            next_event += 1
            return x

        def dma_load_tile(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def dma_store_tile(dst_dram: int, src_sram: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=wait,
                         src_addr=src_sram, dst_addr=dst_dram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]
        # prefetch first tile into A0
        t0 = tiles[0]
        EA0 = new_ev()
        cmds.append(dma_load_tile(A0, addr_A(m, t0.tm*m.tileM, t0.tn*m.tileN), EA0))

        for i, t in enumerate(tiles):
            use_A = A0 if (i % 2 == 0) else A1
            waitA = EA0 if (i % 2 == 0) else EA1

            if i + 1 < len(tiles):
                tnext = tiles[i+1]
                alt_A = A1 if (i % 2 == 0) else A0
                newAev = new_ev()
                if i % 2 == 0:
                    EA1 = newAev
                else:
                    EA0 = newAev
                cmds.append(dma_load_tile(alt_A, addr_A(m, tnext.tm*m.tileM, tnext.tn*m.tileN), newAev))

            ydst = addr_Y(m, t.tm*m.tileM, t.tn*m.tileN)
            cmds.append(dma_store_tile(ydst, use_A, waitA))

        return cmds


# ============================================================
# 8) Simulator (same spirit as v2)
# ============================================================

@dataclass
class EngineState:
    busy_until: int = 0

class CommandSimulator:
    def __init__(self):
        self.events: Dict[int,bool] = {}
        self.t = 0
        self.dma = EngineState()
        self.mac = EngineState()
        self.act = EngineState()

    def _engine(self, c: Cmd32) -> str:
        if c.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            return "DMA"
        if c.opcode == OP_MAC_ACC32:
            return "MAC"
        return "ACT"

    def _dur(self, c: Cmd32) -> int:
        if c.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            return max(1, c.size_or_M // 256)
        if c.opcode == OP_MAC_ACC32:
            M = c.size_or_M; N = c.arg1; K = c.arg2
            return max(1, (M*N*K) // 4096)
        if c.opcode == OP_PACK_B:
            K = c.size_or_M; N = c.arg0
            return max(1, (K*N)//1024)
        return max(1, c.size_or_M // 512)

    def _busy_until(self, eng: str) -> int:
        return {"DMA":self.dma.busy_until, "MAC":self.mac.busy_until, "ACT":self.act.busy_until}[eng]

    def _set_busy(self, eng: str, t: int):
        if eng=="DMA": self.dma.busy_until = t
        elif eng=="MAC": self.mac.busy_until = t
        else: self.act.busy_until = t

    def _waits_ok(self, c: Cmd32) -> bool:
        for w in (c.wait0, c.wait1):
            if w and not self.events.get(w, False):
                return False
        return True

    def run(self, cmds: List[Cmd32]) -> str:
        self.events.clear()
        self.t = 0
        self.dma = EngineState(); self.mac = EngineState(); self.act = EngineState()
        lines: List[str] = []

        for i, c in enumerate(cmds):
            eng = self._engine(c)
            while True:
                if self._waits_ok(c) and self.t >= self._busy_until(eng):
                    break
                # advance
                if self.t < self._busy_until(eng):
                    self.t = self._busy_until(eng)
                else:
                    self.t += 1

            start = self.t
            end = start + self._dur(c)
            self._set_busy(eng, end)

            # signal immediately (toy) – you can change to "signal at end" if you want stricter sim
            if c.sig0: self.events[c.sig0] = True
            if c.sig1: self.events[c.sig1] = True

            lines.append(f"{i:04d} t={start:6d}..{end:6d} {eng:3s} {c.to_trace()}")

            # allow overlap: keep global time
            self.t = start

        makespan = max(self.dma.busy_until, self.mac.busy_until, self.act.busy_until)
        lines.append("")
        lines.append(f"SIM SUMMARY: DMA_end={self.dma.busy_until}, MAC_end={self.mac.busy_until}, ACT_end={self.act.busy_until}, makespan={makespan}")
        lines.append(f"Events signaled: {sorted([k for k,v in self.events.items() if v])}")
        return "\n".join(lines)


# ============================================================
# 9) IO helpers
# ============================================================

def write_bundle(out_dir: str, name: str, cmds: List[Cmd32]) -> None:
    bin_path = os.path.join(out_dir, f"{name}.bin")
    trace_path = os.path.join(out_dir, f"{name}.trace.txt")
    with open(bin_path, "wb") as fb, open(trace_path, "w", encoding="utf-8") as ft:
        for i, c in enumerate(cmds):
            fb.write(c.pack())
            ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 10) Main
# ============================================================

def main():
    out_dir = "./toy_out_v3"
    os.makedirs(out_dir, exist_ok=True)

    # Spec (you can change addresses)
    model = ModelSpec(
        A_base=0x8000_0000,
        B_base=0x8000_2000,
        Y_base=0x8000_4000,
        Skip_base=0x8000_9000,
        Bpack_base=0x8000_6000,
        MS_base=0x8000_7000,
        M=64, N=64, K=64,
        tileM=32, tileN=32, tileK=64,
        zA=0, zB=0, zC=0,
        relu=True,
        ms_entry_size=8,
        sram_size_bytes=256*1024,
        sram_align=64
    )

    # Load graph (if graph.json exists) else use default
    graph_path = "./graph.json"
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    else:
        graph = DEFAULT_GRAPH

    lowerer = GraphLowerer(graph, model)
    norm = lowerer.normalize()
    with open(os.path.join(out_dir, "normalized_graph.json"), "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2)

    plan = lowerer.build_plan()

    emitter = CommandEmitter(model)

    # Liveness + alloc
    intervals = emitter.build_liveness(plan)
    sram_addrs, sram_dump = emitter.allocate_sram(intervals)
    with open(os.path.join(out_dir, "sram_map.txt"), "w", encoding="utf-8") as f:
        f.write(sram_dump + "\n")

    # Emit command lists
    prologue = emitter.emit_pack_b() if plan["use_pack"] else []
    path_compute = emitter.emit_compute(plan, sram_addrs)
    path_bypass = emitter.emit_bypass(plan, sram_addrs)

    write_bundle(out_dir, "prologue_packB", prologue)
    write_bundle(out_dir, "path_compute", path_compute)
    write_bundle(out_dir, "path_bypass", path_bypass)

    dispatch = {
        "phi_strategy": "firmware_select_path",
        "optional_prologue": "prologue_packB.bin" if plan["use_pack"] else None,
        "paths": {
            "if_c_true": "path_compute.bin",
            "if_c_false": "path_bypass.bin"
        },
        "notes": {
            "compute": "tiled MatMul->RequantReLU(per-channel)->(optional)ResidualAdd->Store",
            "bypass": "tiled Store(A)->Y",
            "sram_alloc": "linear-scan liveness-based reuse allocator (toy)"
        }
    }
    with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch, f, indent=2)

    # Simulate
    sim = CommandSimulator()
    with open(os.path.join(out_dir, "sim_path_compute.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(path_compute) + "\n")
    with open(os.path.join(out_dir, "sim_path_bypass.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(path_bypass) + "\n")

    print("Generated:", out_dir)
    print("  - normalized_graph.json")
    print("  - sram_map.txt (liveness-based alloc)")
    print("  - prologue_packB.bin/.trace.txt")
    print("  - path_compute.bin/.trace.txt + sim_path_compute.txt")
    print("  - path_bypass.bin/.trace.txt  + sim_path_bypass.txt")
    print("  - dispatch.json")
    print("\nEN: Put your own graph in ./graph.json to override.")
    print("中: 你可以在当前目录放一个 graph.json 覆盖默认图。")


if __name__ == "__main__":
    main()

