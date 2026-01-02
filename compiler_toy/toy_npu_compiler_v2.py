
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler Skeleton v2
============================
Adds:
  (i) Command interpreter / simulator (events + simple engine overlap checks)
  (ii) Simple SRAM allocator (aligned linear allocator by buffer class)
  (iii) Add an op: RESIDUAL_ADD (OUT = OUT + SKIP) (int8 add w/ clamp) and show lowering

Note:
  - This is still a toy. Real systems have queues, priorities, prefetch depth, bank conflicts, etc.
  - Here we simulate correctness: waits satisfied, events signaled, and rough engine concurrency constraints.

Output:
  ./toy_out_v2/
      prologue_packB.bin/.trace.txt
      path_compute.bin/.trace.txt
      path_bypass.bin/.trace.txt
      dispatch.json
      sim_path_compute.txt
      sim_path_bypass.txt
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
OP_ADD_I8_CLAMP      = 0x40   # NEW: residual add (int8) with clamp, optional ReLU-like clamp min

FLAG_RELU_ENABLE     = 0x01
FLAG_IS_2D           = 0x02
FLAG_ADD_RELU_MIN0   = 0x04   # NEW: in quant domain, enforce >= zC (often 0)

# 32B layout:
# <BBBBBBH> header + 24 bytes payload:
# src_addr(u32), dst_addr(u32), size_or_M(u32), arg0(u32), arg1(u32), arg2(u32)

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
        def u8(x):  return x & 0xFF
        def u16(x): return x & 0xFFFF
        def u32(x): return x & 0xFFFFFFFF

        header = struct.pack(
            "<BBBBBBH",
            u8(self.opcode),
            u8(self.flags),
            u8(self.wait0),
            u8(self.wait1),
            u8(self.sig0),
            u8(self.sig1),
            u16(0),
        )
        payload = struct.pack(
            "<IIIIII",
            u32(self.src_addr),
            u32(self.dst_addr),
            u32(self.size_or_M),
            u32(self.arg0),
            u32(self.arg1),
            u32(self.arg2),
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
# 2) Model / Spec
# ============================================================

@dataclass
class ModelSpec:
    # DRAM base addresses
    A_base: int          # input A (int8)
    B_base: int          # weights B (int8, row-major)
    Y_base: int          # output Y (int8)
    Bpack_base: int      # packed weights
    MS_base: int         # per-channel mult/shift table (for N dimension)
    Skip_base: int       # NEW: residual/skip tensor base in DRAM (int8) same shape as Y

    # Shapes
    M: int = 64
    N: int = 64
    K: int = 64
    tileM: int = 32
    tileN: int = 32
    tileK: int = 64

    # Quant/relu
    zA: int = 0
    zB: int = 0
    zC: int = 0
    relu: bool = True

    # per-channel param table
    ms_entry_size: int = 8  # bytes per output-channel entry

    # SRAM constraints (toy)
    sram_size_bytes: int = 256 * 1024
    sram_align: int = 64


@dataclass
class Events:
    EA0: int = 1
    EB0: int = 2
    EA1: int = 3
    EB1: int = 4
    EACC: int = 5
    EOUT: int = 6
    ESK0: int = 7     # NEW: skip tile ready (buffer0)
    ESK1: int = 8     # NEW: skip tile ready (buffer1)
    EADD: int = 9     # NEW: add done


# ============================================================
# 3) Auto SRAM allocator (toy)
# ============================================================

class SRAMAllocator:
    """
    Very simple allocator:
      - allocate named buffers in order with alignment
      - returns SRAM addresses
    Real allocator would be liveness-based with reuse. This is a scaffold.
    """
    def __init__(self, base: int, size: int, align: int):
        self.base = base
        self.size = size
        self.align = align
        self.cur = base
        self.allocs: Dict[str, Tuple[int, int]] = {}  # name -> (addr, size)

    def _align_up(self, x: int, a: int) -> int:
        return (x + a - 1) // a * a

    def alloc(self, name: str, nbytes: int, align: Optional[int] = None) -> int:
        if name in self.allocs:
            raise ValueError(f"SRAM buffer '{name}' already allocated")
        a = align if align is not None else self.align
        addr = self._align_up(self.cur, a)
        end = addr + nbytes
        if end > self.base + self.size:
            raise MemoryError(f"SRAM overflow: need {nbytes} bytes for {name}, "
                              f"used {addr-self.base} / {self.size}")
        self.allocs[name] = (addr, nbytes)
        self.cur = end
        return addr

    def dump(self) -> str:
        lines = ["SRAM Allocation Map:"]
        for k, (addr, sz) in self.allocs.items():
            lines.append(f"  {k:10s} @ 0x{addr:08X}  size={sz}")
        lines.append(f"  TOTAL used: {self.cur - self.base} / {self.size}")
        return "\n".join(lines)


@dataclass
class SRAMLayout:
    # double buffers for A/B and NEW skip double buffers
    A0: int
    A1: int
    B0: int
    B1: int
    SK0: int
    SK1: int
    ACC: int
    OUT: int


# ============================================================
# 4) Address helpers (row-major)
# ============================================================

def addr_A(model: ModelSpec, m: int, k: int) -> int:
    return model.A_base + (m * model.K + k) * 1

def addr_Skip(model: ModelSpec, m: int, n: int) -> int:
    return model.Skip_base + (m * model.N + n) * 1

def addr_Y(model: ModelSpec, m: int, n: int) -> int:
    return model.Y_base + (m * model.N + n) * 1

def bpack_tile_addr(model: ModelSpec, n_tile_index: int) -> int:
    block_bytes = model.K * model.tileN
    return model.Bpack_base + n_tile_index * block_bytes

def ms_tile_addr(model: ModelSpec, n_tile_index: int) -> int:
    return model.MS_base + n_tile_index * model.tileN * model.ms_entry_size


# ============================================================
# 5) Toy compiler (emits command lists)
# ============================================================

class ToyNPUCompilerV2:
    def __init__(self, model: ModelSpec, sram: SRAMLayout, ev: Events):
        self.m = model
        self.s = sram
        self.e = ev

    def emit_pack_b(self) -> List[Cmd32]:
        m = self.m
        return [Cmd32(
            opcode=OP_PACK_B,
            src_addr=m.B_base,
            dst_addr=m.Bpack_base,
            size_or_M=m.K,
            arg0=m.N,
            arg1=m.tileN,
            arg2=0,
        )]

    def emit_path_compute_with_residual(self) -> List[Cmd32]:
        """
        Compute path now:
          OUT = RequantReLU( MatMul(A,B) )          # OUT in SRAM
          SK  = load Skip tile (int8) into SRAM     # SK in SRAM
          OUT = clamp(OUT + SK)                     # residual add
          store OUT -> Y

        Still tiled + double-buffer overlap for A and B.
        Also double-buffer SK to overlap skip DMA with MAC.
        """
        m, s, e = self.m, self.s, self.e
        cmds: List[Cmd32] = []

        tiles = [(0,0), (0,1), (1,0), (1,1)]

        def dma_load_A(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.K
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.K, arg2=m.K)

        def dma_load_B_linear(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.K * m.tileN
            return Cmd32(OP_DMA_LOAD_LINEAR, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_)

        def dma_load_skip_tile(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            # 32x32 tile from Skip (int8) with stride N
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src_dram, dst_addr=dst_sram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def mac_acc32(dst_acc: int, A_sram: int, B_sram: int, waitA: int, waitB: int, sig: int) -> Cmd32:
            return Cmd32(OP_MAC_ACC32, wait0=waitA, wait1=waitB, sig0=sig,
                         src_addr=A_sram, dst_addr=dst_acc,
                         size_or_M=m.tileM, arg0=B_sram, arg1=m.tileN, arg2=m.tileK)

        def requant_relu_pc(dst_out: int, src_acc: int, ms_addr: int, wait: int, sig: int) -> Cmd32:
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(OP_REQUANT_RELU_PC, flags=flags, wait0=wait, sig0=sig,
                         src_addr=src_acc, dst_addr=dst_out,
                         size_or_M=m.tileM, arg0=m.tileN, arg1=ms_addr, arg2=0)

        def add_i8_clamp_inplace(dst_out: int, src_out: int, src_skip: int, wait0: int, wait1: int, sig: int) -> Cmd32:
            # Convention:
            #  src_addr = src_out (SRAM)
            #  dst_addr = dst_out (SRAM) (can be same as src_out)
            #  size_or_M = bytes
            #  arg0 = src_skip (SRAM)
            #  arg1 = clamp packed: (max<<16)|(min&0xFFFF)
            #  arg2 reserved
            clamp_min = -128 & 0xFFFF
            clamp_max = 127 & 0xFFFF
            clamp_pack = (clamp_max << 16) | clamp_min
            flags = FLAG_ADD_RELU_MIN0 if (m.relu and m.zC == 0) else 0
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_ADD_I8_CLAMP, flags=flags,
                         wait0=wait0, wait1=wait1, sig0=sig,
                         src_addr=src_out, dst_addr=dst_out,
                         size_or_M=bytes_, arg0=src_skip, arg1=clamp_pack, arg2=0)

        def dma_store_Y(dst_dram: int, src_sram: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=wait,
                         src_addr=src_sram, dst_addr=dst_dram,
                         size_or_M=bytes_, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        # Prefetch tile0 into A0,B0, and also prefetch skip tile0 into SK0
        tm0, tn0 = tiles[0]
        cmds.append(dma_load_A(s.A0, addr_A(m, tm0*m.tileM, 0), e.EA0))
        cmds.append(dma_load_B_linear(s.B0, bpack_tile_addr(m, tn0), e.EB0))
        cmds.append(dma_load_skip_tile(s.SK0, addr_Skip(m, tm0*m.tileM, tn0*m.tileN), e.ESK0))

        for i, (tm, tn) in enumerate(tiles):
            use_A  = s.A0  if (i % 2 == 0) else s.A1
            use_B  = s.B0  if (i % 2 == 0) else s.B1
            use_SK = s.SK0 if (i % 2 == 0) else s.SK1

            waitA  = e.EA0  if (i % 2 == 0) else e.EA1
            waitB  = e.EB0  if (i % 2 == 0) else e.EB1
            waitSK = e.ESK0 if (i % 2 == 0) else e.ESK1

            # MAC -> ACC
            cmds.append(mac_acc32(s.ACC, use_A, use_B, waitA, waitB, e.EACC))

            # Prefetch next tile (A,B,Skip) into alternate buffers to overlap
            if i + 1 < len(tiles):
                tm_next, tn_next = tiles[i+1]
                alt_A  = s.A1  if (i % 2 == 0) else s.A0
                alt_B  = s.B1  if (i % 2 == 0) else s.B0
                alt_SK = s.SK1 if (i % 2 == 0) else s.SK0

                sigA  = e.EA1  if (i % 2 == 0) else e.EA0
                sigB  = e.EB1  if (i % 2 == 0) else e.EB0
                sigSK = e.ESK1 if (i % 2 == 0) else e.ESK0

                cmds.append(dma_load_A(alt_A, addr_A(m, tm_next*m.tileM, 0), sigA))
                cmds.append(dma_load_B_linear(alt_B, bpack_tile_addr(m, tn_next), sigB))
                cmds.append(dma_load_skip_tile(alt_SK, addr_Skip(m, tm_next*m.tileM, tn_next*m.tileN), sigSK))

            # REQUANT(+ReLU) -> OUT
            ms_addr = ms_tile_addr(m, tn)
            cmds.append(requant_relu_pc(s.OUT, s.ACC, ms_addr, e.EACC, e.EOUT))

            # RESIDUAL ADD: OUT = OUT + SK (inplace)
            cmds.append(add_i8_clamp_inplace(s.OUT, s.OUT, use_SK, e.EOUT, waitSK, e.EADD))

            # STORE OUT -> Y
            cmds.append(dma_store_Y(addr_Y(m, tm*m.tileM, tn*m.tileN), s.OUT, e.EADD))

        return cmds

    def emit_path_bypass_tiled(self) -> List[Cmd32]:
        """bypass path: tile copy Skip? or identity A? Here we keep earlier identity: Y=A, tiled + overlap."""
        m, s, e = self.m, self.s, self.e
        cmds: List[Cmd32] = []
        tiles = [(0,0), (0,1), (1,0), (1,1)]

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

        # prefetch tile0 into A0
        tm0, tn0 = tiles[0]
        cmds.append(dma_load_tile(s.A0, addr_A(m, tm0*m.tileM, tn0*m.tileN), e.EA0))

        for i, (tm, tn) in enumerate(tiles):
            use_A = s.A0 if (i % 2 == 0) else s.A1
            waitA = e.EA0 if (i % 2 == 0) else e.EA1

            if i + 1 < len(tiles):
                tm_next, tn_next = tiles[i+1]
                alt_A = s.A1 if (i % 2 == 0) else s.A0
                sigA  = e.EA1 if (i % 2 == 0) else e.EA0
                cmds.append(dma_load_tile(alt_A, addr_A(m, tm_next*m.tileM, tn_next*m.tileN), sigA))

            cmds.append(dma_store_tile(addr_Y(m, tm*m.tileM, tn*m.tileN), use_A, waitA))

        return cmds

    # --- emit bundles to disk ---
    def write_bundle(self, out_dir: str, name: str, cmds: List[Cmd32]) -> None:
        bin_path = os.path.join(out_dir, f"{name}.bin")
        trace_path = os.path.join(out_dir, f"{name}.trace.txt")
        with open(bin_path, "wb") as fb, open(trace_path, "w", encoding="utf-8") as ft:
            for i, c in enumerate(cmds):
                fb.write(c.pack())
                ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 6) Command Interpreter / Simulator
# ============================================================

@dataclass
class EngineState:
    # simplistic single-issue per engine (one cmd at a time)
    busy_until: int = 0

class CommandSimulator:
    """
    Simulates:
      - events: cmd can execute when its waits are signaled
      - engines: DMA, MAC, ACT (requant/add)
      - rough duration model to see overlap
      - checks: wait events must be signaled before use; signals recorded

    Not simulating real data. Only control correctness + rough timing.
    """

    def __init__(self):
        self.events: Dict[int, bool] = {}
        self.t = 0
        self.dma = EngineState()
        self.mac = EngineState()
        self.act = EngineState()

    def _engine_for(self, cmd: Cmd32) -> str:
        if cmd.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            return "DMA"
        if cmd.opcode == OP_MAC_ACC32:
            return "MAC"
        if cmd.opcode in (OP_REQUANT_RELU_PC, OP_ADD_I8_CLAMP, OP_PACK_B):
            # PACK_B here treated as ACT/CPU-like; in real life could be CPU or DMA+kernel
            return "ACT"
        return "ACT"

    def _duration(self, cmd: Cmd32) -> int:
        # toy duration model (in "cycles"):
        # DMA proportional to bytes/256, MAC proportional to M*N*K/4096, ACT proportional to bytes/512
        if cmd.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            bytes_ = cmd.size_or_M
            return max(1, bytes_ // 256)
        if cmd.opcode == OP_MAC_ACC32:
            M = cmd.size_or_M
            N = cmd.arg1
            K = cmd.arg2
            work = M * N * K
            return max(1, work // 4096)
        if cmd.opcode == OP_REQUANT_RELU_PC:
            M = cmd.size_or_M
            N = cmd.arg0
            bytes_ = M * N
            return max(1, bytes_ // 512)
        if cmd.opcode == OP_ADD_I8_CLAMP:
            bytes_ = cmd.size_or_M
            return max(1, bytes_ // 512)
        if cmd.opcode == OP_PACK_B:
            # assume heavier
            K = cmd.size_or_M
            N = cmd.arg0
            return max(1, (K*N) // 1024)
        return 1

    def _can_run(self, cmd: Cmd32) -> bool:
        # both waits must be signaled if nonzero
        for w in (cmd.wait0, cmd.wait1):
            if w != 0 and not self.events.get(w, False):
                return False
        return True

    def _engine_busy_until(self, eng: str) -> int:
        if eng == "DMA": return self.dma.busy_until
        if eng == "MAC": return self.mac.busy_until
        return self.act.busy_until

    def _set_engine_busy(self, eng: str, until: int) -> None:
        if eng == "DMA": self.dma.busy_until = until
        elif eng == "MAC": self.mac.busy_until = until
        else: self.act.busy_until = until

    def run(self, cmds: List[Cmd32]) -> str:
        lines = []
        self.t = 0
        self.events.clear()
        self.dma = EngineState()
        self.mac = EngineState()
        self.act = EngineState()

        for i, cmd in enumerate(cmds):
            # advance time until waits satisfied AND engine free
            eng = self._engine_for(cmd)
            while True:
                waits_ok = self._can_run(cmd)
                eng_free_at = self._engine_busy_until(eng)
                if waits_ok and self.t >= eng_free_at:
                    break
                # advance to next interesting time
                next_t = max(self.t + 1, eng_free_at)
                # if waits not ok, we can't know exact time; just increment
                if not waits_ok:
                    next_t = self.t + 1
                self.t = next_t

            dur = self._duration(cmd)
            start = self.t
            end = self.t + dur
            self._set_engine_busy(eng, end)

            # record signals at end
            if cmd.sig0: self.events[cmd.sig0] = True
            if cmd.sig1: self.events[cmd.sig1] = True

            lines.append(f"{i:04d}  t={start:6d}..{end:6d}  {eng:3s}  {cmd.to_trace()}")

            # advance time a bit (in a real pipeline, other engines can run concurrently; we keep global time but engine busy captures overlap)
            self.t = start  # keep global time at start; next cmd will wait on its engine. This allows overlap across different engines.

        # finalize: compute makespan
        makespan = max(self.dma.busy_until, self.mac.busy_until, self.act.busy_until)
        lines.append("")
        lines.append(f"SIM SUMMARY: DMA_end={self.dma.busy_until}, MAC_end={self.mac.busy_until}, ACT_end={self.act.busy_until}, makespan={makespan}")
        lines.append(f"Events signaled: {sorted([k for k,v in self.events.items() if v])}")
        return "\n".join(lines)


# ============================================================
# 7) Main
# ============================================================

def main():
    out_dir = "./toy_out_v2"
    os.makedirs(out_dir, exist_ok=True)

    # Assumed DRAM bases
    model = ModelSpec(
        A_base=0x8000_0000,
        B_base=0x8000_2000,
        Y_base=0x8000_4000,
        Bpack_base=0x8000_6000,
        MS_base=0x8000_7000,
        Skip_base=0x8000_9000,   # NEW: skip tensor
        M=64, N=64, K=64,
        tileM=32, tileN=32, tileK=64,
        zA=0, zB=0, zC=0,
        relu=True,
        ms_entry_size=8,
        sram_size_bytes=256*1024,
        sram_align=64
    )

    # Auto SRAM allocation (toy)
    alloc = SRAMAllocator(base=0x0000, size=model.sram_size_bytes, align=model.sram_align)

    # Sizes:
    A_tile_bytes   = model.tileM * model.K          # 32*64 = 2048
    B_tile_bytes   = model.K * model.tileN          # 64*32 = 2048
    SK_tile_bytes  = model.tileM * model.tileN      # 32*32 = 1024
    ACC_bytes      = model.tileM * model.tileN * 4  # 4096
    OUT_bytes      = model.tileM * model.tileN      # 1024

    # Double buffers for A/B/Skip + single ACC/OUT
    sram = SRAMLayout(
        A0 = alloc.alloc("A0", A_tile_bytes),
        A1 = alloc.alloc("A1", A_tile_bytes),
        B0 = alloc.alloc("B0", B_tile_bytes),
        B1 = alloc.alloc("B1", B_tile_bytes),
        SK0= alloc.alloc("SK0", SK_tile_bytes),
        SK1= alloc.alloc("SK1", SK_tile_bytes),
        ACC= alloc.alloc("ACC", ACC_bytes),
        OUT= alloc.alloc("OUT", OUT_bytes),
    )

    # dump allocation map
    with open(os.path.join(out_dir, "sram_map.txt"), "w", encoding="utf-8") as f:
        f.write(alloc.dump() + "\n")

    ev = Events()

    comp = ToyNPUCompilerV2(model, sram, ev)

    prologue = comp.emit_pack_b()
    pathA = comp.emit_path_compute_with_residual()
    pathB = comp.emit_path_bypass_tiled()

    comp.write_bundle(out_dir, "prologue_packB", prologue)
    comp.write_bundle(out_dir, "path_compute", pathA)
    comp.write_bundle(out_dir, "path_bypass", pathB)

    dispatch = {
        "phi_strategy": "firmware_select_path",
        "paths": {"if_c_true": "path_compute.bin", "if_c_false": "path_bypass.bin"},
        "optional_prologue": "prologue_packB.bin",
        "notes": [
            "Compute path includes: PACK_B optional, tiled MatMul->RequantReLU->ResidualAdd->Store",
            "Bypass path includes: tiled Y=A copy"
        ]
    }
    with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch, f, indent=2)

    # Simulate
    sim = CommandSimulator()
    simA = sim.run(pathA)
    with open(os.path.join(out_dir, "sim_path_compute.txt"), "w", encoding="utf-8") as f:
        f.write(simA + "\n")
    simB = sim.run(pathB)
    with open(os.path.join(out_dir, "sim_path_bypass.txt"), "w", encoding="utf-8") as f:
        f.write(simB + "\n")

    print("Generated in:", out_dir)
    print("  - sram_map.txt")
    print("  - prologue_packB.bin/.trace.txt")
    print("  - path_compute.bin/.trace.txt + sim_path_compute.txt")
    print("  - path_bypass.bin/.trace.txt  + sim_path_bypass.txt")
    print("  - dispatch.json")

    print("\nBilingual notes / 双语说明：")
    print("EN: sim_* reports scheduling overlap (DMA/MAC/ACT) and checks event waits.")
    print("中: sim_* 会显示 DMA/MAC/ACT 的粗略重叠情况，并检查 wait 的 event 是否已被 signal。")


if __name__ == "__main__":
    main()

