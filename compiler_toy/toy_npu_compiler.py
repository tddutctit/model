
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler Skeleton (SSA-ish -> Lowering -> 32B Command Buffer)
====================================================================
Goal:
  - Input: a tiny "graph" with (MatMul->RequantReLU->Store) and an If(c) bypass
  - Output:
      1) Two command lists (Path A compute, Path B bypass)
      2) Optional prologue PACK_B
      3) A binary command buffer (32B per command) for each list
      4) A human-readable trace for validation

Assumed NPU spec (same as we discussed):
  - SRAM 256KB, 64B alignment
  - Engines: DMA_LOAD_2D, DMA_STORE_2D, DMA_LOAD_LINEAR, MAC_ACC32, REQUANT_RELU_PC, PACK_B
  - Sync: wait0/wait1/sig0 events (0-255)
  - Command ABI: fixed 32 bytes, little-endian

How Φ is implemented:
  - Firmware/CPU evaluates condition c, then chooses Path A or Path B command buffer.
  - So the compiler emits 2 buffers (compute vs bypass). Φ is "which buffer runs".
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

FLAG_RELU_ENABLE     = 0x01
FLAG_IS_2D           = 0x02

# 32B layout (little-endian)
# offset 0x00: opcode (u8)
# offset 0x01: flags  (u8)
# offset 0x02: wait0  (u8)
# offset 0x03: wait1  (u8)
# offset 0x04: sig0   (u8)
# offset 0x05: sig1   (u8)
# offset 0x06: rsvd   (u16)
# offset 0x08: src_addr (u32)
# offset 0x0C: dst_addr (u32)
# offset 0x10: size_or_M (u32)
# offset 0x14: arg0 (u32)
# offset 0x18: arg1 (u32)
# offset 0x1C: arg2 (u32)
#
# Total 32 bytes.

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
        # clamp to ranges
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
        body = struct.pack(
            "<IIIIIIII",
            u32(self.src_addr),
            u32(self.dst_addr),
            u32(self.size_or_M),
            u32(self.arg0),
            u32(self.arg1),
            u32(self.arg2),
            u32(0),
            u32(0),
        )
        # header=8 bytes, body=32 bytes (we only need next 24 bytes)
        # We'll take first 24 bytes of body to make total 32B.
        # body layout: src,dst,size,arg0,arg1,arg2,0,0 => 32 bytes
        # We only want src..arg2 => 24 bytes; but our earlier agreed 32B includes exactly src..arg2 (24B) + 8B header = 32B.
        body24 = body[:24]
        return header + body24

    def to_trace(self) -> str:
        return (f"Cmd(op=0x{self.opcode:02X}, flags=0x{self.flags:02X}, "
                f"w0={self.wait0}, w1={self.wait1}, s0={self.sig0}, s1={self.sig1}, "
                f"src=0x{self.src_addr:08X}, dst=0x{self.dst_addr:08X}, "
                f"sz/M=0x{self.size_or_M:08X}, a0=0x{self.arg0:08X}, "
                f"a1=0x{self.arg1:08X}, a2=0x{self.arg2:08X})")


# ============================================================
# 2) Minimal IR / Graph model
# ============================================================

@dataclass
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: str  # "i8", "i32"
    # For simplicity: row-major, contiguous.

@dataclass
class ModelSpec:
    # DRAM base addresses
    A_base: int
    B_base: int
    Y_base: int
    Bpack_base: int
    MS_base: int  # per-channel mult/shift table base

    # Shapes (fixed for this toy)
    M: int = 64
    N: int = 64
    K: int = 64
    tileM: int = 32
    tileN: int = 32
    tileK: int = 64  # full K in one go (toy)

    # Quant params
    zA: int = 0
    zB: int = 0
    zC: int = 0
    relu: bool = True

    # per-channel table format assumptions
    ms_entry_size: int = 8  # bytes per output-channel entry (toy)
    # e.g. entry = [mult32][shift16][zC8][flags8]


@dataclass
class SRAMLayout:
    # double buffers for A/B
    A0: int
    A1: int
    B0: int
    B1: int
    ACC: int
    OUT: int

@dataclass
class Events:
    EA0: int = 1
    EB0: int = 2
    EA1: int = 3
    EB1: int = 4
    EACC: int = 5
    EOUT: int = 6


# ============================================================
# 3) Address helpers (row-major)
# ============================================================

def addr_A(model: ModelSpec, m: int, k: int) -> int:
    # A is int8: bytes = 1
    return model.A_base + (m * model.K + k) * 1

def addr_Y(model: ModelSpec, m: int, n: int) -> int:
    # Y is int8 output: bytes = 1
    return model.Y_base + (m * model.N + n) * 1

def bpack_tile_addr(model: ModelSpec, n_tile_index: int) -> int:
    # each Bpack tile block is K*tileN bytes (int8)
    block_bytes = model.K * model.tileN
    return model.Bpack_base + n_tile_index * block_bytes

def ms_tile_addr(model: ModelSpec, n_tile_index: int) -> int:
    # per-channel params for this tile: tileN entries
    return model.MS_base + n_tile_index * model.tileN * model.ms_entry_size


# ============================================================
# 4) Passes (toy): emit command lists
# ============================================================

class ToyNPUCompiler:
    def __init__(self, model: ModelSpec, sram: SRAMLayout, ev: Events):
        self.model = model
        self.sram = sram
        self.ev = ev

    # ---------- Prologue: PACK_B ----------
    def emit_pack_b(self) -> List[Cmd32]:
        m = self.model
        # PACK_B: src=B_base, dst=Bpack_base, size_or_M=K, arg0=N, arg1=tileN
        return [Cmd32(
            opcode=OP_PACK_B,
            src_addr=m.B_base,
            dst_addr=m.Bpack_base,
            size_or_M=m.K,
            arg0=m.N,
            arg1=m.tileN,
            arg2=0,
        )]

    # ---------- Path A: compute ----------
    def emit_path_compute(self) -> List[Cmd32]:
        m, s, e = self.model, self.sram, self.ev
        cmds: List[Cmd32] = []

        # Tile enumeration in (m_tile, n_tile) order:
        # (0,0), (0,1), (1,0), (1,1) where tile indices multiply by tile sizes.
        tiles = [
            (0, 0),  # m0,n0
            (0, 1),  # m0,n1
            (1, 0),  # m1,n0
            (1, 1),  # m1,n1
        ]

        # Helper: load A tile (2D) into A buffer, signal event
        def dma_load_A(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            # We load A_tile as rows=tileM, cols=K, stride=K (row stride in elements)
            # Because K=64, rows=32, cols=64 => bytes=2048
            bytes_ = m.tileM * m.K * 1
            return Cmd32(
                opcode=OP_DMA_LOAD_2D,
                flags=FLAG_IS_2D,
                sig0=sig,
                src_addr=src_dram,
                dst_addr=dst_sram,
                size_or_M=bytes_,
                arg0=m.tileM,  # rows
                arg1=m.K,      # cols
                arg2=m.K,      # stride (elements)
            )

        # Helper: load Bpack tile (linear) into B buffer, signal event
        def dma_load_B_linear(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.K * m.tileN * 1
            return Cmd32(
                opcode=OP_DMA_LOAD_LINEAR,
                sig0=sig,
                src_addr=src_dram,
                dst_addr=dst_sram,
                size_or_M=bytes_,
            )

        # Helper: MAC_ACC32 tile
        def mac_acc32(dst_acc: int, A_sram: int, B_sram: int, waitA: int, waitB: int, sig: int) -> Cmd32:
            # Convention for this toy ABI:
            # src_addr = A_sram, dst_addr = ACC_sram, arg0 = B_sram, size_or_M=M, arg1=N, arg2=K
            return Cmd32(
                opcode=OP_MAC_ACC32,
                wait0=waitA, wait1=waitB, sig0=sig,
                src_addr=A_sram,
                dst_addr=dst_acc,
                size_or_M=m.tileM,
                arg0=B_sram,
                arg1=m.tileN,
                arg2=m.tileK,
            )

        # Helper: REQUANT_RELU_PC
        def requant_relu_pc(dst_out: int, src_acc: int, ms_addr: int, wait: int, sig: int) -> Cmd32:
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(
                opcode=OP_REQUANT_RELU_PC,
                flags=flags,
                wait0=wait,
                sig0=sig,
                src_addr=src_acc,
                dst_addr=dst_out,
                size_or_M=m.tileM,   # M
                arg0=m.tileN,        # N
                arg1=ms_addr,        # per-channel table addr
                arg2=0,
            )

        # Helper: STORE OUT tile to DRAM
        def dma_store_Y(dst_dram: int, src_sram: int, wait: int) -> Cmd32:
            # store rows=tileM cols=tileN stride=N
            bytes_ = m.tileM * m.tileN * 1
            return Cmd32(
                opcode=OP_DMA_STORE_2D,
                flags=FLAG_IS_2D,
                wait0=wait,
                src_addr=src_sram,
                dst_addr=dst_dram,
                size_or_M=bytes_,
                arg0=m.tileM,
                arg1=m.tileN,
                arg2=m.N,  # stride in elements
            )

        # ---- Double-buffer overlap schedule ----
        # tile0 pref: A0,B0
        (tm0, tn0) = tiles[0]
        A0_src = addr_A(m, tm0 * m.tileM, 0)
        B0_src = bpack_tile_addr(m, tn0)
        cmds.append(dma_load_A(s.A0, A0_src, e.EA0))
        cmds.append(dma_load_B_linear(s.B0, B0_src, e.EB0))

        # For each tile i:
        #   compute tile i using current buffers
        #   prefetch tile i+1 into alternate buffers (if exists)
        #   requant+relu
        #   store
        #
        # Buffer selection:
        #   even i -> use A0,B0; prefetch into A1,B1
        #   odd  i -> use A1,B1; prefetch into A0,B0

        for i, (tm, tn) in enumerate(tiles):
            use_A = s.A0 if (i % 2 == 0) else s.A1
            use_B = s.B0 if (i % 2 == 0) else s.B1
            waitA = e.EA0 if (i % 2 == 0) else e.EA1
            waitB = e.EB0 if (i % 2 == 0) else e.EB1

            # compute
            cmds.append(mac_acc32(s.ACC, use_A, use_B, waitA, waitB, e.EACC))

            # prefetch next tile (overlap) if any
            if i + 1 < len(tiles):
                tm_next, tn_next = tiles[i + 1]
                alt_A = s.A1 if (i % 2 == 0) else s.A0
                alt_B = s.B1 if (i % 2 == 0) else s.B0
                sigA = e.EA1 if (i % 2 == 0) else e.EA0
                sigB = e.EB1 if (i % 2 == 0) else e.EB0

                A_src = addr_A(m, tm_next * m.tileM, 0)
                B_src = bpack_tile_addr(m, tn_next)
                cmds.append(dma_load_A(alt_A, A_src, sigA))
                cmds.append(dma_load_B_linear(alt_B, B_src, sigB))

            # requant + relu per-channel table
            ms_addr = ms_tile_addr(m, tn)
            cmds.append(requant_relu_pc(s.OUT, s.ACC, ms_addr, e.EACC, e.EOUT))

            # store OUT tile
            Y_dst = addr_Y(m, tm * m.tileM, tn * m.tileN)
            cmds.append(dma_store_Y(Y_dst, s.OUT, e.EOUT))

        return cmds

    # ---------- Path B: bypass (tiled copy + overlap) ----------
    def emit_path_bypass(self) -> List[Cmd32]:
        m, s, e = self.model, self.sram, self.ev
        cmds: List[Cmd32] = []

        tiles = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]

        def dma_load_tile_A32x32(dst_sram: int, src_dram: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN * 1
            return Cmd32(
                opcode=OP_DMA_LOAD_2D,
                flags=FLAG_IS_2D,
                sig0=sig,
                src_addr=src_dram,
                dst_addr=dst_sram,
                size_or_M=bytes_,
                arg0=m.tileM,
                arg1=m.tileN,
                arg2=m.N,  # stride in elements for A and Y (both N=64)
            )

        def dma_store_tile_Y32x32(dst_dram: int, src_sram: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN * 1
            return Cmd32(
                opcode=OP_DMA_STORE_2D,
                flags=FLAG_IS_2D,
                wait0=wait,
                src_addr=src_sram,
                dst_addr=dst_dram,
                size_or_M=bytes_,
                arg0=m.tileM,
                arg1=m.tileN,
                arg2=m.N,
            )

        # Prefetch tile0 into A0
        tm0, tn0 = tiles[0]
        src0 = addr_A(m, tm0 * m.tileM, tn0 * m.tileN)
        cmds.append(dma_load_tile_A32x32(s.A0, src0, e.EA0))

        for i, (tm, tn) in enumerate(tiles):
            use_A = s.A0 if (i % 2 == 0) else s.A1
            waitA = e.EA0 if (i % 2 == 0) else e.EA1

            # prefetch next tile into alternate buffer
            if i + 1 < len(tiles):
                tm_next, tn_next = tiles[i + 1]
                alt_A = s.A1 if (i % 2 == 0) else s.A0
                sigA  = e.EA1 if (i % 2 == 0) else e.EA0
                src_next = addr_A(m, tm_next * m.tileM, tn_next * m.tileN)
                cmds.append(dma_load_tile_A32x32(alt_A, src_next, sigA))

            # store current tile to Y
            dstY = addr_Y(m, tm * m.tileM, tn * m.tileN)
            cmds.append(dma_store_tile_Y32x32(dstY, use_A, waitA))

        return cmds

    # ---------- Emit binaries + traces ----------
    def emit(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        prologue = self.emit_pack_b()
        pathA = self.emit_path_compute()
        pathB = self.emit_path_bypass()

        self._write_bundle(out_dir, "prologue_packB", prologue)
        self._write_bundle(out_dir, "path_compute", pathA)
        self._write_bundle(out_dir, "path_bypass", pathB)

        # A small "dispatcher spec" (firmware chooses based on condition c)
        dispatch = {
            "phi_strategy": "firmware_select_path",
            "paths": {
                "if_c_true": "path_compute.bin",
                "if_c_false": "path_bypass.bin"
            },
            "optional_prologue": "prologue_packB.bin"
        }
        with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
            json.dump(dispatch, f, indent=2)

    def _write_bundle(self, out_dir: str, name: str, cmds: List[Cmd32]) -> None:
        bin_path = os.path.join(out_dir, f"{name}.bin")
        txt_path = os.path.join(out_dir, f"{name}.trace.txt")

        with open(bin_path, "wb") as fb, open(txt_path, "w", encoding="utf-8") as ft:
            for idx, c in enumerate(cmds):
                blob = c.pack()
                assert len(blob) == 32, f"Cmd not 32B, got {len(blob)}"
                fb.write(blob)
                ft.write(f"{idx:04d}: {c.to_trace()}\n")


# ============================================================
# 5) Example usage (assumed spec)
# ============================================================

def main():
    # Assumed DRAM bases (same spirit as earlier)
    model = ModelSpec(
        A_base=0x8000_0000,
        B_base=0x8000_2000,
        Y_base=0x8000_4000,
        Bpack_base=0x8000_6000,
        MS_base=0x8000_7000,
        M=64, N=64, K=64,
        tileM=32, tileN=32, tileK=64,
        zA=0, zB=0, zC=0,
        relu=True,
        ms_entry_size=8
    )

    # SRAM layout (toy addresses; should obey 64B alignment in real life)
    sram = SRAMLayout(
        A0=0x0000,
        A1=0x0800,
        B0=0x1000,
        B1=0x1800,
        ACC=0x2000,
        OUT=0x3000
    )

    ev = Events()

    comp = ToyNPUCompiler(model, sram, ev)
    out_dir = "./toy_out"
    comp.emit(out_dir)

    print("Generated:")
    print(f"  {out_dir}/prologue_packB.bin + .trace.txt")
    print(f"  {out_dir}/path_compute.bin  + .trace.txt")
    print(f"  {out_dir}/path_bypass.bin   + .trace.txt")
    print(f"  {out_dir}/dispatch.json")
    print("\nHow to run (conceptually):")
    print("  - firmware evaluates condition c")
    print("  - optionally run prologue_packB once")
    print("  - if c: submit path_compute.bin else path_bypass.bin")


if __name__ == "__main__":
    main()

