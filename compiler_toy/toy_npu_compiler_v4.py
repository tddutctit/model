

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler v4 (Stricter + More Realistic)
===============================================
Upgrades from v3:
1) Simulator now signals events at *command end* (not at start) -> stricter dependency checking.
2) Adds a simple SRAM bank model + bank-conflict checks:
   - SRAM is divided into N banks; each command touches 1~2 SRAM buffers.
   - If two engines overlap and touch the same bank, we can (optionally) serialize or flag a hazard.
   - Here we flag hazards in the sim report (does not automatically fix schedule).
3) PACK_B lowering becomes more realistic:
   - Instead of a single OP_PACK_B, we emit:
       (a) DMA_LOAD_2D blocks of B into a scratch SRAM buffer
       (b) PACK_B_TILE "kernel" that transposes/tiles into Bpack in DRAM (toy ACT engine op)
   - This models "DMA + micro-kernel" split often seen in real NPUs.

Still a toy:
- No real data movement, only control + rough timing/hazards.
- Schedulers/allocators are simplified.

Outputs:
  ./toy_out_v4/
    - normalized_graph.json
    - sram_map.txt
    - prologue_packB.bin/.trace.txt
    - path_compute.bin/.trace.txt + sim_path_compute.txt
    - path_bypass.bin/.trace.txt  + sim_path_bypass.txt
    - dispatch.json
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

OP_PACK_B_TILE       = 0x31   # NEW: pack/transpose micro-kernel for one tile block
OP_ADD_I8_CLAMP      = 0x40

FLAG_RELU_ENABLE     = 0x01
FLAG_IS_2D           = 0x02
FLAG_ADD_RELU_MIN0   = 0x04


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

    zC: int = 0
    relu: bool = True
    ms_entry_size: int = 8

    sram_size_bytes: int = 256 * 1024
    sram_align: int = 64

    # SRAM bank model (toy)
    sram_banks: int = 8
    bank_granularity: int = 256  # bytes per bank "stripe"


@dataclass
class Events:
    base: int = 1


# ============================================================
# 3) Address helpers
# ============================================================

def addr_A(m: ModelSpec, row: int, col: int) -> int:
    return m.A_base + (row * m.K + col) * 1

def addr_Y(m: ModelSpec, row: int, col: int) -> int:
    return m.Y_base + (row * m.N + col) * 1

def addr_Skip(m: ModelSpec, row: int, col: int) -> int:
    return m.Skip_base + (row * m.N + col) * 1

def bpack_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    block_bytes = m.K * m.tileN
    return m.Bpack_base + n_tile_index * block_bytes

def ms_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.MS_base + n_tile_index * m.tileN * m.ms_entry_size

def addr_B(m: ModelSpec, row: int, col: int) -> int:
    return m.B_base + (row * m.N + col) * 1


# ============================================================
# 4) Graph JSON IR (same as v3)
# ============================================================

DEFAULT_GRAPH = {
  "name": "if_matmul_requant_relu_residual",
  "inputs": {
    "A": {"dtype":"i8", "shape":[64,64]},
    "B": {"dtype":"i8", "shape":[64,64]},
    "Skip": {"dtype":"i8", "shape":[64,64]},
    "cond": {"dtype":"bool", "shape":[]}
  },
  "outputs": { "Y": {"dtype":"i8", "shape":[64,64]} },
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
# 5) Plan / Lowering
# ============================================================

@dataclass
class TileTask:
    tm: int
    tn: int
    steps: List[str]

class GraphLowerer:
    def __init__(self, graph: Dict[str, Any], model: ModelSpec):
        self.g = graph
        self.m = model

    def normalize(self) -> Dict[str, Any]:
        g = json.loads(json.dumps(self.g))
        for n in g["if"]["then"]:
            if n["op"] == "PACK_B":
                n.setdefault("attrs", {})
                n["attrs"]["tileN"] = self.m.tileN
        return g

    def build_plan(self) -> Dict[str, Any]:
        g = self.normalize()
        then_ops = [n["op"] for n in g["if"]["then"]]
        use_pack = "PACK_B" in then_ops
        use_residual = "ADD_RESIDUAL" in then_ops

        tm_count = self.m.M // self.m.tileM
        tn_count = self.m.N // self.m.tileN

        tiles: List[TileTask] = []
        for tm in range(tm_count):
            for tn in range(tn_count):
                steps = ["LOAD_A", "LOAD_B", "MAC", "REQUANT"]
                if use_residual:
                    steps += ["LOAD_SKIP", "ADD"]
                steps += ["STORE_Y"]
                tiles.append(TileTask(tm, tn, steps))

        return {
            "use_pack": use_pack,
            "use_residual": use_residual,
            "then_ops": g["if"]["then"],
            "else_ops": g["if"]["else"],
            "tiles": tiles
        }


# ============================================================
# 6) Linear-Scan SRAM allocator (same as v3, but plus scratch for pack)
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
    def __init__(self, base: int, size: int, align: int):
        self.base = base
        self.size = size
        self.default_align = align
        self.free: List[Tuple[int,int]] = []
        self.bump = base
        self.alloc_map: Dict[str, Tuple[int,int]] = {}

    @staticmethod
    def _align_up(x: int, a: int) -> int:
        return (x + a - 1) // a * a

    def _try_alloc_from_free(self, nbytes: int, align: int) -> Optional[int]:
        for i, (addr, sz) in enumerate(self.free):
            aaddr = self._align_up(addr, align)
            pad = aaddr - addr
            if sz >= pad + nbytes:
                alloc_addr = aaddr
                before = pad
                after = sz - (pad + nbytes)
                new_blocks = []
                if before > 0:
                    new_blocks.append((addr, before))
                if after > 0:
                    new_blocks.append((aaddr + nbytes, after))
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
# 7) Emit commands (v4: pack split + strict event wiring)
# ============================================================

class CommandEmitterV4:
    def __init__(self, model: ModelSpec):
        self.m = model

    def _bytes_A_tile(self) -> int:  return self.m.tileM * self.m.K
    def _bytes_B_tile(self) -> int:  return self.m.K * self.m.tileN
    def _bytes_OUT(self) -> int:     return self.m.tileM * self.m.tileN
    def _bytes_ACC(self) -> int:     return self.m.tileM * self.m.tileN * 4
    def _bytes_SKIP(self) -> int:    return self.m.tileM * self.m.tileN

    def build_liveness(self, plan: Dict[str, Any]) -> List[LiveInterval]:
        use_residual = plan["use_residual"]
        total = max(1, len(plan["tiles"]) * (5 + (2 if use_residual else 0)))

        intervals = [
            LiveInterval("A0", 0, total, self._bytes_A_tile(), self.m.sram_align),
            LiveInterval("A1", 0, total, self._bytes_A_tile(), self.m.sram_align),
            LiveInterval("B0", 0, total, self._bytes_B_tile(), self.m.sram_align),
            LiveInterval("B1", 0, total, self._bytes_B_tile(), self.m.sram_align),
            LiveInterval("ACC", 0, total, self._bytes_ACC(), self.m.sram_align),
            LiveInterval("OUT", 0, total, self._bytes_OUT(), self.m.sram_align),
        ]
        if use_residual:
            intervals += [
                LiveInterval("SK0", 0, total, self._bytes_SKIP(), self.m.sram_align),
                LiveInterval("SK1", 0, total, self._bytes_SKIP(), self.m.sram_align),
            ]

        # NEW: scratch for packing B (tileK x tileN bytes)
        intervals.append(LiveInterval("B_SCR", 0, total, self._bytes_B_tile(), self.m.sram_align))
        return intervals

    def allocate_sram(self, intervals: List[LiveInterval]) -> Tuple[Dict[str,int], str]:
        alloc = LinearScanSRAMAllocator(base=0x0000, size=self.m.sram_size_bytes, align=self.m.sram_align)
        intervals_sorted = sorted(intervals, key=lambda it: (it.start, -it.size))
        active: List[LiveInterval] = []

        def expire(t: int):
            nonlocal active
            still = []
            for it in active:
                if it.end <= t:
                    assert it.addr is not None
                    alloc.free_block(it.addr, it.size)
                else:
                    still.append(it)
            active = still

        for it in intervals_sorted:
            expire(it.start)
            it.addr = alloc.alloc(it.name, it.size, it.align)
            active.append(it)

        return {it.name: it.addr for it in intervals_sorted if it.addr is not None}, alloc.dump()

    # ---------------- pack prologue (split) ----------------
    def emit_pack_b_prologue_split(self, sram: Dict[str,int]) -> List[Cmd32]:
        """
        Pack weights B into Bpack in DRAM, tile-by-tile over N tiles.
        For each tn tile:
          - DMA_LOAD_2D: load B[:, tn:tn+tileN] into SRAM scratch
          - PACK_B_TILE: micro-kernel packs SRAM scratch -> DRAM Bpack[tile]
        """
        m = self.m
        B_SCR = sram["B_SCR"]

        cmds: List[Cmd32] = []
        next_ev = 1

        def new_ev() -> int:
            nonlocal next_ev
            x = next_ev
            next_ev += 1
            return x

        def dma_load_Bcolblock(dst_sram: int, tn: int, sig: int) -> Cmd32:
            # load a KxtileN block from B (row-major KxN):
            # interpret as 2D: rows=K, cols=tileN, stride=N
            bytes_ = m.K * m.tileN
            src = addr_B(m, 0, tn*m.tileN)
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=src, dst_addr=dst_sram,
                         size_or_M=bytes_,
                         arg0=m.K, arg1=m.tileN, arg2=m.N)

        def pack_tile(wait: int, tn: int, sig: int) -> Cmd32:
            # PACK_B_TILE: src=SRAM scratch, dst=DRAM bpack_tile_addr
            # size_or_M = K, arg0=tileN, arg1=tn_index
            dst = bpack_tile_addr(m, tn)
            return Cmd32(OP_PACK_B_TILE, wait0=wait, sig0=sig,
                         src_addr=B_SCR, dst_addr=dst,
                         size_or_M=m.K, arg0=m.tileN, arg1=tn, arg2=0)

        tn_count = m.N // m.tileN
        for tn in range(tn_count):
            e_load = new_ev()
            e_pack = new_ev()
            cmds.append(dma_load_Bcolblock(B_SCR, tn, e_load))
            cmds.append(pack_tile(e_load, tn, e_pack))
        return cmds

    # ---------------- compute path ----------------
    def emit_compute(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        use_residual = plan["use_residual"]

        A0, A1 = sram["A0"], sram["A1"]
        B0, B1 = sram["B0"], sram["B1"]
        ACC, OUT = sram["ACC"], sram["OUT"]
        SK0, SK1 = sram.get("SK0", 0), sram.get("SK1", 0)

        cmds: List[Cmd32] = []
        next_ev = 1

        def new_ev() -> int:
            nonlocal next_ev
            x = next_ev
            next_ev += 1
            return x

        def load_A(dst_sram: int, tm: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.K
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m, tm*m.tileM, 0),
                         dst_addr=dst_sram,
                         size_or_M=bytes_,
                         arg0=m.tileM, arg1=m.K, arg2=m.K)

        def load_B(dst_sram: int, tn: int, sig: int) -> Cmd32:
            bytes_ = m.K * m.tileN
            return Cmd32(OP_DMA_LOAD_LINEAR, sig0=sig,
                         src_addr=bpack_tile_addr(m, tn),
                         dst_addr=dst_sram,
                         size_or_M=bytes_)

        def load_skip(dst_sram: int, tm: int, tn: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_Skip(m, tm*m.tileM, tn*m.tileN),
                         dst_addr=dst_sram,
                         size_or_M=bytes_,
                         arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def mac(A_sram: int, B_sram: int, waitA: int, waitB: int, sig: int) -> Cmd32:
            return Cmd32(OP_MAC_ACC32, wait0=waitA, wait1=waitB, sig0=sig,
                         src_addr=A_sram, dst_addr=ACC,
                         size_or_M=m.tileM, arg0=B_sram, arg1=m.tileN, arg2=m.tileK)

        def requant(tn: int, wait: int, sig: int) -> Cmd32:
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(OP_REQUANT_RELU_PC, flags=flags, wait0=wait, sig0=sig,
                         src_addr=ACC, dst_addr=OUT,
                         size_or_M=m.tileM, arg0=m.tileN, arg1=ms_tile_addr(m, tn), arg2=0)

        def add_res(skip_sram: int, wait_out: int, wait_sk: int, sig: int) -> Cmd32:
            clamp_min = (-128) & 0xFFFF
            clamp_max = (127) & 0xFFFF
            clamp_pack = (clamp_max << 16) | clamp_min
            flags = FLAG_ADD_RELU_MIN0 if (m.relu and m.zC == 0) else 0
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_ADD_I8_CLAMP, flags=flags, wait0=wait_out, wait1=wait_sk, sig0=sig,
                         src_addr=OUT, dst_addr=OUT,
                         size_or_M=bytes_, arg0=skip_sram, arg1=clamp_pack, arg2=0)

        def store(tm: int, tn: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=wait,
                         src_addr=OUT, dst_addr=addr_Y(m, tm*m.tileM, tn*m.tileN),
                         size_or_M=bytes_,
                         arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]

        # Prefetch tile0
        t0 = tiles[0]
        EA0 = new_ev(); EB0 = new_ev()
        cmds.append(load_A(A0, t0.tm, EA0))
        cmds.append(load_B(B0, t0.tn, EB0))
        if use_residual:
            ESK0 = new_ev()
            cmds.append(load_skip(SK0, t0.tm, t0.tn, ESK0))
        else:
            ESK0 = 0

        # tile loop (ping-pong)
        EA1 = EB1 = ESK1 = 0
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
                if i % 2 == 0:
                    EA1 = new_ev(); EB1 = new_ev()
                    cmds.append(load_A(alt_A, tnext.tm, EA1))
                    cmds.append(load_B(alt_B, tnext.tn, EB1))
                    if use_residual:
                        ESK1 = new_ev()
                        cmds.append(load_skip(SK1, tnext.tm, tnext.tn, ESK1))
                else:
                    EA0 = new_ev(); EB0 = new_ev()
                    cmds.append(load_A(alt_A, tnext.tm, EA0))
                    cmds.append(load_B(alt_B, tnext.tn, EB0))
                    if use_residual:
                        ESK0 = new_ev()
                        cmds.append(load_skip(SK0, tnext.tm, tnext.tn, ESK0))

            EOUT = new_ev()
            cmds.append(requant(t.tn, EACC, EOUT))

            if use_residual:
                EADD = new_ev()
                cmds.append(add_res(use_SK, EOUT, waitSK, EADD))
                wait_store = EADD
            else:
                wait_store = EOUT

            cmds.append(store(t.tm, t.tn, wait_store))

        return cmds

    # ---------------- bypass path ----------------
    def emit_bypass(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        A0, A1 = sram["A0"], sram["A1"]

        cmds: List[Cmd32] = []
        next_ev = 1

        def new_ev() -> int:
            nonlocal next_ev
            x = next_ev
            next_ev += 1
            return x

        def load_tile(dst_sram: int, tm: int, tn: int, sig: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m, tm*m.tileM, tn*m.tileN),
                         dst_addr=dst_sram,
                         size_or_M=bytes_,
                         arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def store_tile(src_sram: int, tm: int, tn: int, wait: int) -> Cmd32:
            bytes_ = m.tileM * m.tileN
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=wait,
                         src_addr=src_sram,
                         dst_addr=addr_Y(m, tm*m.tileM, tn*m.tileN),
                         size_or_M=bytes_,
                         arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]
        t0 = tiles[0]
        EA0 = new_ev()
        cmds.append(load_tile(A0, t0.tm, t0.tn, EA0))
        EA1 = 0

        for i, t in enumerate(tiles):
            use_A = A0 if (i % 2 == 0) else A1
            waitA = EA0 if (i % 2 == 0) else EA1

            if i + 1 < len(tiles):
                tnext = tiles[i+1]
                alt_A = A1 if (i % 2 == 0) else A0
                newA = new_ev()
                if i % 2 == 0:
                    EA1 = newA
                else:
                    EA0 = newA
                cmds.append(load_tile(alt_A, tnext.tm, tnext.tn, newA))

            cmds.append(store_tile(use_A, t.tm, t.tn, waitA))

        return cmds


# ============================================================
# 8) Strict simulator + bank conflict checks
# ============================================================

@dataclass
class EngineState:
    busy_until: int = 0
    # what SRAM banks this engine is touching in its current interval
    active_banks: Optional[set] = None

class CommandSimulatorV4:
    """
    - Events are signaled at command END.
    - Engine overlap is possible across DMA/MAC/ACT, like before.
    - Bank conflicts:
        compute bank set for SRAM addresses used by a cmd (src/dst that are SRAM-range)
        if two engines overlap in time and their bank sets intersect -> hazard (reported)
    """

    def __init__(self, model: ModelSpec):
        self.m = model
        self.events: Dict[int, bool] = {}
        self.pending_signals: List[Tuple[int,int]] = []  # (time_to_fire, event_id)
        self.t = 0
        self.dma = EngineState()
        self.mac = EngineState()
        self.act = EngineState()
        self.hazards: List[str] = []

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
        if c.opcode == OP_PACK_B_TILE:
            # pack kernel duration proportional to K*tileN
            K = c.size_or_M; tileN = c.arg0
            return max(1, (K*tileN) // 512)
        # REQUANT / ADD
        return max(1, c.size_or_M // 512)

    def _busy_until(self, eng: str) -> int:
        return {"DMA":self.dma.busy_until, "MAC":self.mac.busy_until, "ACT":self.act.busy_until}[eng]

    def _state(self, eng: str) -> EngineState:
        return {"DMA":self.dma, "MAC":self.mac, "ACT":self.act}[eng]

    def _fire_signals_up_to(self, t: int):
        # fire pending signals whose time <= t
        still = []
        for tfire, eid in self.pending_signals:
            if tfire <= t:
                self.events[eid] = True
            else:
                still.append((tfire, eid))
        self.pending_signals = still

    def _waits_ok(self, c: Cmd32) -> bool:
        for w in (c.wait0, c.wait1):
            if w and not self.events.get(w, False):
                return False
        return True

    def _is_sram_addr(self, addr: int) -> bool:
        # Toy heuristic: SRAM is low addresses [0 .. sram_size)
        return 0 <= addr < self.m.sram_size_bytes

    def _banks_for_addr_range(self, addr: int, size: int) -> set:
        # bank index by (addr//granularity) % banks
        banks = set()
        gran = self.m.bank_granularity
        for off in range(0, max(1, size), gran):
            b = ((addr + off) // gran) % self.m.sram_banks
            banks.add(int(b))
        return banks

    def _banks_touched(self, c: Cmd32) -> set:
        banks = set()
        # touch SRAM src/dst if in SRAM range
        if self._is_sram_addr(c.src_addr):
            banks |= self._banks_for_addr_range(c.src_addr, max(1, c.size_or_M))
        if self._is_sram_addr(c.dst_addr):
            banks |= self._banks_for_addr_range(c.dst_addr, max(1, c.size_or_M))
        # some ops have extra SRAM pointer in arg0 (e.g. MAC uses arg0=B_sram, ADD uses arg0=skip_sram)
        if c.opcode == OP_MAC_ACC32 and self._is_sram_addr(c.arg0):
            # B tile size is K*tileN bytes (approx)
            banks |= self._banks_for_addr_range(c.arg0, self.m.K * self.m.tileN)
        if c.opcode == OP_ADD_I8_CLAMP and self._is_sram_addr(c.arg0):
            banks |= self._banks_for_addr_range(c.arg0, c.size_or_M)
        if c.opcode == OP_PACK_B_TILE and self._is_sram_addr(c.src_addr):
            banks |= self._banks_for_addr_range(c.src_addr, self.m.K * self.m.tileN)
        return banks

    def _check_overlap_hazard(self, start: int, end: int, eng: str, banks: set):
        # If any other engine is busy overlapping [start,end) and bank sets intersect -> hazard
        for other_name, other_state in [("DMA", self.dma), ("MAC", self.mac), ("ACT", self.act)]:
            if other_name == eng:
                continue
            obeg = max(0, other_state.busy_until - 1)  # rough; we don't store start time, so hazard is heuristic
            oend = other_state.busy_until
            # If other engine currently busy beyond start, assume overlap
            if oend > start and other_state.active_banks is not None:
                inter = banks & other_state.active_banks
                if inter:
                    self.hazards.append(
                        f"BankHazard: {eng} banks{sorted(banks)} overlaps {other_name} banks{sorted(other_state.active_banks)} "
                        f"intersection={sorted(inter)} at t~{start}..{min(end,oend)}"
                    )

    def run(self, cmds: List[Cmd32]) -> str:
        self.events.clear()
        self.pending_signals.clear()
        self.hazards.clear()
        self.t = 0
        self.dma = EngineState()
        self.mac = EngineState()
        self.act = EngineState()

        lines: List[str] = []

        for i, c in enumerate(cmds):
            eng = self._engine(c)

            # advance until waits satisfied and engine free
            while True:
                self._fire_signals_up_to(self.t)
                if self._waits_ok(c) and self.t >= self._busy_until(eng):
                    break
                # jump to next interesting time:
                next_t = self.t + 1
                next_t = max(next_t, self._busy_until(eng))
                # also jump to next pending signal time if earlier
                if self.pending_signals:
                    next_sig_t = min(tf for tf, _ in self.pending_signals)
                    if next_sig_t < next_t:
                        next_t = next_sig_t
                self.t = next_t

            start = self.t
            dur = self._dur(c)
            end = start + dur

            banks = self._banks_touched(c)
            self._check_overlap_hazard(start, end, eng, banks)

            # set engine busy and record banks touched while active
            st = self._state(eng)
            st.busy_until = end
            st.active_banks = banks

            # schedule signals at END
            if c.sig0:
                self.pending_signals.append((end, c.sig0))
            if c.sig1:
                self.pending_signals.append((end, c.sig1))

            lines.append(f"{i:04d} t={start:6d}..{end:6d} {eng:3s} banks={sorted(banks)} {c.to_trace()}")

            # keep global time at start to allow overlap across engines
            self.t = start

        # fire any remaining signals up to makespan
        makespan = max(self.dma.busy_until, self.mac.busy_until, self.act.busy_until)
        self._fire_signals_up_to(makespan)

        lines.append("")
        lines.append(f"SIM SUMMARY: DMA_end={self.dma.busy_until}, MAC_end={self.mac.busy_until}, ACT_end={self.act.busy_until}, makespan={makespan}")
        lines.append(f"Events signaled: {sorted([k for k,v in self.events.items() if v])}")
        lines.append("")
        lines.append("BANK HAZARDS (heuristic):")
        if not self.hazards:
            lines.append("  (none)")
        else:
            for h in self.hazards:
                lines.append("  " + h)

        return "\n".join(lines)


# ============================================================
# 9) IO helper
# ============================================================

def write_bundle(out_dir: str, name: str, cmds: List[Cmd32]) -> None:
    bin_path = os.path.join(out_dir, f"{name}.bin")
    trace_path = os.path.join(out_dir, f"{name}.trace.txt")
    with open(bin_path, "wb") as fb, open(trace_path, "w", encoding="utf-8") as ft:
        for i, c in enumerate(cmds):
            fb.write(c.pack())
            ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 10) Main (v4)
# ============================================================

def main():
    out_dir = "./toy_out_v4"
    os.makedirs(out_dir, exist_ok=True)

    model = ModelSpec(
        A_base=0x8000_0000,
        B_base=0x8000_2000,
        Y_base=0x8000_4000,
        Skip_base=0x8000_9000,
        Bpack_base=0x8000_6000,
        MS_base=0x8000_7000,
        M=64, N=64, K=64,
        tileM=32, tileN=32, tileK=64,
        zC=0, relu=True,
        ms_entry_size=8,
        sram_size_bytes=256*1024,
        sram_align=64,
        sram_banks=8,
        bank_granularity=256
    )

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

    emitter = CommandEmitterV4(model)
    intervals = emitter.build_liveness(plan)
    sram_addrs, sram_dump = emitter.allocate_sram(intervals)
    with open(os.path.join(out_dir, "sram_map.txt"), "w", encoding="utf-8") as f:
        f.write(sram_dump + "\n")

    # v4 prologue: split pack into DMA + PACK_B_TILE kernel
    prologue = emitter.emit_pack_b_prologue_split(sram_addrs) if plan["use_pack"] else []
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
        "v4_changes": [
            "events signal at command END (strict sim)",
            "bank conflict hazard reporting",
            "PACK_B lowered to DMA_LOAD_2D + PACK_B_TILE micro-kernel"
        ]
    }
    with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch, f, indent=2)

    # Simulate (strict + hazards)
    sim = CommandSimulatorV4(model)
    with open(os.path.join(out_dir, "sim_path_compute.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(path_compute) + "\n")
    with open(os.path.join(out_dir, "sim_path_bypass.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(path_bypass) + "\n")
    with open(os.path.join(out_dir, "sim_prologue_packB.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue) + "\n")

    print("Generated:", out_dir)
    print("  - normalized_graph.json")
    print("  - sram_map.txt")
    print("  - prologue_packB.bin/.trace.txt + sim_prologue_packB.txt")
    print("  - path_compute.bin/.trace.txt + sim_path_compute.txt")
    print("  - path_bypass.bin/.trace.txt  + sim_path_bypass.txt")
    print("  - dispatch.json")
    print("\nEN: bank hazards are heuristic warnings; next step is auto-fixing schedule (v5).")
    print("中: bank hazard 目前是告警（heuristic），下一步 v5 我们做自动 schedule 修复/重排。")


if __name__ == "__main__":
    main()

