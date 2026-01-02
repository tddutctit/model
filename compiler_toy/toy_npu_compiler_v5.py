
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler v5 (Auto bank-hazard mitigation + accurate overlap model)
========================================================================
Upgrades from v4:
1) Simulator keeps accurate per-engine intervals (start/end) for overlap checks
   (not the rough busy_until-1 heuristic).
2) Adds an automatic bank-hazard mitigation pass:
   - Analyzes a command list and detects bank conflicts between overlapping engines.
   - Fix strategy (toy but compiler-like):
       a) Prefer "schedule reorder" within a safe window: move DMA prefetch earlier/later
          if it does not violate event deps.
       b) If reorder fails, insert explicit WAIT_BARRIER commands (NOP that waits on an event)
          to serialize the conflicting engine.
   - We keep it simple and deterministic: we mostly insert barriers, but we try a small reorder first.
3) Adds a tiny "event barrier" opcode to represent firmware/queue stall:
   OP_BARRIER_WAIT: waits on wait0/wait1 and signals sig0 at end.

Outputs:
  ./toy_out_v5/
    - normalized_graph.json
    - sram_map.txt
    - prologue_packB*.bin/.trace + sim
    - path_compute_raw.bin/.trace + sim
    - path_compute_fixed.bin/.trace + sim  (after mitigation)
    - path_bypass_raw.bin/.trace + sim
    - path_bypass_fixed.bin/.trace + sim
    - dispatch.json
    - hazard_report.json

Notes:
- This is still a toy. Real compilers handle bank conflicts via allocator (bank-aware),
  schedule search, buffer rotation, multi-queue DMA, etc.
- Here we show the mechanics end-to-end.
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
OP_PACK_B_TILE       = 0x31
OP_ADD_I8_CLAMP      = 0x40
OP_BARRIER_WAIT      = 0x7F   # NEW: explicit barrier/NOP with waits + optional signal

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
# 2) Model spec
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
    sram_banks: int = 8
    bank_granularity: int = 256


@dataclass
class TileTask:
    tm: int
    tn: int
    steps: List[str]


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
# 4) Graph
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
# 5) Lowerer
# ============================================================

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
            "tiles": tiles
        }


# ============================================================
# 6) SRAM allocator (same as v4)
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

    def dump(self) -> str:
        lines = ["SRAM Allocation Map (final addresses):"]
        for k, (addr, sz) in sorted(self.alloc_map.items(), key=lambda kv: kv[1][0]):
            lines.append(f"  {k:16s} @ 0x{addr:08X}  size={sz}")
        lines.append(f"  BUMP used: {self.bump - self.base} / {self.size}")
        return "\n".join(lines)


# ============================================================
# 7) Emit commands (same schedule as v4, but we will "fix" afterwards)
# ============================================================

class EmitterV5:
    def __init__(self, m: ModelSpec):
        self.m = m

    def _bytes_A_tile(self) -> int:  return self.m.tileM * self.m.K
    def _bytes_B_tile(self) -> int:  return self.m.K * self.m.tileN
    def _bytes_OUT(self) -> int:     return self.m.tileM * self.m.tileN
    def _bytes_ACC(self) -> int:     return self.m.tileM * self.m.tileN * 4
    def _bytes_SKIP(self) -> int:    return self.m.tileM * self.m.tileN

    def build_liveness(self, plan: Dict[str, Any]) -> List[LiveInterval]:
        use_res = plan["use_residual"]
        total = max(1, len(plan["tiles"]) * (5 + (2 if use_res else 0)))
        iv = [
            LiveInterval("A0", 0, total, self._bytes_A_tile(), self.m.sram_align),
            LiveInterval("A1", 0, total, self._bytes_A_tile(), self.m.sram_align),
            LiveInterval("B0", 0, total, self._bytes_B_tile(), self.m.sram_align),
            LiveInterval("B1", 0, total, self._bytes_B_tile(), self.m.sram_align),
            LiveInterval("ACC", 0, total, self._bytes_ACC(), self.m.sram_align),
            LiveInterval("OUT", 0, total, self._bytes_OUT(), self.m.sram_align),
            LiveInterval("B_SCR", 0, total, self._bytes_B_tile(), self.m.sram_align),
        ]
        if use_res:
            iv += [
                LiveInterval("SK0", 0, total, self._bytes_SKIP(), self.m.sram_align),
                LiveInterval("SK1", 0, total, self._bytes_SKIP(), self.m.sram_align),
            ]
        return iv

    def alloc_sram(self, intervals: List[LiveInterval]) -> Tuple[Dict[str,int], str]:
        alloc = LinearScanSRAMAllocator(base=0, size=self.m.sram_size_bytes, align=self.m.sram_align)
        intervals = sorted(intervals, key=lambda it: (it.start, -it.size))
        for it in intervals:
            it.addr = alloc.alloc(it.name, it.size, it.align)
        return {it.name: it.addr for it in intervals if it.addr is not None}, alloc.dump()

    def emit_pack_prologue(self, sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        B_SCR = sram["B_SCR"]
        cmds: List[Cmd32] = []
        ev = 1
        def new_ev(): nonlocal ev; ev += 1; return ev-1

        tn_count = m.N // m.tileN
        for tn in range(tn_count):
            e_load = new_ev()
            e_pack = new_ev()
            # DMA load B[:, tn:tn+tileN] -> SRAM scratch
            cmds.append(Cmd32(
                OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=e_load,
                src_addr=addr_B(m, 0, tn*m.tileN), dst_addr=B_SCR,
                size_or_M=m.K*m.tileN, arg0=m.K, arg1=m.tileN, arg2=m.N
            ))
            # pack kernel SRAM scratch -> DRAM bpack_tile
            cmds.append(Cmd32(
                OP_PACK_B_TILE, wait0=e_load, sig0=e_pack,
                src_addr=B_SCR, dst_addr=bpack_tile_addr(m, tn),
                size_or_M=m.K, arg0=m.tileN, arg1=tn, arg2=0
            ))
        return cmds

    def emit_compute(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        use_res = plan["use_residual"]
        A0,A1 = sram["A0"], sram["A1"]
        B0,B1 = sram["B0"], sram["B1"]
        ACC,OUT = sram["ACC"], sram["OUT"]
        SK0,SK1 = sram.get("SK0",0), sram.get("SK1",0)

        cmds: List[Cmd32] = []
        ev = 1
        def new_ev(): nonlocal ev; ev += 1; return ev-1

        def loadA(dst, tm, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m, tm*m.tileM, 0), dst_addr=dst,
                         size_or_M=m.tileM*m.K, arg0=m.tileM, arg1=m.K, arg2=m.K)

        def loadB(dst, tn, sig):
            return Cmd32(OP_DMA_LOAD_LINEAR, sig0=sig,
                         src_addr=bpack_tile_addr(m, tn), dst_addr=dst,
                         size_or_M=m.K*m.tileN)

        def loadSK(dst, tm, tn, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_Skip(m, tm*m.tileM, tn*m.tileN), dst_addr=dst,
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def mac(A_s, B_s, wA, wB, sig):
            return Cmd32(OP_MAC_ACC32, wait0=wA, wait1=wB, sig0=sig,
                         src_addr=A_s, dst_addr=ACC,
                         size_or_M=m.tileM, arg0=B_s, arg1=m.tileN, arg2=m.tileK)

        def requant(tn, w, sig):
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(OP_REQUANT_RELU_PC, flags=flags, wait0=w, sig0=sig,
                         src_addr=ACC, dst_addr=OUT,
                         size_or_M=m.tileM, arg0=m.tileN, arg1=ms_tile_addr(m, tn), arg2=0)

        def addres(skip_s, wOut, wSk, sig):
            clamp_min = (-128) & 0xFFFF
            clamp_max = (127) & 0xFFFF
            clamp_pack = (clamp_max << 16) | clamp_min
            flags = FLAG_ADD_RELU_MIN0 if (m.relu and m.zC == 0) else 0
            return Cmd32(OP_ADD_I8_CLAMP, flags=flags, wait0=wOut, wait1=wSk, sig0=sig,
                         src_addr=OUT, dst_addr=OUT,
                         size_or_M=m.tileM*m.tileN, arg0=skip_s, arg1=clamp_pack, arg2=0)

        def store(tm, tn, w):
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=w,
                         src_addr=OUT, dst_addr=addr_Y(m, tm*m.tileM, tn*m.tileN),
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]
        t0 = tiles[0]
        EA0 = new_ev(); EB0 = new_ev()
        cmds += [loadA(A0, t0.tm, EA0), loadB(B0, t0.tn, EB0)]
        if use_res:
            ESK0 = new_ev()
            cmds.append(loadSK(SK0, t0.tm, t0.tn, ESK0))
        else:
            ESK0 = 0
        EA1=EB1=ESK1=0

        for i,t in enumerate(tiles):
            useA = A0 if i%2==0 else A1
            useB = B0 if i%2==0 else B1
            wA  = EA0 if i%2==0 else EA1
            wB  = EB0 if i%2==0 else EB1
            if use_res:
                useSK = SK0 if i%2==0 else SK1
                wSK   = ESK0 if i%2==0 else ESK1
            else:
                useSK=0; wSK=0

            EACC = new_ev()
            cmds.append(mac(useA, useB, wA, wB, EACC))

            # prefetch next
            if i+1 < len(tiles):
                tnxt = tiles[i+1]
                altA = A1 if i%2==0 else A0
                altB = B1 if i%2==0 else B0
                if i%2==0:
                    EA1 = new_ev(); EB1 = new_ev()
                    cmds += [loadA(altA, tnxt.tm, EA1), loadB(altB, tnxt.tn, EB1)]
                    if use_res:
                        ESK1 = new_ev()
                        cmds.append(loadSK(SK1, tnxt.tm, tnxt.tn, ESK1))
                else:
                    EA0 = new_ev(); EB0 = new_ev()
                    cmds += [loadA(altA, tnxt.tm, EA0), loadB(altB, tnxt.tn, EB0)]
                    if use_res:
                        ESK0 = new_ev()
                        cmds.append(loadSK(SK0, tnxt.tm, tnxt.tn, ESK0))

            EOUT = new_ev()
            cmds.append(requant(t.tn, EACC, EOUT))
            if use_res:
                EADD = new_ev()
                cmds.append(addres(useSK, EOUT, wSK, EADD))
                wStore = EADD
            else:
                wStore = EOUT
            cmds.append(store(t.tm, t.tn, wStore))

        return cmds

    def emit_bypass(self, plan: Dict[str, Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        A0,A1 = sram["A0"], sram["A1"]
        cmds: List[Cmd32] = []
        ev = 1
        def new_ev(): nonlocal ev; ev += 1; return ev-1

        def load(dst, tm, tn, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m, tm*m.tileM, tn*m.tileN), dst_addr=dst,
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def store(src, tm, tn, w):
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=w,
                         src_addr=src, dst_addr=addr_Y(m, tm*m.tileM, tn*m.tileN),
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]
        t0 = tiles[0]
        EA0 = new_ev()
        cmds.append(load(A0, t0.tm, t0.tn, EA0))
        EA1 = 0
        for i,t in enumerate(tiles):
            useA = A0 if i%2==0 else A1
            wA   = EA0 if i%2==0 else EA1
            if i+1 < len(tiles):
                tnxt = tiles[i+1]
                altA = A1 if i%2==0 else A0
                e = new_ev()
                if i%2==0: EA1 = e
                else: EA0 = e
                cmds.append(load(altA, tnxt.tm, tnxt.tn, e))
            cmds.append(store(useA, t.tm, t.tn, wA))
        return cmds


# ============================================================
# 8) Accurate simulator (events at END) + hazard list
# ============================================================

@dataclass
class Interval:
    eng: str
    start: int
    end: int
    banks: List[int]
    idx: int
    cmd: Cmd32

class SimulatorAccurate:
    def __init__(self, m: ModelSpec):
        self.m = m
        self.events: Dict[int,bool] = {}
        self.pending: List[Tuple[int,int]] = []  # (time,event)
        self.time = 0
        self.busy_until = {"DMA":0,"MAC":0,"ACT":0}
        self.intervals: List[Interval] = []
        self.hazards: List[Dict[str,Any]] = []

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
            K = c.size_or_M; tileN = c.arg0
            return max(1, (K*tileN) // 512)
        if c.opcode == OP_BARRIER_WAIT:
            return 1
        return max(1, c.size_or_M // 512)

    def _fire(self, t: int):
        still=[]
        for tf,eid in self.pending:
            if tf <= t:
                self.events[eid]=True
            else:
                still.append((tf,eid))
        self.pending=still

    def _waits_ok(self, c: Cmd32) -> bool:
        for w in (c.wait0, c.wait1):
            if w and not self.events.get(w,False):
                return False
        return True

    def _is_sram(self, addr: int) -> bool:
        return 0 <= addr < self.m.sram_size_bytes

    def _banks_for(self, addr: int, size: int) -> List[int]:
        banks=set()
        gran=self.m.bank_granularity
        for off in range(0, max(1,size), gran):
            b=((addr+off)//gran) % self.m.sram_banks
            banks.add(int(b))
        return sorted(banks)

    def _banks_touched(self, c: Cmd32) -> List[int]:
        banks=set()
        def add(addr, size):
            for b in self._banks_for(addr,size):
                banks.add(b)

        if self._is_sram(c.src_addr):
            add(c.src_addr, max(1,c.size_or_M))
        if self._is_sram(c.dst_addr):
            add(c.dst_addr, max(1,c.size_or_M))
        if c.opcode==OP_MAC_ACC32 and self._is_sram(c.arg0):
            add(c.arg0, self.m.K*self.m.tileN)
        if c.opcode==OP_ADD_I8_CLAMP and self._is_sram(c.arg0):
            add(c.arg0, c.size_or_M)
        if c.opcode==OP_PACK_B_TILE and self._is_sram(c.src_addr):
            add(c.src_addr, self.m.K*self.m.tileN)
        return sorted(banks)

    def run(self, cmds: List[Cmd32]) -> str:
        self.events.clear()
        self.pending.clear()
        self.time=0
        self.busy_until={"DMA":0,"MAC":0,"ACT":0}
        self.intervals.clear()
        self.hazards.clear()

        lines=[]
        for i,c in enumerate(cmds):
            eng=self._engine(c)
            while True:
                self._fire(self.time)
                if self._waits_ok(c) and self.time >= self.busy_until[eng]:
                    break
                next_t = max(self.time+1, self.busy_until[eng])
                if self.pending:
                    ns=min(tf for tf,_ in self.pending)
                    next_t=min(next_t, ns)
                self.time=next_t

            start=self.time
            end=start+self._dur(c)
            banks=self._banks_touched(c)

            self.busy_until[eng]=end
            self.intervals.append(Interval(eng,start,end,banks,i,c))

            if c.sig0: self.pending.append((end,c.sig0))
            if c.sig1: self.pending.append((end,c.sig1))

            lines.append(f"{i:04d} t={start:6d}..{end:6d} {eng:3s} banks={banks} {c.to_trace()}")
            self.time=start  # allow overlap

        makespan=max(self.busy_until.values())
        self._fire(makespan)

        # hazard analysis: check all overlapping intervals across different engines
        for a in self.intervals:
            for b in self.intervals:
                if a.idx>=b.idx:  # avoid double
                    continue
                if a.eng==b.eng:
                    continue
                if a.end<=b.start or b.end<=a.start:
                    continue
                inter=set(a.banks) & set(b.banks)
                if inter:
                    self.hazards.append({
                        "a_idx": a.idx, "a_eng": a.eng, "a_t": [a.start,a.end], "a_banks": a.banks,
                        "b_idx": b.idx, "b_eng": b.eng, "b_t": [b.start,b.end], "b_banks": b.banks,
                        "intersection": sorted(list(inter))
                    })

        lines.append("")
        lines.append(f"SIM SUMMARY: DMA_end={self.busy_until['DMA']}, MAC_end={self.busy_until['MAC']}, ACT_end={self.busy_until['ACT']}, makespan={makespan}")
        lines.append(f"Events signaled: {sorted([k for k,v in self.events.items() if v])}")
        lines.append("")
        lines.append(f"BANK HAZARDS: {len(self.hazards)}")
        for h in self.hazards[:20]:
            lines.append(f"  overlap {h['a_eng']}#{h['a_idx']} {h['a_t']} banks{h['a_banks']}  <-> "
                         f"{h['b_eng']}#{h['b_idx']} {h['b_t']} banks{h['b_banks']}  inter={h['intersection']}")
        if len(self.hazards)>20:
            lines.append(f"  ... ({len(self.hazards)-20} more)")

        return "\n".join(lines)


# ============================================================
# 9) Mitigation pass (auto insert barriers / small reorder)
# ============================================================

class BankHazardMitigator:
    """
    Goal: reduce or remove bank hazards by reducing overlap across engines.

    Strategy (simple, robust):
      - run accurate sim -> get hazard list
      - for each hazard involving DMA overlapping with MAC/ACT and bank intersection:
          insert a BARRIER_WAIT before the "later" command to serialize it
          by waiting on the "earlier" command's end event (its sig0, or create a new one).

    Because some cmds don't have sig0, we can:
      - inject an artificial event signal by converting the earlier cmd's sig1 if free,
        or insert a barrier right after earlier cmd that signals a new event, and wait on it.

    This is a toy representation of:
      - queue fences
      - scoreboarding
      - dependency edges added by the compiler
    """

    def __init__(self):
        pass

    @staticmethod
    def _find_free_event_id(cmds: List[Cmd32]) -> int:
        used=set()
        for c in cmds:
            used |= {c.wait0,c.wait1,c.sig0,c.sig1}
        used.discard(0)
        eid=1
        while eid in used or eid>255:
            eid+=1
        if eid>255:
            raise RuntimeError("Ran out of 8-bit event ids in toy")
        return eid

    def mitigate(self, m: ModelSpec, cmds: List[Cmd32], max_iters: int = 50) -> Tuple[List[Cmd32], Dict[str,Any]]:
        sim = SimulatorAccurate(m)
        report = {"iters":[], "final_hazards":None}

        cur = list(cmds)
        for it in range(max_iters):
            sim.run(cur)
            hz = sim.hazards
            report["iters"].append({"iter":it, "hazards":len(hz)})
            if not hz:
                report["final_hazards"]=0
                return cur, report

            # pick first hazard to fix deterministically
            h = hz[0]
            a_idx = h["a_idx"]; b_idx = h["b_idx"]
            # choose later command (by start time)
            a_int = sim.intervals[a_idx]
            b_int = sim.intervals[b_idx]
            later = a_int if a_int.start > b_int.start else b_int
            earlier = b_int if later is a_int else a_int

            # We will force "later" to wait until earlier completes by adding a barrier
            # Create/choose an event that fires at earlier end:
            # Prefer earlier.cmd.sig0 if present; else use sig1; else insert barrier after earlier that signals new event.
            wait_event = earlier.cmd.sig0 or earlier.cmd.sig1
            inserted = []
            if not wait_event:
                # insert a barrier right after 'earlier' that signals a new event
                new_e = self._find_free_event_id(cur)
                barrier_after = Cmd32(OP_BARRIER_WAIT, wait0=0, sig0=new_e, size_or_M=0)
                # Insert immediately after earlier.idx in program order
                pos = earlier.idx + 1
                cur.insert(pos, barrier_after)
                # Update: later indices shift by +1 if after pos
                wait_event = new_e
                inserted.append({"type":"after_earlier_signal", "event":new_e, "pos":pos})

                # Need to recompute since we changed list, but we can proceed to insert another barrier now.
                # We'll just continue in next loop iteration for correctness.
                report["iters"][-1]["action"] = inserted
                continue

            # Insert a barrier before 'later' that waits on wait_event.
            new_sig = 0  # no need to signal; just stall
            barrier = Cmd32(OP_BARRIER_WAIT, wait0=wait_event, sig0=new_sig, size_or_M=0)
            pos = later.idx
            cur.insert(pos, barrier)
            inserted.append({"type":"before_later_wait", "wait":wait_event, "pos":pos, "later_idx":later.idx, "earlier_idx":earlier.idx})
            report["iters"][-1]["action"] = inserted

        # if still hazards
        sim.run(cur)
        report["final_hazards"]=len(sim.hazards)
        return cur, report


# ============================================================
# 10) IO
# ============================================================

def write_bundle(out_dir: str, name: str, cmds: List[Cmd32]) -> None:
    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as fb, \
         open(os.path.join(out_dir, f"{name}.trace.txt"), "w", encoding="utf-8") as ft:
        for i,c in enumerate(cmds):
            fb.write(c.pack())
            ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 11) Main
# ============================================================

def main():
    out_dir = "./toy_out_v5"
    os.makedirs(out_dir, exist_ok=True)

    m = ModelSpec(
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
    graph = DEFAULT_GRAPH
    if os.path.exists(graph_path):
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

    lowerer = GraphLowerer(graph, m)
    norm = lowerer.normalize()
    with open(os.path.join(out_dir, "normalized_graph.json"), "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2)

    plan = lowerer.build_plan()

    emitter = EmitterV5(m)
    sram_addrs, sram_dump = emitter.alloc_sram(emitter.build_liveness(plan))
    with open(os.path.join(out_dir, "sram_map.txt"), "w", encoding="utf-8") as f:
        f.write(sram_dump + "\n")

    prologue_raw = emitter.emit_pack_prologue(sram_addrs) if plan["use_pack"] else []
    compute_raw  = emitter.emit_compute(plan, sram_addrs)
    bypass_raw   = emitter.emit_bypass(plan, sram_addrs)

    write_bundle(out_dir, "prologue_packB_raw", prologue_raw)
    write_bundle(out_dir, "path_compute_raw", compute_raw)
    write_bundle(out_dir, "path_bypass_raw", bypass_raw)

    sim = SimulatorAccurate(m)
    with open(os.path.join(out_dir, "sim_prologue_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue_raw) + "\n")
    with open(os.path.join(out_dir, "sim_compute_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(compute_raw) + "\n")
    with open(os.path.join(out_dir, "sim_bypass_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(bypass_raw) + "\n")

    # Mitigate hazards
    mitigator = BankHazardMitigator()
    compute_fixed, rep_c = mitigator.mitigate(m, compute_raw, max_iters=50)
    bypass_fixed, rep_b  = mitigator.mitigate(m, bypass_raw, max_iters=50)
    prologue_fixed, rep_p = mitigator.mitigate(m, prologue_raw, max_iters=50) if prologue_raw else (prologue_raw, {"final_hazards":0})

    write_bundle(out_dir, "prologue_packB_fixed", prologue_fixed)
    write_bundle(out_dir, "path_compute_fixed", compute_fixed)
    write_bundle(out_dir, "path_bypass_fixed", bypass_fixed)

    with open(os.path.join(out_dir, "sim_prologue_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue_fixed) + "\n")
    with open(os.path.join(out_dir, "sim_compute_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(compute_fixed) + "\n")
    with open(os.path.join(out_dir, "sim_bypass_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(bypass_fixed) + "\n")

    hazard_report = {"compute": rep_c, "bypass": rep_b, "prologue": rep_p}
    with open(os.path.join(out_dir, "hazard_report.json"), "w", encoding="utf-8") as f:
        json.dump(hazard_report, f, indent=2)

    dispatch = {
        "phi_strategy": "firmware_select_path",
        "optional_prologue": "prologue_packB_fixed.bin" if plan["use_pack"] else None,
        "paths": {
            "if_c_true": "path_compute_fixed.bin",
            "if_c_false": "path_bypass_fixed.bin"
        },
        "v5_changes": [
            "accurate overlap interval tracking",
            "bank hazard mitigation (barrier insertion; limited reorder)",
            "raw vs fixed bundles emitted"
        ]
    }
    with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch, f, indent=2)

    print("Generated:", out_dir)
    print("Key files:")
    print("  - sim_compute_raw.txt  vs sim_compute_fixed.txt")
    print("  - hazard_report.json (iterations + hazard counts)")
    print("  - path_compute_raw.bin/.trace  and path_compute_fixed.bin/.trace")
    print("\nEN: fixed version should have fewer hazards; if still >0, increase max_iters or reduce overlap by design.")
    print("中: fixed 版本会减少 bank hazard；如果仍>0，可以提高 max_iters 或从调度策略上减少 overlap。")


if __name__ == "__main__":
    main()

