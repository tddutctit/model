
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler v6
===================
What v6 adds (compared to v5):
1) Bank-aware SRAM placement (allocator upgrade)
   - Instead of "first-fit bump", we try to place hot buffers (A0/A1/B0/B1/ACC/OUT/SK0/SK1/B_SCR)
     so their SRAM bank sets overlap as little as possible.
   - This reduces hazards at the root (better than inserting barriers later).

2) Simple list-scheduling style reorder (scheduler upgrade)
   - "Hoist prefetch DMA loads earlier" when it is safe:
     * DMA ops with no waits can be moved earlier in the stream
     * must preserve per-destination order (same dst buffer cannot be reordered relative to itself)
     * must not move past a command that writes/reads the same dst region (toy conservative rule)

3) Keep v5's accurate simulator + hazard report + fallback mitigator
   - Pipeline is now:
       raw_emit -> (bank-aware SRAM) -> (hoist DMA) -> simulate
       if hazards remain -> barrier mitigator -> simulate
   - You get:
       path_compute_raw
       path_compute_sched (bank-aware + hoist)
       path_compute_fixed (sched + barrier fix if needed)

Output:
  ./toy_out_v6/
    - sram_map_bankaware.txt
    - sim_compute_raw.txt / sim_compute_sched.txt / sim_compute_fixed.txt
    - hazard_report.json

This is still a toy, but now it matches real compiler structure:
Allocator (bank-aware) + Scheduler (reorder) + Final hazard fixer (barriers).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import struct
import json
import os
import copy


# ============================================================
# 1) ABI + opcodes
# ============================================================

OP_DMA_LOAD_2D       = 0x01
OP_DMA_STORE_2D      = 0x02
OP_DMA_LOAD_LINEAR   = 0x03
OP_MAC_ACC32         = 0x10
OP_REQUANT_RELU_PC   = 0x21
OP_PACK_B_TILE       = 0x31
OP_ADD_I8_CLAMP      = 0x40
OP_BARRIER_WAIT      = 0x7F

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
    return m.A_base + (row * m.K + col)

def addr_Y(m: ModelSpec, row: int, col: int) -> int:
    return m.Y_base + (row * m.N + col)

def addr_Skip(m: ModelSpec, row: int, col: int) -> int:
    return m.Skip_base + (row * m.N + col)

def bpack_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.Bpack_base + n_tile_index * (m.K * m.tileN)

def ms_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.MS_base + n_tile_index * m.tileN * m.ms_entry_size

def addr_B(m: ModelSpec, row: int, col: int) -> int:
    return m.B_base + (row * m.N + col)


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
    def __init__(self, graph: Dict[str, Any], m: ModelSpec):
        self.g = graph
        self.m = m

    def normalize(self) -> Dict[str, Any]:
        g = copy.deepcopy(self.g)
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

        return {"use_pack": use_pack, "use_residual": use_residual, "tiles": tiles}


# ============================================================
# 6) Bank-aware allocator
# ============================================================

@dataclass
class BufReq:
    name: str
    size: int
    align: int
    hot: bool = True   # hot buffers matter more for bank overlap score


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class BankAwareAllocator:
    """
    Greedy placement with bank-overlap scoring.
    - We place buffers in a chosen order (hot-first).
    - For each buffer, try candidate addresses within a search range, pick lowest score.
    - Score = sum over placed buffers of |banks(new) ∩ banks(old)| (weighted).
    """

    def __init__(self, m: ModelSpec, base: int = 0):
        self.m = m
        self.base = base
        self.limit = base + m.sram_size_bytes
        self.placements: Dict[str, Tuple[int,int,List[int],bool]] = {}  # name -> (addr,size,banks,hot)
        self.used_ranges: List[Tuple[int,int]] = []  # [start,end)

    def banks_for_range(self, addr: int, size: int) -> List[int]:
        banks: Set[int] = set()
        gran = self.m.bank_granularity
        for off in range(0, max(1, size), gran):
            b = ((addr + off) // gran) % self.m.sram_banks
            banks.add(int(b))
        return sorted(banks)

    def overlaps_used(self, addr: int, size: int) -> bool:
        s = addr
        e = addr + size
        for us, ue in self.used_ranges:
            if not (e <= us or ue <= s):
                return True
        return False

    def mark_used(self, addr: int, size: int):
        self.used_ranges.append((addr, addr+size))
        self.used_ranges.sort()

    def score_addr(self, banks: List[int], hot: bool) -> int:
        s = 0
        bset = set(banks)
        for _, (_, _, obanks, ohot) in self.placements.items():
            inter = len(bset & set(obanks))
            w = 3 if (hot and ohot) else 1
            s += w * inter
        return s

    def alloc_one(self, req: BufReq, search_bytes: int = 16*1024) -> int:
        # candidate scan: step by align or bank_granularity (whichever larger), within a window
        step = max(req.align, self.m.bank_granularity)
        # start from base or after max end
        bump = self.base
        if self.used_ranges:
            bump = max(e for _, e in self.used_ranges)
        bump = align_up(bump, req.align)

        best = None
        # try within [base, bump+search_bytes]
        start = self.base
        end = min(self.limit, bump + search_bytes)

        for addr in range(start, end, step):
            addr = align_up(addr, req.align)
            if addr + req.size > self.limit:
                continue
            if self.overlaps_used(addr, req.size):
                continue
            banks = self.banks_for_range(addr, req.size)
            sc = self.score_addr(banks, req.hot)
            cand = (sc, addr, banks)
            if best is None or cand < best:
                best = cand

        if best is None:
            # fallback: bump at end
            addr = bump
            if addr + req.size > self.limit:
                raise MemoryError(f"SRAM overflow placing {req.name}")
            banks = self.banks_for_range(addr, req.size)
        else:
            _, addr, banks = best

        self.placements[req.name] = (addr, req.size, banks, req.hot)
        self.mark_used(addr, req.size)
        return addr

    def alloc_all(self, reqs: List[BufReq]) -> Dict[str,int]:
        # hot first, larger first
        order = sorted(reqs, key=lambda r: (not r.hot, -r.size, r.name))
        for r in order:
            self.alloc_one(r)
        return {k:v[0] for k,v in self.placements.items()}

    def dump(self) -> str:
        lines = ["SRAM Bank-Aware Placement:"]
        for name,(addr,sz,banks,hot) in sorted(self.placements.items(), key=lambda kv: kv[1][0]):
            lines.append(f"  {name:16s} @ 0x{addr:08X}  size={sz:6d}  banks={banks}  hot={hot}")
        return "\n".join(lines)


# ============================================================
# 7) Emitter (same functional behavior as v5)
# ============================================================

class Emitter:
    def __init__(self, m: ModelSpec):
        self.m = m

    def sizes(self, plan: Dict[str,Any]) -> List[BufReq]:
        m = self.m
        use_res = plan["use_residual"]
        reqs = [
            BufReq("A0",   m.tileM*m.K,     m.sram_align, hot=True),
            BufReq("A1",   m.tileM*m.K,     m.sram_align, hot=True),
            BufReq("B0",   m.K*m.tileN,     m.sram_align, hot=True),
            BufReq("B1",   m.K*m.tileN,     m.sram_align, hot=True),
            BufReq("ACC",  m.tileM*m.tileN*4, m.sram_align, hot=True),
            BufReq("OUT",  m.tileM*m.tileN, m.sram_align, hot=True),
            BufReq("B_SCR",m.K*m.tileN,     m.sram_align, hot=False),  # scratch less hot
        ]
        if use_res:
            reqs += [
                BufReq("SK0", m.tileM*m.tileN, m.sram_align, hot=True),
                BufReq("SK1", m.tileM*m.tileN, m.sram_align, hot=True),
            ]
        return reqs

    def emit_pack_prologue(self, sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        B_SCR = sram["B_SCR"]
        cmds: List[Cmd32] = []
        ev = 1
        def new_ev(): nonlocal ev; ev += 1; return ev-1

        for tn in range(m.N // m.tileN):
            e_load = new_ev()
            e_pack = new_ev()
            cmds.append(Cmd32(
                OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=e_load,
                src_addr=addr_B(m, 0, tn*m.tileN), dst_addr=B_SCR,
                size_or_M=m.K*m.tileN, arg0=m.K, arg1=m.tileN, arg2=m.N
            ))
            cmds.append(Cmd32(
                OP_PACK_B_TILE, wait0=e_load, sig0=e_pack,
                src_addr=B_SCR, dst_addr=bpack_tile_addr(m, tn),
                size_or_M=m.K, arg0=m.tileN, arg1=tn, arg2=0
            ))
        return cmds

    def emit_compute(self, plan: Dict[str,Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        use_res = plan["use_residual"]
        A0,A1 = sram["A0"], sram["A1"]
        B0,B1 = sram["B0"], sram["B1"]
        ACC,OUT = sram["ACC"], sram["OUT"]
        SK0,SK1 = sram.get("SK0",0), sram.get("SK1",0)

        cmds: List[Cmd32] = []
        ev=1
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
        EA0=new_ev(); EB0=new_ev()
        cmds += [loadA(A0,t0.tm,EA0), loadB(B0,t0.tn,EB0)]
        if use_res:
            ESK0=new_ev()
            cmds.append(loadSK(SK0,t0.tm,t0.tn,ESK0))
        else:
            ESK0=0
        EA1=EB1=ESK1=0

        for i,t in enumerate(tiles):
            useA = A0 if i%2==0 else A1
            useB = B0 if i%2==0 else B1
            wA   = EA0 if i%2==0 else EA1
            wB   = EB0 if i%2==0 else EB1
            if use_res:
                useSK = SK0 if i%2==0 else SK1
                wSK   = ESK0 if i%2==0 else ESK1
            else:
                useSK=0; wSK=0

            EACC=new_ev()
            cmds.append(mac(useA,useB,wA,wB,EACC))

            # prefetch next tile
            if i+1 < len(tiles):
                tnxt = tiles[i+1]
                altA = A1 if i%2==0 else A0
                altB = B1 if i%2==0 else B0
                if i%2==0:
                    EA1=new_ev(); EB1=new_ev()
                    cmds += [loadA(altA,tnxt.tm,EA1), loadB(altB,tnxt.tn,EB1)]
                    if use_res:
                        ESK1=new_ev()
                        cmds.append(loadSK(SK1,tnxt.tm,tnxt.tn,ESK1))
                else:
                    EA0=new_ev(); EB0=new_ev()
                    cmds += [loadA(altA,tnxt.tm,EA0), loadB(altB,tnxt.tn,EB0)]
                    if use_res:
                        ESK0=new_ev()
                        cmds.append(loadSK(SK0,tnxt.tm,tnxt.tn,ESK0))

            EOUT=new_ev()
            cmds.append(requant(t.tn,EACC,EOUT))

            if use_res:
                EADD=new_ev()
                cmds.append(addres(useSK,EOUT,wSK,EADD))
                wStore=EADD
            else:
                wStore=EOUT

            cmds.append(store(t.tm,t.tn,wStore))

        return cmds

    def emit_bypass(self, plan: Dict[str,Any], sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        A0,A1 = sram["A0"], sram["A1"]
        tiles: List[TileTask] = plan["tiles"]
        cmds: List[Cmd32] = []
        ev=1
        def new_ev(): nonlocal ev; ev+=1; return ev-1

        def load(dst, tm, tn, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m, tm*m.tileM, tn*m.tileN), dst_addr=dst,
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def store(src, tm, tn, w):
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=w,
                         src_addr=src, dst_addr=addr_Y(m, tm*m.tileM, tn*m.tileN),
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        t0=tiles[0]
        EA0=new_ev()
        cmds.append(load(A0,t0.tm,t0.tn,EA0))
        EA1=0
        for i,t in enumerate(tiles):
            useA = A0 if i%2==0 else A1
            wA   = EA0 if i%2==0 else EA1
            if i+1 < len(tiles):
                tnxt=tiles[i+1]
                altA=A1 if i%2==0 else A0
                e=new_ev()
                if i%2==0: EA1=e
                else: EA0=e
                cmds.append(load(altA,tnxt.tm,tnxt.tn,e))
            cmds.append(store(useA,t.tm,t.tn,wA))
        return cmds


# ============================================================
# 8) Accurate simulator + bank hazards (same as v5, kept)
# ============================================================

@dataclass
class Interval:
    eng: str
    start: int
    end: int
    banks: List[int]
    idx: int
    cmd: Cmd32

class Simulator:
    def __init__(self, m: ModelSpec):
        self.m = m
        self.events: Dict[int,bool] = {}
        self.pending: List[Tuple[int,int]] = []
        self.time=0
        self.busy_until={"DMA":0,"MAC":0,"ACT":0}
        self.intervals: List[Interval]=[]
        self.hazards: List[Dict[str,Any]]=[]

    def _engine(self,c:Cmd32)->str:
        if c.opcode in (OP_DMA_LOAD_2D,OP_DMA_STORE_2D,OP_DMA_LOAD_LINEAR):
            return "DMA"
        if c.opcode==OP_MAC_ACC32:
            return "MAC"
        return "ACT"

    def _dur(self,c:Cmd32)->int:
        if c.opcode in (OP_DMA_LOAD_2D,OP_DMA_STORE_2D,OP_DMA_LOAD_LINEAR):
            return max(1, c.size_or_M//256)
        if c.opcode==OP_MAC_ACC32:
            M=c.size_or_M; N=c.arg1; K=c.arg2
            return max(1, (M*N*K)//4096)
        if c.opcode==OP_PACK_B_TILE:
            K=c.size_or_M; tileN=c.arg0
            return max(1, (K*tileN)//512)
        if c.opcode==OP_BARRIER_WAIT:
            return 1
        return max(1, c.size_or_M//512)

    def _fire(self,t:int):
        still=[]
        for tf,eid in self.pending:
            if tf<=t: self.events[eid]=True
            else: still.append((tf,eid))
        self.pending=still

    def _waits_ok(self,c:Cmd32)->bool:
        for w in (c.wait0,c.wait1):
            if w and not self.events.get(w,False):
                return False
        return True

    def _is_sram(self,addr:int)->bool:
        return 0<=addr<self.m.sram_size_bytes

    def _banks_for(self,addr:int,size:int)->List[int]:
        banks=set()
        gran=self.m.bank_granularity
        for off in range(0, max(1,size), gran):
            b=((addr+off)//gran) % self.m.sram_banks
            banks.add(int(b))
        return sorted(banks)

    def _banks_touched(self,c:Cmd32)->List[int]:
        banks=set()
        def add(addr,size):
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

    def run(self,cmds:List[Cmd32])->str:
        self.events.clear(); self.pending.clear(); self.time=0
        self.busy_until={"DMA":0,"MAC":0,"ACT":0}
        self.intervals.clear(); self.hazards.clear()

        lines=[]
        for i,c in enumerate(cmds):
            eng=self._engine(c)
            while True:
                self._fire(self.time)
                if self._waits_ok(c) and self.time>=self.busy_until[eng]:
                    break
                next_t=max(self.time+1, self.busy_until[eng])
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
            self.time=start

        makespan=max(self.busy_until.values())
        self._fire(makespan)

        # hazard analysis
        for a in self.intervals:
            for b in self.intervals:
                if a.idx>=b.idx or a.eng==b.eng:
                    continue
                if a.end<=b.start or b.end<=a.start:
                    continue
                inter=set(a.banks)&set(b.banks)
                if inter:
                    self.hazards.append({
                        "a_idx":a.idx,"a_eng":a.eng,"a_t":[a.start,a.end],"a_banks":a.banks,
                        "b_idx":b.idx,"b_eng":b.eng,"b_t":[b.start,b.end],"b_banks":b.banks,
                        "intersection":sorted(list(inter))
                    })

        lines.append("")
        lines.append(f"SIM SUMMARY: DMA_end={self.busy_until['DMA']}, MAC_end={self.busy_until['MAC']}, ACT_end={self.busy_until['ACT']}, makespan={makespan}")
        lines.append(f"BANK HAZARDS: {len(self.hazards)}")
        for h in self.hazards[:15]:
            lines.append(f"  {h['a_eng']}#{h['a_idx']} {h['a_t']} banks{h['a_banks']}  <-> "
                         f"{h['b_eng']}#{h['b_idx']} {h['b_t']} banks{h['b_banks']}  inter={h['intersection']}")
        if len(self.hazards)>15:
            lines.append(f"  ... ({len(self.hazards)-15} more)")
        return "\n".join(lines)


# ============================================================
# 9) Scheduler: Hoist DMA prefetch
# ============================================================

class DmaHoister:
    """
    Conservative "list scheduling" pass:
    - Identify DMA commands with wait0=wait1=0 (independent prefetch loads)
    - Move them as early as possible while preserving:
        * relative order among DMA commands that write the same dst_addr (same buffer)
        * do not move across a command that reads/writes the same dst region (dst hazard)
    This models classic: "issue prefetch early; compute waits by events anyway".
    """

    DMA_OPS = {OP_DMA_LOAD_2D, OP_DMA_LOAD_LINEAR}

    def __init__(self, m: ModelSpec):
        self.m = m

    def _is_dma_prefetch(self, c: Cmd32) -> bool:
        return c.opcode in self.DMA_OPS and c.wait0 == 0 and c.wait1 == 0

    def _touch_key(self, c: Cmd32) -> Tuple[int,int]:
        # treat dst range as hazard key if dst is SRAM
        return (c.dst_addr, c.size_or_M)

    def hoist(self, cmds: List[Cmd32]) -> List[Cmd32]:
        out = list(cmds)

        # For each command i that is a prefetch DMA, try to swap it leftwards step-by-step.
        last_pos_for_dst: Dict[int,int] = {}  # dst_addr -> last position encountered in scan
        i = 0
        while i < len(out):
            c = out[i]
            if not self._is_dma_prefetch(c):
                i += 1
                continue

            dst = c.dst_addr
            # Cannot move left of last prefetch to same dst (preserve per-dst order)
            min_pos = last_pos_for_dst.get(dst, 0)

            j = i
            while j > min_pos:
                prev = out[j-1]

                # barrier: cannot cross if prev writes same dst (any opcode) or reads same dst (toy conservative)
                if prev.dst_addr == dst:
                    break
                if prev.src_addr == dst:
                    break
                # also don't move above a store (keeps output stores near end, toy)
                if prev.opcode == OP_DMA_STORE_2D:
                    break
                # cannot cross if prev depends on this DMA's signal? (signal is forward anyway)
                # Safe in this toy because waits are explicit events. We only hoist independent DMA.
                out[j], out[j-1] = out[j-1], out[j]
                j -= 1

            # update last_pos_for_dst for this dst at its new position
            last_pos_for_dst[dst] = j
            # continue scan after original i shifts; safest: increment i
            i += 1

        return out


# ============================================================
# 10) Final fixer: barrier insertion (v5 style)
# ============================================================

class BarrierMitigator:
    def _find_free_event_id(self, cmds: List[Cmd32]) -> int:
        used=set()
        for c in cmds:
            used |= {c.wait0,c.wait1,c.sig0,c.sig1}
        used.discard(0)
        eid=1
        while eid in used or eid>255:
            eid+=1
        if eid>255:
            raise RuntimeError("No free event ids")
        return eid

    def mitigate(self, m: ModelSpec, cmds: List[Cmd32], max_iters: int = 60) -> Tuple[List[Cmd32], Dict[str,Any]]:
        sim = Simulator(m)
        cur = list(cmds)
        rep = {"iters":[], "final_hazards":None}

        for it in range(max_iters):
            sim.run(cur)
            hz = sim.hazards
            rep["iters"].append({"iter":it, "hazards":len(hz)})
            if not hz:
                rep["final_hazards"]=0
                return cur, rep

            h = hz[0]
            a = sim.intervals[h["a_idx"]]
            b = sim.intervals[h["b_idx"]]
            later = a if a.start > b.start else b
            earlier = b if later is a else a

            wait_event = earlier.cmd.sig0 or earlier.cmd.sig1
            if not wait_event:
                new_e = self._find_free_event_id(cur)
                cur.insert(earlier.idx+1, Cmd32(OP_BARRIER_WAIT, sig0=new_e))
                rep["iters"][-1]["action"] = {"type":"insert_after_earlier_signal", "event":new_e, "pos":earlier.idx+1}
                continue

            cur.insert(later.idx, Cmd32(OP_BARRIER_WAIT, wait0=wait_event))
            rep["iters"][-1]["action"] = {"type":"insert_before_later_wait", "wait":wait_event, "pos":later.idx}
        sim.run(cur)
        rep["final_hazards"]=len(sim.hazards)
        return cur, rep


# ============================================================
# 11) IO
# ============================================================

def write_bundle(out_dir: str, name: str, cmds: List[Cmd32]) -> None:
    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as fb, \
         open(os.path.join(out_dir, f"{name}.trace.txt"), "w", encoding="utf-8") as ft:
        for i,c in enumerate(cmds):
            fb.write(c.pack())
            ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 12) Main v6
# ============================================================

def main():
    out_dir = "./toy_out_v6"
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

    # ---------- v6 allocator ----------
    emitter = Emitter(m)
    reqs = emitter.sizes(plan)
    alloc = BankAwareAllocator(m)
    sram = alloc.alloc_all(reqs)
    with open(os.path.join(out_dir, "sram_map_bankaware.txt"), "w", encoding="utf-8") as f:
        f.write(alloc.dump() + "\n")

    # ---------- raw emit ----------
    prologue_raw = emitter.emit_pack_prologue(sram) if plan["use_pack"] else []
    compute_raw  = emitter.emit_compute(plan, sram)
    bypass_raw   = emitter.emit_bypass(plan, sram)

    write_bundle(out_dir, "prologue_raw", prologue_raw)
    write_bundle(out_dir, "compute_raw", compute_raw)
    write_bundle(out_dir, "bypass_raw", bypass_raw)

    sim = Simulator(m)
    with open(os.path.join(out_dir, "sim_compute_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(compute_raw) + "\n")
    with open(os.path.join(out_dir, "sim_bypass_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(bypass_raw) + "\n")
    with open(os.path.join(out_dir, "sim_prologue_raw.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue_raw) + "\n")

    # ---------- scheduler hoist ----------
    hoister = DmaHoister(m)
    compute_sched = hoister.hoist(compute_raw)
    bypass_sched  = hoister.hoist(bypass_raw)
    prologue_sched = hoister.hoist(prologue_raw)

    write_bundle(out_dir, "compute_sched", compute_sched)
    write_bundle(out_dir, "bypass_sched", bypass_sched)
    write_bundle(out_dir, "prologue_sched", prologue_sched)

    with open(os.path.join(out_dir, "sim_compute_sched.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(compute_sched) + "\n")
    with open(os.path.join(out_dir, "sim_bypass_sched.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(bypass_sched) + "\n")
    with open(os.path.join(out_dir, "sim_prologue_sched.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue_sched) + "\n")

    # ---------- final barrier mitigation (only if needed) ----------
    mitigator = BarrierMitigator()
    compute_fixed, rep_c = mitigator.mitigate(m, compute_sched, max_iters=80)
    bypass_fixed,  rep_b = mitigator.mitigate(m, bypass_sched, max_iters=80)
    prologue_fixed, rep_p = mitigator.mitigate(m, prologue_sched, max_iters=80) if prologue_sched else (prologue_sched, {"final_hazards":0,"iters":[]})

    write_bundle(out_dir, "compute_fixed", compute_fixed)
    write_bundle(out_dir, "bypass_fixed", bypass_fixed)
    write_bundle(out_dir, "prologue_fixed", prologue_fixed)

    with open(os.path.join(out_dir, "sim_compute_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(compute_fixed) + "\n")
    with open(os.path.join(out_dir, "sim_bypass_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(bypass_fixed) + "\n")
    with open(os.path.join(out_dir, "sim_prologue_fixed.txt"), "w", encoding="utf-8") as f:
        f.write(sim.run(prologue_fixed) + "\n")

    hazard_report = {"compute": rep_c, "bypass": rep_b, "prologue": rep_p}
    with open(os.path.join(out_dir, "hazard_report.json"), "w", encoding="utf-8") as f:
        json.dump(hazard_report, f, indent=2)

    dispatch = {
        "phi_strategy": "firmware_select_path",
        "optional_prologue": "prologue_fixed.bin" if plan["use_pack"] else None,
        "paths": {"if_c_true":"compute_fixed.bin", "if_c_false":"bypass_fixed.bin"},
        "v6_pipeline": ["bank-aware allocation", "dma hoist scheduling", "barrier mitigation fallback"]
    }
    with open(os.path.join(out_dir, "dispatch.json"), "w", encoding="utf-8") as f:
        json.dump(dispatch, f, indent=2)

    print("Generated:", out_dir)
    print("Compare these three:")
    print("  - sim_compute_raw.txt")
    print("  - sim_compute_sched.txt")
    print("  - sim_compute_fixed.txt")
    print("Bank-aware placement is in: sram_map_bankaware.txt")
    print("Hazard iterations in: hazard_report.json")


if __name__ == "__main__":
    main()

