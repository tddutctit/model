
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy NPU Compiler v7
===================
v7 goals (more "real compiler"):
A) True list scheduling (DAG-based) with resource constraints (DMA/MAC/ACT) + cost function.
B) Bank-aware *temporal* planning:
   - not only "place buffers on different banks", but also "at the time of overlap,
     DMA(dst banks) should avoid MAC(B banks) and ACT(OUT/SK banks)".
C) Buffer rotation / bank swizzling:
   - For ping-pong buffers (A0/A1, B0/B1, SK0/SK1), we allow choosing among multiple
     candidate SRAM addresses per logical buffer and select the one that minimizes
     hazards under expected overlap.
D) Still keep v6 pipeline + final barrier fallback:
   raw_emit -> choose bank-swizzle mapping -> build DAG -> list schedule -> simulate -> (if needed) barrier fix.

Outputs:
  ./toy_out_v7/
    - sram_candidates.txt
    - swizzle_choice.json
    - compute_raw / compute_sched / compute_fixed (+ traces + sims)
    - hazard_report.json
    - dispatch.json

This is a toy, but structurally matches how real NPU compilers do it:
  placement/allocator -> schedule (DAG) -> final legality pass / fences.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Set
import struct
import json
import os
import copy
import math


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
# 2) Model + tiles
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
# 3) Address helpers (DRAM)
# ============================================================

def addr_A(m: ModelSpec, row: int, col: int) -> int:
    return m.A_base + (row * m.K + col)

def addr_Y(m: ModelSpec, row: int, col: int) -> int:
    return m.Y_base + (row * m.N + col)

def addr_Skip(m: ModelSpec, row: int, col: int) -> int:
    return m.Skip_base + (row * m.N + col)

def addr_B(m: ModelSpec, row: int, col: int) -> int:
    return m.B_base + (row * m.N + col)

def bpack_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.Bpack_base + n_tile_index * (m.K * m.tileN)

def ms_tile_addr(m: ModelSpec, n_tile_index: int) -> int:
    return m.MS_base + n_tile_index * m.tileN * m.ms_entry_size


# ============================================================
# 4) Graph + lowerer
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

        tiles: List[TileTask] = []
        for tm in range(self.m.M // self.m.tileM):
            for tn in range(self.m.N // self.m.tileN):
                steps = ["LOAD_A", "LOAD_B", "MAC", "REQUANT"]
                if use_residual:
                    steps += ["LOAD_SKIP", "ADD"]
                steps += ["STORE_Y"]
                tiles.append(TileTask(tm, tn, steps))

        return {"use_pack": use_pack, "use_residual": use_residual, "tiles": tiles}


# ============================================================
# 5) SRAM bank utilities
# ============================================================

def is_sram(m: ModelSpec, addr: int) -> bool:
    return 0 <= addr < m.sram_size_bytes

def banks_for_range(m: ModelSpec, addr: int, size: int) -> List[int]:
    banks: Set[int] = set()
    gran = m.bank_granularity
    for off in range(0, max(1, size), gran):
        b = ((addr + off) // gran) % m.sram_banks
        banks.add(int(b))
    return sorted(banks)

def banks_touched(m: ModelSpec, c: Cmd32) -> List[int]:
    banks: Set[int] = set()
    def add(addr: int, size: int):
        for b in banks_for_range(m, addr, size):
            banks.add(b)

    if is_sram(m, c.src_addr):
        add(c.src_addr, max(1, c.size_or_M))
    if is_sram(m, c.dst_addr):
        add(c.dst_addr, max(1, c.size_or_M))
    # MAC uses arg0 as B SRAM pointer
    if c.opcode == OP_MAC_ACC32 and is_sram(m, c.arg0):
        add(c.arg0, m.K*m.tileN)
    # ADD uses arg0 as SK pointer
    if c.opcode == OP_ADD_I8_CLAMP and is_sram(m, c.arg0):
        add(c.arg0, c.size_or_M)
    # PACK uses src SRAM scratch
    if c.opcode == OP_PACK_B_TILE and is_sram(m, c.src_addr):
        add(c.src_addr, m.K*m.tileN)
    return sorted(banks)


# ============================================================
# 6) Raw emitter (same as v6/v5)
# ============================================================

@dataclass
class BufReq:
    name: str
    size: int
    align: int
    hot: bool = True

class Emitter:
    def __init__(self, m: ModelSpec):
        self.m = m

    def reqs(self, plan: Dict[str,Any]) -> List[BufReq]:
        m = self.m
        use_res = plan["use_residual"]
        out = [
            BufReq("A0",   m.tileM*m.K,       m.sram_align, True),
            BufReq("A1",   m.tileM*m.K,       m.sram_align, True),
            BufReq("B0",   m.K*m.tileN,       m.sram_align, True),
            BufReq("B1",   m.K*m.tileN,       m.sram_align, True),
            BufReq("ACC",  m.tileM*m.tileN*4, m.sram_align, True),
            BufReq("OUT",  m.tileM*m.tileN,   m.sram_align, True),
            BufReq("B_SCR",m.K*m.tileN,       m.sram_align, False),
        ]
        if use_res:
            out += [
                BufReq("SK0", m.tileM*m.tileN, m.sram_align, True),
                BufReq("SK1", m.tileM*m.tileN, m.sram_align, True),
            ]
        return out

    def emit_pack_prologue(self, sram: Dict[str,int]) -> List[Cmd32]:
        m = self.m
        B_SCR = sram["B_SCR"]
        cmds: List[Cmd32] = []
        ev=1
        def new_ev(): nonlocal ev; ev+=1; return ev-1
        for tn in range(m.N // m.tileN):
            e_load=new_ev()
            cmds.append(Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=e_load,
                              src_addr=addr_B(m,0,tn*m.tileN), dst_addr=B_SCR,
                              size_or_M=m.K*m.tileN, arg0=m.K, arg1=m.tileN, arg2=m.N))
            e_pack=new_ev()
            cmds.append(Cmd32(OP_PACK_B_TILE, wait0=e_load, sig0=e_pack,
                              src_addr=B_SCR, dst_addr=bpack_tile_addr(m,tn),
                              size_or_M=m.K, arg0=m.tileN, arg1=tn, arg2=0))
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
        def new_ev(): nonlocal ev; ev+=1; return ev-1

        def loadA(dst, tm, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m,tm*m.tileM,0), dst_addr=dst,
                         size_or_M=m.tileM*m.K, arg0=m.tileM, arg1=m.K, arg2=m.K)

        def loadB(dst, tn, sig):
            return Cmd32(OP_DMA_LOAD_LINEAR, sig0=sig,
                         src_addr=bpack_tile_addr(m,tn), dst_addr=dst,
                         size_or_M=m.K*m.tileN)

        def loadSK(dst, tm, tn, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_Skip(m,tm*m.tileM,tn*m.tileN), dst_addr=dst,
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def mac(A_s, B_s, wA, wB, sig):
            return Cmd32(OP_MAC_ACC32, wait0=wA, wait1=wB, sig0=sig,
                         src_addr=A_s, dst_addr=ACC,
                         size_or_M=m.tileM, arg0=B_s, arg1=m.tileN, arg2=m.tileK)

        def requant(tn, w, sig):
            flags = FLAG_RELU_ENABLE if m.relu else 0
            return Cmd32(OP_REQUANT_RELU_PC, flags=flags, wait0=w, sig0=sig,
                         src_addr=ACC, dst_addr=OUT,
                         size_or_M=m.tileM, arg0=m.tileN, arg1=ms_tile_addr(m,tn), arg2=0)

        def addres(skip_s, wOut, wSk, sig):
            clamp_min = (-128) & 0xFFFF
            clamp_max = (127) & 0xFFFF
            clamp_pack = (clamp_max << 16) | clamp_min
            flags = FLAG_ADD_RELU_MIN0 if (m.relu and m.zC==0) else 0
            return Cmd32(OP_ADD_I8_CLAMP, flags=flags, wait0=wOut, wait1=wSk, sig0=sig,
                         src_addr=OUT, dst_addr=OUT,
                         size_or_M=m.tileM*m.tileN, arg0=skip_s, arg1=clamp_pack, arg2=0)

        def store(tm, tn, w):
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=w,
                         src_addr=OUT, dst_addr=addr_Y(m,tm*m.tileM,tn*m.tileN),
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        tiles: List[TileTask] = plan["tiles"]
        t0=tiles[0]
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
                tnxt=tiles[i+1]
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
        m=self.m
        A0,A1=sram["A0"], sram["A1"]
        tiles: List[TileTask]=plan["tiles"]
        cmds=[]
        ev=1
        def new_ev(): nonlocal ev; ev+=1; return ev-1

        def load(dst, tm, tn, sig):
            return Cmd32(OP_DMA_LOAD_2D, flags=FLAG_IS_2D, sig0=sig,
                         src_addr=addr_A(m,tm*m.tileM,tn*m.tileN), dst_addr=dst,
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        def store(src, tm, tn, w):
            return Cmd32(OP_DMA_STORE_2D, flags=FLAG_IS_2D, wait0=w,
                         src_addr=src, dst_addr=addr_Y(m,tm*m.tileM,tn*m.tileN),
                         size_or_M=m.tileM*m.tileN, arg0=m.tileM, arg1=m.tileN, arg2=m.N)

        t0=tiles[0]
        EA0=new_ev()
        cmds.append(load(A0,t0.tm,t0.tn,EA0))
        EA1=0
        for i,t in enumerate(tiles):
            useA=A0 if i%2==0 else A1
            wA=EA0 if i%2==0 else EA1
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
# 7) SRAM candidate generation + swizzle selection (buffer rotation)
# ============================================================

def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a

class CandidateSRAMPlanner:
    """
    For each buffer, generate multiple candidate addresses:
      - we step by bank_granularity to produce different bank sets
      - ensure no overlap among chosen candidates (handled later by selection)
    """

    def __init__(self, m: ModelSpec):
        self.m = m

    def gen_candidates(self, req: BufReq, max_cand: int = 8) -> List[int]:
        m=self.m
        step=max(req.align, m.bank_granularity)
        # simple: scan from 0 upward
        cands=[]
        addr=0
        while addr + req.size <= m.sram_size_bytes and len(cands) < max_cand:
            a=align_up(addr, req.align)
            if a + req.size > m.sram_size_bytes:
                break
            cands.append(a)
            addr = a + step
        return cands

def ranges_overlap(a0:int,a1:int,b0:int,b1:int)->bool:
    return not (a1<=b0 or b1<=a0)

class SwizzleSelector:
    """
    Choose 1 candidate address per buffer to minimize predicted overlap cost.
    We use a small brute-force over ping-pong groups:
      - Group A: A0/A1
      - Group B: B0/B1
      - Group SK: SK0/SK1 (optional)
    ACC/OUT usually fixed (1 candidate) to simplify, but we still allow multiple if generated.

    Cost model (temporal-aware, approximate):
      - We estimate overlap pairs that are likely concurrent:
          DMA(load A_next/B_next/SK_next) overlaps MAC(current) and ACT(current)
        So we penalize intersections between:
          banks(DMA.dst) and banks(MAC.B_sram)   [B banks]
          banks(DMA.dst) and banks(ACT.OUT/SK)   [OUT/SK banks]
    """

    def __init__(self, m: ModelSpec):
        self.m=m

    def bankset(self, addr:int, size:int)->Set[int]:
        return set(banks_for_range(self.m, addr, size))

    def choose(self,
               reqs: Dict[str,BufReq],
               cands: Dict[str,List[int]]) -> Dict[str,int]:
        m=self.m

        # convenience
        def bs(name, addr):
            return self.bankset(addr, reqs[name].size)

        # fixed buffers first: ACC, OUT, B_SCR
        choice: Dict[str,int] = {}
        for fixed in ["ACC","OUT","B_SCR"]:
            if fixed in cands:
                choice[fixed] = cands[fixed][0]

        # enumerate small combos for ping-pong groups
        A0c=cands.get("A0",[0]); A1c=cands.get("A1",[0])
        B0c=cands.get("B0",[0]); B1c=cands.get("B1",[0])
        SK0c=cands.get("SK0",[0]); SK1c=cands.get("SK1",[0])

        best=None
        for a0 in A0c:
            for a1 in A1c:
                if ranges_overlap(a0,a0+reqs["A0"].size,a1,a1+reqs["A1"].size): continue
                for b0 in B0c:
                    for b1 in B1c:
                        if ranges_overlap(b0,b0+reqs["B0"].size,b1,b1+reqs["B1"].size): continue
                        # avoid overlap across groups
                        bad=False
                        for x0,xs in [(a0,reqs["A0"].size),(a1,reqs["A1"].size)]:
                            for y0,ys in [(b0,reqs["B0"].size),(b1,reqs["B1"].size)]:
                                if ranges_overlap(x0,x0+xs,y0,y0+ys): bad=True
                        if bad: continue

                        # optional SK
                        sk_pairs=[(None,None)]
                        if "SK0" in reqs and "SK1" in reqs:
                            sk_pairs=[]
                            for sk0 in SK0c:
                                for sk1 in SK1c:
                                    if ranges_overlap(sk0,sk0+reqs["SK0"].size,sk1,sk1+reqs["SK1"].size): continue
                                    # avoid overlap with A/B
                                    ok=True
                                    for x0,xs in [(a0,reqs["A0"].size),(a1,reqs["A1"].size),
                                                  (b0,reqs["B0"].size),(b1,reqs["B1"].size)]:
                                        if ranges_overlap(x0,x0+xs,sk0,sk0+reqs["SK0"].size): ok=False
                                        if ranges_overlap(x0,x0+xs,sk1,sk1+reqs["SK1"].size): ok=False
                                    if ok:
                                        sk_pairs.append((sk0,sk1))
                        for sk0,sk1 in sk_pairs:
                            # compute temporal-aware cost (approx)
                            # likely overlaps:
                            # - when MAC uses B0, DMA loads into A1/B1/SK1
                            # - when MAC uses B1, DMA loads into A0/B0/SK0
                            out_bs = bs("OUT", choice["OUT"]) if "OUT" in choice else set()
                            acc_bs = bs("ACC", choice["ACC"]) if "ACC" in choice else set()

                            A0bs=bs("A0",a0); A1bs=bs("A1",a1)
                            B0bs=bs("B0",b0); B1bs=bs("B1",b1)
                            SK0bs=bs("SK0",sk0) if sk0 is not None else set()
                            SK1bs=bs("SK1",sk1) if sk1 is not None else set()

                            # cost terms: overlap between prefetch-dst banks and current compute banks
                            # phase0: MAC uses (A0,B0)->ACC then ACT uses OUT then next prefetch goes to (A1,B1,SK1)
                            # phase1: MAC uses (A1,B1)->ACC ... next prefetch -> (A0,B0,SK0)
                            def inter_cost(dst_bs, compute_bs, w):
                                return w * len(dst_bs & compute_bs)

                            # MAC likely stresses B banks and ACC banks, ACT stresses OUT and SK banks.
                            cost=0
                            # phase0 overlaps:
                            cost += inter_cost(A1bs, B0bs, 5)   # DMA A1 vs MAC B0
                            cost += inter_cost(B1bs, B0bs, 6)   # DMA B1 vs MAC B0 (most important)
                            cost += inter_cost(A1bs, acc_bs, 2)
                            cost += inter_cost(B1bs, acc_bs, 2)
                            cost += inter_cost(A1bs, out_bs, 2)
                            cost += inter_cost(B1bs, out_bs, 2)
                            cost += inter_cost(SK1bs, out_bs, 3)
                            cost += inter_cost(SK1bs, B0bs, 2)

                            # phase1 overlaps:
                            cost += inter_cost(A0bs, B1bs, 5)
                            cost += inter_cost(B0bs, B1bs, 6)
                            cost += inter_cost(A0bs, acc_bs, 2)
                            cost += inter_cost(B0bs, acc_bs, 2)
                            cost += inter_cost(A0bs, out_bs, 2)
                            cost += inter_cost(B0bs, out_bs, 2)
                            cost += inter_cost(SK0bs, out_bs, 3)
                            cost += inter_cost(SK0bs, B1bs, 2)

                            cand = (cost, a0,a1,b0,b1,sk0,sk1)
                            if best is None or cand < best:
                                best=cand

        if best is None:
            raise RuntimeError("No non-overlapping swizzle choice found (increase SRAM or reduce candidates)")

        cost, a0,a1,b0,b1,sk0,sk1 = best
        choice.update({"A0":a0,"A1":a1,"B0":b0,"B1":b1})
        if sk0 is not None:
            choice.update({"SK0":sk0,"SK1":sk1})
        return choice


# ============================================================
# 8) Build a DAG from commands + resource-constrained list scheduler
# ============================================================

@dataclass
class Node:
    idx: int
    cmd: Cmd32
    eng: str
    dur: int
    banks: List[int]
    preds: Set[int]
    succs: Set[int]

class Scheduler:
    """
    Build dependencies:
      1) Event deps: if node B waits on event produced by node A => A -> B.
      2) Same-engine program order (preserve original order within DMA/MAC/ACT queues), optional:
         - In real hardware each engine has its own queue; order within queue must be preserved.
         - We'll preserve per-engine order to avoid inventing a multi-queue model.

    List scheduling:
      - Maintain ready set; pick next node to issue in the global stream.
      - Use cost function that predicts bank conflicts with currently-running other engines.
      - We simulate time while building order (like a greedy scheduler).

    This is a toy approximation of modulo scheduling / software pipelining.
    """

    def __init__(self, m: ModelSpec):
        self.m=m

    def engine(self, c: Cmd32) -> str:
        if c.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            return "DMA"
        if c.opcode == OP_MAC_ACC32:
            return "MAC"
        return "ACT"

    def dur(self, c: Cmd32) -> int:
        if c.opcode in (OP_DMA_LOAD_2D, OP_DMA_STORE_2D, OP_DMA_LOAD_LINEAR):
            return max(1, c.size_or_M // 256)
        if c.opcode == OP_MAC_ACC32:
            M=c.size_or_M; N=c.arg1; K=c.arg2
            return max(1, (M*N*K)//4096)
        if c.opcode == OP_PACK_B_TILE:
            K=c.size_or_M; tileN=c.arg0
            return max(1, (K*tileN)//512)
        if c.opcode == OP_BARRIER_WAIT:
            return 1
        return max(1, c.size_or_M // 512)

    def build_dag(self, cmds: List[Cmd32]) -> List[Node]:
        # map event -> producer idx
        prod: Dict[int,int] = {}
        for i,c in enumerate(cmds):
            if c.sig0: prod[c.sig0]=i
            if c.sig1: prod[c.sig1]=i

        nodes=[]
        for i,c in enumerate(cmds):
            preds=set()
            for w in (c.wait0, c.wait1):
                if w and w in prod:
                    preds.add(prod[w])
            n=Node(i,c,self.engine(c),self.dur(c),banks_touched(self.m,c),preds,set())
            nodes.append(n)

        # succ fill
        for n in nodes:
            for p in n.preds:
                nodes[p].succs.add(n.idx)

        # preserve per-engine order (queue semantics)
        last_by_eng: Dict[str,int] = {}
        for n in nodes:
            if n.eng in last_by_eng:
                p=last_by_eng[n.eng]
                if p not in n.preds:
                    n.preds.add(p)
                    nodes[p].succs.add(n.idx)
            last_by_eng[n.eng]=n.idx

        return nodes

    def schedule(self, nodes: List[Node]) -> List[Cmd32]:
        # Kahn-style scheduling with time-aware selection
        indeg = {n.idx: len(n.preds) for n in nodes}
        ready: Set[int] = {n.idx for n in nodes if indeg[n.idx]==0}

        # track per-engine "when last issued finishes" and which banks are active
        t=0
        eng_busy_until={"DMA":0,"MAC":0,"ACT":0}
        eng_active_banks={"DMA":set(),"MAC":set(),"ACT":set()}

        scheduled=[]
        node_map={n.idx:n for n in nodes}

        def update_active_banks(now:int):
            # when engine is idle (now >= busy), clear active banks
            for e in ["DMA","MAC","ACT"]:
                if now >= eng_busy_until[e]:
                    eng_active_banks[e]=set()

        def pick_best(now:int, candidates:Set[int]) -> int:
            update_active_banks(now)

            best=None
            for idx in candidates:
                n=node_map[idx]
                e=n.eng
                # can't issue if engine busy at 'now'
                if now < eng_busy_until[e]:
                    continue

                # score: bank intersection with other engines currently active
                inter=0
                nb=set(n.banks)
                for oe in ["DMA","MAC","ACT"]:
                    if oe==e: continue
                    inter += len(nb & eng_active_banks[oe])

                # prefer earlier original order slightly to keep stable output
                score = (inter*10) + (idx*0.001)
                cand=(score, idx)
                if best is None or cand < best:
                    best=cand
            if best is None:
                # no engine available; advance time to earliest engine free
                return -1
            return best[1]

        # We also need to respect "wait event times" but event deps already enforce order.
        # Because we preserved per-engine order, time evolution is coarse but consistent.

        while ready:
            idx = pick_best(t, ready)
            if idx == -1:
                t = min(eng_busy_until.values())
                continue

            n=node_map[idx]
            ready.remove(idx)
            scheduled.append(n.cmd)

            # issue: start at t, end at t+dur, update engine active banks
            end = t + n.dur
            eng_busy_until[n.eng]=end
            eng_active_banks[n.eng]=set(n.banks)

            # advance time for next selection: allow overlap -> do NOT advance; keep t
            # but if all engines busy, we will advance in pick_best(-1)
            # (this approximates a global issue stream for multiple queues).
            for s in n.succs:
                indeg[s]-=1
                if indeg[s]==0:
                    ready.add(s)

        return scheduled


# ============================================================
# 9) Accurate simulator + hazard detection
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
        self.m=m
        self.events: Dict[int,bool]={}
        self.pending: List[Tuple[int,int]]=[]
        self.time=0
        self.busy={"DMA":0,"MAC":0,"ACT":0}
        self.intervals: List[Interval]=[]
        self.hazards: List[Dict[str,Any]]=[]

    def engine(self,c:Cmd32)->str:
        if c.opcode in (OP_DMA_LOAD_2D,OP_DMA_STORE_2D,OP_DMA_LOAD_LINEAR):
            return "DMA"
        if c.opcode==OP_MAC_ACC32:
            return "MAC"
        return "ACT"

    def dur(self,c:Cmd32)->int:
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

    def fire(self,t:int):
        still=[]
        for tf,eid in self.pending:
            if tf<=t: self.events[eid]=True
            else: still.append((tf,eid))
        self.pending=still

    def waits_ok(self,c:Cmd32)->bool:
        for w in (c.wait0,c.wait1):
            if w and not self.events.get(w,False):
                return False
        return True

    def run(self,cmds:List[Cmd32])->str:
        self.events.clear(); self.pending.clear(); self.time=0
        self.busy={"DMA":0,"MAC":0,"ACT":0}
        self.intervals.clear(); self.hazards.clear()
        lines=[]

        for i,c in enumerate(cmds):
            e=self.engine(c)
            while True:
                self.fire(self.time)
                if self.waits_ok(c) and self.time>=self.busy[e]:
                    break
                nxt=max(self.time+1, self.busy[e])
                if self.pending:
                    nxt=min(nxt, min(tf for tf,_ in self.pending))
                self.time=nxt

            st=self.time
            en=st+self.dur(c)
            b=banks_touched(self.m,c)
            self.busy[e]=en
            self.intervals.append(Interval(e,st,en,b,i,c))
            if c.sig0: self.pending.append((en,c.sig0))
            if c.sig1: self.pending.append((en,c.sig1))
            lines.append(f"{i:04d} t={st:6d}..{en:6d} {e:3s} banks={b} {c.to_trace()}")
            self.time=st

        makespan=max(self.busy.values())
        self.fire(makespan)

        # hazards
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
        lines.append(f"SIM SUMMARY: DMA_end={self.busy['DMA']}, MAC_end={self.busy['MAC']}, ACT_end={self.busy['ACT']}, makespan={makespan}")
        lines.append(f"BANK HAZARDS: {len(self.hazards)}")
        for h in self.hazards[:15]:
            lines.append(f"  {h['a_eng']}#{h['a_idx']} {h['a_t']} banks{h['a_banks']}  <-> "
                         f"{h['b_eng']}#{h['b_idx']} {h['b_t']} banks{h['b_banks']}  inter={h['intersection']}")
        if len(self.hazards)>15:
            lines.append(f"  ... ({len(self.hazards)-15} more)")
        return "\n".join(lines)


# ============================================================
# 10) Barrier fallback (same as v6)
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

    def mitigate(self, m: ModelSpec, cmds: List[Cmd32], max_iters: int = 80) -> Tuple[List[Cmd32], Dict[str,Any]]:
        sim=Simulator(m)
        cur=list(cmds)
        rep={"iters":[], "final_hazards":None}
        for it in range(max_iters):
            sim.run(cur)
            hz=sim.hazards
            rep["iters"].append({"iter":it,"hazards":len(hz)})
            if not hz:
                rep["final_hazards"]=0
                return cur, rep

            h=hz[0]
            a=sim.intervals[h["a_idx"]]
            b=sim.intervals[h["b_idx"]]
            later=a if a.start>b.start else b
            earlier=b if later is a else a

            wait_ev = earlier.cmd.sig0 or earlier.cmd.sig1
            if not wait_ev:
                new_e=self._find_free_event_id(cur)
                cur.insert(earlier.idx+1, Cmd32(OP_BARRIER_WAIT, sig0=new_e))
                rep["iters"][-1]["action"]={"type":"insert_after_earlier_signal","event":new_e,"pos":earlier.idx+1}
                continue

            cur.insert(later.idx, Cmd32(OP_BARRIER_WAIT, wait0=wait_ev))
            rep["iters"][-1]["action"]={"type":"insert_before_later_wait","wait":wait_ev,"pos":later.idx}
        sim.run(cur)
        rep["final_hazards"]=len(sim.hazards)
        return cur, rep


# ============================================================
# 11) IO helpers
# ============================================================

def write_bundle(out_dir: str, name: str, cmds: List[Cmd32]) -> None:
    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as fb, \
         open(os.path.join(out_dir, f"{name}.trace.txt"), "w", encoding="utf-8") as ft:
        for i,c in enumerate(cmds):
            fb.write(c.pack())
            ft.write(f"{i:04d}: {c.to_trace()}\n")


# ============================================================
# 12) Main v7
# ============================================================

def main():
    out_dir="./toy_out_v7"
    os.makedirs(out_dir, exist_ok=True)

    m=ModelSpec(
        A_base=0x8000_0000,
        B_base=0x8000_2000,
        Y_base=0x8000_4000,
        Skip_base=0x8000_9000,
        Bpack_base=0x8000_6000,
        MS_base=0x8000_7000,
        M=64,N=64,K=64,
        tileM=32,tileN=32,tileK=64,
        zC=0,relu=True,
        ms_entry_size=8,
        sram_size_bytes=256*1024,
        sram_align=64,
        sram_banks=8,
        bank_granularity=256
    )

    graph_path="./graph.json"
    graph=DEFAULT_GRAPH
    if os.path.exists(graph_path):
        with open(graph_path,"r",encoding="utf-8") as f:
            graph=json.load(f)

    lowerer=GraphLowerer(graph,m)
    norm=lowerer.normalize()
    with open(os.path.join(out_dir,"normalized_graph.json"),"w",encoding="utf-8") as f:
        json.dump(norm,f,indent=2)

    plan=lowerer.build_plan()
    emitter=Emitter(m)
    req_list=emitter.reqs(plan)
    reqs={r.name:r for r in req_list}

    # ---- generate SRAM candidates
    cand_planner=CandidateSRAMPlanner(m)
    cands: Dict[str,List[int]]={}
    for r in req_list:
        # give more candidates for ping-pong hot buffers
        mc = 10 if r.name in ("A0","A1","B0","B1","SK0","SK1") else 4
        cands[r.name]=cand_planner.gen_candidates(r, max_cand=mc)

    # dump candidates
    with open(os.path.join(out_dir,"sram_candidates.txt"),"w",encoding="utf-8") as f:
        for name,addrs in cands.items():
            f.write(f"{name} size={reqs[name].size}:\n")
            for a in addrs[:12]:
                f.write(f"  0x{a:08X} banks={banks_for_range(m,a,reqs[name].size)}\n")
            f.write("\n")

    # ---- choose swizzle mapping (buffer rotation / bank swizzle)
    selector=SwizzleSelector(m)
    sram_choice=selector.choose(reqs, cands)
    with open(os.path.join(out_dir,"swizzle_choice.json"),"w",encoding="utf-8") as f:
        json.dump({k:f"0x{v:08X}" for k,v in sram_choice.items()}, f, indent=2)

    # ---- emit raw command streams using chosen SRAM mapping
    prologue_raw = emitter.emit_pack_prologue(sram_choice) if plan["use_pack"] else []
    compute_raw  = emitter.emit_compute(plan, sram_choice)
    bypass_raw   = emitter.emit_bypass(plan, sram_choice)

    write_bundle(out_dir,"compute_raw",compute_raw)
    write_bundle(out_dir,"bypass_raw",bypass_raw)
    write_bundle(out_dir,"prologue_raw",prologue_raw)

    sim=Simulator(m)
    with open(os.path.join(out_dir,"sim_compute_raw.txt"),"w",encoding="utf-8") as f:
        f.write(sim.run(compute_raw)+"\n")

    # ---- DAG list scheduling (true scheduler)
    sched=Scheduler(m)
    nodes = sched.build_dag(compute_raw)
    compute_sched = sched.schedule(nodes)
    write_bundle(out_dir,"compute_sched",compute_sched)
    with open(os.path.join(out_dir,"sim_compute_sched.txt"),"w",encoding="utf-8") as f:
        f.write(sim.run(compute_sched)+"\n")

    # (we keep bypass/prologue as-is; you can also schedule them similarly if you want)
    # ---- Final barrier fallback
    mitig=BarrierMitigator()
    compute_fixed, rep = mitig.mitigate(m, compute_sched, max_iters=120)
    write_bundle(out_dir,"compute_fixed",compute_fixed)
    with open(os.path.join(out_dir,"sim_compute_fixed.txt"),"w",encoding="utf-8") as f:
        f.write(sim.run(compute_fixed)+"\n")

    hazard_report={"compute":rep}
    with open(os.path.join(out_dir,"hazard_report.json"),"w",encoding="utf-8") as f:
        json.dump(hazard_report,f,indent=2)

    dispatch={
        "phi_strategy":"firmware_select_path",
        "optional_prologue":"prologue_raw.bin" if plan["use_pack"] else None,
        "paths":{"if_c_true":"compute_fixed.bin","if_c_false":"bypass_raw.bin"},
        "v7_pipeline":[
            "SRAM candidate generation",
            "bank+temporal swizzle selection (buffer rotation)",
            "DAG list scheduling with resource constraints",
            "barrier mitigation fallback"
        ]
    }
    with open(os.path.join(out_dir,"dispatch.json"),"w",encoding="utf-8") as f:
        json.dump(dispatch,f,indent=2)

    print("Generated:", out_dir)
    print("Compare:")
    print("  sim_compute_raw.txt")
    print("  sim_compute_sched.txt")
    print("  sim_compute_fixed.txt")
    print("Swizzle mapping:", os.path.join(out_dir,"swizzle_choice.json"))
    print("Candidates:", os.path.join(out_dir,"sram_candidates.txt"))
    print("Hazard fix iterations:", os.path.join(out_dir,"hazard_report.json"))


if __name__ == "__main__":
    main()

