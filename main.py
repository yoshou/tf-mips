#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np

from regfile import *
from constants import *
from utils import *
from control import *
from alu_ctrl import *
from alu import *
from datamem import *

mem_size = 1024 * 20  # 20k
insts_file = "./mips-test.bin"

insts = np.fromfile(insts_file, dtype=">i")

insts_mem = tf.constant(insts, dtype=tf.int32)
data_mem = tf.Variable(np.zeros((mem_size,)), dtype=tf.int32)

regfile = Regfile(32)

wires = obj_dict()
regs = obj_dict()
sync = sync_update(regs)

regs.regfile = regfile.regfile
regs.data_mem = data_mem


# =================================================================================
# Register definitions
# =================================================================================

# pc
regs.s0_pc = tf.Variable(0)

# s0 - s1
regs.s0_s1_inst = tf.Variable(0)
regs.s0_s1_pc_plus_4 = tf.Variable(0)

# s1 - s2
regs.s1_s2_reg_r1_data_fwd = tf.Variable(0)
regs.s1_s2_reg_r2_data_fwd = tf.Variable(0)
regs.s1_s2_inst_imm_ext = tf.Variable(0)
regs.s1_s2_inst_shamt = tf.Variable(0)
regs.s1_s2_inst_funct = tf.Variable(0)
regs.s1_s2_inst_rs = tf.Variable(0)
regs.s1_s2_inst_rt = tf.Variable(0)
regs.s1_s2_inst_rd = tf.Variable(0)

regs.s1_s2_alu_op = tf.Variable(0)
regs.s1_s2_alu_src_a_sel = tf.Variable(0)
regs.s1_s2_alu_src_b_sel = tf.Variable(0)


regs.s1_s2_reg_src = tf.Variable(0)
regs.s1_s2_reg_dst = tf.Variable(0)
regs.s1_s2_reg_write = tf.Variable(False)

regs.s1_s2_pc_plus_4 = tf.Variable(0)
regs.s1_s2_data_write = tf.Variable(False)


# s2 - s3
regs.s2_s3_alu_y = tf.Variable(0)
regs.s2_s3_reg_r2_data_fwd = tf.Variable(0)
regs.s2_s3_reg_src = tf.Variable(0)
regs.s2_s3_reg_dst = tf.Variable(0)
regs.s2_s3_reg_write = tf.Variable(False)
regs.s2_s3_reg_w_addr = tf.Variable(0)
regs.s2_s3_pc_plus_4 = tf.Variable(0)
regs.s2_s3_data_write = tf.Variable(False)

# s3 - s4
regs.s3_s4_reg_src = tf.Variable(0)
regs.s3_s4_reg_dst = tf.Variable(0)
regs.s3_s4_reg_write = tf.Variable(False)
regs.s3_s4_reg_w_addr = tf.Variable(0)
regs.s3_s4_data_out = tf.Variable(0)
regs.s3_s4_alu_y = tf.Variable(0)
regs.s3_s4_pc_plus_4 = tf.Variable(0)

# =================================================================================
# 4. WB stage
# =================================================================================

wires.s4_reg_src = regs.s3_s4_reg_src
wires.s4_reg_dst = regs.s3_s4_reg_dst
wires.s4_reg_write = regs.s3_s4_reg_write
wires.s4_reg_w_addr = regs.s3_s4_reg_w_addr
wires.s4_data_out = regs.s3_s4_data_out
wires.s4_alu_y = regs.s3_s4_alu_y
wires.s4_pc_plus_4 = regs.s3_s4_pc_plus_4

# Select write register source
wires.s4_reg_w_data = tf.case({
    tf.equal(wires.s4_reg_src, reg_src_alu): lambda: wires.s4_alu_y,
    tf.equal(wires.s4_reg_src, reg_src_mem): lambda: wires.s4_data_out,
    tf.equal(wires.s4_reg_src, reg_src_pc): lambda: wires.s4_pc_plus_4 + 1,
}, default=lambda: 0, exclusive=True)

# =================================================================================
# 3. MEM stage
# =================================================================================

wires.s3_alu_y = regs.s2_s3_alu_y
wires.s3_reg_r2_data_fwd = regs.s2_s3_reg_r2_data_fwd
wires.s3_reg_src = regs.s2_s3_reg_src
wires.s3_reg_dst = regs.s2_s3_reg_dst
wires.s3_reg_write = regs.s2_s3_reg_write
wires.s3_reg_w_addr = regs.s2_s3_reg_w_addr
wires.s3_pc_plus_4 = regs.s2_s3_pc_plus_4
wires.s3_data_write = regs.s2_s3_data_write

wires.s3_data_in = wires.s3_reg_r2_data_fwd

wires.s3_data_out, = datamem_connect(
    data_mem, wires.s3_data_write, wires.s3_alu_y, wires.s3_data_in, sync)


def s3_s4_sync(regs):
    yield regs.s3_s4_reg_src.assign(wires.s3_reg_src)
    yield regs.s3_s4_reg_dst.assign(wires.s3_reg_dst)
    yield regs.s3_s4_reg_write.assign(wires.s3_reg_write)
    yield regs.s3_s4_reg_w_addr.assign(wires.s3_reg_w_addr)
    yield regs.s3_s4_data_out.assign(wires.s3_data_out)
    yield regs.s3_s4_alu_y.assign(wires.s3_alu_y)
    yield regs.s3_s4_pc_plus_4.assign(wires.s3_pc_plus_4)


sync.append(s3_s4_sync)

# =================================================================================
# 2. EX stage
# =================================================================================


wires.s2_reg_r1_data = regs.s1_s2_reg_r1_data_fwd
wires.s2_reg_r2_data = regs.s1_s2_reg_r2_data_fwd
wires.s2_inst_imm_ext = regs.s1_s2_inst_imm_ext
wires.s2_inst_shamt = regs.s1_s2_inst_shamt
wires.s2_inst_funct = regs.s1_s2_inst_funct
wires.s2_inst_rs = regs.s1_s2_inst_rs
wires.s2_inst_rt = regs.s1_s2_inst_rt
wires.s2_inst_rd = regs.s1_s2_inst_rd

wires.s2_alu_op = regs.s1_s2_alu_op
wires.s2_alu_src_a_sel = regs.s1_s2_alu_src_a_sel
wires.s2_alu_src_b_sel = regs.s1_s2_alu_src_b_sel

wires.s2_reg_src = regs.s1_s2_reg_src
wires.s2_reg_dst = regs.s1_s2_reg_dst
wires.s2_reg_write = regs.s1_s2_reg_write

wires.s2_pc_plus_4 = regs.s1_s2_pc_plus_4
wires.s2_data_write = regs.s1_s2_data_write

wires.s2_inst_imm_hi = tf.bitwise.left_shift(
    get_bits(wires.s2_inst_imm_ext, 0, 15), 16)

wires.s2_reg_r1_data_fwd_sel = tf.case([
    (tf.not_equal(wires.s2_inst_rs, 0) & tf.equal(wires.s2_inst_rs,
                                                  wires.s3_reg_w_addr) & wires.s3_reg_write, lambda: 1),
    (tf.not_equal(wires.s2_inst_rs, 0) & tf.equal(wires.s2_inst_rs,
                                                  wires.s4_reg_w_addr) & wires.s4_reg_write, lambda: 2),
], default=lambda: 0)

wires.s2_reg_r2_data_fwd_sel = tf.case([
    (tf.not_equal(wires.s2_inst_rt, 0) & tf.equal(wires.s2_inst_rt,
                                                  wires.s3_reg_w_addr) & wires.s3_reg_write, lambda: 1),
    (tf.not_equal(wires.s2_inst_rt, 0) & tf.equal(wires.s2_inst_rt,
                                                  wires.s4_reg_w_addr) & wires.s4_reg_write, lambda: 2),
], default=lambda: 0)

wires.s2_reg_r1_data_fwd = tf.case({
    tf.equal(wires.s2_reg_r1_data_fwd_sel, 1): lambda: wires.s3_alu_y,
    tf.equal(wires.s2_reg_r1_data_fwd_sel, 2): lambda: wires.s4_reg_w_data,
}, default=lambda: wires.s2_reg_r1_data, exclusive=True)

wires.s2_reg_r2_data_fwd = tf.case({
    tf.equal(wires.s2_reg_r2_data_fwd_sel, 1): lambda: wires.s3_alu_y,
    tf.equal(wires.s2_reg_r2_data_fwd_sel, 2): lambda: wires.s4_reg_w_data,
}, default=lambda: wires.s2_reg_r2_data, exclusive=True)

wires.s2_alu_src_a = wires.s2_reg_r1_data_fwd


# Select register or immediate value for alu source

wires.s2_alu_src_b = tf.case({
    tf.equal(wires.s2_alu_src_b_sel, alu_src_b_reg): lambda: wires.s2_reg_r2_data_fwd,
    tf.equal(wires.s2_alu_src_b_sel, alu_src_b_imm): lambda: wires.s2_inst_imm_ext,
    tf.equal(wires.s2_alu_src_b_sel, alu_src_b_imm_hi): lambda: wires.s2_inst_imm_hi,
    tf.equal(wires.s2_alu_src_b_sel, alu_src_b_shamt): lambda: wires.s2_inst_shamt,
    tf.equal(wires.s2_alu_src_b_sel, alu_src_b_zero): lambda: 0,
}, default=lambda: wires.s2_reg_r2_data, exclusive=True)

wires.s2_alu_ctrl_sig = alu_ctrl(wires.s2_alu_op, wires.s2_inst_funct)

wires.s2_alu_y, wires.s2_alu_oflow, wires.s2_alu_zero = alu(
    wires.s2_alu_src_a, wires.s2_alu_src_b, wires.s2_alu_ctrl_sig)

# Select register to write
wires.s2_reg_w_addr = tf.case({
    tf.equal(wires.s2_reg_dst, reg_dst_rt): lambda: wires.s2_inst_rt,
    tf.equal(wires.s2_reg_dst, reg_dst_rd): lambda: wires.s2_inst_rd,
    tf.equal(wires.s2_reg_dst, reg_dst_ra): lambda: bit("11111"),
}, default=lambda: 0, exclusive=True)


# For debug
regs.s2_s3_alu_src_a = tf.Variable(0)
regs.s2_s3_alu_src_b = tf.Variable(0)


def s2_s3_sync(regs):
    yield regs.s2_s3_alu_y.assign(wires.s2_alu_y)
    yield regs.s2_s3_reg_r2_data_fwd.assign(wires.s2_reg_r2_data_fwd)
    yield regs.s2_s3_reg_src.assign(wires.s2_reg_src)
    yield regs.s2_s3_reg_dst.assign(wires.s2_reg_dst)
    yield regs.s2_s3_reg_write.assign(wires.s2_reg_write)
    yield regs.s2_s3_reg_w_addr.assign(wires.s2_reg_w_addr)
    yield regs.s2_s3_pc_plus_4.assign(wires.s2_pc_plus_4)
    yield regs.s2_s3_data_write.assign(wires.s2_data_write)

    yield regs.s2_s3_alu_src_a.assign(wires.s2_alu_src_a)
    yield regs.s2_s3_alu_src_b.assign(wires.s2_alu_src_b)


sync.append(s2_s3_sync)

# =================================================================================
# 1. ID stage
# =================================================================================

wires.s1_inst = regs.s0_s1_inst
wires.s1_pc_plus_4 = regs.s0_s1_pc_plus_4

wires.s1_inst_op = get_bits(wires.s1_inst, 26, 31)
wires.s1_inst_rs = get_bits(wires.s1_inst, 21, 25)
wires.s1_inst_rt = get_bits(wires.s1_inst, 16, 20)
wires.s1_inst_rd = get_bits(wires.s1_inst, 11, 15)
wires.s1_inst_funct = get_bits(wires.s1_inst, 0, 5)
wires.s1_inst_imm = get_bits(wires.s1_inst, 0, 15)
wires.s1_inst_shamt = get_bits(wires.s1_inst, 6, 10)
wires.s1_inst_addr = get_bits(wires.s1_inst, 0, 25)
wires.s1_inst_regimm = get_bits(wires.s1_inst, 16, 20)

wires.s1_reg_r1_addr = wires.s1_inst_rs
wires.s1_reg_r2_addr = wires.s1_inst_rt

# Register file

wires.s1_reg_r1_data, wires.s1_reg_r2_data = regfile.port_map(
    wires.s4_reg_write, wires.s1_reg_r1_addr, wires.s1_reg_r2_addr, wires.s4_reg_w_addr, wires.s4_reg_w_data, sync)

# Immediate sign extension
wires.s1_inst_imm_ext = sign_extend(wires.s1_inst_imm, 16)

# Forwarding from s3, s4 for branch condition
wires.s1_reg_r1_data_fwd_sel = tf.not_equal(wires.s1_inst_rs, 0) & tf.equal(
    wires.s1_inst_rs, wires.s3_reg_w_addr) & wires.s3_reg_write

wires.s1_reg_r2_data_fwd_sel = tf.not_equal(wires.s1_inst_rt, 0) & tf.equal(
    wires.s1_inst_rs, wires.s3_reg_w_addr) & wires.s3_reg_write

wires.s1_reg_r1_data_fwd = tf.case([
    (tf.equal(wires.s1_reg_r1_addr, 0), lambda: 0),
    (tf.equal(wires.s1_reg_r1_addr, wires.s4_reg_w_addr), lambda: wires.s4_reg_w_data),
    (wires.s1_reg_r1_data_fwd_sel, lambda: wires.s3_alu_y),
], default=lambda: wires.s1_reg_r1_data)

wires.s1_reg_r2_data_fwd = tf.case([
    (tf.equal(wires.s1_reg_r2_addr, 0), lambda: 0),
    (tf.equal(wires.s1_reg_r2_addr, wires.s4_reg_w_addr), lambda: wires.s4_reg_w_data),
    (wires.s1_reg_r2_data_fwd_sel, lambda: wires.s3_alu_y),
], default=lambda: wires.s1_reg_r2_data)

# Branch condition
wires.s1_branch_ltz = tf.less(wires.s1_reg_r1_data_fwd, 0)

wires.s1_branch_gez = tf.greater_equal(wires.s1_reg_r1_data_fwd, 0)

wires.s1_branch_lez = tf.less_equal(wires.s1_reg_r1_data_fwd, 0)

wires.s1_branch_gtz = tf.greater(wires.s1_reg_r1_data_fwd, 0)

wires.s1_branch_eq = tf.equal(
    wires.s1_reg_r1_data_fwd, wires.s1_reg_r2_data_fwd)

wires.s1_branch_ne = tf.not_equal(
    wires.s1_reg_r1_data_fwd, wires.s1_reg_r2_data_fwd)

wires.s1_branch_cond = tf.case({
    tf.equal(wires.s1_inst_op, bit("000001")) & tf.equal(get_bits(wires.s1_inst_regimm, 0, 3), 0): lambda: wires.s1_branch_ltz,
    tf.equal(wires.s1_inst_op, bit("000001")) & tf.equal(get_bits(wires.s1_inst_regimm, 0, 3), 1): lambda: wires.s1_branch_gez,
    tf.equal(wires.s1_inst_op, bit("000110")) & tf.equal(get_bits(wires.s1_inst_regimm, 0, 3), 0): lambda: wires.s1_branch_lez,
    tf.equal(wires.s1_inst_op, bit("000111")) & tf.equal(get_bits(wires.s1_inst_regimm, 0, 3), 0): lambda: wires.s1_branch_gtz,
    tf.equal(wires.s1_inst_op, bit("000100")): lambda: wires.s1_branch_eq,
    tf.equal(wires.s1_inst_op, bit("000101")): lambda: wires.s1_branch_ne
}, default=lambda: False, exclusive=True)

wires.s1_alu_op, wires.s1_alu_src_a_sel, wires.s1_alu_src_b_sel, wires.s1_reg_dst, wires.s1_reg_write, wires.s1_data_write, wires.s1_reg_src, wires.s1_pc_src = control(
    wires.s1_inst_op, wires.s1_inst_funct, wires.s1_inst_regimm)

wires.s1_branch = tf.equal(wires.s1_pc_src, pc_src_reg) | tf.equal(wires.s1_pc_src, pc_src_addr) | (
    tf.equal(wires.s1_pc_src, pc_src_imm) & wires.s1_branch_cond)

wires.s1_pc_add = wires.s1_pc_plus_4 + wires.s1_inst_imm_ext

wires.s1_pc_jump = tf.case({
    tf.equal(wires.s1_pc_src, pc_src_pc_add_4): lambda: wires.s1_pc_plus_4,
    tf.equal(wires.s1_pc_src, pc_src_reg): lambda: wires.s1_reg_r1_data_fwd,
    tf.equal(wires.s1_pc_src, pc_src_addr): lambda: tf.bitwise.bitwise_or(tf.bitwise.left_shift(get_bits(wires.s1_pc_plus_4, 28, 31), 28), wires.s1_inst_addr),
    tf.equal(wires.s1_pc_src, pc_src_imm): lambda: wires.s1_pc_add
}, default=lambda: wires.s1_pc_plus_4, exclusive=True)

wires.lw_stall = tf.equal(wires.s2_reg_src, reg_src_mem) & (
    tf.equal(wires.s1_inst_rs, wires.s2_reg_w_addr) | tf.equal(wires.s1_inst_rt, wires.s2_reg_w_addr))

wires.branch_stall = (wires.s1_branch & tf.equal(wires.s2_reg_src, reg_src_alu) & (tf.equal(wires.s1_inst_rs, wires.s2_reg_w_addr) | tf.equal(wires.s1_inst_rt, wires.s2_reg_w_addr))) | (
    wires.s1_branch & tf.equal(wires.s3_reg_src, reg_src_mem) & (tf.equal(wires.s1_inst_rs, wires.s3_reg_w_addr) | tf.equal(wires.s1_inst_rt, wires.s3_reg_w_addr)))

wires.s0_stall = wires.lw_stall | wires.branch_stall
wires.s1_stall = wires.s0_stall
wires.s2_flush = wires.s0_stall  # insert nop after lw stall or branch stall

wires.s1_flush = wires.s1_branch  # replace instruction after branch to nop


def s1_s2_sync(regs):
    yield regs.s1_s2_reg_r1_data_fwd.assign(clr(wires.s1_reg_r1_data_fwd, wires.s2_flush))
    yield regs.s1_s2_reg_r2_data_fwd.assign(clr(wires.s1_reg_r2_data_fwd, wires.s2_flush))
    yield regs.s1_s2_inst_imm_ext.assign(clr(wires.s1_inst_imm_ext, wires.s2_flush))
    yield regs.s1_s2_inst_shamt.assign(clr(wires.s1_inst_shamt, wires.s2_flush))
    yield regs.s1_s2_inst_funct.assign(clr(wires.s1_inst_funct, wires.s2_flush))
    yield regs.s1_s2_inst_rs.assign(clr(wires.s1_inst_rs, wires.s2_flush))
    yield regs.s1_s2_inst_rt.assign(clr(wires.s1_inst_rt, wires.s2_flush))
    yield regs.s1_s2_inst_rd.assign(clr(wires.s1_inst_rd, wires.s2_flush))

    yield regs.s1_s2_alu_op.assign(clr(wires.s1_alu_op, wires.s2_flush))
    yield regs.s1_s2_alu_src_a_sel.assign(clr(wires.s1_alu_src_a_sel, wires.s2_flush))
    yield regs.s1_s2_alu_src_b_sel.assign(clr(wires.s1_alu_src_b_sel, wires.s2_flush))

    yield regs.s1_s2_reg_src.assign(clr(wires.s1_reg_src, wires.s2_flush))
    yield regs.s1_s2_reg_dst.assign(clr(wires.s1_reg_dst, wires.s2_flush))
    yield regs.s1_s2_reg_write.assign(clr(wires.s1_reg_write, wires.s2_flush))

    yield regs.s1_s2_pc_plus_4.assign(clr(wires.s1_pc_plus_4, wires.s2_flush))
    yield regs.s1_s2_data_write.assign(clr(wires.s1_data_write, wires.s2_flush))


sync.append(s1_s2_sync)

# =================================================================================
# 0. IF stage
# =================================================================================


wires.s0_pc_plus_4 = regs.s0_pc + 1
wires.s0_pc = regs.s0_pc
wires.s0_pc_temp = regs.s0_pc


wires.s0_pc_next = tf.cond(
    wires.s1_branch, lambda: wires.s1_pc_jump, lambda: wires.s0_pc_plus_4)

wires.s0_inst = tf.gather_nd(insts_mem, [wires.s0_pc])


def s0_s1_sync(regs):
    yield regs.s0_pc.assign(tf.cond(wires.s0_stall, lambda: wires.s0_pc, lambda: wires.s0_pc_next))

    yield regs.s0_s1_inst.assign(clr_en(regs.s0_s1_inst,
                                        wires.s0_inst, wires.s1_flush, tf.logical_not(wires.s1_stall)))
    yield regs.s0_s1_pc_plus_4.assign(clr_en(regs.s0_s1_pc_plus_4,
                                             wires.s0_pc_plus_4, wires.s1_flush, tf.logical_not(wires.s1_stall)))


sync.append(s0_s1_sync)

update_ops = sync.update(list(wires.values()))


def unsigned(value):
    return value & ((1 << 32) - 1)


import time


with tf.compat.v1.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)

    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)

    start = time.time()

    iters = 400

    print("{0:6}, {1:32}, {2:6}, {3:6}, {4:6}, {5:8}".format(
        "s0_pc", "s0_s1_inst", "reg_sp", "reg_ra", "reg_s8", "data[1024*10]"))

    for i in range(iters):

        result = sess.run(list(regs.values()) +
                          list(wires.values()) + update_ops)

        result = obj_dict({k: v for k, v in zip(
            list(regs.keys()) + list(wires.keys()), result)})

        regmap = result.regfile
        data = result.data_mem

        reg_sp = regmap[reg_no_sp]
        reg_ra = regmap[reg_no_ra]
        reg_s8 = regmap[reg_no_s8]

        print("{0:=6x}, {1:0=32b}, {2:=6x}, {3:=6x}, {4:=6x}, {5:=8}".format(
            result.s0_pc * 4, unsigned(result.s0_s1_inst), reg_sp, reg_ra, reg_s8, data[1024*10]))

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("cycles per second:{0}".format(iters / elapsed_time))

    writer.close()
