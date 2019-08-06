#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from constants import *
from utils import *


def control(op, funct, regimm):
    alu_op = tf.case({
        tf.equal(op, bit("000000")): lambda: alu_op_funct,
        tf.equal(op, bit("001000")): lambda: alu_op_add,
        tf.equal(op, bit("001001")): lambda: alu_op_add,
        tf.equal(op, bit("101011")): lambda: alu_op_add,
        tf.equal(op, bit("100011")): lambda: alu_op_add,
        tf.equal(op, bit("001111")): lambda: alu_op_add,
        tf.equal(op, bit("000001")): lambda: alu_op_slt,
        tf.equal(op, bit("000100")): lambda: alu_op_slt,  # Ignore
        tf.equal(op, bit("000011")): lambda: alu_op_add,
    }, default=lambda: alu_op_add, exclusive=True)

    reg_dst = tf.case({
        tf.equal(op, bit("000000")): lambda: reg_dst_rd,
        tf.equal(op, bit("001000")): lambda: reg_dst_rt,
        tf.equal(op, bit("001001")): lambda: reg_dst_rt,
        tf.equal(op, bit("101011")): lambda: reg_dst_rt,
        tf.equal(op, bit("100011")): lambda: reg_dst_rt,
        tf.equal(op, bit("001111")): lambda: reg_dst_rt,
        tf.equal(op, bit("000001")): lambda: reg_dst_ra,
        tf.equal(op, bit("000100")): lambda: reg_dst_ra,  # Ignore
        tf.equal(op, bit("000011")): lambda: reg_dst_ra,
    }, default=lambda: reg_dst_rd, exclusive=True)

    reg_write = tf.case({
        tf.equal(op, bit("000000")): lambda: True,
        tf.equal(op, bit("001000")): lambda: True,
        tf.equal(op, bit("001001")): lambda: True,
        tf.equal(op, bit("101011")): lambda: False,
        tf.equal(op, bit("100011")): lambda: True,
        tf.equal(op, bit("001111")): lambda: True,
        tf.equal(op, bit("000001")): lambda: tf.equal(get_bit(regimm, 4), 1),
        tf.equal(op, bit("000100")): lambda: False,
        tf.equal(op, bit("000011")): lambda: True,
    }, default=lambda: False, exclusive=True)

    alu_src_a = tf.constant(0)

    alu_src_b = tf.case({
        tf.equal(op, bit("000000")): lambda: alu_src_b_reg,
        tf.equal(op, bit("001000")): lambda: alu_src_b_imm,
        tf.equal(op, bit("001001")): lambda: alu_src_b_imm,
        tf.equal(op, bit("101011")): lambda: alu_src_b_imm,
        tf.equal(op, bit("100011")): lambda: alu_src_b_imm,
        tf.equal(op, bit("001111")): lambda: alu_src_b_imm_hi,
        tf.equal(op, bit("000001")): lambda: alu_src_b_zero,
        tf.equal(op, bit("000100")): lambda: alu_src_b_zero,  # Ignore
        tf.equal(op, bit("000011")): lambda: alu_src_b_reg,
    }, default=lambda: alu_src_b_reg, exclusive=True)

    mem_write = tf.case({
        tf.equal(op, bit("000000")): lambda: False,
        tf.equal(op, bit("001000")): lambda: False,
        tf.equal(op, bit("001001")): lambda: False,
        tf.equal(op, bit("101011")): lambda: True,
        tf.equal(op, bit("100011")): lambda: False,
        tf.equal(op, bit("001111")): lambda: False,
        tf.equal(op, bit("000001")): lambda: False,
        tf.equal(op, bit("000100")): lambda: False,
        tf.equal(op, bit("000011")): lambda: False,
    }, default=lambda: False, exclusive=True)

    reg_src = tf.case({
        tf.equal(op, bit("000000")): lambda: reg_src_alu,
        tf.equal(op, bit("001000")): lambda: reg_src_alu,
        tf.equal(op, bit("001001")): lambda: reg_src_alu,
        tf.equal(op, bit("101011")): lambda: reg_src_alu,
        tf.equal(op, bit("100011")): lambda: reg_src_mem,
        tf.equal(op, bit("001111")): lambda: reg_src_alu,
        tf.equal(op, bit("000001")): lambda: reg_src_pc,
        tf.equal(op, bit("000100")): lambda: reg_src_pc,
        tf.equal(op, bit("000011")): lambda: reg_src_pc,
    }, default=lambda: reg_src_alu, exclusive=True)

    pc_src = tf.case({
        tf.equal(op, bit("000000")): lambda: tf.case({
            tf.equal(funct, bit("001000")): lambda: pc_src_reg,
        }, default=lambda: pc_src_pc_add_4),
        tf.equal(op, bit("001000")): lambda: pc_src_pc_add_4,
        tf.equal(op, bit("001001")): lambda: pc_src_pc_add_4,
        tf.equal(op, bit("101011")): lambda: pc_src_pc_add_4,
        tf.equal(op, bit("100011")): lambda: pc_src_pc_add_4,
        tf.equal(op, bit("001111")): lambda: pc_src_pc_add_4,
        tf.equal(op, bit("000001")): lambda: pc_src_imm,
        tf.equal(op, bit("000100")): lambda: pc_src_imm,
        tf.equal(op, bit("000011")): lambda: pc_src_addr,
    }, default=lambda: pc_src_pc_add_4, exclusive=True)

    return [alu_op, alu_src_a, alu_src_b, reg_dst, reg_write, mem_write, reg_src, pc_src]
