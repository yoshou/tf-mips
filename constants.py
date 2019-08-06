#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import bit

alu_op_add = 0
alu_op_sub = 1
alu_op_funct = 2
alu_op_slt = 3

reg_dst_rt = 0
reg_dst_rd = 2
reg_dst_ra = 1

alu_src_b_reg = 0
alu_src_b_imm = 1
alu_src_b_shamt = 2
alu_src_b_imm_hi = 3
alu_src_b_zero = 4

pc_src_pc_add_4 = 0
pc_src_addr = 1
pc_src_reg = 2
pc_src_imm = 3

reg_src_alu = 0
reg_src_mem = 1
reg_src_pc = 2

alu_ctrl_zero = bit("0000")
alu_ctrl_add = bit("0010")
alu_ctrl_sub = bit("0011")
alu_ctrl_slt = bit("0101")
alu_ctrl_and = bit("1000")
alu_ctrl_or = bit("1001")
alu_ctrl_xor = bit("1010")
alu_ctrl_sll = bit("1100")
alu_ctrl_srl = bit("1101")
alu_ctrl_sra = bit("1110")

alu_funct_add = bit("100000")
alu_funct_addu = bit("100001")
alu_funct_sub = bit("100010")
alu_funct_subu = bit("100011")
alu_funct_and = bit("100100")
alu_funct_or = bit("100101")
alu_funct_jr = bit("001000")
alu_funct_invalid = bit("000000")
