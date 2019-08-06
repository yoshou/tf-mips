#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from constants import *
from utils import *


def alu_ctrl(op, funct):
    f = tf.case({
        tf.equal(funct, alu_funct_add): lambda: alu_ctrl_add,
        tf.equal(funct, alu_funct_addu): lambda: alu_ctrl_add,
        tf.equal(funct, alu_funct_sub): lambda: alu_ctrl_sub,
        tf.equal(funct, alu_funct_subu): lambda: alu_ctrl_sub,
        tf.equal(funct, alu_funct_and): lambda: alu_ctrl_and,
        tf.equal(funct, alu_funct_or): lambda: alu_ctrl_or,
        tf.equal(funct, alu_funct_jr): lambda: alu_ctrl_or,
    }, default=lambda: alu_ctrl_zero, exclusive=True)

    ctrl = tf.case({
        tf.equal(op, alu_op_add): lambda: alu_ctrl_add,
        tf.equal(op, alu_op_sub): lambda: alu_ctrl_sub,
        tf.equal(op, alu_op_funct): lambda: f,
    }, default=lambda: alu_ctrl_zero, exclusive=True)

    return ctrl
