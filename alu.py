#!/usr/bin/env python
# -*- coding: utf-8 -*-


import tensorflow as tf
from constants import *
from utils import *


def alu(a, b, ctrl):
    sub = get_bit(ctrl, 0)

    and_out = tf.bitwise.bitwise_and(a, b)
    or_out = tf.bitwise.bitwise_or(a, b)
    xor_out = tf.bitwise.bitwise_xor(a, b)

    add_sub_out = tf.cond(tf.equal(sub, 1), lambda: a - b, lambda: a + b)

    oflow_out = (tf.equal(get_bit(a, 31), 0) & tf.equal(get_bit(b, 31), 0) & tf.equal(get_bit(add_sub_out, 31), 1)) | (
        tf.equal(get_bit(a, 31), 1) & tf.equal(get_bit(b, 31), 1) & tf.equal(get_bit(add_sub_out, 31), 0))

    slt_out = tf.bitwise.bitwise_xor(get_bit(add_sub_out, 31), tf.cond(
        oflow_out, lambda: 1, lambda: 0))

    y_out = tf.case({
        tf.equal(ctrl, alu_ctrl_add): lambda: a + b,
        tf.equal(ctrl, alu_ctrl_sub): lambda: a - b,
        tf.equal(ctrl, alu_ctrl_slt): lambda: slt_out,
        tf.equal(ctrl, alu_ctrl_and): lambda: and_out,
        tf.equal(ctrl, alu_ctrl_or): lambda: or_out,
        tf.equal(ctrl, alu_ctrl_xor): lambda: xor_out,
        tf.equal(ctrl, alu_ctrl_sll): lambda: tf.bitwise.left_shift(a, b),
        tf.equal(ctrl, alu_ctrl_srl): lambda: tf.bitwise.right_shift(a, b),
        tf.equal(ctrl, alu_ctrl_sra): lambda: tf.bitwise.right_shift(a, b),
    }, default=lambda: 0, exclusive=True)

    zero_out = tf.equal(y_out, 0)

    return [y_out, oflow_out, zero_out]
