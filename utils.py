#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class obj_dict(dict):
    __getattr__ = dict.get

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__


class sync_update:
    def __init__(self, regs):
        self.__dict__['regs'] = regs
        self.__dict__['funcs'] = []

    def append(self, func):
        self.funcs.append(func)

    def update(self, deps):
        temp_regs = obj_dict()

        # Allocate temporary variables to update without overwrite current values.
        for name, reg in self.regs.items():
            temp_regs[name] = tf.Variable(reg.initial_value,
                                          dtype=tf.identity(reg).dtype, shape=reg.shape)

        updated_regs = []
        with tf.control_dependencies(deps):
            for func in self.funcs:
                for updated in func(temp_regs):
                    updated_regs.append(updated)

        # Assign temporary variables to actual variables.
        operations = []
        with tf.control_dependencies(deps + updated_regs):
            for name, reg in self.regs.items():
                op = reg.assign(temp_regs[name])
                operations.append(op)

        return operations


class sync_exec:

    def __init__(self, name):
        self.lst = []
        self.name = name

    def query(self, func):
        self.lst.append(func)

    def execute(self, deps, ops):
        operations = []

        with tf.control_dependencies(deps):
            for i, func in enumerate(self.lst):
                op = func(ops)
                operations.extend(op)

        return operations


def bit(s):
    return int(s, 2)


def get_bits(value, from_pos: int, to_pos: int, name=None):
    width = to_pos - from_pos + 1
    mask = (1 << width) - 1
    return tf.bitwise.bitwise_and(tf.bitwise.right_shift(value, from_pos), mask, name)


def get_bit(value, pos: int):
    return tf.bitwise.bitwise_and(tf.bitwise.right_shift(value, pos), 1)


def sign_extend(x, bits):
    sign_bit = tf.constant(1 << (bits - 1), dtype=tf.int32)
    y = tf.bitwise.bitwise_and(x, (sign_bit - 1)) - \
        tf.bitwise.bitwise_and(x, sign_bit)
    return y


def clr(regin, clr):
    return tf.cond(clr, lambda: tf.cast(tf.constant(0), dtype=regin.dtype), lambda: regin)


def clr_en(reg, regin, clr, en):
    return tf.cond(clr, lambda: tf.constant(0), lambda: tf.cond(en, lambda: regin, lambda: reg))
