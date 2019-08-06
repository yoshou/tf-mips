
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from utils import *

reg_no_zero = 0
reg_no_at = 1
reg_no_v0 = 2
reg_no_v1 = 3
reg_no_a0 = 4
reg_no_a1 = 5
reg_no_a2 = 6
reg_no_a3 = 7
reg_no_t0 = 8
reg_no_t1 = 9
reg_no_t2 = 10
reg_no_t3 = 11
reg_no_t4 = 12
reg_no_t5 = 13
reg_no_t6 = 14
reg_no_t7 = 15
reg_no_s0 = 16
reg_no_s1 = 17
reg_no_s2 = 18
reg_no_s3 = 19
reg_no_s4 = 20
reg_no_s5 = 21
reg_no_s6 = 22
reg_no_s7 = 23
reg_no_t8 = 24
reg_no_t9 = 25
reg_no_k0 = 26
reg_no_k1 = 27
reg_no_gp = 28
reg_no_sp = 29
reg_no_s8 = 30
reg_no_ra = 31


class Regfile:
    def __init__(self, size, initial_sp=1024):
        initial = np.zeros((size,))

        initial[reg_no_sp] = initial_sp

        self.regfile = tf.Variable(initial, dtype=tf.int32)

    def port_map(self, we, r1_addr, r2_addr, w_addr, w_data, sync):

        def write(regs):
            yield regs.regfile.assign(tf.cond(we, lambda: tf.scatter_update(
                self.regfile, [get_bits(w_addr, 0, 4)], [w_data]), lambda: self.regfile))

        sync.append(write)

        r1_addr = get_bits(r1_addr, 0, 4)
        r2_addr = get_bits(r2_addr, 0, 4)

        r1_data = tf.cond(tf.equal(r1_addr, 0),
                          lambda: 0, lambda: tf.gather_nd(self.regfile, [r1_addr]))

        r2_data = tf.cond(tf.equal(r2_addr, 0),
                          lambda: 0, lambda: tf.gather_nd(self.regfile, [r2_addr]))

        return [r1_data, r2_data]
