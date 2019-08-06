from unittest import TestCase, main

import numpy as np
from alu_ctrl import *
from alu import *
import tensorflow as tf


class AluTest(TestCase):

    def test_add(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 1, b: 2, ctrl: alu_ctrl_add})

            self.assertEqual(result_y, 3)
            self.assertEqual(result_oflow, False)
            self.assertEqual(result_zero, False)

    def test_add_zero(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 0, b: 0, ctrl: alu_ctrl_add})

            self.assertEqual(result_zero, True)

    def test_add_oflow(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 2147483647, b: 1, ctrl: alu_ctrl_add})

            self.assertEqual(result_oflow, True)

    def test_sub(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 904, b: 0, ctrl: alu_ctrl_sub})

            self.assertEqual(result_y, 904)
            self.assertEqual(result_oflow, False)
            self.assertEqual(result_zero, False)

    def test_and(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 52435234, b: 765754, ctrl: alu_ctrl_and})

            self.assertEqual(result_y, 52435234 & 765754)

    def test_sll(self):
        a = tf.placeholder(tf.int32)
        b = tf.placeholder(tf.int32)
        ctrl = tf.placeholder(tf.int32)

        y, oflow, zero = alu(
            a, b, ctrl)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, result_oflow, result_zero = sess.run(
                [y, oflow, zero], {a: 52435234, b: 4, ctrl: alu_ctrl_sll})

            self.assertEqual(result_y, 52435234 << 4)


class SignExtendTest(TestCase):

    def test_sign_extend(self):
        x = tf.placeholder(tf.int32)

        y = sign_extend(x, 16)

        with tf.compat.v1.Session() as sess:
            init_op = tf.compat.v1.global_variables_initializer()
            sess.run(init_op)

            result_y, = sess.run(
                [y], {x: -20000 & ((1 << 16) - 1)})

            self.assertEqual(result_y, -20000)


if __name__ == "__main__":
    main()
