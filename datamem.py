
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from utils import *


def datamem_connect(mem, we, addr, d_in, sync):
    def write(regs):
        yield regs.data_mem.assign(tf.cond(we, lambda: tf.scatter_update(
            mem, [addr], [d_in]), lambda: mem))

    sync.append(write)

    addr = tf.maximum(addr, 0)
    addr = tf.minimum(addr, mem.shape[0] - 1)

    d_out = tf.gather_nd(mem, [addr])

    return [d_out]
