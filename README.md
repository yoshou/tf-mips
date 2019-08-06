tf-mips
========

About
-----
This project implements a simple pipeline CPU with TensorFlow, which ISA is subset of MIPS Ⅰ.
The purposes for this implementations is studying tensorflow as a dataflow programming.
As the result, achieved to running a infinite counting program compiled GCC C compiler.

Features of the techniques are:

* Register is implemented with tf.Variable
* Pipeline stages are described from downstream in order to forwarding wires to up stages.
* Next states of registers are temporary save to new variables and copy to actual registers to
synchronize.

Reference
---------

TensorFlow:
  https://github.com/tensorflow/tensorflow

Chisel and Firrtl:
  https://github.com/freechipsproject/chisel3
  https://github.com/freechipsproject/firrtl


Contact
-------
If you encounter any issues or have any questions, please use the
GitHub issues page:

Tensorflow
  https://github.com/yoshou/tf-mips/issues
