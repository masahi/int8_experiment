import numpy as np
import tvm
from tvm import relay


def test_dense_vnni():
    for m, n, k in [
        (64, 800, 320),
        (64, 768, 512),
        (16, 256, 512),
        (128, 128, 128),
        (256, 512, 256),
        (1024, 1024, 1024),
    ]:
        data_shape = (m, k)
        weight_shape = (n, k)

        data = relay.var("data", shape=data_shape, dtype="uint8")
        weight = relay.var("weight", shape=weight_shape, dtype="int8")
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        dense = relay.nn.dense(data, weight, out_dtype="int32")
        out = relay.nn.bias_add(dense, bias)
        mod = tvm.IRModule.from_expr(out)

        target = "llvm -mcpu=cascadelake"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target)

        dev = tvm.device(target, 0)
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

        a = np.random.uniform(1, 10, size=data_shape).astype("uint8")
        b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
        c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

        runtime.set_input("data", a)
        runtime.set_input("weight", b)
        runtime.set_input("bias", c)
        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = np.dot(a, b.transpose()) + c

        np.testing.assert_equal(out, ref)

        print(runtime.benchmark(dev, number=1, repeat=500))


test_dense_vnni()
