import numpy as np
import tvm
from tvm import relay


fbgemm_workloads = [
    (64, 800, 320),
    (64, 768, 512),
    (16, 256, 512),
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
]

bert_workloads = [(128, 768, 3072), (128, 768, 768), (128, 3072, 768)]


def test_dense_vnni():
    for m, n, k in bert_workloads:
        data_shape = (m, k)
        weight_shape = (n, k)

        with_bias = False

        data = relay.var("data", shape=data_shape, dtype="uint8")
        weight = relay.var("weight", shape=weight_shape, dtype="int8")
        bias = relay.var("bias", shape=(weight_shape[0],), dtype="int32")
        dense = relay.nn.dense(data, weight, out_dtype="int32")

        if with_bias:
            out = relay.nn.bias_add(dense, bias)
        else:
            out = dense

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

        if with_bias:
            runtime.set_input("bias", c)
        else:
            c = np.zeros_like(c)

        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c.astype("int32")

        np.testing.assert_equal(out, ref)

        gops_per_mm = 2 * m * n * k
        elapsed = runtime.benchmark(dev, number=1, repeat=500).mean
        print(m, n, k, gops_per_mm / elapsed / 1e9)


test_dense_vnni()
