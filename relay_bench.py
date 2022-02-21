import numpy as np
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, RandomTuner
import tvm.topi.testing


fbgemm_workloads = [
    (64, 800, 320),
    (64, 768, 512),
    (16, 256, 512),
    (128, 128, 128),
    (256, 512, 256),
    (1024, 1024, 1024),
]

bert_workloads = [(128, 768, 3072), (128, 768, 768), (128, 3072, 768)]

log_file = "dense_vnni.log"
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(
        number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
    ),
)


def tune(task, use_random=False):
    if use_random:
        tuner_obj = RandomTuner(task)
    else:
        tuner_obj = XGBTuner(task, loss_type="rank")

    n_trial = len(task.config_space)
    tuner_obj.tune(
        n_trial=n_trial,
        early_stopping=None,
        measure_option=measure_option,
        callbacks=[
            autotvm.callback.progress_bar(n_trial),
            autotvm.callback.log_to_file(log_file),
        ],
    )


def test(mod, params, input_names, np_inputs, np_out, do_tune=True):
    target = "llvm -mcpu=cascadelake"

    if do_tune:
        with tvm.transform.PassContext(opt_level=3):
            opt_mod, opt_params = relay.optimize(mod, params=params, target=target)

        tasks = autotvm.task.extract_from_program(
            opt_mod["main"],
            target=target,
            params=opt_params,
        )
        tune(tasks[0])

        with autotvm.apply_history_best(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, params=params, target=target)
                # print(lib.lib.get_source("asm"))
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, params=params, target=target)

    dev = tvm.device(target, 0)
    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    for name, inp in zip(input_names, np_inputs):
        runtime.set_input(name, inp)

    runtime.run()

    out = runtime.get_output(0).numpy()
    np.testing.assert_equal(out, np_out)

    return runtime.benchmark(dev, number=1, repeat=500).mean



def test_dense_vnni():
    for m, n, k in fbgemm_workloads + bert_workloads:
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

        a = np.random.uniform(1, 10, size=data_shape).astype("uint8")
        b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
        c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

        params = {"weight": b, "bias": c}

        if not with_bias:
            c = np.zeros_like(c)

        ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c.astype(
            "int32"
        )

        gops_per_mm = 2 * m * n * k
        elapsed = test(mod, params, ["data"], [a], ref)
        print(m, n, k, gops_per_mm / elapsed / 1e9)


def test_batch_matmul_vnni():
    batch = 8
    for m, n, k in fbgemm_workloads + bert_workloads:
        x_shape = (batch, m, k)
        y_shape = (batch, n, k)

        x = relay.var("x", shape=x_shape, dtype="uint8")
        y = relay.var("y", shape=y_shape, dtype="int8")
        out = relay.nn.batch_matmul(x, y, out_dtype="int32")

        mod = tvm.IRModule.from_expr(out)

        a = np.random.uniform(1, 10, size=x_shape).astype("uint8")
        b = np.random.uniform(1, 10, size=y_shape).astype("int8")

        params = {}

        gops_per_mm = 2 * m * n * k * batch
        ref = tvm.topi.testing.batch_matmul(a, b, out_dtype="int32")

        elapsed = test(mod, params, ["x", "y"], [a, b], ref)
        print(m, n, k, gops_per_mm / elapsed / 1e9)


test_dense_vnni()
# test_batch_matmul_vnni()
