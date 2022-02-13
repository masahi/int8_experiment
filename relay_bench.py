import numpy as np
import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, RandomTuner


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

        target = "llvm -mcpu=cascadelake"

        a = np.random.uniform(1, 10, size=data_shape).astype("uint8")
        b = np.random.uniform(1, 10, size=weight_shape).astype("int8")
        c = np.random.uniform(1, 10, size=(weight_shape[0],)).astype("int32")

        params = {"weight": b, "bias": c}

        do_tune = True

        if do_tune:
            tasks = autotvm.task.extract_from_program(
                mod["main"],
                target=target,
                params=params,
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

        runtime.set_input("data", a)

        if not with_bias:
            c = np.zeros_like(c)

        runtime.run()

        out = runtime.get_output(0).numpy()
        ref = np.dot(a.astype("int32"), b.transpose().astype("int32")) + c.astype(
            "int32"
        )

        np.testing.assert_equal(out, ref)

        gops_per_mm = 2 * m * n * k
        elapsed = runtime.benchmark(dev, number=1, repeat=500).mean
        print(m, n, k, gops_per_mm / elapsed / 1e9)


test_dense_vnni()
