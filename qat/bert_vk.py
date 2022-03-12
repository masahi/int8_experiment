import numpy as np

import tvm
from tvm import relay, autotvm
from tvm.autotvm.tuner import XGBTuner, RandomTuner

import warnings
warnings.simplefilter('ignore')


with open("/home/masa/projects/dev/tensorir-experiment/models/bert_base_int8.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open("/home/masa/projects/dev/tensorir-experiment/models/bert_base_int8.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())


target = "vulkan -from_device=0"


measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(
        number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
    ),
)

def tune(log_file, task, use_random=False):
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

# tasks = autotvm.task.extract_from_program(
#     mod["main"],
#     target=target,
#     params=params,
# )

# log_file = "bert_base_int8_vk.log"

# for task in tasks[:-1]:
#     tune(log_file, task)

# with autotvm.apply_history_best(log_file):
with tvm.transform.PassContext(opt_level=3):
    mod = relay.transform.FastMath()(mod)
    lib = relay.build(mod, params=params, target=target)

dev = tvm.device(target, 0)
runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

batch_size = 1
seq_len = 384

shape_dict = {
    "input_ids": (batch_size, seq_len),
    "segment_ids": (batch_size, seq_len),
    "input_mask": (batch_size, seq_len),
}

for name, shape in shape_dict.items():
    arr = np.random.uniform(1, 10, size=shape).astype("int64")
    runtime.set_input(name, arr)

runtime.run()

# out = runtime.get_output(0).numpy()

print(runtime.benchmark(dev, number=1, repeat=50).mean)
