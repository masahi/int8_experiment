import numpy as np

import tvm
from tvm import relay


with open("models/bert_large_int8.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())

with open("models/bert_large_int8.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())


target = "llvm -mcpu=cascadelake"
with tvm.transform.PassContext(opt_level=3):
    # opt_mod, opt_params = relay.optimize(mod, target=target, params=params)
    # print(opt_mod["main"])

    lib = relay.build(mod, params=params, target=target)

# asm = lib.lib.get_source("asm")
# assert "vpdpbusd" in asm

# print(asm)


# dev = tvm.device(target, 0)
# runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

# batch_size = 1
# seq_len = 128

# shape_dict = {
#     "input_ids": (batch_size, seq_len),
#     "attention_mask": (batch_size, seq_len),
#     "token_type_ids": (batch_size, seq_len),
# }

# for name, shape in shape_dict.items():
#     arr = np.random.uniform(1, 10, size=shape).astype("int64")
#     runtime.set_input(name, arr)

# runtime.run()

# out = runtime.get_output(0).numpy()

# print(runtime.benchmark(dev, number=1, repeat=50).mean)
