# UbuntuPC CPU tests

..................                                                       [100%]
=============================== warnings summary ===============================
tools/tests/test_deepseek_sboed.py::test_dad_raw_encoder_empty_history_and_order_symmetry
  /home/qiqi/Documents/objective_power_grid_v1_specialization/tools/tests/test_deepseek_sboed.py:32: UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior.
  Consider using tensor.detach() first. (Triggered internally at /pytorch/torch/csrc/autograd/generated/python_variable_methods.cpp:836.)
    if stage>0:assert float(m.input_norm(m.encode(f,stage)).square().mean().sqrt())<1.01

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
18 passed, 1 warning in 2.10s
