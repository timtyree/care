#from https://documen.tician.de/pycuda/metaprog.html#why-metaprogramming
from jinja2 import Template

tpl = Template("""
    __global__ void add(
            {{ type_name }} *tgt,
            {{ type_name }} *op1,
            {{ type_name }} *op2)
    {
      int idx = threadIdx.x +
        {{ thread_block_size }} * {{block_size}}
        * blockIdx.x;

      {% for i in range(block_size) %}
          {% set offset = i*thread_block_size %}
          tgt[idx + {{ offset }}] =
            op1[idx + {{ offset }}]
            + op2[idx + {{ offset }}];
      {% endfor %}
    }""")

rendered_tpl = tpl.render(
    type_name="float", block_size=block_size,
    thread_block_size=thread_block_size)

mod = SourceModule(rendered_tpl)

# #also see:
# https://docs.dask.org/en/latest/gpu.html
# https://wiki.tiker.net/PyCuda/Examples/MultipleThreads/
# https://wiki.tiker.net/PyCuda/Examples/KernelConcurrency/
# https://wiki.tiker.net/PyCuda/Examples/#existing-examples
# https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
# https://github.com/fjarri/reikna
# https://documen.tician.de/pycuda/array.html#module-pycuda.reduction