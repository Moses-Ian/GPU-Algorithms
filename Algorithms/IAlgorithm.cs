using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

namespace GPU_Algorithms.Algorithms
{
    public interface IAlgorithm
    {
        void InitCpu();
        void InitGpu(Context context, Accelerator device, bool forceCPU = false);
        void InitBuffers();
        void CompileKernels();
        void Run();
        float[] GetOutputs();
    }
}
