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
    internal class Hello : IAlgorithm
    {
        #region Members

        // gpu things
        Context context;
        Accelerator device;

        // architecture things
        public int size = 1;

        // data
        public float[] inputs;
        public float[] outputs;

        // buffers
        protected MemoryBuffer1D<float, Stride1D.Dense> inputsBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> outputsBuffer;

        // kernels
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> kernel;

        #endregion

        #region Initialize

        public void InitCpu()
        {
            inputs = new float[size];
            outputs = new float[size];

            for (int i = 0; i < size; i++)
                inputs[i] = i;
        }

        public void InitGpu(Context context, Accelerator device, bool forceCPU = false)
        {
            // set up the gpu
            this.context = context;
            this.device = device;
        }

        public void InitBuffers()
        {
            inputsBuffer = device.Allocate1D<float>(size);
            outputsBuffer = device.Allocate1D<float>(size);
        }

        public void CompileKernels()
        {
            kernel = device.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(addOne);
        }

        #endregion

        #region Run

        public void Run()
        {
            kernel(size, inputsBuffer, outputsBuffer);
        }

        #endregion

        #region Getters

        public float[] GetOutputs()
        {
            outputs = outputsBuffer.GetAsArray1D();
            return outputs;
        }

        #endregion

        #region Kernels

        public static void addOne(Index1D index, ArrayView1D<float, Stride1D.Dense> inputs, ArrayView1D<float, Stride1D.Dense> outputs)
        {
            outputs[index] = inputs[index] + 1;
        }

        #endregion

    }
}
