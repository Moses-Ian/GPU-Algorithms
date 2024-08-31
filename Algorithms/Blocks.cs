// This uses blocks and prints those numbers
// This isn't working because I can't find a good example to borrow from

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
    internal class Blocks : IAlgorithm
    {
        #region Members

        // gpu things
        Context context;
        Accelerator device;

        // architecture things
        public int blocks = 4;
        public int blockSize = 4;

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
            inputs = new float[blocks * blockSize];
            outputs = new float[blocks * blockSize];

            for (int i = 0; i < blocks; i++)
                for (int j = 0; j < blockSize; j++)
                    inputs[i * blockSize + j] = j;
        }

        public void InitGpu(Context context, Accelerator device, bool forceCPU = false)
        {
            // set up the gpu
            this.context = context;
            this.device = device;
        }

        public void InitBuffers()
        {
            inputsBuffer = device.Allocate1D<float>(blocks * blockSize);
            outputsBuffer = device.Allocate1D<float>(blocks * blockSize);
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

        public void Load()
        {
            inputsBuffer.CopyFromCPU(inputs);
        }

        public void Run()
        {
            // I do not understand what part of this isn't working

            //device.Launch <
            //Index1D,
            //ArrayView1D<float, Stride1D.Dense>,
            //ArrayView1D<float, Stride1D.Dense> > (kernel, new KernelConfig(blocks, blockSize), inputsBuffer, outputsBuffer);
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

        #region Cpu Implementation

        public float[] RunCpu()
        {
            for (int i = 0; i < inputs.Length; i++)
                outputs[i] = inputs[i] + 1;

            return outputs;
        }

        #endregion
    }
}
