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
    internal class Sum : IAlgorithm
    {
        #region Members

        // gpu things
        Context context;
        Accelerator device;

        // architecture things
        public int size = 1024;

        // data
        public float[] inputs;
        public float[] inputs2;
        public float[] outputs;

        // buffers
        protected MemoryBuffer1D<float, Stride1D.Dense> aBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> bBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> cBuffer;

        // kernels
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>> kernel;

        #endregion

        #region Initialize

        public void InitCpu()
        {
            inputs = new float[size];
            inputs2 = new float[size];
            outputs = new float[size];

            for (int i = 0; i < size; i++)
                inputs[i] = i;

            for (int i = 0; i < size; i++)
                inputs2[i] = size-i;
        }

        public void InitGpu(Context context, Accelerator device, bool forceCPU = false)
        {
            // set up the gpu
            this.context = context;
            this.device = device;
        }

        public void InitBuffers()
        {
            aBuffer = device.Allocate1D<float>(size);
            bBuffer = device.Allocate1D<float>(size);
            cBuffer = device.Allocate1D<float>(size);
        }

        public void CompileKernels()
        {
            kernel = device.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>,
            ArrayView1D<float, Stride1D.Dense>>(add);
        }

        #endregion

        #region Run

        public void Load()
        {
            aBuffer.CopyFromCPU(inputs);
            bBuffer.CopyFromCPU(inputs2);
        }

        public void Run()
        {
            kernel(size, aBuffer, bBuffer, cBuffer);
        }

        #endregion

        #region Getters

        public float[] GetOutputs()
        {
            outputs = cBuffer.GetAsArray1D();
            return outputs;
        }

        #endregion

        #region Kernels

        public static void add(Index1D index, ArrayView1D<float, Stride1D.Dense> A, ArrayView1D<float, Stride1D.Dense> B, ArrayView1D<float, Stride1D.Dense> C)
        {
            C[index] = A[index] + B[index];
        }

        #endregion

        #region Cpu Implementation

        public float[] RunCpu()
        {
            for (int i = 0; i < inputs.Length; i++)
                outputs[i] = inputs[i] + inputs2[i];

            return outputs;
        }

        #endregion
    }
}
