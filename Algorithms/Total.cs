// This works, but I think it's only through luck. This algorithm violates the super important rule:
// Don't let the output of one thread depend on the output of another thread

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;

namespace GPU_Algorithms.Algorithms
{
    /// <summary>
    /// Adds up all of the numbers in an array. 
    /// The result is an array where each element k is the sum of all elements 0 through k.
    /// </summary>
    internal class Total : IAlgorithm
    {
        #region Members

        // gpu things
        Context context;
        Accelerator device;

        // architecture things
        public int size = 512;  // arrays can't be bigger than 512 or else they break
        
        // data
        public float[] inputs;
        public float[] outputsGpu;
        public float[] outputsCpu;

        // buffers
        protected MemoryBuffer1D<float, Stride1D.Dense> aBuffer;

        // kernels
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>> kernel;

        #endregion

        #region Initialize

        public void InitCpu()
        {
            inputs = new float[size];
            outputsCpu = new float[size];

            for (int i = 0; i < size; i++)
                inputs[i] = i;
        }

        public void InitGpu(Context context, Accelerator device, bool forceCPU = false)
        {
            // set up the gpu
            this.context = context;
            this.device = device;

            outputsGpu = new float[size];
        }

        public void InitBuffers()
        {
            aBuffer = device.Allocate1D<float>(size);
        }

        public void CompileKernels()
        {
            kernel = device.LoadAutoGroupedStreamKernel<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>>(total);
        }

        #endregion

        #region Run

        public void Load()
        {
            aBuffer.CopyFromCPU(inputs);
        }

        public void Run()
        {
            kernel(size, aBuffer);
        }

        #endregion

        #region Getters

        public float[] GetOutputs()
        {
            outputsGpu = aBuffer.GetAsArray1D();
            return outputsGpu;
        }

        #endregion

        #region Kernels

        public static void total(Index1D index, ArrayView1D<float, Stride1D.Dense> A)
        {
            for (int offset = 1; offset <= A.Length; offset *= 2)
            {
                if (index - offset >= 0)
                    A[index] += A[index - offset];

                // wait until every thread gets through this iteration
                Group.Barrier();
            }
        }

        #endregion

        #region Cpu Implementation

        public float[] RunCpu()
        {
            for (int i = 1; i < inputs.Length; i++)
                outputsCpu[i] = outputsCpu[i-1] + inputs[i];

            return outputsCpu;
        }

        #endregion
    }
}
