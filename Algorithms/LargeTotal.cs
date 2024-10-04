// I can't get this one to work. I can only assume that it's because the kernel violates the super important rule:
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
    internal class LargeTotal : IAlgorithm
    {
        #region Members

        // gpu things
        Context context;
        Accelerator device;

        // architecture things
        public static int size = 16;//1024;
        public static int blockSize = 8;//512; // we can only pass 512 elements to a block or else it will break
        public static int numberOfBlocks = (int) Math.Ceiling( (decimal) size / blockSize );
        
        // data
        public float[] inputs;
        public float[] outputsGpu;
        public float[] outputsCpu;

        // hidden data
        private float[,] inputs2D;
        private float[,] outputs2D;

        // buffers
        protected MemoryBuffer2D<float, Stride2D.DenseX> aBuffer;
        protected MemoryBuffer1D<float, Stride1D.Dense> holdingBuffer;

        // kernels
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>> stepOneKernel;
        public Action<
            Index1D,
            ArrayView1D<float, Stride1D.Dense>> stepTwoKernel;
        public Action<
            Index2D,
            ArrayView2D<float, Stride2D.DenseX>,
            ArrayView1D<float, Stride1D.Dense>> stepThreeKernel;

        #endregion

        #region Initialize

        public void InitCpu()
        {
            inputs = new float[size];
            outputsCpu = new float[size];
            inputs2D = new float[numberOfBlocks, blockSize];

            for (int i = 0; i < size / 2; i++)
                inputs[i] = i;
            for (int i = size / 2; i < size; i++)
                inputs[i] = i - size / 2;
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
            aBuffer = device.Allocate2DDenseX<float>(new Index2D(numberOfBlocks, blockSize));
            holdingBuffer = device.Allocate1D<float>(numberOfBlocks);
        }

        public void CompileKernels()
        {
            stepOneKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>>(stepOne);
            stepTwoKernel = device.LoadAutoGroupedStreamKernel<
                Index1D,
                ArrayView1D<float, Stride1D.Dense>>(stepTwo);
            stepThreeKernel = device.LoadAutoGroupedStreamKernel<
                Index2D,
                ArrayView2D<float, Stride2D.DenseX>,
                ArrayView1D<float, Stride1D.Dense>>(stepThree);
        }

        #endregion

        #region Run

        public void Load()
        {
            for (int i = 0; i < numberOfBlocks; i++)
                for (int j = 0; j < blockSize; j++)
                    inputs2D[i, j] = inputs[i * blockSize + j];

            aBuffer.CopyFromCPU(inputs2D);
        }

        public void Run()
        {
            stepOneKernel(new Index2D(numberOfBlocks, blockSize), aBuffer, holdingBuffer);
            device.Synchronize();
            //stepTwoKernel(numberOfBlocks, holdingBuffer);
            //device.Synchronize();
            //stepThreeKernel(new Index2D(numberOfBlocks, blockSize), aBuffer, holdingBuffer);
            //device.Synchronize();
        }

        #endregion

        #region Getters

        public float[] GetOutputs()
        {
            outputs2D = aBuffer.GetAsArray2D();
            for (int i = 0; i < numberOfBlocks; i++)
                for (int j = 0; j < blockSize; j++)
                    outputsGpu[i * blockSize + j] = outputs2D[i, j];
            return outputsGpu;
        }

        #endregion

        #region Kernels

        public static void stepOne(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView1D<float, Stride1D.Dense> holding)
        {
            float temp = 0;
            // do each individual block
            //for (int offset = 1; offset <= 2; offset *= 2)
            int offset = 1;
            {
                if (index.Y - offset >= 0)
                    temp = A[index.X, index.Y - offset];
                Group.Barrier();

                if (index.Y - offset >= 0)
                    A[index.X, index.Y] += temp;
                Group.Barrier();
            }

            offset = 2;
            {
                if (index.Y - offset >= 0)
                    temp = A[index.X, index.Y - offset];
                Group.Barrier();

                if (index.Y - offset >= 0)
                    A[index.X, index.Y] = temp;
                Group.Barrier();
            }

            // store the last element of each block in the holding array
            //if (index.Y == A.Extent.Y - 1)
            //{
            //    holding[index.X] = A[index.X, index.Y];
            //}
        }

        public static void stepTwo(Index1D index, ArrayView1D<float, Stride1D.Dense> holding)
        {
            for (int offset = 1; offset <= holding.Length; offset *= 2)
            {
                if (index - offset >= 0)
                    holding[index] += holding[index - offset];

                // wait until every thread gets through this iteration
                Group.Barrier();
            }

            // lazilly shift them all down one
            if (index > 0)
                holding[index] = holding[index - 1];
            else
                holding[index] = 0;
            Group.Barrier();
        }

        public static void stepThree(Index2D index, ArrayView2D<float, Stride2D.DenseX> A, ArrayView1D<float, Stride1D.Dense> holding)
        {
            A[index.X, index.Y] += holding[index.X];
        }



        #endregion

        #region Cpu Implementation

        public float[] RunCpu()
        {
            outputsCpu[0] = inputs[0];
            for (int i = 1; i < inputs.Length; i++)
                outputsCpu[i] = outputsCpu[i-1] + inputs[i];

            return outputsCpu;
        }

        #endregion
    }
}
