using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ILGPU.Runtime;
using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.CPU;
using GPU_Algorithms.Algorithms;

namespace GPU_Algorithms
{
    internal class Program
    {
        public static Context context;
        public static Accelerator device;

        static void Main(string[] args)
        {
            IAlgorithm algorithm = new Hello();

            try
            {
                InitGpu();

                algorithm.InitCpu();
                algorithm.InitGpu(context, device);
                algorithm.InitBuffers();
                algorithm.CompileKernels();
                algorithm.Run();
                float[] outputs = algorithm.GetOutputs();

                foreach (var item in outputs)
                {
                    Console.WriteLine(item);
                }
            }
            catch (Exception e)
            {
                Console.WriteLine(e);
            }

            Console.WriteLine("done");
            Console.ReadKey();
        }

        public static void InitGpu(bool forceCPU = false)
        {
            // set up the gpu
            context = Context.Create(builder => builder.Cuda().CPU().EnableAlgorithms());
            device = context.GetPreferredDevice(forceCPU).CreateAccelerator(context);
        }
    }
}
