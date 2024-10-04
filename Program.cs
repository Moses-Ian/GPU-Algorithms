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
using System.Diagnostics;

namespace GPU_Algorithms
{
    internal class Program
    {
        // parameters
        static bool logDeviceInfo = false;
        static bool logOutputs = true;

        static Context context;
        static Accelerator device;

        static void Main(string[] args)
        {
            //IAlgorithm algorithm = new Hello();
            //IAlgorithm algorithm = new Sum();
            IAlgorithm algorithm = new Total();

            Stopwatch stopwatch = new Stopwatch();
            // time tracking
            long loadTime;
            long runTime;
            long readTime;
            long cpuTime;

            try
            {
                InitGpu();

                algorithm.InitCpu();
                algorithm.InitGpu(context, device);
                algorithm.CompileKernels();

                stopwatch.Restart();
                    algorithm.InitBuffers();
                    algorithm.Load();
                stopwatch.Stop();
                loadTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                    algorithm.Run();
                    device.Synchronize();
                stopwatch.Stop();
                runTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                    float[] outputs = algorithm.GetOutputs();
                stopwatch.Stop();
                readTime = stopwatch.ElapsedMilliseconds;

                stopwatch.Restart();
                    float[] correct = algorithm.RunCpu();
                stopwatch.Stop();
                cpuTime = stopwatch.ElapsedMilliseconds;

                if (logOutputs)
                {
                    Console.WriteLine("Gpu:");
                    Console.WriteLine(string.Join(", ", outputs));
                    Console.WriteLine("Cpu:");
                    Console.WriteLine(string.Join(", ", correct));
                }
                Console.WriteLine(string.Format("Gpu: {0} | {1} | {2}", loadTime, runTime, readTime));
                Console.WriteLine(string.Format("Total: {0}", loadTime + runTime + readTime));
                Console.WriteLine("Cpu: " + cpuTime);

                Console.WriteLine(Compare(outputs, correct) ? "Correct" : "Wrong");

                //foreach (var item in outputs)
                //{
                //    Console.WriteLine(item);
                //}
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

            if (logDeviceInfo)
            {
                Console.WriteLine(device.Device.Name);
                Console.WriteLine($"{device.Device.MemorySize / (1024 * 1024)} MB");
                Console.WriteLine($"{device.Device.MaxNumThreads} Threads");
                Console.WriteLine();
            }
        }

        private static bool Compare(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; i++)
                if (a[i] != b[i])
                    return false;

            return true;
        }
    }
}
