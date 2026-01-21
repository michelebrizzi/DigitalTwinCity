config = coder.gpuConfig();
config.GpuConfig.EnableMemoryManager = true;
config.GpuConfig.ComputeCapability =  gpuDevice().ComputeCapability;
config.GpuConfig.StackLimitPerThread = intmax;

codegen -config config -args {imgList, pcList, poses} scan2map.m
