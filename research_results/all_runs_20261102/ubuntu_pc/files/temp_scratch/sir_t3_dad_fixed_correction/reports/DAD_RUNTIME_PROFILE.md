# Native DAD profile

{
  "status": "PASS",
  "physical_predictions": true,
  "loglikelihood_and_terminal_sPCE": true,
  "time_parameter_initial_state_gradients": true,
  "policy_gradients": true,
  "high_rate_boundary_cases": true,
  "records": [
    {
      "implementation": "reference",
      "repeat": 0,
      "forward_seconds": 2.0217340290546417,
      "backward_seconds": 16.716398176038638,
      "optimizer_seconds": 0.030272745992988348,
      "checkpoint_seconds": 0.0019519201014190912,
      "CPU_process_seconds": 18.83732366,
      "peak_VRAM_bytes": 1216310784,
      "ODE_particle_stages": 12006000,
      "effective_RK4_steps": 6147536176,
      "likelihood_evaluations": 12006000,
      "total_seconds": 18.768404951086268,
      "training_sPCE": 4.826828374212273,
      "saturation_fraction": 0.109
    },
    {
      "implementation": "reference",
      "repeat": 1,
      "forward_seconds": 1.9046942270360887,
      "backward_seconds": 16.775338522158563,
      "optimizer_seconds": 0.0006988910026848316,
      "checkpoint_seconds": 0.0018723700195550919,
      "CPU_process_seconds": 18.695147119999998,
      "peak_VRAM_bytes": 1224944128,
      "ODE_particle_stages": 12006000,
      "effective_RK4_steps": 6147536176,
      "likelihood_evaluations": 12006000,
      "total_seconds": 18.680731640197337,
      "training_sPCE": 4.826828374212273,
      "saturation_fraction": 0.109
    },
    {
      "implementation": "count_sorted",
      "repeat": 0,
      "forward_seconds": 1.9088238631375134,
      "backward_seconds": 16.75266565196216,
      "optimizer_seconds": 0.000773122999817133,
      "checkpoint_seconds": 0.0016961959190666676,
      "CPU_process_seconds": 18.676175239000003,
      "peak_VRAM_bytes": 1770901504,
      "ODE_particle_stages": 12006000,
      "effective_RK4_steps": 6147536176,
      "likelihood_evaluations": 12006000,
      "total_seconds": 18.66226263809949,
      "training_sPCE": 4.826828374212273,
      "saturation_fraction": 0.109
    },
    {
      "implementation": "count_sorted",
      "repeat": 1,
      "forward_seconds": 1.9062776200007647,
      "backward_seconds": 16.75204328005202,
      "optimizer_seconds": 0.0007076452020555735,
      "checkpoint_seconds": 0.001782541861757636,
      "CPU_process_seconds": 18.674093658000004,
      "peak_VRAM_bytes": 1770900992,
      "ODE_particle_stages": 12006000,
      "effective_RK4_steps": 6147536176,
      "likelihood_evaluations": 12006000,
      "total_seconds": 18.65902854525484,
      "training_sPCE": 4.826828374212273,
      "saturation_fraction": 0.109
    }
  ],
  "profiler": [
    {
      "name": "cudaDeviceSynchronize",
      "calls": 4,
      "CPU_us": 1354808.406,
      "device_us": 0
    },
    {
      "name": "Activity Buffer Request",
      "calls": 1,
      "CPU_us": 1583.802,
      "device_us": 0
    },
    {
      "name": "aten::empty",
      "calls": 74,
      "CPU_us": 505.9730000047134,
      "device_us": 0
    },
    {
      "name": "cudaStreamIsCapturing",
      "calls": 33,
      "CPU_us": 15.603999999801545,
      "device_us": 295.2669999999998
    },
    {
      "name": "aten::randn",
      "calls": 5,
      "CPU_us": 423.40200000021514,
      "device_us": 715.5249999998614
    },
    {
      "name": "aten::normal_",
      "calls": 5,
      "CPU_us": 95.48199999993676,
      "device_us": 715.5249999998614
    },
    {
      "name": "cudaLaunchKernel",
      "calls": 365,
      "CPU_us": 15379019.199000016,
      "device_us": 1066.4389999997702
    },
    {
      "name": "void at::native::(anonymous namespace)::distribution_elementwise_grid_stride_kernel<double, 2, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1}, at::native::(anonymous namespace)::distribution_nullary_kernel<double, double, double2, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1}, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1} const&, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(int, double)#1}>(long, at::PhiloxCudaState, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1}, at::native::(anonymous namespace)::distribution_nullary_kernel<double, double, double2, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1}, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_and_transform<double, double, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::CUDAGeneratorImpl*, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(curandStatePhilox4_32_10*)#1} const&, at::native::templates::cuda::normal_kernel<at::CUDAGeneratorImpl*>(at::TensorBase const&, double, double, at::CUDAGeneratorImpl*)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1})::{lambda(int, double)#1})",
      "calls": 5,
      "CPU_us": 0,
      "device_us": 715.5249999998614
    },
    {
      "name": "aten::mul",
      "calls": 34,
      "CPU_us": 390.3680000006939,
      "device_us": 1036.1689999965215
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::AUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul>)",
      "calls": 14,
      "CPU_us": 0,
      "device_us": 403.5249999966959
    },
    {
      "name": "aten::to",
      "calls": 11,
      "CPU_us": 58.73100000146633,
      "device_us": 91.5530000001163
    },
    {
      "name": "aten::add",
      "calls": 25,
      "CPU_us": 261.0700000013817,
      "device_us": 1547.8529999920502
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<double> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<double> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<double> >(at::TensorIteratorBase&, at::native::CUDAFunctor_add<double> const&)::{lambda(int)#1})",
      "calls": 26,
      "CPU_us": 0,
      "device_us": 2513.717000001583
    },
    {
      "name": "aten::exp",
      "calls": 5,
      "CPU_us": 48.69899999914696,
      "device_us": 835.6210000018959
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::exp_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 8,
      "CPU_us": 0,
      "device_us": 1372.4260000018307
    },
    {
      "name": "cudaMalloc",
      "calls": 25,
      "CPU_us": 1545.4849999992116,
      "device_us": 0
    },
    {
      "name": "aten::unsqueeze",
      "calls": 24,
      "CPU_us": 57.0430000031447,
      "device_us": 0
    },
    {
      "name": "aten::as_strided",
      "calls": 197,
      "CPU_us": 105.93400000926295,
      "device_us": 0
    },
    {
      "name": "aten::cat",
      "calls": 15,
      "CPU_us": 460.05399999991823,
      "device_us": 3182.8429999998652
    },
    {
      "name": "Lazy Function Loading",
      "calls": 28,
      "CPU_us": 916.0879999987615,
      "device_us": 1276015.2939999998
    },
    {
      "name": "aten::zeros",
      "calls": 36,
      "CPU_us": 478.32399998997516,
      "device_us": 1217.8980000029042
    },
    {
      "name": "aten::zero_",
      "calls": 37,
      "CPU_us": 254.02999999635313,
      "device_us": 1219.3370000029045
    },
    {
      "name": "aten::fill_",
      "calls": 54,
      "CPU_us": 352.5790000065672,
      "device_us": 2019.2470000028643
    },
    {
      "name": "aten::lift_fresh",
      "calls": 2,
      "CPU_us": 0.4810000000002219,
      "device_us": 0
    },
    {
      "name": "aten::select",
      "calls": 41,
      "CPU_us": 138.9089999978287,
      "device_us": 0
    },
    {
      "name": "aten::item",
      "calls": 20,
      "CPU_us": 1899773.8220000004,
      "device_us": 10.591999999800464
    },
    {
      "name": "aten::_local_scalar_dense",
      "calls": 20,
      "CPU_us": 1899756.363,
      "device_us": 10.591999999800464
    },
    {
      "name": "aten::full",
      "calls": 4,
      "CPU_us": 27.890000000055807,
      "device_us": 3.327999999997701
    },
    {
      "name": "DAD_POLICY_FORWARD",
      "calls": 3,
      "CPU_us": 82509.32000000007,
      "device_us": 358.3379999993931
    },
    {
      "name": "aten::new_zeros",
      "calls": 1,
      "CPU_us": 12.886999999999716,
      "device_us": 1.4390000000003056
    },
    {
      "name": "aten::new_empty",
      "calls": 1,
      "CPU_us": 3.963999999999942,
      "device_us": 0
    },
    {
      "name": "aten::linear",
      "calls": 7,
      "CPU_us": 71100.20400000003,
      "device_us": 325.72999999974763
    },
    {
      "name": "aten::t",
      "calls": 34,
      "CPU_us": 111.32199999788736,
      "device_us": 0
    },
    {
      "name": "aten::transpose",
      "calls": 34,
      "CPU_us": 43.32699999909619,
      "device_us": 0
    },
    {
      "name": "aten::addmm",
      "calls": 5,
      "CPU_us": 70655.96600000025,
      "device_us": 225.15400000009595
    },
    {
      "name": "aten::expand",
      "calls": 12,
      "CPU_us": 41.93099999931928,
      "device_us": 0
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy_vectorized<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 2, 128, 1, 16, 2>(char*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "calls": 1,
      "CPU_us": 0,
      "device_us": 135.29799999999977
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<double>, std::array<char*, 1ul> >(int, at::native::FillFunctor<double>, std::array<char*, 1ul>)",
      "calls": 49,
      "CPU_us": 0,
      "device_us": 1228.7780000028706
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<double> >(at::TensorIteratorBase&, at::native::FillFunctor<double> const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::FillFunctor<double> >(at::TensorIteratorBase&, at::native::FillFunctor<double> const&)::{lambda(int)#1})",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 701.9249999999993
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy_alignedK_contig<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 2, 128, 1, 16>(at::native::(anonymous namespace)::OpaqueType<8u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 128, 1>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "calls": 11,
      "CPU_us": 0,
      "device_us": 11.45699999982844
    },
    {
      "name": "DAD_POLICY_FORWARD",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 80539.53099999986
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1})",
      "calls": 28,
      "CPU_us": 0,
      "device_us": 3471.7399999946783
    },
    {
      "name": "cudaFree",
      "calls": 2,
      "CPU_us": 719.4829999991061,
      "device_us": 0
    },
    {
      "name": "cudaDeviceGetAttribute",
      "calls": 33,
      "CPU_us": 1506.2879999969136,
      "device_us": 0
    },
    {
      "name": "cudaGetDriverEntryPoint",
      "calls": 2,
      "CPU_us": 0.34500000000116415,
      "device_us": 0
    },
    {
      "name": "cudaGetSymbolAddress",
      "calls": 1,
      "CPU_us": 4065.822,
      "device_us": 0
    },
    {
      "name": "Runtime Triggered Module Loading",
      "calls": 22,
      "CPU_us": 15382283.189,
      "device_us": 80.06399999998393
    },
    {
      "name": "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      "calls": 3,
      "CPU_us": 5248.908000000432,
      "device_us": 0
    },
    {
      "name": "void gemv2T_kernel_val<int, int, double, double, double, double, 128, 16, 2, 2, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double>, double, double)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 9.18399999992107
    },
    {
      "name": "aten::full_like",
      "calls": 4,
      "CPU_us": 166.89099999991595,
      "device_us": 89.3119999999908
    },
    {
      "name": "aten::empty_like",
      "calls": 25,
      "CPU_us": 471.722000005102,
      "device_us": 0
    },
    {
      "name": "aten::empty_strided",
      "calls": 20,
      "CPU_us": 366.60599999752594,
      "device_us": 0
    },
    {
      "name": "aten::ge",
      "calls": 3,
      "CPU_us": 54.00099999998929,
      "device_us": 3.936000000074273
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<4, at::native::compare_scalar_kernel<double>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, double)::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::compare_scalar_kernel<double>(at::TensorIteratorBase&, at::native::(anonymous namespace)::OpType, double)::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 9,
      "CPU_us": 0,
      "device_us": 449.1870000001363
    },
    {
      "name": "aten::any",
      "calls": 9,
      "CPU_us": 124.76199999960954,
      "device_us": 43.07199999947625
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::or_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(bool, bool)#1}>, unsigned int, bool, 4, 4> >(at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::or_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(bool, bool)#1}>, unsigned int, bool, 4, 4>)",
      "calls": 9,
      "CPU_us": 0,
      "device_us": 40.95999999968626
    },
    {
      "name": "aten::is_nonzero",
      "calls": 12,
      "CPU_us": 1898534.27,
      "device_us": 7.551999999821419
    },
    {
      "name": "cudaMemcpyAsync",
      "calls": 32,
      "CPU_us": 193.88399999881221,
      "device_us": 0
    },
    {
      "name": "cudaStreamSynchronize",
      "calls": 18,
      "CPU_us": 1899595.3450000002,
      "device_us": 0
    },
    {
      "name": "Memcpy DtoH (Device -> Pinned)",
      "calls": 18,
      "CPU_us": 0,
      "device_us": 10.591999999800464
    },
    {
      "name": "aten::sigmoid",
      "calls": 3,
      "CPU_us": 3995.8730000001524,
      "device_us": 34.75199999992037
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::sigmoid_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 17.37599999991653
    },
    {
      "name": "aten::rsub",
      "calls": 3,
      "CPU_us": 94.88899999987916,
      "device_us": 4.800000000017462
    },
    {
      "name": "aten::sub",
      "calls": 24,
      "CPU_us": 466.1969999979774,
      "device_us": 1287.3059999992984
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnOther_add<double>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnOther_add<double>, std::array<char*, 2ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 3.6160000000236323
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 3ul>)",
      "calls": 14,
      "CPU_us": 0,
      "device_us": 500.4519999989716
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<double>, std::array<char*, 3ul>)",
      "calls": 37,
      "CPU_us": 0,
      "device_us": 1645.5189999981085
    },
    {
      "name": "SIR_ODE_PROPAGATION",
      "calls": 3,
      "CPU_us": 1904393.6400000001,
      "device_us": 3178917.306
    },
    {
      "name": "SIR_ODE_PROPAGATION",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 1906135.348
    },
    {
      "name": "aten::detach",
      "calls": 12,
      "CPU_us": 25.380000000441214,
      "device_us": 0
    },
    {
      "name": "detach",
      "calls": 12,
      "CPU_us": 15.639999997860286,
      "device_us": 0
    },
    {
      "name": "aten::sum",
      "calls": 27,
      "CPU_us": 522.1110000019835,
      "device_us": 875.4619999956485
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::FillFunctor<long>, std::array<char*, 1ul> >(int, at::native::FillFunctor<long>, std::array<char*, 1ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 88.54399999999441
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::sum_functor<double, double, double>::operator()(at::TensorIterator&)::{lambda(double, double)#1}>, unsigned int, double, 4, 4> >(at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::sum_functor<double, double, double>::operator()(at::TensorIterator&)::{lambda(double, double)#1}>, unsigned int, double, 4, 4>)",
      "calls": 15,
      "CPU_us": 0,
      "device_us": 570.8829999939189
    },
    {
      "name": "aten::div",
      "calls": 28,
      "CPU_us": 264.61000000475906,
      "device_us": 299.0430000060878
    },
    {
      "name": "aten::ceil",
      "calls": 3,
      "CPU_us": 31.887000000089756,
      "device_us": 91.10399999997753
    },
    {
      "name": "aten::_to_copy",
      "calls": 3,
      "CPU_us": 49.063000000154716,
      "device_us": 91.5530000001163
    },
    {
      "name": "aten::copy_",
      "calls": 42,
      "CPU_us": 338.92600000160746,
      "device_us": 3947.484999996028
    },
    {
      "name": "aten::maximum",
      "calls": 3,
      "CPU_us": 33.34200000004785,
      "device_us": 255.87499999988358
    },
    {
      "name": "cudaMemsetAsync",
      "calls": 19,
      "CPU_us": 57.224999999205465,
      "device_us": 0
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul> >(int, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> >, std::array<char*, 2ul>)",
      "calls": 21,
      "CPU_us": 0,
      "device_us": 291.4590000025637
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::ceil_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::ceil_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 91.10399999997753
    },
    {
      "name": "void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1> >(int, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithCast<1>, at::native::memory::StoreWithCast<1>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 91.5530000001163
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<long, long, long, at::native::maximum_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, long)#1}>, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<long, long, long, at::native::maximum_kernel_cuda(at::TensorIteratorBase&)::{lambda()#1}::operator()() const::{lambda()#4}::operator()() const::{lambda(long, long)#1}>, std::array<char*, 3ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 255.87499999988358
    },
    {
      "name": "Memset (Device)",
      "calls": 17,
      "CPU_us": 0,
      "device_us": 5.985000001906883
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::sum_functor<long, long, long>::operator()(at::TensorIterator&)::{lambda(long, long)#1}>, unsigned int, long, 4, 4> >(at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::sum_functor<long, long, long>::operator()(at::TensorIterator&)::{lambda(long, long)#1}>, unsigned int, long, 4, 4>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 30.84800000001269
    },
    {
      "name": "aten::max",
      "calls": 3,
      "CPU_us": 57.0,
      "device_us": 31.617000000012922
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::MaxNanFunctor<long> >, unsigned int, long, 4, 4> >(at::native::ReduceOp<long, at::native::func_wrapper_t<long, at::native::MaxNanFunctor<long> >, unsigned int, long, 4, 4>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 30.849000000031083
    },
    {
      "name": "aten::lt",
      "calls": 6,
      "CPU_us": 69.61400000024878,
      "device_us": 445.25100000006205
    },
    {
      "name": "aten::slice",
      "calls": 44,
      "CPU_us": 97.93699999232194,
      "device_us": 0
    },
    {
      "name": "_DiscreteRK4",
      "calls": 3,
      "CPU_us": 2885.3309999999765,
      "device_us": 3172335.5590000004
    },
    {
      "name": "aten::reshape",
      "calls": 34,
      "CPU_us": 169.83500000086497,
      "device_us": 2.4319999993313104
    },
    {
      "name": "aten::view",
      "calls": 40,
      "CPU_us": 91.72500000490982,
      "device_us": 0
    },
    {
      "name": "aten::contiguous",
      "calls": 3,
      "CPU_us": 106.86300000011397,
      "device_us": 739.5900000001566
    },
    {
      "name": "aten::clone",
      "calls": 8,
      "CPU_us": 176.3070000042062,
      "device_us": 745.7660000022297
    },
    {
      "name": "cuLaunchKernelEx",
      "calls": 6,
      "CPU_us": 62.20599999955448,
      "device_us": 0
    },
    {
      "name": "_forward",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 1896643.53
    },
    {
      "name": "aten::isfinite",
      "calls": 3,
      "CPU_us": 387.1379999999335,
      "device_us": 1344.3300000000745
    },
    {
      "name": "aten::abs",
      "calls": 12,
      "CPU_us": 591.3909999997995,
      "device_us": 1244.871999999974
    },
    {
      "name": "aten::resize_",
      "calls": 9,
      "CPU_us": 209.94000000035157,
      "device_us": 0
    },
    {
      "name": "aten::ne",
      "calls": 3,
      "CPU_us": 44.72599999977683,
      "device_us": 321.6020000000717
    },
    {
      "name": "aten::eq",
      "calls": 6,
      "CPU_us": 58.54199999976845,
      "device_us": 360.0350000000326
    },
    {
      "name": "aten::all",
      "calls": 3,
      "CPU_us": 55.4059999998135,
      "device_us": 22.944000000017695
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::AbsFunctor<double>, std::array<char*, 2ul> >(int, at::native::AbsFunctor<double>, std::array<char*, 2ul>)",
      "calls": 6,
      "CPU_us": 0,
      "device_us": 622.435999999987
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<double, double, bool, at::native::(anonymous namespace)::CompareEqFunctor<double> >, std::array<char*, 2ul> >(int, at::native::AUnaryFunctor<double, double, bool, at::native::(anonymous namespace)::CompareEqFunctor<double> >, std::array<char*, 2ul>)",
      "calls": 6,
      "CPU_us": 0,
      "device_us": 325.7300000000978
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<double, double, bool, at::native::(anonymous namespace)::CompareEqFunctor<double> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<double, double, bool, at::native::(anonymous namespace)::CompareEqFunctor<double> >, std::array<char*, 3ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 355.9070000000065
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<bool, bool, bool, at::native::binary_internal::MulFunctor<bool> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<bool, bool, bool, at::native::binary_internal::MulFunctor<bool> >, std::array<char*, 3ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 48.35200000007171
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::and_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(bool, bool)#1}>, unsigned int, bool, 4, 4> >(at::native::ReduceOp<bool, at::native::func_wrapper_t<bool, at::native::and_kernel_cuda(at::TensorIterator&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(bool, bool)#1}>, unsigned int, bool, 4, 4>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 21.152000000118278
    },
    {
      "name": "void at::native::(anonymous namespace)::CatArrayBatchedCopy<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 3, 64, 64>(at::native::(anonymous namespace)::OpaqueType<8u>*, at::native::(anonymous namespace)::CatArrInputTensorMetadata<at::native::(anonymous namespace)::OpaqueType<8u>, unsigned int, 64, 64>, at::native::(anonymous namespace)::TensorSizeStride<unsigned int, 4u>, int, unsigned int)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 2174.7050000000745
    },
    {
      "name": "GAUSSIAN_LOG_LIKELIHOOD",
      "calls": 3,
      "CPU_us": 17378.3890000002,
      "device_us": 1141.8629999997793
    },
    {
      "name": "aten::square",
      "calls": 3,
      "CPU_us": 16965.86400000006,
      "device_us": 130.0810000000056
    },
    {
      "name": "aten::pow",
      "calls": 6,
      "CPU_us": 17026.888000000617,
      "device_us": 373.665999999037
    },
    {
      "name": "aten::result_type",
      "calls": 6,
      "CPU_us": 4.5150000017602,
      "device_us": 0
    },
    {
      "name": "GAUSSIAN_LOG_LIKELIHOOD",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 17274.980999999796
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<double, double>(at::TensorIteratorBase&, double)::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::pow_tensor_scalar_kernel_impl<double, double>(at::TensorIteratorBase&, double)::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 71.71299999998882
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<double>, std::array<char*, 2ul>)",
      "calls": 8,
      "CPU_us": 0,
      "device_us": 102.68800000019837
    },
    {
      "name": "void at::native::reduce_kernel<128, 4, at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::sum_functor<double, double, double>::operator()(at::TensorIterator&)::{lambda(double, double)#1}>, unsigned int, double, 4, 4> >(at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::sum_functor<double, double, double>::operator()(at::TensorIterator&)::{lambda(double, double)#1}>, unsigned int, double, 4, 4>)",
      "calls": 9,
      "CPU_us": 0,
      "device_us": 221.79399999952875
    },
    {
      "name": "aten::logsumexp",
      "calls": 6,
      "CPU_us": 13836.237000000197,
      "device_us": 1655.3719999995083
    },
    {
      "name": "aten::real",
      "calls": 3,
      "CPU_us": 1.5919999999459833,
      "device_us": 0
    },
    {
      "name": "aten::amax",
      "calls": 3,
      "CPU_us": 78.11499999975786,
      "device_us": 128.2249999998603
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::MaxNanFunctor<double> >, unsigned int, double, 4, 4> >(at::native::ReduceOp<double, at::native::func_wrapper_t<double, at::native::MaxNanFunctor<double> >, unsigned int, double, 4, 4>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 96.35299999988638
    },
    {
      "name": "aten::squeeze",
      "calls": 12,
      "CPU_us": 37.103999995742925,
      "device_us": 0
    },
    {
      "name": "aten::masked_fill_",
      "calls": 3,
      "CPU_us": 198.22599999979138,
      "device_us": 5.408999999752268
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, bool)#1}, std::array<char*, 3ul> >(int, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double, bool)#1}, std::array<char*, 3ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 4.128999999840744
    },
    {
      "name": "aten::exp_",
      "calls": 3,
      "CPU_us": 30.58800000010524,
      "device_us": 536.8049999999348
    },
    {
      "name": "aten::log_",
      "calls": 3,
      "CPU_us": 4675.912999999942,
      "device_us": 43.48800000036135
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::log_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 21.888000000151806
    },
    {
      "name": "aten::add_",
      "calls": 25,
      "CPU_us": 142.97900000424124,
      "device_us": 1431.5650000085589
    },
    {
      "name": "aten::ones",
      "calls": 6,
      "CPU_us": 61.49400000006426,
      "device_us": 4.609000000171363
    },
    {
      "name": "aten::_reshape_alias",
      "calls": 3,
      "CPU_us": 11.71300000010524,
      "device_us": 0
    },
    {
      "name": "aten::matmul",
      "calls": 2,
      "CPU_us": 322.55899999989197,
      "device_us": 91.1679999998305
    },
    {
      "name": "aten::mm",
      "calls": 15,
      "CPU_us": 7669886.6000000015,
      "device_us": 2418.163999996148
    },
    {
      "name": "cuLaunchKernel",
      "calls": 12,
      "CPU_us": 45.047999999136664,
      "device_us": 0
    },
    {
      "name": "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_tn_align1>(cutlass_80_tensorop_d884gemm_64x32_16x4_tn_align1::Params)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 67.07199999992736
    },
    {
      "name": "aten::_unsafe_view",
      "calls": 4,
      "CPU_us": 14.485000000335276,
      "device_us": 0
    },
    {
      "name": "aten::relu",
      "calls": 2,
      "CPU_us": 10785.25400000019,
      "device_us": 23.520999999716878
    },
    {
      "name": "aten::clamp_min",
      "calls": 2,
      "CPU_us": 10762.45000000007,
      "device_us": 23.520999999716878
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::(anonymous namespace)::launch_clamp_scalar(at::TensorIteratorBase&, c10::Scalar, c10::Scalar, at::native::detail::ClampLimits)::{lambda()#1}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 10.944999999948777
    },
    {
      "name": "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_tn_align1>(cutlass_80_tensorop_d884gemm_32x32_16x5_tn_align1::Params)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 159.26500000001397
    },
    {
      "name": "aten::stack",
      "calls": 1,
      "CPU_us": 26.625,
      "device_us": 1.247999999905005
    },
    {
      "name": "aten::mean",
      "calls": 1,
      "CPU_us": 6048.701000000117,
      "device_us": 13.567999999970198
    },
    {
      "name": "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<double, at::native::MeanOps<double, double, double, double>, unsigned int, double, 4, 4> >(at::native::ReduceOp<double, at::native::MeanOps<double, double, double, double>, unsigned int, double, 4, 4>)",
      "calls": 1,
      "CPU_us": 0,
      "device_us": 3.3919999999925494
    },
    {
      "name": "aten::neg",
      "calls": 12,
      "CPU_us": 135.32500000065193,
      "device_us": 251.04099999321625
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul> >(int, at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}, std::array<char*, 2ul>)",
      "calls": 9,
      "CPU_us": 0,
      "device_us": 246.88199999649078
    },
    {
      "name": "aten::ones_like",
      "calls": 1,
      "CPU_us": 20.68900000001304,
      "device_us": 0.735999999800697
    },
    {
      "name": "autograd::engine::evaluate_function: NegBackward0",
      "calls": 1,
      "CPU_us": 33.74499999987893,
      "device_us": 0.9609999998938292
    },
    {
      "name": "NegBackward0",
      "calls": 1,
      "CPU_us": 27.584000000031665,
      "device_us": 0.9609999998938292
    },
    {
      "name": "autograd::engine::evaluate_function: MeanBackward0",
      "calls": 1,
      "CPU_us": 66.01199999987148,
      "device_us": 1.1200000001117587
    },
    {
      "name": "MeanBackward0",
      "calls": 1,
      "CPU_us": 62.99499999987893,
      "device_us": 1.1200000001117587
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BUnaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1})",
      "calls": 7,
      "CPU_us": 0,
      "device_us": 7.584000003524125
    },
    {
      "name": "autograd::engine::evaluate_function: SelectBackward0",
      "calls": 17,
      "CPU_us": 577.39100000076,
      "device_us": 3027.7360000086483
    },
    {
      "name": "SelectBackward0",
      "calls": 17,
      "CPU_us": 450.88099999935366,
      "device_us": 1377.6410000030883
    },
    {
      "name": "aten::select_backward",
      "calls": 17,
      "CPU_us": 425.7929999975022,
      "device_us": 1377.6410000030883
    },
    {
      "name": "autograd::engine::evaluate_function: StackBackward0",
      "calls": 1,
      "CPU_us": 6.037000000011176,
      "device_us": 0
    },
    {
      "name": "StackBackward0",
      "calls": 1,
      "CPU_us": 3.9850000001024455,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: SubBackward0",
      "calls": 15,
      "CPU_us": 176.60199999972247,
      "device_us": 299.5199999916367
    },
    {
      "name": "SubBackward0",
      "calls": 15,
      "CPU_us": 100.44999999692664,
      "device_us": 246.4959999928251
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1}>(at::TensorIteratorBase&, at::native::neg_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#6}::operator()() const::{lambda(double)#1} const&)::{lambda(int)#1})",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 4.15899999672547
    },
    {
      "name": "autograd::engine::evaluate_function: LogsumexpBackward0",
      "calls": 3,
      "CPU_us": 162.49900000006892,
      "device_us": 880.1020000043791
    },
    {
      "name": "LogsumexpBackward0",
      "calls": 3,
      "CPU_us": 132.54599999869242,
      "device_us": 736.8680000014137
    },
    {
      "name": "autograd::engine::evaluate_function: AddBackward0",
      "calls": 13,
      "CPU_us": 75.86700000124983,
      "device_us": 22.4660000000149
    },
    {
      "name": "AddBackward0",
      "calls": 13,
      "CPU_us": 7.343999998643994,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: SumBackward1",
      "calls": 5,
      "CPU_us": 36.45500000054017,
      "device_us": 0
    },
    {
      "name": "SumBackward1",
      "calls": 5,
      "CPU_us": 29.227999999420717,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: MulBackward0",
      "calls": 6,
      "CPU_us": 91.37100000074133,
      "device_us": 95.61699999519624
    },
    {
      "name": "MulBackward0",
      "calls": 6,
      "CPU_us": 73.19999999925494,
      "device_us": 95.61699999519624
    },
    {
      "name": "autograd::engine::evaluate_function: PowBackward0",
      "calls": 3,
      "CPU_us": 135.1170000017155,
      "device_us": 600.8680000000168
    },
    {
      "name": "PowBackward0",
      "calls": 3,
      "CPU_us": 125.8480000004638,
      "device_us": 600.8680000000168
    },
    {
      "name": "autograd::engine::evaluate_function: DivBackward0",
      "calls": 9,
      "CPU_us": 128.06599999940954,
      "device_us": 90.9780000038445
    },
    {
      "name": "DivBackward0",
      "calls": 9,
      "CPU_us": 79.24799999385141,
      "device_us": 87.13800000585616
    },
    {
      "name": "autograd::engine::evaluate_function: UnsqueezeBackward0",
      "calls": 9,
      "CPU_us": 118.02999999839813,
      "device_us": 11.776000007987022
    },
    {
      "name": "UnsqueezeBackward0",
      "calls": 9,
      "CPU_us": 35.432000001193956,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: SliceBackward0",
      "calls": 11,
      "CPU_us": 329.8489999980666,
      "device_us": 3274.872999995714
    },
    {
      "name": "SliceBackward0",
      "calls": 11,
      "CPU_us": 284.7239999973681,
      "device_us": 2509.6509999947157
    },
    {
      "name": "aten::slice_backward",
      "calls": 11,
      "CPU_us": 268.18800000310875,
      "device_us": 2509.6509999947157
    },
    {
      "name": "autograd::engine::evaluate_function: _DiscreteRK4Backward",
      "calls": 3,
      "CPU_us": 19839.654000000795,
      "device_us": 16679371.984000003
    },
    {
      "name": "_DiscreteRK4Backward",
      "calls": 3,
      "CPU_us": 19779.3079999981,
      "device_us": 16679371.984000003
    },
    {
      "name": "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<double, double, double, at::native::binary_internal::MulFunctor<double> > const&)::{lambda(int)#1})",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 83.84000000078231
    },
    {
      "name": "Memcpy DtoD (Device -> Device)",
      "calls": 14,
      "CPU_us": 0,
      "device_us": 387.3610000014305
    },
    {
      "name": "_backward",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 16679371.984000003
    },
    {
      "name": "autograd::engine::evaluate_function: ExpandBackward0",
      "calls": 3,
      "CPU_us": 56.6269999993965,
      "device_us": 53.82399999909103
    },
    {
      "name": "ExpandBackward0",
      "calls": 3,
      "CPU_us": 49.92800000263378,
      "device_us": 53.82399999909103
    },
    {
      "name": "autograd::engine::evaluate_function: RsubBackward1",
      "calls": 2,
      "CPU_us": 28.040999999502674,
      "device_us": 5.216000001877546
    },
    {
      "name": "RsubBackward1",
      "calls": 2,
      "CPU_us": 10.759999999776483,
      "device_us": 2.400000000372529
    },
    {
      "name": "autograd::engine::evaluate_function: SigmoidBackward0",
      "calls": 3,
      "CPU_us": 7663996.336,
      "device_us": 5.631000004708767
    },
    {
      "name": "SigmoidBackward0",
      "calls": 3,
      "CPU_us": 7663967.137999999,
      "device_us": 5.631000004708767
    },
    {
      "name": "aten::sigmoid_backward",
      "calls": 3,
      "CPU_us": 7663822.902000003,
      "device_us": 5.631000004708767
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::sigmoid_backward_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double, double)#1}, std::array<char*, 3ul> >(int, at::native::sigmoid_backward_kernel_cuda(at::TensorIteratorBase&)::{lambda()#2}::operator()() const::{lambda()#1}::operator()() const::{lambda(double, double)#1}, std::array<char*, 3ul>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 5.631000004708767
    },
    {
      "name": "autograd::engine::evaluate_function: AddmmBackward0",
      "calls": 5,
      "CPU_us": 3271.3169999998063,
      "device_us": 1362.0279999878258
    },
    {
      "name": "AddmmBackward0",
      "calls": 5,
      "CPU_us": 3155.1449999976903,
      "device_us": 1317.2599999960512
    },
    {
      "name": "void gemmk1_kernel<int, double, 256, 5, false, false, false, false, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double, 0>(cublasGemmk1Params<double, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double, biasType<cublasGemvTensorStridedBatched<double>::value_type, double>::type>)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 3.0720000006258488
    },
    {
      "name": "void gemv2N_kernel<int, int, double, double, double, double, 128, 32, 4, 4, 1, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double const>, cublasGemvTensorStridedBatched<double>, double>)",
      "calls": 3,
      "CPU_us": 0,
      "device_us": 38.880999997258186
    },
    {
      "name": "autograd::engine::evaluate_function: TBackward0",
      "calls": 7,
      "CPU_us": 77.34399999678135,
      "device_us": 5.343999993056059
    },
    {
      "name": "TBackward0",
      "calls": 7,
      "CPU_us": 11.14100000075996,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: ViewBackward0",
      "calls": 4,
      "CPU_us": 37.90699999779463,
      "device_us": 1.2799999993294477
    },
    {
      "name": "ViewBackward0",
      "calls": 4,
      "CPU_us": 31.512000000104308,
      "device_us": 1.2799999993294477
    },
    {
      "name": "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1>(cutlass_80_tensorop_d884gemm_64x32_16x4_nn_align1::Params)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 67.39299999922514
    },
    {
      "name": "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nt_align1>(cutlass_80_tensorop_d884gemm_32x32_16x5_nt_align1::Params)",
      "calls": 4,
      "CPU_us": 0,
      "device_us": 2058.927999999374
    },
    {
      "name": "autograd::engine::evaluate_function: ReluBackward0",
      "calls": 2,
      "CPU_us": 5122.277999999002,
      "device_us": 9.951999997720122
    },
    {
      "name": "ReluBackward0",
      "calls": 2,
      "CPU_us": 5113.593000002205,
      "device_us": 9.951999997720122
    },
    {
      "name": "aten::threshold_backward",
      "calls": 2,
      "CPU_us": 5098.680999999866,
      "device_us": 9.951999997720122
    },
    {
      "name": "void at::native::vectorized_elementwise_kernel<2, at::native::BinaryFunctor<double, double, double, at::native::(anonymous namespace)::threshold_kernel_impl<double>(at::TensorIteratorBase&, double, double)::{lambda(double, double)#1}>, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<double, double, double, at::native::(anonymous namespace)::threshold_kernel_impl<double>(at::TensorIteratorBase&, double, double)::{lambda(double, double)#1}>, std::array<char*, 3ul>)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 9.951999997720122
    },
    {
      "name": "autograd::engine::evaluate_function: UnsafeViewBackward0",
      "calls": 3,
      "CPU_us": 8.975999999791384,
      "device_us": 0
    },
    {
      "name": "UnsafeViewBackward0",
      "calls": 3,
      "CPU_us": 3.8440000005066395,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: MmBackward0",
      "calls": 2,
      "CPU_us": 7666611.357000001,
      "device_us": 1010.8880000002682
    },
    {
      "name": "MmBackward0",
      "calls": 2,
      "CPU_us": 7666583.446999999,
      "device_us": 1010.8880000002682
    },
    {
      "name": "autograd::engine::evaluate_function: CloneBackward0",
      "calls": 1,
      "CPU_us": 1.5930000003427267,
      "device_us": 0
    },
    {
      "name": "CloneBackward0",
      "calls": 1,
      "CPU_us": 0.2570000011473894,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: ReshapeAliasBackward0",
      "calls": 3,
      "CPU_us": 19.68100000359118,
      "device_us": 0
    },
    {
      "name": "ReshapeAliasBackward0",
      "calls": 3,
      "CPU_us": 12.390000000596046,
      "device_us": 0
    },
    {
      "name": "autograd::engine::evaluate_function: CatBackward0",
      "calls": 7,
      "CPU_us": 59.18400000408292,
      "device_us": 0
    },
    {
      "name": "CatBackward0",
      "calls": 7,
      "CPU_us": 44.002000000327826,
      "device_us": 0
    },
    {
      "name": "aten::narrow",
      "calls": 21,
      "CPU_us": 30.664999997243285,
      "device_us": 0
    },
    {
      "name": "void cutlass::Kernel2<cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1>(cutlass_80_tensorop_d884gemm_32x32_16x5_nn_align1::Params)",
      "calls": 2,
      "CPU_us": 0,
      "device_us": 157.82599999941885
    },
    {
      "name": "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad",
      "calls": 6,
      "CPU_us": 34.36799999512732,
      "device_us": 0
    },
    {
      "name": "torch::autograd::AccumulateGrad",
      "calls": 6,
      "CPU_us": 22.210999995470047,
      "device_us": 0
    },
    {
      "name": "void cublasLt::splitKreduce_kernel<32, 16, int, double, double, double, double, false, double, double, double, true, false, false>(cublasLt::cublasSplitKParams<double>, double const*, double const*, double*, double*, double const*, double const*, double const*, double const*, double*, void*, long, double*, int*, double*, double*, double const*, double const*, double const*, double const*, double const*)",
      "calls": 1,
      "CPU_us": 0,
      "device_us": 2.0480000004172325
    }
  ],
  "speedup": 1.001163141740732,
  "sort_only_no_physics_change": true
}