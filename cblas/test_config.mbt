///|
test "Output Config" {
  //let num_threads = openblas_get_num_threads()
  //let num_procs = openblas_get_num_procs()
  //let config = openblas_get_config()
  //let corename = openblas_get_corename()
  //let parallel = openblas_get_parallel()
  //
  //println("Number of Threads: \{num_threads}")
  //println("Number of Processors: \{num_procs}")
  //println("OpenBLAS Configuration:\{config}")
  //println("Core Name: \{corename}")
  //println("Parallel Support: \{parallel}")

  ignore(openblas_get_num_threads)
  ignore(openblas_get_num_procs)
  ignore(openblas_get_config)
  ignore(openblas_get_corename)
  ignore(openblas_get_parallel)

  //ignore(openblas_set_num_threads_local)
  ignore(openblas_set_num_threads)
  //ignore(openblas_set_threads_callback_function)
  ignore(goto_set_num_threads)
}
