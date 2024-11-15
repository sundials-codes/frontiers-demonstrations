set(sources ark_sod_lsrk.cpp)

# set the target name
set(target ark_sod_lsrk)

# create executable
add_executable(${target} ${sources})

target_include_directories(${target} PRIVATE)

target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial SUNDIALS::nvecmanyvector MPI::MPI_CXX)
