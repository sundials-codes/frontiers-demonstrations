###
# Linear Advection problem

#   source files
set(sources linear_advection.cpp)

#   extra installation files
set(extras plot_advection.py)

#   set the target name
set(target linear_advection)

#   set the executable, listing the source files on which it depends
add_executable(${target} ${sources})

#   set the include directories
target_include_directories(${target} PRIVATE)

#   set the include libraries to link against
target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial SUNDIALS::nvecmanyvector MPI::MPI_CXX)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target} DESTINATION ${demonstration_bin})

#   request installation of the extra files in the overall "bin" folder
install(FILES ${extras} DESTINATION ${demonstration_bin})
