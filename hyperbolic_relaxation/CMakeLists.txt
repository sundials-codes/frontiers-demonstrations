###
# Hyperbolic Relaxation problem

#   source files
set(source_main hyperbolic_relaxation.cpp)

set(source_ref hyperbolic_relaxation_refSol.cpp)

#   extra installation files
set(extras 
    plot_hyperbolic_relaxation.py
    runtests_hyperbolic_relaxation.py
    runtests_hyperbolic_relaxation_refSol.py)

#   set the target name
set(target_main hyperbolic_relaxation)

set(target_ref hyperbolic_relaxation_refSol)

#   set the executable, listing the source files on which it depends
add_executable(${target_main} ${source_main})
add_executable(${target_ref} ${source_ref})

#   set the include directories
target_include_directories(${target_main} PRIVATE)
target_include_directories(${target_ref} PRIVATE)

#   set the include libraries to link against
target_link_libraries(${target_main} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial SUNDIALS::nvecmanyvector MPI::MPI_CXX)

  target_link_libraries(${target_ref} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial SUNDIALS::nvecmanyvector MPI::MPI_CXX)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_main} DESTINATION ${demonstration_bin})
install(TARGETS ${target_ref} DESTINATION ${demonstration_bin})

#   request installation of the extra files in the overall "bin" folder
install(FILES ${extras} DESTINATION ${demonstration_bin})
