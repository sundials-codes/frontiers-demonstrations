###
# Population density test problem

#   source files
set(source_main
    population.cpp)

set(source_ref
    population_refSoln.cpp)

#   extra installation files
set(extras
    runtests_population.py
    plot_population.py
    runtests_population_refSol.py
    plot_population_finalStepSoln.py)

#   set the target name
set(target_main 
    population)

set(target_ref 
    population_refSoln)

#   set the executable, listing the source files on which it depends
add_executable(${target_main} ${source_main})
add_executable(${target_ref} ${source_ref})

#   set the include directories
target_include_directories(${target_main} PRIVATE)
target_include_directories(${target_ref} PRIVATE)

#   set the include libraries to link against
target_link_libraries(${target_main} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

target_link_libraries(${target_ref} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target_main} DESTINATION ${demonstration_bin})
install(TARGETS ${target_ref} DESTINATION ${demonstration_bin})

#   request installation of the extra files in the overall "bin" folder
install(FILES ${extras} DESTINATION ${demonstration_bin})
