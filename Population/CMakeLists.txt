###
# Population density test problem

#   source files
set(sources population_density.cpp)

#   set the target name
set(target population_density)

#   source files
set(sources population_density_imex.cpp)

#   set the target name
set(target population_density_imex)

#   set the executable, listing the source files on which it depends
add_executable(${target} ${sources})

#   set the include directories
target_include_directories(${target} PRIVATE)

#   set the include libraries to link against
target_link_libraries(${target} PRIVATE
  SUNDIALS::arkode SUNDIALS::nvecserial)

#   request installation of the target into the overall "bin" folder
install(TARGETS ${target} DESTINATION ${demonstration_bin})

#   request installation of the extra files in the overall "bin" folder
install(FILES ${extras} DESTINATION ${demonstration_bin})
