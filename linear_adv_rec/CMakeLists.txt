###
# Linear Advection Reaction problem

#   source files
set(sources linear_adv_rec.cpp)

#   extra installation files
set(extras runtests_linear_adv_rec.py)

#   set the target name
set(target linear_advection_reaction)

#   source files
set(sources linear_adv_rec.cpp)

#   set the target name
set(target linear_adv_rec)

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
