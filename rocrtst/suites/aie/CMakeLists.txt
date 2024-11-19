find_package(hsa-runtime64 CONFIG REQUIRED NAMES hsa_runtime64 hsa-runtime64)

# hsa test
add_executable(aie_hsa_dispatch_test aie_hsa_dispatch_test.cc)
target_link_libraries(aie_hsa_dispatch_test PUBLIC hsa-runtime64::hsa-runtime64)
