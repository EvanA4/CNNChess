cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cnnChess)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(cnnChess uci.hpp uci.cpp driver.cpp thc-chess-library/thc.h thc-chess-library/thc.cpp)
target_link_libraries(cnnChess "${TORCH_LIBRARIES}")
set_property(TARGET cnnChess PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET cnnChess
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:cnnChess>)
endif (MSVC)

#[[
cmake -DCMAKE_PREFIX_PATH=/your/path/here/CNNChess/libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121 -DTorch_DIR=/your/path/here/CNNChess/libtorch-cxx11-abi-shared-with-deps-2.2.1+cu121/libtorch/share/cmake/Torch ..
cmake --build . --config Release
]]