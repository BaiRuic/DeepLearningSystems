cmake_minimum_required(VERSION 3.2)
project(needle C CXX)

# 注：Python使用了系统默认的
# Pybind11使用了 dlsys环境下面的

# find correct version of Python
# 找系统默认的 Python，在linux中，默认python位于 usr/bin
execute_process(COMMAND python3-config --prefix
  OUTPUT_VARIABLE Python_ROOT_DIR)

# 参考https://cmake.org/cmake/help/latest/module/FindPython.html
# 不想使用系统目录下的 Python，因此通过指定 Python3_ROOT_DIR 改变查找路径
# find_package(Python ..) 会定义 Python_INCLUDE_DIRS 等变量
# 注意：如果 Interpreter 和 Development 组件都被指定的话，
# 这个模块只会搜索 Cmake 配置的平台架构的解释器。
# 如果仅指定了 Interpreter 组件的话，这个约束不会生效。
# 因此，此处依旧是搜索默认解释器。
set(Python3_ROOT_DIR /home/brc/miniconda3/envs/dlsys)
find_package(Python COMPONENTS Development Interpreter REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# find pybind
execute_process(COMMAND python3 -m pybind11 --cmakedir
  RESULT_VARIABLE __pybind_exit_code
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE) # 可以删除运行命令的标准输出中的任何尾随空 格
# set pybind11 dir 
# 这里手动设置 pybind11_DIR。具体值为通过 python3 -m pybind11 --cmakedir 在对应环境下 运行得到的
set(pybind11_DIR /home/brc/miniconda3/envs/dlsys/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11)


if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
else()
  set(CMAKE_CXX_FLAGS "/std:c++11 -O2 -march=native ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 14)
endif()

# 是通过 find_package(pybind11) 创建的变量 
include_directories(SYSTEM ${pybind11_INCLUDE_DIRS})
# 引入一个变量 LINKER_LIBS ，包括 ${pybind11_LIBRARIES}
list(APPEND LINKER_LIBS ${pybind11_LIBRARIES})


###################
### CPU BACKEND ###
###################
# 将指定的源码编译到库中
# MODULE 为创建库的一种：不链接到项目中的任何目标，不过可以进行动态加载
add_library(ndarray_backend_cpu MODULE src/ndarray_backend_cpu.cc)
# 将库链接到可执行文件
target_link_libraries(ndarray_backend_cpu PUBLIC ${LINKER_LIBS})

pybind11_extension(ndarray_backend_cpu)
pybind11_strip(ndarray_backend_cpu)


# directly output to ffi folder
# 设置 链接库的输出位置
set_target_properties(ndarray_backend_cpu
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
  CXX_VISIBILITY_PRESET "hidden"
)

# 针对达尔文(Macos)操作系统专门设置属性
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set_property(TARGET ndarray_backend_cpu PROPERTY LINK_OPTIONS -undefined dynamic_lookup)
endif()



####################
### CUDA BACKEND ###
####################
find_package(CUDA)
if(CUDA_FOUND)
  message(STATUS "Found cuda, building cuda backend")

  include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
  list(APPEND LINKER_LIBS ${CUDA_CUDART_LIBRARY})

  # invoke nvidia smi to detect if we really have a GPU
  execute_process(COMMAND "nvidia-smi" ERROR_QUIET  RESULT_VARIABLE NV_RET)
  if(NV_RET EQUAL "0")
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
  else()
    # set to 3.7 the flag of K80
    CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 3.7)
  endif()

  # set arch flags properly
  CUDA_ADD_LIBRARY(ndarray_backend_cuda MODULE src/ndarray_backend_cuda.cu OPTIONS ${ARCH_FLAGS})

  target_link_libraries(ndarray_backend_cuda ${LINKER_LIBS})
  pybind11_extension(ndarray_backend_cuda)
  pybind11_strip(ndarray_backend_cuda)

  # directly output to ffi folder
  set_target_properties(ndarray_backend_cuda
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/needle/backend_ndarray
    CXX_VISIBILITY_PRESET "hidden"
    CUDA_VISIBILITY_PRESET "hidden"
)

endif()

