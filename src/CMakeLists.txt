set(SOURCE_FILES
  memory_space.cpp  
  matrix_multiply.cu
  norm.cu
  search.cu
  select.cu
)

cuda_add_library(${PROJECT_NAME} ${LIB_TYPE} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
#  ${CUDA_curand_LIBRARY}
)
