set(SOURCE_FILES
  memory_space.cpp
)

set(INTERNAL_SOURCE_FILES
  internal/matrix_multiply.cu
  internal/norm.cu
  internal/search.cu
  internal/select.cu
)

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_library(${PROJECT_NAME} ${LIB_TYPE} ${SOURCE_FILES} ${INTERNAL_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
  ${CUDA_LIBRARIES}
  ${CUDA_CUBLAS_LIBRARIES}
#  ${CUDA_curand_LIBRARY}
)
