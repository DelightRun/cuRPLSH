set(CPP_SOURCE_FILES
  index.cpp
  index_bf.cpp
)

set(CUDA_SOURCE_FILES
  internal/algorithms.cu
  internal/broadcast.cu
  internal/norm.cu
  internal/matrix_ops.cu
  internal/search.cu
  internal/select.cu
)

include_directories(${PROJECT_SOURCE_DIR}/include)

cuda_add_library(${PROJECT_NAME} ${LIB_TYPE} ${CPP_SOURCE_FILES} ${CUDA_SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${CUDA_LINK_LIBS})
