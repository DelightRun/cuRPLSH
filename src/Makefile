default: test_norm

CXX=g++
CXXFLAGS=-std=c++14 -m64 -O3
CXXINCLUDE=-I/usr/local/cuda-9.2/include
CXXLDFLAGS=-L/usr/local/cuda/lib64 -lcudart -lcurand -lcublas

NVCC=colornvcc
NVCCFLAGS=-gencode=arch=compute_61,code=sm_61 --use_fast_math

TEST_NORM_OBJS=test_norm.o norm.o memory_space.o
TEST_SEARCH_OBJS=test_search.o search.o norm.o matrix_multiply.o select.o memory_space.o

%.o : %.cu
	$(NVCC) $(CXXFLAGS) $(CXXLDFLAGS) $(NVCCFLAGS) -c $< -o $@

memory_space.o : memory_space.cpp
	$(CXX) $(CXXINCLUDE) $(CXXFLAGS) $(CXXLDFLAGS) -c memory_space.cpp -o $@

test_norm : $(TEST_NORM_OBJS)
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) $(TEST_NORM_OBJS) -o $@
	
test_search : $(TEST_SEARCH_OBJS)
	$(NVCC) $(CXXFLAGS) $(CXXLDFLAGS) $(NVCCFLAGS) $(TEST_SEARCH_OBJS) -o $@

clean:
	rm -f *.cubin *.ii *.cudafe1.* *.fatbin *.fatbin.c *.o *.module_id *.ptx
