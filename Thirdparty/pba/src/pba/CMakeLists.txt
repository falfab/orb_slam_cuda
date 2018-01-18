

IF(CUDA_FOUND)

  INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

  CUDA_COMPILE(PROGRAM_CU ProgramCU.cu)

  SET(SRCS pba.cpp
           ConfigBA.cpp
           CuTexImage.cpp
          #  SparseBundleCPU.cpp
           SparseBundleCU.cpp)

  CUDA_ADD_LIBRARY(pba SHARED ${SRCS} ${PROGRAM_CU} ${CUDA_LIBRARY})

  TARGET_LINK_LIBRARIES(pba ${CMAKE_THREAD_LIBS_INIT})

  SET_TARGET_PROPERTIES(pba PROPERTIES
                            SOVERSION 1
                            VERSION 1.5.0)
                            
    # TODO panato
  CUDA_ADD_EXECUTABLE(pba_driver ../driver/driver.cpp ${SRCS} ${PROGRAM_CU} ${CUDA_LIBRARY})
ELSE()

  SET(SRCS pba.cpp
           ConfigBA.cpp
           SparseBundleCPU.cpp)

  ADD_LIBRARY(pba SHARED ${SRCS})

  SET_TARGET_PROPERTIES(pba PROPERTIES
                            SOVERSION 1
                            VERSION 1.5.0)
  TARGET_LINK_LIBRARIES(pba ${CMAKE_THREAD_LIBS_INIT})
  
  # TODO panato
  ADD_EXECUTABLE(pba_driver ../driver/driver.cpp ${SRCS})

ENDIF()

INSTALL(TARGETS pba DESTINATION "lib${LIB_SUFFIX}")
