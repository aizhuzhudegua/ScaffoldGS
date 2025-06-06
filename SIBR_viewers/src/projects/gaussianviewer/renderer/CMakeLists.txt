# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# 
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# 
# For inquiries contact sibr@inria.fr and/or George.Drettakis@inria.fr

set(SIBR_PROJECT "gaussian")
project(sibr_${SIBR_PROJECT} LANGUAGES CXX)

sibr_gitlibrary(TARGET CudaRasterizer
    GIT_REPOSITORY 	"https://github.com/tongji-rkr/diff-gaussian-rasterization.git"
    GIT_TAG			"10c7167ccb7162ae81ad6e174933805325c95047"
)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
source_group("Source Files" FILES ${SOURCES})

file(GLOB SHADERS "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")
source_group("Source Files\\shaders" FILES ${SHADERS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp" "shaders/*.frag" "shaders/*.vert" "shaders/*.geom")

## Specify target rules
add_library(${PROJECT_NAME} SHARED ${SOURCES})

include_directories(${Boost_INCLUDE_DIRS} .)
if (WIN32)
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glfw3
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
)
else()
target_link_libraries(${PROJECT_NAME}
	${Boost_LIBRARIES}
	${ASSIMP_LIBRARIES}
	${GLEW_LIBRARIES}
	${OPENGL_LIBRARIES}
	${OpenCV_LIBRARIES}
	glfw
	sibr_system
	sibr_view
	sibr_assets
	sibr_renderer
	sibr_basic
	CUDA::cudart
	CudaRasterizer
)
endif()

add_definitions( -DSIBR_EXP_ULR_EXPORTS -DBOOST_ALL_DYN_LINK  )

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "projects/${SIBR_PROJECT}/renderer")

## High level macro to install in an homogen way all our ibr targets
include(install_runtime)
ibr_install_target(${PROJECT_NAME}
    INSTALL_PDB                         ## mean install also MSVC IDE *.pdb file (DEST according to target type)
	SHADERS ${SHADERS}
	RSC_FOLDER ${SIBR_PROJECT}

    #STANDALONE  ${INSTALL_STANDALONE}   ## mean call install_runtime with bundle dependencies resolution
    COMPONENT   ${PROJECT_NAME}_install ## will create custom target to install only this project
)
