# nestkernel/H5Synapse/CMakeLists.txt
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


set( h5import_sources
   H5Synapses.h H5Synapses.cpp
   H5Neurons.h H5Neurons.cpp
   h5reader.h h5reader.cpp  
   )
    
    
add_library( h5import ${h5import_sources} )
target_link_libraries( h5import nestutil sli_lib nestkernel ${MPI_CXX_LIBRARIES} ${HDF5_LIBRARIES} )

target_include_directories( h5import PRIVATE
    ${PROJECT_SOURCE_DIR}/librandom
    ${PROJECT_SOURCE_DIR}/libnestutil
    ${PROJECT_BINARY_DIR}/libnestutil
    ${PROJECT_SOURCE_DIR}/sli
    ${PROJECT_SOURCE_DIR}/nestkernel
    ${HDF5_INCLUDE_DIRS}
    )

install( TARGETS h5import
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
