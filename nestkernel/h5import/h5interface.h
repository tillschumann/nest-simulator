/*
 * h5file.h
 *
 *  Created on: Oct 12, 2016
 *      Author: schumann
 */

#include <hdf5.h>

#ifndef H5FILE_H_
#define H5FILE_H_

namespace h5import {

class H5File
{
protected:
		// hdf5 file pointer
		hid_t file_id_, gid_;
public:
	H5File( const std::string& path )
	{
		hid_t fapl_id = H5Pcreate( H5P_FILE_ACCESS );
		file_id_ = H5Fopen( path.c_str(), H5F_ACC_RDONLY, fapl_id );
		H5Pclose( fapl_id );

		gid_ = H5Gopen( file_id_, "/", H5P_DEFAULT );
	}
        
        H5File( const std::string& path, MPI_Comm comm )
        {
                hid_t fapl_id = H5Pcreate( H5P_FILE_ACCESS );

                MPI_Info info  = MPI_INFO_NULL;;
                H5Pset_fapl_mpio( fapl_id, comm, info );
                
                file_id_ = H5Fopen( path.c_str(), H5F_ACC_RDONLY, fapl_id );
                H5Pclose( fapl_id );

                gid_ = H5Gopen( file_id_, "/", H5P_DEFAULT );
        }

	virtual ~H5File()
	{
		H5Gclose( gid_ );
		H5Fclose( file_id_ );
	}

	hid_t id() const {
		return file_id_;
	}
};

struct H5View
{
	hsize_t offset[ 1 ];
	hsize_t stride[ 1 ];
	hsize_t count[ 1 ];
	hsize_t block[ 1 ];

	H5View( hsize_t icount = 0,
			hsize_t ioffset = 0,
			hsize_t istride = 1,
			hsize_t iblock = 1 )
	{
		offset[ 0 ] = ioffset;
		stride[ 0 ] = istride;
		count[ 0 ] = icount;
		block[ 0 ] = iblock;
	}
	inline hsize_t
	view2dataset( const hsize_t& v_idx ) const
	{
		return offset[ 0 ] + ( v_idx / block[ 0 ] ) * ( stride[ 0 ] - 1 ) + v_idx;
	}
};


class H5Dataset
{
private:
	hid_t id_;

public:
	H5Dataset( const H5File& file, const std::string& name )
	{
		id_ = H5Dopen2( file.id(), name.c_str(), H5P_DEFAULT );
	}

	~H5Dataset()
	{
	  H5Dclose( id_ );
	}

	hid_t id()
	{
	  return id_;
	}

	hsize_t size() const
	{
		hid_t dataspace_id = H5Dget_space( id_ );
		hsize_t count;
		H5Sget_simple_extent_dims(dataspace_id, &count, NULL );
		H5Sclose( dataspace_id );
		return count;
	}
};


};

#endif /* H5FILE_H_ */
