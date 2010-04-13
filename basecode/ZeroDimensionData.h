/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZERO_DIMENSION_DATA_H
#define _ZERO_DIMENSION_DATA_H

/**
 * This class manages the data part of Elements having just one
 * data entry.
 */
class ZeroDimensionData: public DataHandler
{
	public:
		ZeroDimensionData( const DinfoBase* dinfo )
			: DataHandler( dinfo ), data_( 0 )
		{;}

		~ZeroDimensionData();

		/**
		 * Returns the data on the specified index.
		 */
		char* data( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the data at one level up of indexing.
		 * Here there isn't any.
		 */
		char* data1( DataId index ) const {
			return data_;
		}

		/**
		 * Returns the number of data entries.
		 */
		unsigned int numData() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 1.
		 */
		unsigned int numData1() const {
			return 1;
		}

		/**
		 * Returns the number of data entries at index 2, if present.
		 * For regular Elements and 1-D arrays this is always 1.
		 */
		 unsigned int numData2( unsigned int index1 ) const {
		 	return 1;
		 }

		/**
		 * Returns the number of dimensions of the data.
		 */
		unsigned int numDimensions() const {
			return 0;
		}

		/**
		 * Assigns the sizes of all array field entries at once.
		 * This is ignored for regular Elements.
		 */
		void setArraySizes( const vector< unsigned int >& sizes ) {
			;
		}


		/**
		 * Looks up the sizes of all array field entries at once. Returns
		 * all ones for regular Elements. 
		 * Note that a single Element may have more than one array field.
		 * However, each FieldElement instance will refer to just one of
		 * these array fields, so there is no ambiguity.
		 */
		void getArraySizes( vector< unsigned int >& sizes ) const;

		/**
		 * Returns true if the node decomposition has the data on the
		 * current node
		 */
		bool isDataHere( DataId index ) const;

		virtual bool isAllocated() const;

		void allocate();

	protected:

	private:
		char* data_;
};

#endif // _ZERO_DIMENSION_DATA_H
