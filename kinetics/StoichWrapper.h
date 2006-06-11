/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _StoichWrapper_h
#define _StoichWrapper_h
class StoichWrapper: 
	public Stoich, public Neutral
{
	friend Element* integrateConnStoichLookup( const Conn* );
    public:
		StoichWrapper(const string& n)
		:
			Neutral( n ),
			allocateSrc_( &integrateConn_ )
			// integrateConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setPath( Conn* c, string value ) {
			static_cast< StoichWrapper* >( c->parent() )->
				setPathLocal( value );
		}
		static string getPath( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->path_;
		}
		static int getNMols( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nMols_;
		}
		static int getNVarMols( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nVarMols_;
		}
		static int getNSumTot( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nSumTot_;
		}
		static int getNBuffered( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nBuffered_;
		}
		static int getNReacs( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nReacs_;
		}
		static int getNEnz( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nEnz_;
		}
		static int getNMmEnz( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nMmEnz_;
		}
		static int getNExternalRates( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->nExternalRates_;
		}
		static int getRateVectorSize( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->
				rates_.size();
		}
		static void setUseOneWayReacs( Conn* c, int value ) {
			static_cast< StoichWrapper* >( c->parent() )->useOneWayReacs_ = value;
		}
		static int getUseOneWayReacs( const Element* e ) {
			return static_cast< const StoichWrapper* >( e )->useOneWayReacs_;
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getAllocateSrc( Element* e ) {
			return &( static_cast< StoichWrapper* >( e )->allocateSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  ) {
			S_ = Sinit_;
			allocateSrc_.send( &S_ );
		}
		static void reinitFunc( Conn* c ) {
			static_cast< StoichWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void integrateFuncLocal( vector< double >*  yprime, double dt ) {
			updateRates( yprime, dt );
		}
		static void integrateFunc( Conn* c, vector< double >*  yprime, double dt ) {
			static_cast< StoichWrapper* >( c->parent() )->
				integrateFuncLocal( yprime, dt );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getIntegrateConn( Element* e ) {
			return &( static_cast< StoichWrapper* >( e )->integrateConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Stoich* p = dynamic_cast<const Stoich *>(proto);
			// if (p)... and so on. 
			return new StoichWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc1< vector< double >*  > allocateSrc_;
		UniConn< integrateConnStoichLookup > integrateConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
		void setPathLocal( const string& value );
		void setupMols(
			vector< Element* >& varMolVec,
			vector< Element* >& bufVec,
			vector< Element* >& sumTotVec);
		void addSumTot( Element* e );
		void addReac( Element* e );
		void addEnz( Element* e );
		void addMmEnz( Element* e );
		void addTab( Element* e );
		void addRate( Element* e );
		void setupReacSystem();
		unsigned int findReactants( Element* e,
			const string& msgFieldName, vector< const double* >& ret );
		unsigned int findProducts( Element* e,
			const string& msgFieldName, vector< const double* >& ret );
		map< const Element*, int > molMap_;
		void fillStoich( const double* baseptr, 
			vector< const double* >& sub, 
			vector< const double* >& prd, 
			int reacNum );
		void fillHalfStoich( const double* baseptr, 
			vector< const double* >& reactants, 
			int sign, int reacNum );
		bool checkEnz( Element* e,
			vector< const double* >& sub, 
			vector< const double* >& prd, 
			vector< const double* >& enz, 
			vector< const double* >& cplx,
			double& k1, double& k2, double& k3,
			bool isMM
			);
		static const double EPSILON;

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _StoichWrapper_h
