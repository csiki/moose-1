/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MARKOVGSLSOLVER_H
#define _MARKOVGSLSOLVER_H

// Author : Vishaka Datta S, 2011, NCBS.
//
// The GslIntegrator class in ksolve deals with a system whose coefficients stay
// constant. In the case of a Markov channel, the coefficients of the system
// vary with time. 
//
// This makes it necessary for the system to keep track of changes in the system
// matrix, which is implemented by the message handler. 
//

class MarkovGslSolver
{
	public:
		MarkovGslSolver();
		~MarkovGslSolver();

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////
		bool getIsInitialized() const;
		string getMethod() const;
		void setMethod( string method );
		double getRelativeAccuracy() const;
		void setRelativeAccuracy( double value );
		double getAbsoluteAccuracy() const;
		void setAbsoluteAccuracy( double value );
		double getInternalDt() const;
		void setInternalDt( double value );

///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr info );
		void reinit( const Eref& e, ProcPtr info );

		void setParams( Id );
//		void handleState( vector< double > );

		static const Cinfo* initCinfo();
	private:
		bool isInitialized_;
		string method_;
		double absAccuracy_;
		double relAccuracy_;
		double internalStepSize_;
		vector< double > state_;
		double* stateGsl_;
		unsigned int nVars_;

		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};
#endif 