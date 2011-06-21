/*******************************************************************
 * File:            STPNMDAChan.h
 * Description:      
 * Author:          Subhasis Ray
 * E-mail:          
 * Created:         2011-06-15 14:45:05 (+0530)
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _STPNMDACHAN_H
#define _STPNMDACHAN_H


class STPNMDAChan: public STPSynChan
{    
  public:
    // Functions duplicated from SynChan
    STPNMDAChan();
    virtual ~STPNMDAChan(){};
    ///////////////////////////////////////////////////
    // Dest function definitions
    ///////////////////////////////////////////////////

    virtual void innerSynapseFunc( const Conn* c, double time );
    virtual void innerProcessFunc( Eref e, ProcInfo p );
    virtual void innerReinitFunc( Eref e,  ProcInfo p );

    ///////////////////////////////////////
    // Functions specific to STPNMDAChan
    ///////////////////////////////////////
    static void setTransitionParam(const Conn* c, double value, const unsigned int& index);
    void innerSetTransitionParam(double value, const unsigned int index);
    static double getTransitionParam(Eref e, const unsigned int& index);
    double innerGetTransitionParam(unsigned int index);
    static double getUnblocked(Eref e);
    double innerGetUnblocked();
    static double getSaturation(Eref e);
    double innerGetSaturation();
    static void setSaturation(const Conn * conn, double value);
    void innerSetSaturation(double value);
    static void setMgConc(const Conn* conn, double conc);
    void innerSetMgConc(double value);
    static double getMgConc(Eref e);
    double innerGetMgConc();

///////////////////////////////////////////////////
// Protected fields.
///////////////////////////////////////////////////

  protected:
    double saturation_, unblocked_, Mg_, A_, B1_, B2_, decayFactor_;
    vector<double> c_;
    priority_queue<SynInfo> oldEvents_;
    
};

#endif