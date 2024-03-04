/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 * 
 */


#ifndef _SIMINTERFACE_H_
#define _SIMINTERFACE_H_

#include "PomdpInterface.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
#include <cmath>
using namespace std;

class SimInterface
{
private:
    /* data */
public:
    SimInterface(){};
    virtual ~SimInterface(){};

    // ------- obligate functions ----------
    virtual tuple<int, int, double, bool> Step(int sI, int aI) = 0; // sI_next, oI, Reward, Done
    virtual int SampleStartState() = 0;
    virtual int GetSizeOfObs() const = 0;
    virtual int GetSizeOfA() const = 0;
    virtual double GetDiscount() const = 0;
    virtual int GetNbAgent() const = 0;
    // --------------------------------------------------------

    // Maybe add visulization functions? :)

};

#endif