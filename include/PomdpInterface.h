/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * 
 */

#ifndef _POMDPINTERFACE_H_
#define _POMDPINTERFACE_H_

#include <vector>
#include <string>
#include <map>
#include <unordered_map>

using namespace std;

class PomdpInterface
{
private:
    /* data */
public:
    PomdpInterface(/* args */){};
    virtual ~PomdpInterface(){};
    virtual double GetDiscount() const = 0;
    virtual int GetSizeOfS() const = 0;
    virtual int GetSizeOfA() const = 0;
    virtual int GetSizeOfObs() const = 0;
    virtual double TransFunc(int sI, int aI, int s_newI) const = 0;
    virtual double ObsFunc(int oI, int s_newI, int aI) const = 0;
    virtual double Reward(int sI, int aI) const = 0;
    virtual const std::vector<string> &GetAllStates() const = 0;
    virtual const std::vector<string> &GetAllActions() const = 0;
    virtual const std::vector<string> &GetAllObservations() const = 0;
    // for sparse representation
    virtual const map<int, double> *GetTransProbDist(int sI, int aI) const
    {
        (void)(sI);
        (void)(aI);
        return nullptr;
    };

    virtual const map<int, double> *GetObsFuncProbDist(int s_newI, int aI) const
    {
        (void)(s_newI);
        (void)(aI);
        return nullptr;
    };
    virtual const map<int, double> *GetInitBeliefSparse() const
    {
        return nullptr;
    };
};

#endif