/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 * 
 */

#ifndef _POMDPINTERFACE_H_
#define _POMDPINTERFACE_H_

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <random>

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

	// sI_next, oI, Reward, Done
    virtual tuple<int, int, double, bool> Step(int sI, int aI) const{
	std::mt19937_64 rng(random_device{}());
	uniform_real_distribution<double> unif(0, 1);

	// sample next state
  	const double u_s = unif(rng);
	int s_next = -1;
	double p_s = 0.0;
	for (int s=0; s<GetSizeOfS(); ++s){
		p_s += TransFunc(sI, aI, s);
		if (p_s > u_s) {
			s_next = s; 
			break;
		}
	}

	// sample observation
	const double u_o = unif(rng);
	int obs = -1;
	double p_o = 0.0;
	for (int o=0; o<GetSizeOfObs(); ++o){
		p_o += ObsFunc(o, s_next, aI);
		if (p_o > u_o) {
			obs = o; 
			break;
		}
	}

	return {s_next, obs, Reward(sI, aI), false};
	}

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