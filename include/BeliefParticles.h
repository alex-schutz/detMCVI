/* This file has been written and/or modified by the following people:
 *
 * Yang You
 * Alex Schutz
 * 
 */

#ifndef _BELIEFPARTICLES_H_
#define _BELIEFPARTICLES_H_

#include <iostream>
#include <vector>
#include <map>
#include <any> // from C++ 17

using namespace std;

class BeliefParticles
{
private:
    vector<any> particles;           // a vector of Any state particles
    int size_particles = -1;
public:
    BeliefParticles(){};
    ~BeliefParticles(){};
    BeliefParticles(vector<any> &particles);
    any SampleOneState() const;
    int GetParticleSize();
    double operator[](int i);
    bool operator==(BeliefParticles &o);
    void BuildBeliefSparse();
};

#endif