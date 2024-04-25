#include "ParserPOMDPSparse.h"

namespace MCVI {

/** builds a POMDP from file **/
ParsedPOMDPSparse::ParsedPOMDPSparse(const std::string filename) {
  cout << "#### SPARSE POMDP ####" << endl;

  // #### STEP 1 : extract structure of POMDP by ging through the whole file ###

  // open input file containing a pomdp file
  ifstream infile;
  infile.open(filename);
  if (!infile.is_open()) cout << "open file failure" << endl;

  // TODO rename temp as 'line'
  std::string temp;
  bool ReadStart = false;  // end at here

  // go through the file line by line
  // First Get discount and all the state, action and observation space
  while (getline(infile, temp) && !ReadStart) {
    // analyse the line
    istd::stringstream is(temp);
    std::string s;

    // values corresponding to parsing steps
    int64_t temp_num = 0;
    bool ReadDiscount = false;
    bool ReadStates = false;
    bool ReadActions = false;
    bool ReadObservations = false;

    // read the POMDP structure (discount, states, actions and observation)
    // analyse the extracted line token by token
    while (is >> s) {
      // ### STEP 1.1 Line structure ###
      // TODO can be improved by removing the first token from the loop !!

      // if read "discount" => future value = discount factor
      if (s == "discount:") {
        ReadDiscount = true;
      }

      // if read "states" => future values = list of states
      else if (s == "states:") {
        ReadStates = true;
      }

      // if read "actions" => future values = list of states
      else if (s == "actions:") {
        ReadActions = true;
      }

      // if read "observations" => future values = list of observations
      else if (s == "observations:") {
        ReadObservations = true;
      }

      // if read "start" => future values => values of probabilities of b0
      // (switch to next loop)
      else if (s == "start:") {
        ReadStart = true;
      }

      // ### STEP 1.2 : VALUES depending on structure (start of the line) ###
      // TODO no need to consider temp_num = 1

      // Get discount factor
      if (ReadDiscount && temp_num == 1) {
        this->discount = stod(s);
      }

      // Get all the States
      if (ReadStates && temp_num > 0) {
        this->States.push_back(s);
      }

      // Get all actions
      if (ReadActions && temp_num > 0) {
        this->Actions.push_back(s);
      }

      // Get all observations
      if (ReadObservations && temp_num > 0) {
        this->Observations.push_back(s);
      }

      // Get initial belief
      if (ReadStart && temp_num > 0) {
        double pb = stod(s);
        b0.push_back(pb);
        if (pb > 0) {
          b0_sparse[b0.size() - 1] = pb;
        }
      }

      // update number of times file hase been read
      temp_num += 1;
    }
  }

  // close the file
  infile.close();

  // ### STEP 2 : read the values of probabilities by goign throug the whole
  // POMDP file ###
  // TODO no need to close and open the file again !!! could be possible to
  // begin from where we stopped from previous loop temporary variables to store
  // information

  this->TransFuncVecs.resize(
      Actions.size(), std::vector<std::map<int64_t, double>>(States.size()));

  // observation as A -> S' -> O -> proba
  // std::vector< std::vector< std::vector<double> > > O(Actions.size(),
  // std::vector<std::vector<double> >(States.size(),
  // std::vector<double>(Observations.size()) ));
  this->ObsFuncVecs.resize(
      Actions.size(), std::vector<std::map<int64_t, double>>(States.size()));
  // reawrd as A -> S -> reward value
  this->RewardFuncVecs.resize(Actions.size(),
                              std::vector<double>(States.size()));
  // std::vector< std::vector<double> > R(Actions.size(),
  // std::vector<double>(States.size()));

  // open pomdp file
  infile.open(filename);

  // go through the entire file to store probabilities, line by line
  while (getline(infile, temp)) {
    // to anayse the line
    istd::stringstream is(temp);
    std::string s;

    // information for deciding which line to consider
    int64_t temp_num = 0;
    bool buildTrans = false;
    bool buildObs = false;
    bool buildReward = false;
    int64_t aI = 0;
    int64_t sI = 0;
    int64_t oI = 0;
    int64_t snewI = 0;
    double pb = 0;

    // Get T,O and R depending on the line
    while (is >> s) {
      // ### STEP 2.1 : anlyse the start of the line to know the type of
      // information ###
      // TODO possible to extract this from the loop

      // Next values corresponds to transition function
      if (s == "T:") {
        buildTrans = true;
      }
      // Next values correspond to observation probabilities
      else if (s == "O:") {
        buildObs = true;
      }
      // Next values correspond to a reward
      else if (s == "R:") {
        buildReward = true;
      }

      // ### STEP 2.2 : store the corresponding information through the line ###
      // TODO : no need to do it that way : easier to separte outside the loop
      // depending on the type of info.

      // first value of the line : always an action
      if (temp_num == 1) {
        if (buildTrans || buildObs || buildReward) {
          aI = stoi(s);
        }
      }

      // second value of the line : always a state
      else if (temp_num == 3) {
        if (buildTrans || buildObs || buildReward) {
          sI = stoi(s);
        }
      }

      // third value of the line depends on the type of information
      else if (temp_num == 5) {
        // if transition => arrival state
        if (buildTrans) {
          snewI = stoi(s);
        }

        // if observation => observation
        if (buildObs) {
          oI = stoi(s);
        }
      }

      // 4th value of the line
      else if (temp_num == 6) {
        // if transition => transition probability, we can store it in T
        if (buildTrans) {
          pb = stod(s);
          // add in the std::map with insert (since it is a new key)
          this->TransFuncVecs[aI][sI].insert(std::make_pair(snewI, pb));
        }

        // if observation => observation probability, we can store it in O
        if (buildObs) {
          pb = stod(s);
          // O[aI][sI][oI] = pb;
          this->ObsFuncVecs[aI][sI].insert(std::make_pair(oI, pb));
        }
      }

      // 5th value of the line : should be reward
      else if (temp_num == 8) {
        if (buildReward) {
          pb = stod(s);
          this->RewardFuncVecs[aI][sI] = pb;
        }
      }

      // increase the token number
      temp_num += 1;

    }  // end of line parsing
  }  // end of file parsing

  // all probabilities stored, close file
  infile.close();

  // store informations as attributes
  // this->TransFuncVecs = T;
  // this->ObsFuncVecs = O;
  // this->RewardFuncVecs = R;
  this->S_size = this->States.size();
  this->Obs_size = this->Observations.size();
  this->A_size = this->Actions.size();
}

/* returns discount factor */
double ParsedPOMDPSparse::GetDiscount() const { return this->discount; };

/* returns number of states */
int64_t ParsedPOMDPSparse::GetSizeOfS() const { return this->S_size; };

/* returns number of actions */
int64_t ParsedPOMDPSparse::GetSizeOfA() const { return this->A_size; };

/* returns number of observations */
int64_t ParsedPOMDPSparse::GetSizeOfObs() const { return this->Obs_size; };

// /* returns initial belief */
// std::std::vector<double> ParsedPOMDPSparse::GetInitBelief(){
// 	return this->b0;
// };

const std::map<int64_t, double> *ParsedPOMDPSparse::GetInitBeliefSparse()
    const {
  return &this->b0_sparse;
};

/* returns transition function probability */
double ParsedPOMDPSparse::TransFunc(int64_t sI, int64_t aI,
                                    int64_t s_newI) const {
  // if key absent
  if ((this->TransFuncVecs[aI][sI]).find(s_newI) ==
      this->TransFuncVecs[aI][sI].end()) {
    // returns proba 0
    return 0.;
  }
  // key present
  else {
    // returns associated value
    return this->TransFuncVecs[aI][sI].find(s_newI)->second;
  }
};

/* returns observation probabilities */
double ParsedPOMDPSparse::ObsFunc(int64_t oI, int64_t s_newI,
                                  int64_t aI) const {
  return this->ObsFuncVecs[aI][s_newI].find(oI)->second;
};

/* returns reward */
double ParsedPOMDPSparse::Reward(int64_t sI, int64_t aI) const {
  return this->RewardFuncVecs[aI][sI];
};

/* destructor */
ParsedPOMDPSparse::~ParsedPOMDPSparse() {}

/* return a prob distribution for transition function*/
// std::map<int64_t,double>* ParsedPOMDPSparse::GetTransProbDist(int64_t sI,
// int64_t aI){ 	return &this->TransFuncVecs[aI][sI];
// };
const std::map<int64_t, double> *ParsedPOMDPSparse::GetTransProbDist(
    int64_t sI, int64_t aI) const {
  return &this->TransFuncVecs[aI][sI];
};

const std::map<int64_t, double> *ParsedPOMDPSparse::GetObsFuncProbDist(
    int64_t s_newI, int64_t aI) const {
  return &this->ObsFuncVecs[aI][s_newI];
};

const std::std::vector<std::string> &ParsedPOMDPSparse::GetAllStates() const {
  return this->States;
}
const std::std::vector<std::string> &ParsedPOMDPSparse::GetAllActions() const {
  return this->Actions;
}
const std::std::vector<std::string> &ParsedPOMDPSparse::GetAllObservations()
    const {
  return this->Observations;
}

}  // namespace MCVI
