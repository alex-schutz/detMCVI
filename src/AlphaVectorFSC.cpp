#include "../include/AlphaVectorFSC.h"

int64_t AlphaVectorFSC::GetEtaValue(int64_t nI, int64_t action,
                                    int64_t observation) const {
  const std::unordered_map<std::pair<int64_t, int64_t>, int64_t>& m = _eta[nI];
  const auto it = m.find({action, observation});
  if (it != m.cend()) return it->second;
  return -1;
}
