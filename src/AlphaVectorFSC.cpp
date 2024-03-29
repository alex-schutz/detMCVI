#include "../include/AlphaVectorFSC.h"

int64_t AlphaVectorFSC::GetEtaValue(int64_t nI, int64_t action,
                                    int64_t observation) const {
  const EdgeMap& m = _eta[nI];
  const auto it = m.find({action, observation});
  if (it != m.cend()) return it->second;
  return -1;
}

int64_t AlphaVectorFSC::AddNode(const AlphaVectorNode& node) {
  _nodes.emplace_back(node);
  return _nodes.size() - 1;
}

void AlphaVectorFSC::UpdateEta(int64_t nI, int64_t a, int64_t o,
                               int64_t nI_new) {
  _eta[nI][{a, o}] = nI_new;
}

void AlphaVectorFSC::UpdateEta(int64_t nI, const AlphaVectorFSC::EdgeMap& eta) {
  _eta[nI] = eta;
}
