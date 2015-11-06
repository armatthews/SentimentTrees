#pragma once
#include <vector>
#include <string>
#include "cnn/dict.h"
//#include "utils.h"

using namespace std;
using namespace cnn;

typedef int WordId;

class SyntaxTree {
public:
  SyntaxTree();
  SyntaxTree(string tree, Dict* dict);

  bool IsTerminal() const;
  unsigned NumChildren() const;
  unsigned NumNodes() const;
  unsigned MaxBranchCount() const;
  unsigned MinDepth() const;
  unsigned MaxDepth() const;
  WordId label() const;
  unsigned id() const;
  unsigned sentiment() const;
  vector<WordId> GetTerminals() const;

  SyntaxTree& GetChild(unsigned i);
  const SyntaxTree& GetChild(unsigned i) const;

  string ToString() const;
  unsigned AssignNodeIds(unsigned start = 0);
private:
  Dict* dict;
  WordId label_;
  unsigned id_;
  unsigned sentiment_;
  vector<SyntaxTree> children;
};

ostream& operator<< (ostream& stream, const SyntaxTree& tree);
