#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "treelstm.h"
#include "syntax_tree.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

struct MLP {
  vector<Expression> i_IH;
  Expression i_Hb;
  Expression i_HO;
  Expression i_Ob;

  Expression Feed(vector<Expression> input) const;
};

class SentimentModel {
public:
  SentimentModel();
  SentimentModel(Model& model, unsigned vocab_size);
  void InitializeParameters(Model& model, unsigned vocab_size);

  Expression BuildGraph(const SyntaxTree& tree, ComputationGraph& cg);
  Expression CalculateLoss(const vector<tuple<SyntaxTree*, Expression>>& results, ComputationGraph& cg);
  void CalculateOutputs(const SyntaxTree& tree, const vector<Expression>& annotations, const MLP& final_mlp, ComputationGraph& cg, vector<tuple<SyntaxTree*, Expression>>* results);
  vector<tuple<SyntaxTree*, Expression>> Predict(const SyntaxTree& tree, ComputationGraph& cg);
  vector<Expression> BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg);
  vector<Expression> BuildLinearAnnotationVectors(const SyntaxTree& tree, ComputationGraph& cg);
  vector<Expression> BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg);

  MLP GetFinalMLP(ComputationGraph& cg) const;

private:
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  TreeLSTMBuilder tree_builder;
  LookupParameters* p_E;
  Parameters* p_fIH;
  Parameters* p_fHb;
  Parameters* p_fHO;
  Parameters* p_fOb;

  vector<cnn::real> zero_annotation;

  unsigned lstm_layer_count = 1;
  unsigned word_embedding_dim = 50;
  unsigned node_embedding_dim = 50;
  unsigned final_hidden_dim = 50;

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & word_embedding_dim;
    ar & node_embedding_dim;
    ar & final_hidden_dim;
  }
};
