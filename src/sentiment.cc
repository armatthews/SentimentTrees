#include "sentiment.h"

Expression MLP::Feed(vector<Expression> inputs) const {
  assert (inputs.size() == i_IH.size());
  vector<Expression> xs(2 * inputs.size() + 1);
  xs[0] = i_Hb;
  for (unsigned i = 0; i < inputs.size(); ++i) {
    xs[2 * i + 1] = i_IH[i];
    xs[2 * i + 2] = inputs[i];
  }
  Expression hidden1 = affine_transform(xs);
  Expression hidden2 = tanh({hidden1});
  Expression output = affine_transform({i_Ob, i_HO, hidden2});
  return output;
}

SentimentModel::SentimentModel() {
}

SentimentModel::SentimentModel(Model& model, unsigned vocab_size) {
  InitializeParameters(model, vocab_size);
}

void SentimentModel::InitializeParameters(Model& model, unsigned vocab_size) {
  assert (node_embedding_dim % 2 == 0);
  const unsigned half_node_embedding_dim = node_embedding_dim / 2;
  forward_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim, half_node_embedding_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim, half_node_embedding_dim, &model); 
  tree_builder = TreeLSTMBuilder(5, lstm_layer_count, node_embedding_dim, node_embedding_dim, &model);

  p_E = model.add_lookup_parameters(vocab_size, {word_embedding_dim});

  p_fIH = model.add_parameters({final_hidden_dim, node_embedding_dim});
  p_fHb = model.add_parameters({final_hidden_dim});
  p_fHO = model.add_parameters({5, final_hidden_dim});
  p_fOb = model.add_parameters({5});

  zero_annotation.resize(node_embedding_dim);
}

Expression SentimentModel::CalculateLoss(const vector<tuple<SyntaxTree*, Expression>>& results, ComputationGraph& cg) {
  vector<Expression> losses(results.size());
  for (unsigned i = 0; i < results.size(); ++i) {
    const SyntaxTree* tree = get<0>(results[i]);
    Expression prediction = get<1>(results[i]);
    losses[i] = pickneglogsoftmax(prediction, tree->sentiment());
  }
  return sum(losses);
}

void SentimentModel::CalculateOutputs(const SyntaxTree& tree, const vector<Expression>& annotations, const MLP& final_mlp, ComputationGraph& cg, vector<tuple<SyntaxTree*, Expression>>* results) {
  if (tree.NumChildren() > 0) {
    for (unsigned i = 0; i < tree.NumChildren(); ++i) {
      const SyntaxTree& child = tree.GetChild(i);
      CalculateOutputs(child, annotations, final_mlp, cg, results);
    }

    assert (tree.id() < annotations.size());
    Expression my_output = final_mlp.Feed({annotations[tree.id()]});
    results->push_back(make_tuple((SyntaxTree*)&tree, my_output));
  }
}

vector<Expression> SentimentModel::BuildLinearAnnotationVectors(const SyntaxTree& tree, ComputationGraph& cg) {
  const bool use_bidirectional = false;
  if (use_bidirectional) {
    vector<Expression> forward_annotations = BuildForwardAnnotations(tree.GetTerminals(), cg);
    vector<Expression> reverse_annotations = BuildReverseAnnotations(tree.GetTerminals(), cg);
    vector<Expression> linear_annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
    return linear_annotations;
  }
  else {
    vector<Expression> linear_annotations;
    for (WordId w : tree.GetTerminals()) {
      linear_annotations.push_back(lookup(cg, p_E, w));
    }
    return linear_annotations;
  }
}

vector<tuple<SyntaxTree*, Expression>> SentimentModel::Predict(const SyntaxTree& tree, ComputationGraph& cg) {
  vector<Expression> linear_annotations = BuildLinearAnnotationVectors(tree, cg);
  vector<Expression> tree_annotations = BuildTreeAnnotationVectors(tree, linear_annotations, cg);
  assert (tree_annotations.size() == tree.NumNodes());

  vector<tuple<SyntaxTree*, Expression>> outputs;
  const MLP& final_mlp = GetFinalMLP(cg);
  CalculateOutputs(tree, tree_annotations, final_mlp, cg, &outputs);
  return outputs;
}

Expression SentimentModel::BuildGraph(const SyntaxTree& tree, ComputationGraph& cg) {
  vector<tuple<SyntaxTree*, Expression>> outputs = Predict(tree, cg);
  return CalculateLoss(outputs, cg);
}

vector<Expression> SentimentModel::BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  forward_builder.new_graph(cg);
  forward_builder.start_new_sequence();
  vector<Expression> forward_annotations(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    Expression i_x_t = lookup(cg, p_E, sentence[t]);
    Expression i_y_t = forward_builder.add_input(i_x_t);
    forward_annotations[t] = i_y_t;
  }
  return forward_annotations;
}

vector<Expression> SentimentModel::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  reverse_builder.new_graph(cg);
  reverse_builder.start_new_sequence();
  vector<Expression> reverse_annotations(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    Expression i_x_t = lookup(cg, p_E, sentence[t]);
    Expression i_y_t = reverse_builder.add_input(i_x_t);
    reverse_annotations[t] = i_y_t;
  }
  return reverse_annotations;
}

vector<Expression> SentimentModel::BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg) {
  vector<Expression> annotations(forward_annotations.size());
  for (unsigned t = 0; t < forward_annotations.size(); ++t) {
    const Expression& i_f = forward_annotations[t];
    const Expression& i_r = reverse_annotations[t];
    Expression i_h = concatenate({i_f, i_r});
    annotations[t] = i_h;
  }
  return annotations;
}

vector<Expression> SentimentModel::BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg) {
  tree_builder.new_graph(cg);
  tree_builder.start_new_sequence();
  vector<Expression> annotations;
  vector<Expression> tree_annotations;
  vector<const SyntaxTree*> node_stack = {&source_tree};
  vector<unsigned> index_stack = {0};
  unsigned terminal_index = 0;

  while (node_stack.size() > 0) {
    assert (node_stack.size() == index_stack.size());
    const SyntaxTree* node = node_stack.back();
    unsigned i = index_stack.back();
    if (i >= node->NumChildren()) {
      assert (tree_annotations.size() == node->id());
      vector<int> children(node->NumChildren());
      for (unsigned j = 0; j < node->NumChildren(); ++j) {
        unsigned child_id = node->GetChild(j).id();
        assert (child_id < tree_annotations.size());
        assert (child_id < (unsigned)INT_MAX);
        children[j] = (int)child_id;
      }

      Expression input_expr;
      if (node->NumChildren() == 0) {
        assert (terminal_index < linear_annotations.size());
        input_expr = linear_annotations[terminal_index];
        terminal_index++;
      }
      else {
        input_expr = input(cg, {(long)zero_annotation.size()}, &zero_annotation);
      }
      Expression node_annotation = tree_builder.add_input((int)node->id(), children, input_expr);
      tree_annotations.push_back(node_annotation);
      index_stack.pop_back();
      node_stack.pop_back();
    }
    else {
      index_stack[index_stack.size() - 1] += 1;
      node_stack.push_back(&node->GetChild(i));
      index_stack.push_back(0);
      ++i;
    }
  }
  assert (node_stack.size() == index_stack.size());

  return tree_annotations;
}

MLP SentimentModel::GetFinalMLP(ComputationGraph& cg) const {
  Expression i_fIH = parameter(cg, p_fIH);
  Expression i_fHb = parameter(cg, p_fHb);
  Expression i_fHO = parameter(cg, p_fHO);
  Expression i_fOb = parameter(cg, p_fOb);
  MLP final_mlp = {{i_fIH}, i_fHb, i_fHO, i_fOb};
  return final_mlp;
}
