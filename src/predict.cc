#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <vector>

#include "sentiment.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

bool ctrlc_pressed = false;
void ctrlc_handler(int signal) {
  if (ctrlc_pressed) {
    exit(1);
  }
  else {
    ctrlc_pressed = true;
  }
}

tuple<Dict*, Model*, SentimentModel*> LoadModel(string model_filename) {
  ifstream model_file(model_filename);
  if (!model_file.is_open()) {
    cerr << "ERROR: Unable to open " << model_filename << endl;
    exit(1);
  }
  boost::archive::text_iarchive ia(model_file);

  Dict* vocab = new Dict();
  ia & *vocab;
  vocab->Freeze();

  Model* cnn_model = new Model();
  SentimentModel* sentiment_model = new SentimentModel();

  ia & *sentiment_model;
  sentiment_model->InitializeParameters(*cnn_model, vocab->size());

  ia & *cnn_model;

  return make_tuple(vocab, cnn_model, sentiment_model);
}

unsigned argmax(const vector<float>& probs) {
  assert (probs.size() > 0);
  unsigned m = 0;
  for (unsigned i = 1; i < probs.size(); ++ i) {
    if (probs[i] > probs[m]) {
      m = i;
    }
  }
  return m;
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "model file, as output by train")
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  string model_filename = vm["model"].as<string>();
  cnn::Initialize(argc, argv);

  Dict* vocab = nullptr;
  Model* cnn_model = nullptr;
  SentimentModel* sentiment_model = nullptr;
  tie(vocab, cnn_model, sentiment_model) = LoadModel(model_filename);

  vocab->Freeze();

  string line;
  unsigned sentence_number = 0;
  while(getline(cin, line)) {
    SyntaxTree tree(line, vocab);
    tree.AssignNodeIds();

    ComputationGraph cg;
    vector<tuple<SyntaxTree*, Expression>> predictions = sentiment_model->Predict(tree, cg);
    cg.forward();

    for (auto t : predictions) {
      SyntaxTree* tree;
      Expression predictions;
      tie(tree, predictions) = t;
      vector<float> p = as_vector(predictions.value());
      cout << sentence_number << " ||| ";
      for (WordId w : tree->GetTerminals()) {
        cout << vocab->Convert(w) << " ";
      }
      cout << "||| " << tree->sentiment() << " ||| " << argmax(p) << " |||";
      for (float v : p) {
        cout << " " << v;
      }
      cout << "\n";
    }

    sentence_number++;

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
