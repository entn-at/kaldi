// nnet3/nnet-training.cc

// Copyright      2015    Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "nnet3/nnet-cctc-training.h"
#include "nnet3/nnet-utils.h"
#include <ctime>
#include <time.h>
std::clock_t    startX;
// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime2() {
  
     std::cout << "Time: " << (std::clock() - startX) / (double)(CLOCKS_PER_SEC / 1000) << " ms" << std::endl;
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

     startX = std::clock();
    return buf;
}

namespace kaldi {
namespace nnet3 {

NnetCctcTrainer::NnetCctcTrainer(const NnetCctcTrainerOptions &config,
                                 const ctc::CctcTransitionModel &trans_model,
                                 Nnet *nnet):
    config_(config),
    trans_model_(trans_model),
    nnet_(nnet),
    compiler_(*nnet, config_.optimize_config),
    num_minibatches_processed_(0) {
  if (config.zero_component_stats)
    ZeroComponentStats(nnet);
  if (config.momentum == 0.0 && config.max_param_change == 0.0) {
    delta_nnet_= NULL;
  } else {
    KALDI_ASSERT(config.momentum >= 0.0 &&
                 config.max_param_change >= 0.0);
    delta_nnet_ = nnet_->Copy();
    bool is_gradient = false;  // setting this to true would disable the
                               // natural-gradient updates.
    SetZero(is_gradient, delta_nnet_);
  }
  if (config_.read_cache != "") {
    bool binary;
    try {
      Input ki(config_.read_cache, &binary);
      compiler_.ReadCache(ki.Stream(), binary);
      KALDI_LOG << "Read computation cache from " << config_.read_cache;
    } catch (...) {
      KALDI_WARN << "Could not open cached computation. "
                    "Probably this is the first training iteration.";
    }
  } 
  trans_model_.ComputeWeights(&cu_weights_);
     startX = std::clock();
}


void NnetCctcTrainer::Train(const NnetCctcExample &cctc_eg) {
  bool need_model_derivative = true;
  ComputationRequest request;
    KALDI_LOG << "GetCctcComputationRequest b " << currentDateTime2();
  GetCctcComputationRequest(*nnet_, cctc_eg, need_model_derivative,
                            config_.store_component_stats,
                            &request);
    KALDI_LOG << "GetCctcComputationRequest " << currentDateTime2();
  const NnetComputation *computation = compiler_.Compile(request);
    KALDI_LOG << "Compile " << currentDateTime2();

  NnetComputer computer(config_.compute_config, *computation,
                        *nnet_,
                        (delta_nnet_ == NULL ? nnet_ : delta_nnet_));
  // give the inputs to the computer object.
    KALDI_LOG << "computer " << currentDateTime2();
  computer.AcceptInputs(*nnet_, cctc_eg.inputs);
    KALDI_LOG << "AcceptInputs " << currentDateTime2();
  computer.Forward();
    KALDI_LOG << "Forward " << currentDateTime2();

  this->ProcessOutputs(cctc_eg, &computer);
    KALDI_LOG << "ProcessOutputs " << currentDateTime2();
  computer.Backward();
    KALDI_LOG << "Backward " << currentDateTime2();

  if (delta_nnet_ != NULL) {
    BaseFloat scale = (1.0 - config_.momentum);
    if (config_.max_param_change != 0.0) {
      BaseFloat param_delta =
          std::sqrt(DotProduct(*delta_nnet_, *delta_nnet_)) * scale;
      if (param_delta > config_.max_param_change) {
        if (param_delta - param_delta != 0.0) {
          KALDI_WARN << "Infinite parameter change, will not apply.";
          SetZero(false, delta_nnet_);
        } else {
          scale *= config_.max_param_change / param_delta;
          KALDI_LOG << "Parameter change too big: " << param_delta << " > "
                    << "--max-param-change=" << config_.max_param_change
                    << ", scaling by " << config_.max_param_change / param_delta;
        }
      }
    }
    KALDI_LOG << "if (delta_nnet_ != NULL) " << currentDateTime2();
    AddNnet(*delta_nnet_, scale, nnet_);
    KALDI_LOG << "AddNnet " << currentDateTime2();
    ScaleNnet(config_.momentum, delta_nnet_);
    KALDI_LOG << "ScaleNnet " << currentDateTime2();
  }
    KALDI_LOG << "Train END " << currentDateTime2();
}


void NnetCctcTrainer::ProcessOutputs(const NnetCctcExample &eg,
                                     NnetComputer *computer) {
  // There will normally be just one output here, named 'output',
  // but the code is more general than this.
  std::vector<NnetCctcSupervision>::const_iterator iter = eg.outputs.begin(),
      end = eg.outputs.end();
  for (; iter != end; ++iter) {
    const NnetCctcSupervision &sup = *iter;
    int32 node_index = nnet_->GetNodeIndex(sup.name);
    if (node_index < 0 ||
        !nnet_->IsOutputNode(node_index))
      KALDI_ERR << "Network has no output named " << sup.name;

    const CuMatrixBase<BaseFloat> &nnet_output = computer->GetOutput(sup.name);
    CuMatrix<BaseFloat> nnet_output_deriv(nnet_output.NumRows(),
                                          nnet_output.NumCols(),
                                          kUndefined);

    BaseFloat tot_weight, tot_objf;
    sup.ComputeObjfAndDerivs(config_.cctc_training_config,
                             trans_model_,
                             cu_weights_, nnet_output,
                             &tot_weight, &tot_objf, &nnet_output_deriv);

    computer->AcceptOutputDeriv(sup.name, &nnet_output_deriv);

    objf_info_[sup.name].UpdateStats(sup.name, config_.print_interval,
                                     num_minibatches_processed_++,
                                     tot_weight, tot_objf);
  }
}


bool NnetCctcTrainer::PrintTotalStats() const {
  unordered_map<std::string, ObjectiveFunctionInfo>::const_iterator
      iter = objf_info_.begin(),
      end = objf_info_.end();
  bool ans = false;
  for (; iter != end; ++iter) {
    const std::string &name = iter->first;
    const ObjectiveFunctionInfo &info = iter->second;
    ans = ans || info.PrintTotalStats(name);
  }
  return ans;
}


NnetCctcTrainer::~NnetCctcTrainer() {
  if (config_.write_cache != "") {
    Output ko(config_.write_cache, config_.binary_write_cache);
    compiler_.WriteCache(ko.Stream(), config_.binary_write_cache);
    KALDI_LOG << "Wrote computation cache to " << config_.write_cache;
  } 
  delete delta_nnet_;
}


} // namespace nnet3
} // namespace kaldi
