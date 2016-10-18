// nnet3/nnet-ctc-test.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)

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

#include "ctc/cctc-transition-model.h"
#include "ctc/cctc-graph.h"
#include "ctc/language-model.h"
#include "ctc/cctc-supervision.h"
#include "ctc/cctc-training.h"
#include "ctc/cctc-test-utils.h"
#include "nnet3/nnet-cctc-example.h"

// This test program tests things declared in ctc-supervision.h and
// cctc-training.h

namespace kaldi {
namespace nnet3 {


// This function creates a compact lattice with phones as the labels,
// and strings (which would normally be transition-ids) containing
// repetitions of '1', for the specified duration.  It's used for
// testing the CCTC supervsion code.
void CreateCompactLatticeFromPhonesAndDurations(
    const std::vector<int32> &phones,
    const std::vector<int32> &durations,
    CompactLattice *lat_out) {
  KALDI_ASSERT(phones.size() == durations.size());
  lat_out->DeleteStates();
  int32 current_state = lat_out->AddState();
  lat_out->SetStart(current_state);
  for (size_t i = 0; i < phones.size(); i++) {
    int32 next_state = lat_out->AddState(),
        phone = phones[i], duration = durations[i];
    std::vector<int32> repeated_ones(duration, 1);
    CompactLatticeWeight weight(LatticeWeight(RandUniform(), RandUniform()),
                                repeated_ones);
    CompactLatticeArc arc(phone, phone, weight, next_state);
    lat_out->AddArc(current_state, arc);
    current_state = next_state;
  }
  CompactLatticeWeight weight(LatticeWeight(RandUniform(), RandUniform()),
                              std::vector<int32>());
  lat_out->SetFinal(current_state, weight);    
}


void ExcludeRangeFromVector(const std::vector<int32> &in,
                            int32 first, int32 last,
                            std::vector<int32> *out) {
  out->clear();
  out->reserve(in.size());
  for (std::vector<int32>::const_iterator iter = in.begin(),
           end = in.end(); iter != end; ++iter) {
    int32 i = *iter;
    if (i < first || i > last)
      out->push_back(i);
  }
}

void TestCctcSupervisionTraining(const ctc::CctcTransitionModel &trans_model,
                                 const ctc::CctcSupervision &supervision) {
  using namespace kaldi::ctc;
  BaseFloat delta = 1.0e-04;
  int32 num_frames = supervision.num_frames,
      nnet_output_dim = trans_model.NumOutputIndexes();
  CuMatrix<BaseFloat> nnet_output(num_frames, nnet_output_dim);
  nnet_output.SetRandn();
  CuMatrix<BaseFloat> cu_weights;
  trans_model.ComputeWeights(&cu_weights);
  CctcTrainingOptions opts;
  CctcComputation computation(opts, trans_model, cu_weights,
                              supervision, nnet_output);
  BaseFloat log_like = computation.Forward();
  KALDI_LOG << "log-like of CCTC computation is " << log_like;

  CuMatrix<BaseFloat> nnet_output_deriv(num_frames, nnet_output_dim,
                                        kUndefined);
  nnet_output_deriv.SetRandn();  // <- the class requires only that it not have
                                 // NaN's or infs, so we set it random to test
                                 // that it ignores the existing data.
  computation.Backward(&nnet_output_deriv);
  int32 num_offsets = 3;
  Vector<BaseFloat> predicted_objf_changes(num_offsets),
      measured_objf_changes(num_offsets);
  for (int32 i = 0; i < num_offsets; i++) {
    CuMatrix<BaseFloat> modified_output(num_frames, nnet_output_dim);
    modified_output.SetRandn();
    modified_output.Scale(delta);
    predicted_objf_changes(i) = TraceMatMat(modified_output,
                                            nnet_output_deriv, kTrans);
    modified_output.AddMat(1.0, nnet_output);
    CctcComputation computation(opts, trans_model, cu_weights,
                                supervision, modified_output);
    BaseFloat modified_log_like = computation.Forward();
    // no need to do backward.
    measured_objf_changes(i) = modified_log_like - log_like;
  }
  KALDI_LOG << "predicted_objf_changes = " << predicted_objf_changes;
  KALDI_LOG << "measured_objf_changes = " << measured_objf_changes;
  KALDI_ASSERT(predicted_objf_changes.ApproxEqual(measured_objf_changes, 0.1));
}

void TestNnetCctcExampleIo(const NnetCctcExample &eg) {
  bool binary = (RandInt(0, 1) == 0);
  std::ostringstream os;
  eg.Write(os, binary);
  std::istringstream is(os.str());
  NnetCctcExample eg2;
  if (RandInt(0, 1) == 0)
    eg2 = eg;  // test reading already-existing object.
  eg2.Read(is, binary);
  std::ostringstream os2;
  eg2.Write(os2, binary);
  if (binary) {
    KALDI_ASSERT(os.str() == os2.str());
    KALDI_ASSERT(eg == eg2);
  }
}

void TestNnetCctcExampleMerge(const NnetCctcExample &eg) {
  int32 num_merge = RandInt(1, 10);
  bool compress = (RandInt(0, 1) == 0);
  bool compactify = (RandInt(0, 1) == 0);
  KALDI_LOG << "num_merge = " << num_merge << ", compress = "
            << (compress ? "true" : "false") << ", compactify = "
            << (compactify ? "true" : "false");
  std::vector<NnetCctcExample> egs(num_merge);
  for (int32 i = 0; i < num_merge; i++)
    egs[i] = eg;
  NnetCctcExample merged_eg;
  MergeCctcExamples(compress, compactify, &egs, &merged_eg);
  /*
  {
    std::ostringstream os;
    eg.Write(os, false);
    KALDI_LOG << "Original eg is " << os.str();
  }
  {
    std::ostringstream os;
    merged_eg.Write(os, false);
    KALDI_LOG << "Merged eg is " << os.str();
  }
  */
}


void TestCctcSupervisionNnet(const ctc::CctcSupervision &supervision) {
  using namespace ctc;
  CctcSupervisionSplitter splitter(supervision);
  int32 num_frames = supervision.num_frames,
      frames_per_range = RandInt(3, 10);
  std::vector<int32> range_starts;
  SplitIntoRanges(num_frames, frames_per_range, &range_starts);
  int32 num_ranges = range_starts.size();
  std::vector<CctcSupervision> split_supervision(num_ranges);
  for (int32 i = 0; i < num_ranges; i++)
    splitter.GetFrameRange(range_starts[i], frames_per_range,
                           &split_supervision[i]);
  if (num_ranges > 0) {
    int32 input_dim = RandInt(10, 80),
        left_context = RandInt(0, 10),
        right_context = RandInt(0, 5);
    int32 first_frame = 0, // 1st supervised frame of each eg
        frame_skip = RandInt(1, 3);
    std::vector<NnetCctcExample> nnet_examples(num_ranges);
    for (int32 i = 0; i < num_ranges; i++) {
      NnetCctcExample &nnet_eg = nnet_examples[i];
      const CctcSupervision &cctc_sup = split_supervision[i];
      int32 num_ctc_frames = cctc_sup.num_frames;
      NnetCctcSupervision nnet_ctc_sup(cctc_sup, "output", first_frame,
                                       frame_skip);
      nnet_eg.outputs.resize(1);
      nnet_eg.outputs[0].Swap(&nnet_ctc_sup);
      nnet_eg.inputs.resize(0);
      int32 num_input_frames = left_context + num_ctc_frames * frame_skip +
          right_context;
      Matrix<BaseFloat> feats(num_input_frames, input_dim);
      feats.SetRandn();
      NnetIo input_io("input", -left_context, feats);
      nnet_eg.inputs.resize(1);
      nnet_eg.inputs[0].Swap(&input_io);
      TestNnetCctcExampleIo(nnet_eg);
      TestNnetCctcExampleMerge(nnet_eg);
    }
  }
}



std::ostream &operator << (std::ostream &os, const ctc::CctcProtoSupervision &p) {
  p.Write(os, false);
  return os;
}
std::ostream &operator << (std::ostream &os, const ctc::CctcSupervision &s) {
  s.Write(os, false);
  return os;
}

void TestCctcSupervision(const ctc::CctcTransitionModel &trans_model) {
  using namespace kaldi::ctc;
  int32 num_phones = trans_model.NumPhones(),
      tot_frames = 0, subsample_factor = RandInt(1, 3);
  std::vector<int32> phones, durations;
  int32 sequence_length = RandInt(1, 20);
  for (int32 i = 0; i < sequence_length; i++) {
    int32 phone = RandInt(1, num_phones),
        duration = RandInt(1, 6);
    phones.push_back(phone);
    durations.push_back(duration);
    tot_frames += duration;
  }
  if (tot_frames < subsample_factor) {
    // Don't finish the test because it would fail.  we run this multiple times.
    return;
  }  
  CctcSupervisionOptions sup_opts;
  sup_opts.frame_subsampling_factor = subsample_factor;
  // keep the following two lines in sync and keep the silence
  // range contiguous for the test to work.
  int32 first_silence_phone = 1, last_silence_phone = 3;
  sup_opts.silence_phones = "1:2:3";
  bool start_from_lattice = (RandInt(0, 1) == 0);
  CctcProtoSupervision proto_supervision;
  if (start_from_lattice) {
    CompactLattice clat;
    CreateCompactLatticeFromPhonesAndDurations(phones, durations, &clat);
    PhoneLatticeToProtoSupervision(clat, &proto_supervision);
  } else {
    AlignmentToProtoSupervision(phones, durations, &proto_supervision);
  }
  KALDI_LOG << "Original proto-supervision is: " << proto_supervision;
  MakeSilencesOptional(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after making silences optional is: " << proto_supervision;
  ModifyProtoSupervisionTimes(sup_opts, &proto_supervision);
  KALDI_LOG << "Proto-supervision after modifying times is: " << proto_supervision;
  AddBlanksToProtoSupervision(&proto_supervision);
  KALDI_LOG << "Proto-supervision after adding blanks is: " << proto_supervision;
  CctcSupervision supervision;
  if (!MakeCctcSupervisionNoContext(proto_supervision, num_phones,
                                   &supervision)) {
    // the only way this should fail is if we had too many phones for
    // the number of subsampled frames.    
    KALDI_ASSERT(sequence_length > tot_frames / subsample_factor);
    KALDI_LOG << "Failed to create CctcSupervision because too many "
              << "phones for too few frames.";
    return;
  }
  KALDI_LOG << "Supervision without context is: " << supervision;
  AddContextToCctcSupervision(trans_model, &supervision);
  KALDI_LOG << "Supervision after adding context is: " << supervision;


  fst::StdVectorFst one_path;
  // ShortestPath effectively chooses an arbitrary path, because all paths have
  // unit weight / zero cost.
  ShortestPath(supervision.fst, &one_path);

  std::vector<int32> graph_label_seq_in, graph_label_seq_out;
  fst::TropicalWeight tot_weight;
  GetLinearSymbolSequence(one_path, &graph_label_seq_in,
                          &graph_label_seq_out, &tot_weight);
  KALDI_ASSERT(tot_weight == fst::TropicalWeight::One() &&
               graph_label_seq_in == graph_label_seq_out);

  { // basic testing of ComputeFstStateTimes (it has a lot of asserts).
    std::vector<int32> state_times;
    int32 length = ComputeFstStateTimes(supervision.fst, &state_times);
    KALDI_ASSERT(static_cast<size_t>(length) == graph_label_seq_out.size());
    for (size_t i = 0; i + 1 < state_times.size(); i++)
      KALDI_ASSERT(state_times[i] <= state_times[i+1]);
  }
  

  std::vector<int32> phones_from_graph;
  for (size_t i = 0; i < graph_label_seq_in.size(); i++) {
    int32 this_phone = trans_model.GraphLabelToPhone(graph_label_seq_in[i]);
    if (this_phone != 0)
      phones_from_graph.push_back(this_phone);
  }
  // phones_from_graph should equal 'phones', except that silences may be
  // deleted.  Check this.
  std::vector<int32> phone_nosil, phones_from_graph_nosil;
  ExcludeRangeFromVector(phones, first_silence_phone, last_silence_phone,
                         &phone_nosil);
  ExcludeRangeFromVector(phones_from_graph, first_silence_phone,
                         last_silence_phone, &phones_from_graph_nosil);
  KALDI_ASSERT(phone_nosil == phones_from_graph_nosil);

  TestCctcSupervisionNnet(supervision);
}

void NnetCctcSupervisionTest() {
  ctc::CctcTransitionModel trans_model;
  ctc::GenerateCctcTransitionModel(&trans_model);
  TestCctcSupervision(trans_model);
}



}  // namespace ctc
}  // namespace kaldi


int main() {
  using namespace kaldi;
  using namespace kaldi::nnet3;

  for (int32 loop = 0; loop < 2; loop++) {
#if HAVE_CUDA == 1
    if (loop == 0)
      CuDevice::Instantiate().SelectGpuId("no");
    else
      CuDevice::Instantiate().SelectGpuId("yes");
#endif
    for (int32 i = 0; i < 5; i++)
      kaldi::nnet3::NnetCctcSupervisionTest();
  }

  KALDI_LOG << "Tests succeeded.";

  return 0;
}

