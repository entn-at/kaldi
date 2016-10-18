// ctc/cctc-function.cc

// Copyright      2015   Johns Hopkins University (author: Daniel Povey)

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

#include "ctc/cctc-supervision.h"
#include "lat/lattice-functions.h"
#include "util/text-utils.h"
#include "ctc/cctc-graph.h"

namespace kaldi {
namespace ctc {

// Note: testing code for these functions is in cctc-transition-model-test.cc
void CctcProtoSupervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CctcProtoSupervision>");
  if (!binary) os << "\n";
  WriteToken(os, binary, "<NumFrames>");
  WriteBasicType(os, binary, num_frames);
  if (!binary) os << "\n";
  WriteFstKaldi(os, binary, fst);
  WriteToken(os, binary, "<PhoneInstances>");
  int32 num_pinst = phone_instances.size() - 1;
  WriteBasicType(os, binary, num_pinst);
  if (!binary) os << "\n";
  for (int32 p = 1; p <= num_pinst; p++) {
    if (!binary) { // this is just for ease of interpretation.
      os << "label" << p << ": ";
    }
    WriteBasicType(os, binary, phone_instances[p].phone_or_blank);
    WriteBasicType(os, binary, phone_instances[p].begin_frame);
    WriteBasicType(os, binary, phone_instances[p].end_frame);
    if (!binary) os << "\n";
  }
  WriteToken(os, binary, "</CctcProtoSupervision>");
  if (!binary) os << "\n";
}

void AlignmentToProtoSupervision(const std::vector<int32> &phones,
                                 const std::vector<int32> &durations,
                                 CctcProtoSupervision *proto_supervision) {
  KALDI_ASSERT(phones.size() > 0 && phones.size() == durations.size());
  // we don't use element zero of the phone_instances array.
  proto_supervision->phone_instances.resize(phones.size() + 1);
  std::vector<int32> labels(phones.size());
  int32 current_frame = 0;
  for (size_t i = 0; i < phones.size(); i++) {
    int32 phone = phones[i], duration = durations[i];
    KALDI_ASSERT(phone > 0 && duration > 0);
    proto_supervision->phone_instances[i+1].phone_or_blank = phone;
    proto_supervision->phone_instances[i+1].begin_frame = current_frame;
    current_frame += duration;
    proto_supervision->phone_instances[i+1].end_frame = current_frame;
    labels[i] = i + 1;  // will become labels in the FST.
  }
  proto_supervision->num_frames = current_frame;
  fst::MakeLinearAcceptor(labels, &(proto_supervision->fst));
}

void AlignmentToProtoSupervision(
    const std::vector<std::pair<int32, int32> > &phones_durations,
    CctcProtoSupervision *proto_supervision) {
  KALDI_ASSERT(phones_durations.size() > 0);
  // we don't use element zero of the phone_instances array.
  proto_supervision->phone_instances.resize(phones_durations.size() + 1);
  std::vector<int32> labels(phones_durations.size());
  int32 current_frame = 0;
  for (size_t i = 0; i < phones_durations.size(); i++) {
    int32 phone = phones_durations[i].first,
        duration = phones_durations[i].second;
    KALDI_ASSERT(phone > 0 && duration > 0);
    proto_supervision->phone_instances[i+1].phone_or_blank = phone;
    proto_supervision->phone_instances[i+1].begin_frame = current_frame;
    current_frame += duration;
    proto_supervision->phone_instances[i+1].end_frame = current_frame;
    labels[i] = i + 1;  // will become labels in the FST.
  }
  proto_supervision->num_frames = current_frame;
  fst::MakeLinearAcceptor(labels, &(proto_supervision->fst));
}


void PhoneLatticeToProtoSupervision(const CompactLattice &lat,
                                    CctcProtoSupervision *proto_supervision) {
  static bool warned_eps = false, warned_final = false;
  KALDI_ASSERT(lat.NumStates() != 0);
  int32 num_states = lat.NumStates();
  proto_supervision->fst.DeleteStates();
  proto_supervision->fst.ReserveStates(num_states);
  proto_supervision->phone_instances.resize(1);  // zeroth element is unused.
  proto_supervision->phone_instances.reserve(num_states * 2);
  std::vector<int32> state_times;
  proto_supervision->num_frames = CompactLatticeStateTimes(lat, &state_times);
  for (int32 state = 0; state < num_states; state++)
    proto_supervision->fst.AddState();
  proto_supervision->fst.SetStart(lat.Start());
  for (int32 state = 0; state < num_states; state++) {
    int32 state_time = state_times[state];
    for (fst::ArcIterator<CompactLattice> aiter(lat, state); !aiter.Done();
         aiter.Next()) {
      const CompactLatticeArc &lat_arc = aiter.Value();
      int32 next_state_time = state_time + lat_arc.weight.String().size();
      int32 phone = lat_arc.ilabel;  // It's an acceptor so ilabel == ollabel.
      if (phone == 0) {
        if (!warned_eps) {
          KALDI_WARN << "CompactLattice has epsilon arc.  Unexpected.";
          warned_eps = true;
        }
        // add epsilon arc to the proto_supervision.
        proto_supervision->fst.AddArc(state,
                                      fst::StdArc(0, 0,
                                                  fst::TropicalWeight::One(),
                                                  lat_arc.nextstate));
      } else {
        int32 label = proto_supervision->phone_instances.size();
        proto_supervision->phone_instances.push_back(
            PhoneInstance(phone, state_time, next_state_time));
        proto_supervision->fst.AddArc(state,
                                      fst::StdArc(label, label,
                                                  fst::TropicalWeight::One(),
                                                  lat_arc.nextstate));
      }
    }
    if (lat.Final(state) != CompactLatticeWeight::Zero()) {
      proto_supervision->fst.SetFinal(state, fst::TropicalWeight::One());
      if (state_times[state] != proto_supervision->num_frames &&
          !warned_final) {
        KALDI_WARN << "Time of final state " << state << " in lattice is "
                   << "not equal to number of frames "
                   << proto_supervision->num_frames << ".  Are you sure "
                   << "the lattice is phone-aligned?";
        warned_final = true;
      }
    }
  }
}


bool TimeEnforcerFst::GetArc(StateId s, Label ilabel, fst::StdArc* oarc) {
  KALDI_ASSERT(ilabel > 0 && static_cast<size_t>(ilabel) <
               proto_supervision_.phone_instances.size());
  const PhoneInstance &instance =
      proto_supervision_.phone_instances[ilabel];
  if (s < instance.begin_frame || s >= instance.end_frame) {
    // we don't allow that phone-instance at that time.
    return false;
  }
  oarc->ilabel = ilabel;
  // Output labels have an offset of 1 so that blank doesn't get
  // confused with epsilon.
  oarc->olabel = instance.phone_or_blank + 1;
  oarc->weight = Weight::One();
  oarc->nextstate = s + 1;
  return true;
}


bool MakeCctcSupervisionNoContext(
    const CctcProtoSupervision &proto_supervision,
    int32 num_phones,
    CctcSupervision *supervision) {
  TimeEnforcerFst enforcer(proto_supervision);

  ComposeDeterministicOnDemand(proto_supervision.fst,
                               &enforcer, &(supervision->fst));
  fst::Project(&(supervision->fst), fst::PROJECT_OUTPUT);
  fst::Connect(&(supervision->fst));
  fst::RmEpsilon(&(supervision->fst));
  fst::StdVectorFst det_fst;
  // Determinization will make sure that there are no duplicate paths with
  // the same label sequence; it will reduce the size of the FST.
  fst::Determinize(supervision->fst, &det_fst);
  supervision->fst = det_fst;  // shallow copy.
  if (supervision->fst.NumStates() == 0)
    return false;
  SortBreadthFirstSearch(&(supervision->fst));
  supervision->weight = 1.0;
  supervision->num_frames = proto_supervision.num_frames;
  supervision->label_dim = num_phones + 1;
  return (supervision->fst.NumStates() > 0);
}

void MakeSilencesOptional(const CctcSupervisionOptions &opts,
                          CctcProtoSupervision *proto_supervision) {

  if (opts.silence_phones.empty())
    return;  // Nothing to do.
  std::vector<int32> silence_phones;
  if (!SplitStringToIntegers(opts.silence_phones, ":,", false,
                             &silence_phones) || silence_phones.empty())
    KALDI_ERR << "Invalid --silence-phones option: '"
              << opts.silence_phones << "'";
  int32 max_silence_phone = *std::max_element(silence_phones.begin(),
                                              silence_phones.end()),
      num_phone_instances = proto_supervision->phone_instances.size() - 1;
  KALDI_ASSERT(max_silence_phone > 0);
  std::vector<bool> is_silence_phone(max_silence_phone + 1, false);
  for (size_t i = 0; i < silence_phones.size(); i++) {
    KALDI_ASSERT(silence_phones[i] > 0);
    is_silence_phone[silence_phones[i]] = true;
  }
  int32 num_states = proto_supervision->fst.NumStates();
  for (int32 state = 0; state < num_states; state++) {
    // arcs_to_add is the pairs (nextstate, label) for blank arcs that we have
    // to add starting from this state.  We don't add them inside the loop over
    // arcs, to avoid invalidating the iterator.
    std::vector<std::pair<int32,int32> > arcs_to_add;

    typedef fst::ArcIterator<fst::StdVectorFst> IterType;
    for (IterType aiter(proto_supervision->fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      int32 label = arc.ilabel;  // it's an acceptor, ilabel == olabel.
      if (label == 0)
        continue;  // We don't do anything for epsilon arcs.
      KALDI_ASSERT(label > 0 && label <= num_phone_instances);
      const PhoneInstance &pinst = proto_supervision->phone_instances[label];
      int32 phone = pinst.phone_or_blank,
          duration = pinst.end_frame - pinst.begin_frame;
      KALDI_ASSERT(phone > 0 && "Making silences optional after already "
                   "adding optional blanks?");
      bool is_silence = (phone <= max_silence_phone &&
                         is_silence_phone[phone]);
      if (is_silence && duration < opts.optional_silence_cutoff) {
        // add a phone-instance label for the blank phone with the
        // same time values as the silence arc.
        int32 new_pinst_label = proto_supervision->phone_instances.size();
        proto_supervision->phone_instances.push_back(
            PhoneInstance(0, pinst.begin_frame, pinst.end_frame));
        arcs_to_add.push_back(std::pair<int32,int32>(arc.nextstate,
                                                     new_pinst_label));
      }
    }
    for (size_t i = 0; i < arcs_to_add.size(); i++) {
      int32 nextstate = arcs_to_add[i].first, label = arcs_to_add[i].second;
      fst::StdArc arc(label, label, fst::TropicalWeight::One(), nextstate);
      proto_supervision->fst.AddArc(state, arc);
    }
  }
}

void ModifyProtoSupervisionTimes(const CctcSupervisionOptions &opts,
                                 CctcProtoSupervision *proto_supervision) {
  const int32 num_frames = proto_supervision->num_frames,
      subsampling_factor = opts.frame_subsampling_factor,
      left_tolerance = opts.left_tolerance,
      right_tolerance = opts.right_tolerance;
  KALDI_ASSERT(num_frames >= subsampling_factor);
  // if the following is not true, we'd likely get disconnected FST.
  KALDI_ASSERT(left_tolerance + right_tolerance >= subsampling_factor - 1 &&
               "Insufficient left + right tolerance.");
  KALDI_ASSERT(left_tolerance >= 0 && right_tolerance >= 0 &&
               subsampling_factor > 0);
  // First modify all start-times and end-times.
  std::vector<PhoneInstance>::iterator
      iter = proto_supervision->phone_instances.begin(),
      end = proto_supervision->phone_instances.end();
  for (; iter != end; ++iter) {
    iter->begin_frame = std::max<int32>(0,
                                        iter->begin_frame - left_tolerance) /
        subsampling_factor;
    iter->end_frame = std::min<int32>(num_frames,
                                      iter->end_frame + right_tolerance) /
        subsampling_factor;
  }
  proto_supervision->num_frames = num_frames / subsampling_factor;
}

void AddBlanksToProtoSupervision(CctcProtoSupervision *proto_supervision) {
  int32 num_states = proto_supervision->fst.NumStates(),
      orig_num_phone_instances = proto_supervision->phone_instances.size() - 1;
  for (int32 state = 0; state < num_states; state++) {
    // arcs_to_add is the pairs (nextstate, label) for blank self-loop arcs that
    // we have to add both at this state and nextstate.  We don't add them
    // inside the loop over arcs, to avoid invalidating the iterator.
    std::vector<std::pair<int32,int32> > arcs_to_add;

    typedef fst::ArcIterator<fst::StdVectorFst> IterType;
    for (IterType aiter(proto_supervision->fst, state); !aiter.Done();
         aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      int32 label = arc.ilabel;  // it's an acceptor, ilabel == olabel.
      if (label == 0)
        continue;  // We don't do anything for epsilon arcs.
      if (label > orig_num_phone_instances) {
        // this means the arc was one we already added inside this function,
        // corresponding to blank.  We might as well break from the
        // loop, because any further arcs on this state will also be
        // blank self-loops added within this function.
        break;
      }
      // && label <= num_phone_instances);
      const PhoneInstance &pinst = proto_supervision->phone_instances[label];
      int32 phone = pinst.phone_or_blank;
      if (phone == 0)
        continue;  // If this arc is for the blank phone (e.g. added by
                   // MakeSilencesOptional) we don't need to do anything.

      // add a phone-instance label for the blank phone with the
      // same time values as the current arc.
      int32 new_pinst_label = proto_supervision->phone_instances.size();
      proto_supervision->phone_instances.push_back(
          PhoneInstance(0, pinst.begin_frame, pinst.end_frame));
      arcs_to_add.push_back(std::pair<int32,int32>(arc.nextstate,
                                                   new_pinst_label));
    }
    for (size_t i = 0; i < arcs_to_add.size(); i++) {
      int32 nextstate = arcs_to_add[i].first, label = arcs_to_add[i].second;
      // First add a self-loop at this state
      proto_supervision->fst.AddArc(
          state, fst::StdArc(label, label, fst::TropicalWeight::One(), state));
      // next, one at 'nextstate' (if it's != state)
      if (nextstate != state)
        proto_supervision->fst.AddArc(
            nextstate, fst::StdArc(label, label, fst::TropicalWeight::One(),
                                   nextstate));

    }
  }
}

CctcSupervisionSplitter::CctcSupervisionSplitter(
    const CctcSupervision &supervision):
    supervision_(supervision),
    frame_(supervision_.fst.NumStates(), -1) {
  const fst::StdVectorFst &fst(supervision_.fst);
  // The fst in struct CctcSupervision is supposed to be epsilon-free and
  // topologically sorted; this function relies on those properties to
  // set up the frame_ vector (which maps each state in the
  // FST to a frame-index 0 <= t < num_frames), and it checks them.
  int32 num_states = fst.NumStates(),
      num_frames = supervision_.num_frames;
  KALDI_ASSERT(num_states > 0);
  int32 start_state = fst.Start();
  // FST should be top-sorted and connected, so start-state must be 0.
  KALDI_ASSERT(start_state == 0 && "Expecting start-state to be 0");
  frame_[start_state] = 0;
  for (int32 state = 0; state < num_states; state++) {
    int32 cur_frame = frame_[state];
    if (cur_frame == -1) {
      // If this happens it means the CctcSupervision does not have the required
      // properties, e.g. being top-sorted and connected.
      KALDI_ERR << "Error computing frame indexes for CctcSupervision";
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      // The FST is supposed to be an epsilon-free acceptor.
      KALDI_ASSERT(arc.ilabel == arc.olabel && arc.ilabel > 0);
      int32 nextstate = arc.nextstate;
      KALDI_ASSERT(nextstate >= 0 && nextstate < num_states);
      // all arcs go from some t to t + 1.
      int32 &next_frame = frame_[nextstate];
      if (next_frame == -1)
        next_frame = cur_frame + 1;
      else
        KALDI_ASSERT(next_frame == cur_frame + 1);
    }
  }
  // The following assert checks that the number of frames in the FST
  // matches the num_frames stored in the supervision object; it also relies
  // on the topological sorting and connectedness of the FST.
  KALDI_ASSERT(frame_.back() == num_frames);
  std::vector<int32>::iterator iter = frame_.begin(),
      end = iter + (frame_.size() - 1);
  // check that the frame-indexes of states are monotonically non-decreasing, as
  // they should be based on the top-sorting.  We rely on this property to
  // compute the frame ranges while splitting.
  while (iter != end) {
    int32 cur_t = *iter;
    ++iter;
    int32 next_t = *iter;
    KALDI_ASSERT(next_t >= cur_t);
  }
}

void CctcSupervisionSplitter::GetFrameRange(int32 begin_frame, int32 num_frames,
                                            CctcSupervision *out_supervision) const {
  int32 end_frame = begin_frame + num_frames;
  // Note: end_frame is not included in the range of frames that the
  // output supervision object covers; it's one past the end.
  KALDI_ASSERT(num_frames > 0 && begin_frame >= 0 &&
               begin_frame + num_frames <= supervision_.num_frames);
  std::vector<int32>::const_iterator begin_iter =
      std::lower_bound(frame_.begin(), frame_.end(), begin_frame),
      end_iter = std::lower_bound(begin_iter, frame_.end(), end_frame);
  KALDI_ASSERT(*begin_iter == begin_frame &&
               (begin_iter == frame_.begin() || begin_iter[-1] < begin_frame));
  // even if end_frame == supervision_.num_frames, there should be a state with
  // that frame index.
  KALDI_ASSERT(end_iter[-1] < end_frame &&
               (end_iter < frame_.end() || *end_iter == end_frame));
  int32 begin_state = begin_iter - frame_.begin(),
      end_state = end_iter - frame_.begin();

  CreateRangeFst(begin_frame, end_frame,
                 begin_state, end_state, &(out_supervision->fst));

  fst::RmEpsilon(&(out_supervision->fst));
  fst::StdVectorFst det_fst;
  // Determinization will make sure that there are no duplicate paths with
  // the same label sequence; it will reduce the size of the FST.
  fst::Determinize(out_supervision->fst, &det_fst);
  out_supervision->fst = det_fst;  // shallow copy.
  KALDI_ASSERT(out_supervision->fst.NumStates() > 0);
  SortBreadthFirstSearch(&(out_supervision->fst));
  out_supervision->weight = supervision_.weight;
  out_supervision->num_frames = num_frames;
  out_supervision->label_dim = supervision_.label_dim;
}

void CctcSupervisionSplitter::CreateRangeFst(
    int32 begin_frame, int32 end_frame,
    int32 begin_state, int32 end_state,
    fst::StdVectorFst *fst) const {
  // There will be a special pre-start state that has epsilon transitions to all
  // states whose frame equals begin_frame; we'll later do RmEpsilon to remove
  // these.  Next we will include all states begin_state <= s < end_state in the
  // output FST, plus (if end_frame != supervision_.num_frames) a special final
  // state.  All transitions to states >= end_state will be turned into
  // a transition to the special final state.  There should be no final-probs
  // on the states begin_state <= s < end_state.
  KALDI_ASSERT(end_state > begin_state);
  fst->DeleteStates();
  fst->ReserveStates(end_state - begin_state + 2);
  int32 start_state = fst->AddState();
  fst->SetStart(start_state);
  for (int32 i = begin_state; i < end_state; i++)
    fst->AddState();
  // Add the special final-state.
  int32 final_state = fst->AddState();
  fst->SetFinal(final_state, fst::TropicalWeight::One());
  for (int32 state = begin_state; state < end_state; state++) {
    int32 output_state = state - begin_state + 1;
    if (frame_[state] == begin_frame) {
      // we'd like to make this an initial state, but OpenFst doesn't allow
      // multiple initial states.  Instead we add an epsilon transition to it
      // from our actual initial state; we'll later do RmEpsilon and
      // determinize.
      fst->AddArc(start_state,
                  fst::StdArc(0, 0, fst::TropicalWeight::One(),
                              output_state));
    } else {
      KALDI_ASSERT(frame_[state] < end_frame);
    }
    typedef fst::ArcIterator<fst::StdVectorFst> IterType;
    for (IterType aiter(supervision_.fst, state); !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc(aiter.Value());
      int32 nextstate = arc.nextstate;
      if (nextstate >= end_state) {
        // A transition to any state outside the range becomes a transition to
        // our special final-state.
        fst->AddArc(output_state,
                    fst::StdArc(arc.ilabel, arc.olabel,
                                arc.weight, final_state));
      } else {
        int32 output_nextstate = arc.nextstate - begin_state + 1;
        // note: arc.ilabel should equal arc.olabel and arc.weight should equal
        // fst::TropicalWeight::One().
        fst->AddArc(output_state,
                    fst::StdArc(arc.ilabel, arc.olabel,
                                arc.weight, output_nextstate));
      }
    }
  }
}


// I couldn't figure out how to do this with OpenFST's native 'visitor' and
// queue mechanisms so I'm just coding this myself.
void SortBreadthFirstSearch(fst::StdVectorFst *fst) {
  std::vector<int32> state_order(fst->NumStates(), -1);
  std::vector<bool> seen(fst->NumStates(), false);
  int32 start_state = fst->Start();
  KALDI_ASSERT(start_state >= 0);
  std::deque<int32> queue;
  queue.push_back(start_state);
  seen[start_state] = true;
  int32 num_output = 0;
  while (!queue.empty()) {
    int32 state = queue.front();
    state_order[state] = num_output++;
    queue.pop_front();
    for (fst::ArcIterator<fst::StdVectorFst> aiter(*fst, state);
         !aiter.Done(); aiter.Next()) {
      int32 nextstate = aiter.Value().nextstate;
      if (!seen[nextstate]) {
        seen[nextstate] = true;
        queue.push_back(nextstate);
      }
    }
  }
  if (num_output != fst->NumStates())
    KALDI_ERR << "Input to SortBreadthFirstSearch must be connected.";
  fst::StateSort(fst, state_order);
}


void AddContextToCctcSupervision(
    const CctcTransitionModel &trans_model,
    CctcSupervision *supervision) {
  KALDI_ASSERT(supervision->label_dim = trans_model.NumPhones() + 1);
  fst::StdVectorFst phone_plus_blank_fst = supervision->fst;
  BaseFloat phone_language_model_weight = 0.0;
  CreateCctcDecodingFst(trans_model, phone_language_model_weight,
                        phone_plus_blank_fst, &(supervision->fst));
  // at this point we only have the 'graph-labels' on the input side only; so we
  // need to project on the input.
  fst::Project(&(supervision->fst), fst::PROJECT_INPUT);
  SortBreadthFirstSearch(&(supervision->fst));
  if (supervision->fst.NumStates() == 0)
    KALDI_ERR << "Supervision FST is empty after context expansion.";
  supervision->label_dim = trans_model.NumGraphLabels();
}


BaseFloat CctcSupervision::ComputeExtraLogprob(
    const CctcTransitionModel &trans_mdl) const {
  BaseFloat extra_logprob = 0.0;

  // extra_history_states would logically appear inside the loop, but we want to
  // avoid the initializer and destructor code.
  std::set<int32> extra_history_states;

  /*
    A simpler, non-optimized version of this code is:
   for (int32 state = 0; state < fst.NumStates(); state++) {
     std::set<int32> history_states;
     for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state); !aiter.Done();
          aiter.Next()) {
       history_states.insert(trans_mdl.GraphLabelToHistoryState(arc.ilabel));
     }
     if (history_states.size() > 1)
       extra_logprob += -log(history_states.size());
   }
  */
  for (int32 state = 0; state < fst.NumStates(); state++) {
    int32 first_history_state;  // an optimization to avoid set operations in
                                // the common case.
    bool saw_extra = false;
    fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
    if (aiter.Done())
      continue;
    // handle the first arc separately-- an optimization.
    int32 graph_label = aiter.Value().ilabel;
    first_history_state = trans_mdl.GraphLabelToHistoryState(graph_label);
    aiter.Next();
    for (; !aiter.Done(); aiter.Next()) {
      int32 graph_label = aiter.Value().ilabel;
      int32 this_history_state = trans_mdl.GraphLabelToHistoryState(graph_label);
      if (this_history_state != first_history_state) {
        saw_extra = true;
        extra_history_states.insert(this_history_state);
      }
    }
    if (saw_extra) {
      BaseFloat size = 1.0 + extra_history_states.size();
      extra_logprob += -log(size);
      extra_history_states.clear();
    }
  }
  return extra_logprob;
}

void CctcSupervision::Write(std::ostream &os, bool binary) const {
  WriteToken(os, binary, "<CctcSupervision>");
  WriteToken(os, binary, "<Weight>");
  WriteBasicType(os, binary, weight);
  WriteToken(os, binary, "<Frames>");
  WriteBasicType(os, binary, num_frames);
  WriteToken(os, binary, "<LabelDim>");
  WriteBasicType(os, binary, label_dim);
  KALDI_ASSERT(num_frames > 0 && label_dim > 0);
  if (binary == false) {
    // In text mode, write the FST without any compactification.
    WriteFstKaldi(os, binary, fst);
  } else {
    // Write using StdUnweightedAcceptorCompactFst, making use of the fact that
    // it's an unweighted acceptor.
    fst::FstWriteOptions write_options("<unknown>");
    fst::StdCompactUnweightedAcceptorFst::WriteFst(
        fst, fst::UnweightedAcceptorCompactor<fst::StdArc>(), os,
        write_options);
  }
  WriteToken(os, binary, "</CctcSupervision>");
}

void CctcSupervision::Read(std::istream &is, bool binary) {
  ExpectToken(is, binary, "<CctcSupervision>");
  ExpectToken(is, binary, "<Weight>");
  ReadBasicType(is, binary, &weight);
  ExpectToken(is, binary, "<Frames>");
  ReadBasicType(is, binary, &num_frames);
  ExpectToken(is, binary, "<LabelDim>");
  ReadBasicType(is, binary, &label_dim);
  if (!binary) {
    ReadFstKaldi(is, binary, &fst);
  } else {
    fst::StdCompactUnweightedAcceptorFst *compact_fst =
        fst::StdCompactUnweightedAcceptorFst::Read(
            is, fst::FstReadOptions(std::string("[unknown]")));
    if (compact_fst == NULL)
      KALDI_ERR << "Error reading compact FST from disk";
    fst = *compact_fst;
    delete compact_fst;
  }
    // ReadFstKaldi will work even though we wrote using a compact format.
  ExpectToken(is, binary, "</CctcSupervision>");
}

int32 ComputeFstStateTimes(const fst::StdVectorFst &fst,
                           std::vector<int32> *state_times) {
  if (fst.Start() != 0)  // this is implied by our properties.
    KALDI_ERR << "Expecting input FST start state to be zero";
  int32 num_states = fst.NumStates();
  int32 total_length = -1;
  state_times->clear();
  state_times->resize(num_states, -1);
  (*state_times)[0] = 0;
  for (int32 state = 0; state < num_states; state++) {
    int32 next_state_time = (*state_times)[state] + 1;
    if (next_state_time <= 0)  // i.e. (*state_times)[state] < 0
      KALDI_ERR << "Input FST does not have required properties.";
    for (fst::ArcIterator<fst::StdVectorFst> aiter(fst, state);
         !aiter.Done(); aiter.Next()) {
      const fst::StdArc &arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel != 0);
      int32 &next_state_ref = (*state_times)[arc.nextstate];
      if (next_state_ref == -1)
        next_state_ref = next_state_time;
      else if (next_state_ref != next_state_time)
        KALDI_ERR << "Input FST does not have required properties.";
    }
    if (fst.Final(state) != fst::TropicalWeight::Zero()) {
      if (total_length == -1)
        total_length = next_state_time - 1;
      else if (total_length != next_state_time - 1)
        KALDI_ERR << "Input FST does not have required properties.";
    }
  }
  if (total_length < 0)
    KALDI_ERR << "Input FST does not have required properties.";
  return total_length;
}

CctcSupervision::CctcSupervision(const CctcSupervision &other):
    weight(other.weight), num_frames(other.num_frames),
    label_dim(other.label_dim), fst(other.fst) { }

void AppendCctcSupervision(const std::vector<const CctcSupervision*> &input,
                          bool compactify,
                          std::vector<CctcSupervision> *output_supervision) {
  KALDI_ASSERT(!input.empty());
  int32 label_dim = input[0]->label_dim,
      num_inputs = input.size();
  if (num_inputs == 1) {
    output_supervision->resize(1);
    (*output_supervision)[0] = *(input[0]);
    return;
  }
  std::vector<bool> output_was_merged;
  for (int32 i = 1; i < num_inputs; i++)
    KALDI_ASSERT(input[i]->label_dim == label_dim &&
                 "Trying to append incompatible CctcSupervision objects");
  output_supervision->clear();
  output_supervision->reserve(input.size());
  BaseFloat cur_weight = -1.0;
  for (int32 i = 0; i < input.size(); i++) {
    const CctcSupervision &src = *(input[i]);
    if (compactify && src.weight == cur_weight) {
      // Combine with current output
      KALDI_ASSERT(!output_supervision->empty());
      // append src.fst to output_supervision->fst.
      fst::Concat(&output_supervision->back().fst, src.fst);
      output_supervision->back().num_frames += src.num_frames;
      output_was_merged.back() = true;
    } else {
      output_supervision->resize(output_supervision->size() + 1);
      output_supervision->back() = src;
      cur_weight = src.weight;
      output_was_merged.push_back(false);
    }
  }
  KALDI_ASSERT(output_was_merged.size() == output_supervision->size());
  for (size_t i = 0; i < output_supervision->size(); i++) {
    if (output_was_merged[i]) {
      fst::StdVectorFst &out_fst = (*output_supervision)[i].fst;
      // The process of concatenation will have introduced epsilons.
      fst::RmEpsilon(&out_fst);
      SortBreadthFirstSearch(&out_fst);
    }
  }
}

void SplitIntoRanges(int32 num_frames,
                     int32 frames_per_range,
                     std::vector<int32> *range_starts) {
  if (frames_per_range > num_frames) {
    range_starts->clear();
    return;  // there is no room for even one range.
  }
  int32 num_ranges = num_frames  / frames_per_range,
      extra_frames = num_frames % frames_per_range;
  // this is a kind of heuristic.  If the number of frames we'd
  // be skipping is less than 1/4 of the frames_per_range, then
  // skip frames; otherwise, duplicate frames.
  // it's important that this is <=, not <, so that if
  // extra_frames == 0 and frames_per_range is < 4, we
  // don't insert an extra range.
  if (extra_frames <= frames_per_range / 4) {
    // skip frames.  we do this at start or end, or between ranges.
    std::vector<int32> num_skips(num_ranges + 1, 0);
    for (int32 i = 0; i < extra_frames; i++)
      num_skips[RandInt(0, num_ranges)]++;
    range_starts->resize(num_ranges);
    int32 cur_start = num_skips[0];
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start += num_skips[i + 1];
    }
    KALDI_ASSERT(cur_start == num_frames);
  } else {
    // duplicate frames.
    num_ranges++;
    int32 num_duplicated_frames = frames_per_range - extra_frames;
    // the way we handle the 'extra_frames' frames of output is that we
    // backtrack zero or more frames between outputting each pair of ranges, and
    // the total of these backtracks equals 'extra_frames'.
    std::vector<int32> num_backtracks(num_ranges, 0);
    for (int32 i = 0; i < num_duplicated_frames; i++) {
      // num_ranges - 2 below is not a bug.  we only want to backtrack
      // between ranges, not past the end of the last range (i.e. at
      // position num_ranges - 1).  we make the vector one longer to
      // simplify the loop below.
      num_backtracks[RandInt(0, num_ranges - 2)]++;
    }
    range_starts->resize(num_ranges);
    int32 cur_start = 0;
    for (int32 i = 0; i < num_ranges; i++) {
      (*range_starts)[i] = cur_start;
      cur_start += frames_per_range;
      cur_start -= num_backtracks[i];
    }
    KALDI_ASSERT(cur_start == num_frames);
  }
}

bool CctcSupervision::operator == (const CctcSupervision &other) const {
  return weight == other.weight && num_frames == other.num_frames &&
      label_dim == other.label_dim && fst::Equal(fst, other.fst);
}

void CctcSupervision::Check(const CctcTransitionModel &trans_mdl) const {
  if (weight <= 0.0)
    KALDI_ERR << "Weight should be positive.";
  if (num_frames <= 0)
    KALDI_ERR << "Invalid num-frames: " << num_frames;
  if (label_dim != trans_mdl.NumGraphLabels())
    KALDI_ERR << "Invalid label-dim: " << label_dim
              << ", expected " << trans_mdl.NumGraphLabels();
  std::vector<int32> state_times;
  if (num_frames != ComputeFstStateTimes(fst, &state_times))
    KALDI_ERR << "Num-frames does not match fst.";
}



}  // namespace ctc
}  // namespace kaldi
