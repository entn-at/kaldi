#!/bin/bash
# Copyright 2019 Ewald Enzinger
#
# Apache 2.0.

. ./cmd.sh
. ./path.sh
set -e

stage=-1
sad_nnet_dir=exp/segmentation_1a/tdnn_stats_asr_sad_1a
dia_nnet_dir=exp/xvector_nnet_1a
asr_nnet_dir=exp/kaldi-generic-de-tdnn_f-r20190328
nj=1

# Prepara data
if [ $stage -le -1 ]; then
  for name in cba; do
    mkdir -p data/${name}
    touch data/${name}/wav.scp
    touch data/${name}/utt2spk
    touch data/${name}/spk2utt
    touch data/${name}/reco2num_spk
    touch data/${name}/reco2file_and_channel
    for f in /home/ewald.enzinger/atspeech/${name}/*.mp3; do 
      TMP=$(basename "$f" .mp3 | awk -F"[-_ ]" '{print $1""$2""$3""$4""$5}')
      echo "$TMP sox \"$f\" -t wav -r 16k -e signed-integer -b 16 - channels 1 |" >>data/${name}/wav.scp
      echo "$TMP $TMP" >>data/${name}/utt2spk
      echo "$TMP $TMP" >>data/${name}/spk2utt
      echo "$TMP 2" >>data/${name}/reco2num_spk
      echo "$TMP $TMP 1" >>data/${name}/reco2file_and_channel
    done
  done
fi

# Perform Speech Activity Detection
if [ $stage -le 0 ]; then
  for name in cba; do
    steps/segmentation/detect_speech_activity.sh --cmd "$train_cmd" --stage 0 --nj $nj \
      --extra-left-context 79 --extra-right-context 21 \
      --extra-left-context-initial 0 --extra-right-context-final 0 \
      --frames-per-chunk 150 --mfcc-config conf/mfcc_sad.conf \
      data/${name} $sad_nnet_dir \
      mfcc_sad tmp/${name} data/${name}
    utils/fix_data_dir.sh data/${name}_seg
  done
fi

# Extract features for diarization
if [ $stage -le 1 ]; then
  for name in cba; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc_dia.conf --nj $nj \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/${name}_seg exp/make_mfcc mfcc_dia
    utils/fix_data_dir.sh data/${name}_seg
  done
fi

# Prepare features for diarization
if [ $stage -le 2 ]; then
  for name in cba; do
    local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
      data/${name}_seg data/${name}_cmn exp/${name}_cmn
    cp data/${name}_seg/segments data/${name}_cmn/
    cp data/${name}_seg/reco2file_and_channel data/${name}_cmn/
    utils/fix_data_dir.sh data/${name}_cmn
  done
fi

# Extract DNN speaker embeddings from short sections of speech
if [ $stage -le 3 ]; then
  for name in cba; do
    diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" \
      --nj $nj --window 1.5 --period 0.75 --apply-cmn false --use-gpu false \
     --min-segment 0.5 $dia_nnet_dir \
      data/${name}_cmn exp/xvector_nnet_1a/xvectors_${name}
  done
fi

# Calculate comparison scores between all DNN speaker embeddings in each recording
if [ $stage -le 4 ]; then
  for name in cba; do
    diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      --nj $nj exp/xvector_nnet_1a/xvectors_${name} exp/xvector_nnet_1a/xvectors_${name} \
      exp/xvector_nnet_1a/xvectors_${name}/plda_scores
  done
fi

# Cluster embeddings based on scores to determine speaker clusters
if [ $stage -le 5 ]; then
  for name in cba; do
    diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $nj \
      --reco2num-spk data/${name}/reco2num_spk \
      exp/xvector_nnet_1a/xvectors_${name}/plda_scores exp/xvector_nnet_1a/xvectors_${name}/plda_scores_num_spk
  done
fi

# Perform speech recognition
if [ $stage -le 6 ]; then
  for name in cba; do
    steps/online/nnet3/decode.sh --cmd "$train_cmd" --nj $nj --acwt 1.0 --post-decode-acwt 10.0 \
      --skip-scoring true \
      $asr_nnet_dir/graph data/${name}_cmn $asr_nnet_dir/decode_${name}
  done
fi

# Get automatic transcripts in Conversation Time Mark (CTM) format
if [ $stage -le 7 ]; then
  for data in cba; do
    mkdir -p exp/ctm_${data}
    local/get_ctm.sh --frame-shift 0.03 data/${data}_cmn $asr_nnet_dir/data/lang $asr_nnet_dir/decode_${data}
  done
fi

# Reformat CTM files and split them according to recording IDs
if [ $stage -le 8 ]; then
  for data in cba; do
    mkdir -p exp/ctm_${data}
    awk '{print $2}' data/${data}_cmn/segments | sort -u |\
    while read rec; do
      grep "$rec" $asr_nnet_dir/decode_${data}/score_10/${data}_cmn.ctm | tr "-" "_" >exp/ctm_${data}/$rec.ctm
    done
  done
fi
