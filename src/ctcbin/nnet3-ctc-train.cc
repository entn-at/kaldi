// nnet3bin/nnet3-ctc-train.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-cctc-training.h"
#include <ctime>
#include <time.h>
// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

int main(int argc, char *argv[]) {
  try {
    std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+ctc neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-merge-egs in\n"
        "the input pipeline.  This training program is single-threaded (best to\n"
        "use it with a GPU).  The --write-raw option controls whether the entire\n"
        "model including the transition-model, or just the neural net, is output.\n"
        "\n"
        "Usage:  nnet3-ctc-train [options] <model-in> <ctc-training-examples-in> (<model-out>|<raw-model-out>)\n"
        "\n"
        "nnet3-ctc-train 1.mdl 'ark:nnet3-merge-egs 1.cegs ark:-|' 2.raw\n";

    bool binary_write = true;
    bool write_raw = false;
    std::string use_gpu = "yes";
    NnetCctcTrainerOptions train_config;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("write-raw", &write_raw, "If true, write just the raw neural-net "
                "and not also the transition-model");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");

    train_config.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    KALDI_LOG << "Select GPU " << currentDateTime();
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    KALDI_LOG << "Read files " << currentDateTime();
    std::string cctc_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxfilename = po.GetArg(3);

    CctcTransitionModel trans_model;
    Nnet nnet;
    {
      bool binary;
      Input input(cctc_nnet_rxfilename, &binary);
      trans_model.Read(input.Stream(), binary);
      nnet.Read(input.Stream(), binary);
    }
    KALDI_LOG << "Trainer " << currentDateTime();
    
    NnetCctcTrainer trainer(train_config, trans_model, &nnet);

    KALDI_LOG << "SequentialNnetCctcExampleReader " << currentDateTime();
    SequentialNnetCctcExampleReader example_reader(examples_rspecifier);
std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    int i = 0;
    for (; !example_reader.Done(); example_reader.Next()) {
      KALDI_LOG << "Train " << i << "  " << currentDateTime();
      i++;
      trainer.Train(example_reader.Value());
    }
std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    KALDI_LOG << "Stats " << currentDateTime();
    bool ok = trainer.PrintTotalStats();

    KALDI_LOG << "Profile " << currentDateTime();
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    KALDI_LOG << "Write " << currentDateTime();
    if (write_raw) {
      WriteKaldiObject(nnet, nnet_wxfilename, binary_write);
      KALDI_LOG << "Wrote raw model to " << nnet_wxfilename << "  " << currentDateTime();
    } else {
      Output output(nnet_wxfilename, binary_write);
      trans_model.Write(output.Stream(), binary_write);
      nnet.Write(output.Stream(), binary_write);
      KALDI_LOG << "Wrote model to " << nnet_wxfilename << "  " << currentDateTime();
    }
    std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
//~ #if HAVE_CUDA==1
    //~ CuDevice::Instantiate().DeviceReset();
//~ #endif
std::cout << "currentDateTime()=" << currentDateTime() << std::endl;
    KALDI_LOG << "Ending after DeviceReset " << currentDateTime();
    return (ok ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
