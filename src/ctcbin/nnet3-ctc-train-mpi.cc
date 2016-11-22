// nnet3bin/nnet3-ctc-train.cc

// Copyright 2015  Johns Hopkins University (author: Daniel Povey)
// Copyright 2016  VoiceLab.ai (author: Paweł Rościszewski)

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

#include <mpi.h>
#include <string.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-cctc-training.h"
#include "nnet3/nnet-utils.h"

#define MAX_ARCHIVE_DIGITS  5
#define MPI_TAG 1

#define COMM_DIE 0
#define COMM_TWO_WAY 1
#define COMM_FORWARD 2
#define COMM_BACKWARD 3
#define COMM_NO_MODEL 4

int main(int argc, char *argv[]) {

    MPI_Status status;
    int my_rank, proc_count;

    MPI_Init(&argc-1, &argv+1);

    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using namespace kaldi::ctc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Train nnet3+ctc neural network parameters with backprop and stochastic\n"
        "gradient descent.  Minibatches are to be created by nnet3-merge-egs in\n"
        "the input pipeline. A C-like '%d' placeholder in the pipeline will be filled with consecutive\n"
        "integer values from the range 1-<num-archives>. This is a MPI version of single threaded nnet3-ctc-train,\n"
        "it spawns np - 1 (np is the MPI comm size, one process is master) slaves, each of which \n"
        "uses a GPU. The number of slaves actually performing computations, learning rate and shrink value \n"
        "are determined dynamically for each iteration. The master will average \n"
        "models from the slaves every <avg-freq> iteration.\n"
        "\n"
        "Usage:  nnet3-train-mpi [options] <raw-model-in> <training-examples-in> <out-model-dir> "
		"<current-iter> <num-iters> <num-archives-to-process> <num-archives-processed> <num-archives> "
        "<num-jobs-initial> <num-jobs-final> <initial-learning-rate> <final-learning-rate>"
        "<shrink> <shrink-threshold> <frame-subsampling-factor> <avg-freq>\n";

    bool binary_read;
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

    if (po.NumArgs() != 16) {
      po.PrintUsage();
      exit(1);
    }

    std::string cctc_nnet_rxfilename = po.GetArg(1),
        examples_rspecifier = po.GetArg(2),
        nnet_wxdir = po.GetArg(3);

    CctcTransitionModel trans_model;

    Input input(cctc_nnet_rxfilename, &binary_read);
    trans_model.Read(input.Stream(), binary_read);

    std::stringstream nnet_buffer_ostream;

    int comm_mode, archive, nnet_size, frame_shift;

    // MASTER
    if(!my_rank) {

        MPI_Request *requests = (MPI_Request *) calloc(proc_count - 1, sizeof(MPI_Request));
        int *nnet_sizes = (int *) calloc(proc_count - 1, sizeof(int));

        int current_iter = std::atoi(po.GetArg(4).c_str());
        int num_iters = std::atoi(po.GetArg(5).c_str());

        int num_archives_to_process = std::atoi(po.GetArg(6).c_str());
        int num_archives_processed = std::atoi(po.GetArg(7).c_str());
        int num_archives = std::atoi(po.GetArg(8).c_str());

        int num_jobs_initial = std::atoi(po.GetArg(9).c_str());
        int num_jobs_final = std::atoi(po.GetArg(10).c_str());

        double ilr = std::atof(po.GetArg(11).c_str());
        double flr = std::atof(po.GetArg(12).c_str());

        double shrink = std::atof(po.GetArg(13).c_str());
        double shrink_threshold = std::atof(po.GetArg(14).c_str());

        int frame_subsampling_factor = std::atoi(po.GetArg(15).c_str());

        int avg_freq = std::atoi(po.GetArg(16).c_str());

        Nnet nnet;
        nnet.Read(input.Stream(), binary_read);

        time_t start = time(NULL);

        int avg_cnt = 0;
        for(; current_iter != num_iters; current_iter++) {

            // TODO: find formula for the optimal number of jobs
            //int this_num_jobs = (int) (0.5 + num_jobs_initial + (num_jobs_final - num_jobs_initial) * (double) current_iter / (double) num_iters);
            int this_num_jobs = (int) (0.5 + num_jobs_initial + (num_jobs_final - num_jobs_initial) * (double) current_iter / ((double) num_iters / 2.0));
            if(this_num_jobs > num_jobs_final) this_num_jobs = num_jobs_final;

            KALDI_LOG << "Training neural net (pass " << current_iter << ") - inner loop, num jobs: " << this_num_jobs;

            double this_learning_rate = (current_iter + 1 >= num_iters ? flr : ilr*exp(num_archives_processed*log(flr/ilr)/num_archives_to_process)) * this_num_jobs;
            KALDI_LOG << "ilr " << ilr << " flr " << flr << " np " << num_archives_processed << " nt " << num_archives_to_process << " x " << current_iter;

            double this_shrink = 1.0;
            double sigmoid_mean = nnet.ComputeNonlinearityMean("SigmoidComponent");
        
            KALDI_LOG << "sigmoid mean " << sigmoid_mean << ", shrink threshold " << shrink_threshold << ", shrink " << shrink;
            if(sigmoid_mean > shrink_threshold) this_shrink = shrink;

            if(avg_freq == 1)
                comm_mode = COMM_TWO_WAY;
            else if((avg_cnt % avg_freq) == 0)
                comm_mode = COMM_FORWARD;
            else if((avg_cnt % avg_freq) == avg_freq - 1)
                comm_mode = COMM_BACKWARD;
            else comm_mode = COMM_NO_MODEL;

            KALDI_LOG << "On iteration " << current_iter << ", learning rate is " << this_learning_rate
                << ", shrink value is " << this_shrink << " and comm mode is " << comm_mode;

            nnet.Write(nnet_buffer_ostream, true);
            nnet_buffer_ostream.seekg(0, std::ios::end);
            nnet_size = nnet_buffer_ostream.tellg();
            nnet_buffer_ostream.seekg(0, std::ios::beg);

            for(int s = 1; s <= this_num_jobs; s++) {
                KALDI_LOG << "Master sending package to slave" << s << " out of " << this_num_jobs;
                int k = (num_archives_processed + s - 1);
                archive = k%num_archives + 1;
                frame_shift = (k/num_archives)%frame_subsampling_factor;

                MPI_Send(&comm_mode, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
                MPI_Send(&archive, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
                MPI_Send(&frame_shift, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
                MPI_Send(&current_iter, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
                MPI_Send(&this_learning_rate, 1, MPI_DOUBLE, s, MPI_TAG, MPI_COMM_WORLD);

                if(comm_mode == COMM_FORWARD || comm_mode == COMM_TWO_WAY) {
                    MPI_Send(&nnet_size, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
                    MPI_Send((void *) nnet_buffer_ostream.str().c_str(), nnet_size, MPI_BYTE, s, MPI_TAG, MPI_COMM_WORLD);
                }
            }
            nnet_buffer_ostream.str(std::string());
            num_archives_processed += this_num_jobs;

            for(int s = 1; s <= this_num_jobs; s++)
                MPI_Irecv(nnet_sizes + s - 1, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD, requests + s - 1);

            for(int result_count = 0; result_count < this_num_jobs; result_count++) {
                int request_completed;
                MPI_Waitany(this_num_jobs, requests, &request_completed, &status);

                nnet_size = nnet_sizes[request_completed];
                if(comm_mode == COMM_BACKWARD || comm_mode == COMM_TWO_WAY) {
                    char *nnet_buffer = new char[nnet_size];
                    MPI_Recv(nnet_buffer, nnet_size, MPI_BYTE, request_completed + 1, MPI_TAG, MPI_COMM_WORLD, &status);
                    std::istringstream nnet_buffer_istream(std::string(nnet_buffer, nnet_size));

		    KALDI_LOG << "< iteration " << current_iter << " averaging or setting next model starting after " << (time(NULL) - start) << " seconds";
                    if(result_count > 0) {
                        Nnet src_nnet;
                        src_nnet.Read(nnet_buffer_istream, true);
                        AddNnet(src_nnet, 1.0/this_num_jobs, &nnet);
                    }
                    else {
                        nnet.Read(nnet_buffer_istream, true);
                        ScaleNnet(1.0/this_num_jobs, &nnet);
                    }
		    KALDI_LOG << "< iteration " << current_iter << " averaging or setting next model ended after " << (time(NULL) - start) << " seconds";

                    delete nnet_buffer;
                }
            }

            ScaleNnet(this_shrink, &nnet);

            KALDI_LOG << "< iteration " << current_iter << " ends after " << (time(NULL) - start) << " seconds";

            if(comm_mode == COMM_BACKWARD || comm_mode == COMM_TWO_WAY) {                
                std::stringstream filename;
                filename << nnet_wxdir << (current_iter + 1) << ".mdl";
                
                if (write_raw) {
                    WriteKaldiObject(nnet, filename.str(), binary_write);
                }
                else {
                    Output output(filename.str(), binary_write);
                    trans_model.Write(output.Stream(), binary_write);
                    nnet.Write(output.Stream(), binary_write);
                }
                KALDI_LOG << "Wrote model to " << filename;
            }

            avg_cnt++;
        }

        int die_command = COMM_DIE;
        for(int s = 1; s < proc_count; s++)
            MPI_Send(&die_command, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD);
    }
    // SLAVE
    else {
        NnetCctcTrainer *trainer;
        Nnet *nnet = new Nnet();
        bool gpu_instantiated = false;

        while(true) {
            MPI_Recv(&comm_mode, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);

            if(comm_mode == COMM_DIE) {
                break;
            }
            
#if HAVE_CUDA==1
            if(!gpu_instantiated) {
                CuDevice::Instantiate().SelectGpuId(use_gpu);
                gpu_instantiated = true;
            }
#endif

            MPI_Recv(&archive, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);

            MPI_Recv(&frame_shift, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);
            int current_iter;
            MPI_Recv(&current_iter, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);
            double this_learning_rate;
            MPI_Recv(&this_learning_rate, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);
            
            KALDI_LOG << "Slave " << my_rank << " processes archive " << archive << ", current iter "
                      << current_iter << ", frame shift " << frame_shift << ", learning rate " << this_learning_rate;

            if(comm_mode == COMM_FORWARD || comm_mode == COMM_TWO_WAY) {
                MPI_Recv(&nnet_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);
                char *nnet_buffer = new char[nnet_size];
                MPI_Recv(nnet_buffer, nnet_size, MPI_BYTE, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, &status);
                std::istringstream nnet_buffer_istream(std::string(nnet_buffer, nnet_size));

                nnet->Read(nnet_buffer_istream, true);

                if(!trainer) trainer = new NnetCctcTrainer(train_config, trans_model, nnet);
                else trainer->setupNewIteration(nnet);

                delete nnet_buffer;
            }
            else trainer->setupNewIteration(NULL);

            char *examples_rspecifier_formatted = new char[examples_rspecifier.length() + MAX_ARCHIVE_DIGITS];
            sprintf(examples_rspecifier_formatted, examples_rspecifier.c_str(), frame_shift, archive, current_iter);

            SetLearningRate(this_learning_rate, nnet);

            SequentialNnetCctcExampleReader *example_reader = new SequentialNnetCctcExampleReader(examples_rspecifier_formatted);

            for (; !example_reader->Done(); example_reader->Next())
                trainer->Train(example_reader->Value());

            delete example_reader;
            delete examples_rspecifier_formatted;

            trainer->PrintTotalStats();

            nnet->Write(nnet_buffer_ostream, true);
            nnet_buffer_ostream.seekg(0, std::ios::end);
            nnet_size = nnet_buffer_ostream.tellg();
            nnet_buffer_ostream.seekg(0, std::ios::beg);

            MPI_Send(&nnet_size, 1, MPI_INT, 0, MPI_TAG, MPI_COMM_WORLD);
            if(comm_mode == COMM_BACKWARD || comm_mode == COMM_TWO_WAY) {
                MPI_Send((void *)nnet_buffer_ostream.str().c_str(), nnet_size, MPI_BYTE, 0, MPI_TAG, MPI_COMM_WORLD);
            }

            nnet_buffer_ostream.str(std::string());

#if HAVE_CUDA == 1
                CuDevice::Instantiate().PrintProfile();
#endif
        }

        delete trainer;
        KALDI_LOG << "Slave " << my_rank << " finished";

#if HAVE_CUDA==1
    	CuDevice::Instantiate().DeviceReset();
#endif
    }

} catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
}

fflush(stdout);
MPI_Finalize();
exit(0);
}
