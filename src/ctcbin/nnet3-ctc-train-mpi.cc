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
#include <pthread.h>
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "nnet3/nnet-cctc-training.h"
#include "nnet3/nnet-utils.h"

#define MAX_ARCHIVE_DIGITS  5
#define MAX_EXAMPLES_IN_ARCHIVE 5000

#define MPI_TAG 1

#define COMM_DIE -1

typedef struct iter_params_t {
    int num_jobs;
    int archive;
    int iter_no;
    int frame_shift;
    double learning_rate;
    kaldi::nnet3::SequentialNnetCctcExampleReader *reader;
    kaldi::nnet3::NnetCctcExample *buffer;
    int egs_cnt;
} iter_params_t;


int compute_num_jobs(int iter_no, int num_jobs_initial, int num_jobs_final, int max_iters) {
    int num_jobs = (int) (0.5 + num_jobs_initial + (num_jobs_final - num_jobs_initial) * (double) iter_no / ((double) max_iters / 2.0));
    if(num_jobs > num_jobs_final) num_jobs = num_jobs_final;
    return num_jobs;
}

void send_iter_params(iter_params_t iter_params, int slave_no) {
    KALDI_LOG << " sending ar " << iter_params.archive << " fs " << iter_params.frame_shift
                << " itno " << iter_params.iter_no << " lr " << iter_params.learning_rate;
    MPI_Send(&(iter_params.archive), 1, MPI_INT, slave_no, MPI_TAG, MPI_COMM_WORLD);
    MPI_Send(&(iter_params.frame_shift), 1, MPI_INT, slave_no, MPI_TAG, MPI_COMM_WORLD);
    MPI_Send(&(iter_params.iter_no), 1, MPI_INT, slave_no, MPI_TAG, MPI_COMM_WORLD);
    MPI_Send(&(iter_params.learning_rate), 1, MPI_DOUBLE, slave_no, MPI_TAG, MPI_COMM_WORLD);    
}

void receive_iter_params(iter_params_t *iter_params, int source) {
    MPI_Recv(&(iter_params->archive), 1, MPI_INT, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(iter_params->frame_shift), 1, MPI_INT, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(iter_params->iter_no), 1, MPI_INT, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&(iter_params->learning_rate), 1, MPI_DOUBLE, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int write_nnet_to_buffer(kaldi::nnet3::Nnet *nnet, std::stringstream *buffer) {
    nnet->Write(*buffer, true);
    buffer->seekg(0, std::ios::end);
    int nnet_size = buffer->tellg();
    buffer->seekg(0, std::ios::beg);
    return nnet_size;
}

void write_nnet_to_file(kaldi::nnet3::Nnet *nnet, kaldi::ctc::CctcTransitionModel *trans_model,
                        std::string filename, bool write_raw, bool binary_write) {
    if (write_raw) {
        kaldi::WriteKaldiObject(*nnet, filename, binary_write);
    }
    else {
        kaldi::Output output(filename, binary_write);
        trans_model->Write(output.Stream(), binary_write);
        nnet->Write(output.Stream(), binary_write);
    }
    KALDI_LOG << "Wrote model to " << filename;
}

void reset_buffer(std::stringstream *buffer) {
    buffer->str(std::string());
}

void send_nnet(int size, std::stringstream *buffer, int destination) {
    MPI_Send(&size, 1, MPI_INT, destination, MPI_TAG, MPI_COMM_WORLD);
    MPI_Send((void *) buffer->str().c_str(), size, MPI_BYTE, destination, MPI_TAG, MPI_COMM_WORLD);
}

void receive_nnet(int size, int source, kaldi::nnet3::Nnet *dest_nnet) {
    char *nnet_buffer = new char[size];
    MPI_Recv(nnet_buffer, size, MPI_BYTE, source, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::istringstream nnet_buffer_istream(std::string(nnet_buffer, size));
    dest_nnet->Read(nnet_buffer_istream, true);
    delete nnet_buffer;
}

void *prefetch_egs(void *voidArgs) {
    iter_params_t *args = (iter_params_t *) voidArgs;

    for (args->egs_cnt = 0; !args->reader->Done(); args->reader->Next(), args->egs_cnt++)
        args->buffer[args->egs_cnt] = args->reader->Value();
        
    args->reader->Close();
    
    delete args->reader;
    pthread_exit(NULL);
}

void start_egs_prefetching(std::string examples_rspecifier, iter_params_t *iter_params, kaldi::nnet3::NnetCctcExample* buffer, pthread_t *thread) {
    char *examples_rspecifier_formatted = new char[examples_rspecifier.length() + MAX_ARCHIVE_DIGITS];
    sprintf(examples_rspecifier_formatted, examples_rspecifier.c_str(), iter_params->frame_shift, iter_params->archive, iter_params->iter_no);
    kaldi::nnet3::SequentialNnetCctcExampleReader *example_reader = new kaldi::nnet3::SequentialNnetCctcExampleReader(examples_rspecifier_formatted);
    delete examples_rspecifier_formatted;

    iter_params->reader = example_reader;
    iter_params->buffer = buffer;
    pthread_create(thread, NULL, &prefetch_egs, iter_params);

    KALDI_LOG << " thread created";
}

void compute_training_params(int iter_no, int num_jobs, int num_archives_processed, int num_archives_to_process,
                             double ilr, double flr, int max_iters, iter_params_t *ret) {
    ret->iter_no = iter_no;
    ret->learning_rate = (iter_no + 1 >= max_iters ? flr : ilr*exp(num_archives_processed*log(flr/ilr)/num_archives_to_process)) * num_jobs;
    KALDI_LOG << "ilr " << ilr << " flr " << flr << " np " << num_archives_processed << " nt " << num_archives_to_process << " x " << iter_no;
    KALDI_LOG << "Computed parameters for iteration " << iter_no << " - inner loop, num jobs: " << num_jobs;
}

void compute_archive_params(int slave_no, int num_archives_processed, int num_archives,
                            int frame_subsampling_factor, iter_params_t *ret) {
    int k = (num_archives_processed + slave_no - 1);
    ret->archive = k%num_archives + 1;
    ret->frame_shift = (k/num_archives)%frame_subsampling_factor;
}

int distribute_archive_params(int iter_no, int *num_archives_processed, int num_archives, int num_archives_to_process,
                              double initial_learning_rate, double final_learning_rate, int max_iters,
                              int num_jobs_initial, int num_jobs_final, int frame_subsampling_factor) {
    int num_jobs = compute_num_jobs(iter_no, num_jobs_initial, num_jobs_final, max_iters);
    
    iter_params_t iter_params;
    compute_training_params(iter_no, num_jobs, *num_archives_processed, num_archives_to_process,
                            initial_learning_rate, final_learning_rate, max_iters, &iter_params);
            
    for(int s = 1; s <= num_jobs; s++) {
        compute_archive_params(s, *num_archives_processed, num_archives, frame_subsampling_factor, &iter_params);
        send_iter_params(iter_params, s);
    }
    *num_archives_processed += num_jobs;

    return num_jobs;
}

int main(int argc, char *argv[]) {
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
        "are determined dynamically for each iteration.\n"
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

    if (po.NumArgs() != 15) {
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

    int nnet_size;
    
    time_t start = time(NULL);

    // MASTER
    if(!my_rank) {

        MPI_Request *requests = (MPI_Request *) calloc(proc_count - 1, sizeof(MPI_Request));
        int *nnet_sizes = (int *) calloc(proc_count - 1, sizeof(int));

        int start_iter = std::atoi(po.GetArg(4).c_str());
        int max_iters = std::atoi(po.GetArg(5).c_str());

        int num_archives_to_process = std::atoi(po.GetArg(6).c_str());
        int num_archives_processed = std::atoi(po.GetArg(7).c_str());
        int num_archives = std::atoi(po.GetArg(8).c_str());

        int num_jobs_initial = std::atoi(po.GetArg(9).c_str());
        int num_jobs_final = std::atoi(po.GetArg(10).c_str());

        double initial_learning_rate = std::atof(po.GetArg(11).c_str());
        double final_learning_rate = std::atof(po.GetArg(12).c_str());

        double shrink = std::atof(po.GetArg(13).c_str());
        double shrink_threshold = std::atof(po.GetArg(14).c_str());

        int frame_subsampling_factor = std::atoi(po.GetArg(15).c_str());

        Nnet nnet;
        nnet.Read(input.Stream(), binary_read);

        int num_jobs_computing = distribute_archive_params(start_iter, &num_archives_processed, num_archives,
                                                           num_archives_to_process, initial_learning_rate, final_learning_rate,
                                                           max_iters, num_jobs_initial, num_jobs_final, frame_subsampling_factor);

        for(int current_iter = start_iter; current_iter != max_iters; current_iter++) {

            int num_jobs_prefetching;

            if(current_iter == (max_iters - 1)) {
                iter_params_t iter_params;
                iter_params.iter_no = COMM_DIE;
                for(int s = 1; s < proc_count; s++)
                    send_iter_params(iter_params, s);
            }
            else {
                num_jobs_prefetching = distribute_archive_params(current_iter + 1, &num_archives_processed, num_archives,
                                                                 num_archives_to_process, initial_learning_rate, final_learning_rate,
                                                                 max_iters, num_jobs_initial, num_jobs_final, frame_subsampling_factor);
            }

            double this_shrink = 1.0;
            double sigmoid_mean = nnet.ComputeNonlinearityMean("SigmoidComponent");
        
            KALDI_LOG << "sigmoid mean " << sigmoid_mean << ", shrink threshold " << shrink_threshold << ", shrink " << shrink;
            if(sigmoid_mean > shrink_threshold) this_shrink = shrink;
            KALDI_LOG << "On iteration " << current_iter << ", shrink value is " << this_shrink;

            int nnet_size = write_nnet_to_buffer(&nnet, &nnet_buffer_ostream);

            for (int s = 1; s <= num_jobs_computing; s++)
                send_nnet(nnet_size, &nnet_buffer_ostream, s);

            reset_buffer(&nnet_buffer_ostream);

            for(int s = 1; s <= num_jobs_computing; s++)
                MPI_Irecv(nnet_sizes + s - 1, 1, MPI_INT, s, MPI_TAG, MPI_COMM_WORLD, requests + s - 1);

            for(int result_count = 0; result_count < num_jobs_computing; result_count++) {
                int request_completed;
                MPI_Waitany(num_jobs_computing, requests, &request_completed, MPI_STATUS_IGNORE);

                nnet_size = nnet_sizes[request_completed];

                if(result_count > 0) {
                    Nnet src_nnet;
                    receive_nnet(nnet_size, request_completed + 1, &src_nnet);
                    AddNnet(src_nnet, 1.0/num_jobs_computing, &nnet);
                }
                else {
                    receive_nnet(nnet_size, request_completed + 1, &nnet);
                    ScaleNnet(1.0/num_jobs_computing, &nnet);
                }
            }

            ScaleNnet(this_shrink, &nnet);

            KALDI_LOG << "< iteration " << current_iter << " ends after " << (time(NULL) - start) << " seconds";

            std::stringstream filename;
            filename << nnet_wxdir << (current_iter + 1) << ".mdl";
            write_nnet_to_file(&nnet, &trans_model, filename.str(), write_raw, binary_write);

            num_jobs_computing = num_jobs_prefetching;
        }
    }
    // SLAVE
    else {
        NnetCctcTrainer *trainer = NULL;
        Nnet *nnet = new Nnet();
        NnetCctcExample *prefetching_egs_buffer = (NnetCctcExample *) calloc(MAX_EXAMPLES_IN_ARCHIVE, sizeof(NnetCctcExample)),
                         *computing_egs_buffer = (NnetCctcExample *) calloc(MAX_EXAMPLES_IN_ARCHIVE, sizeof(NnetCctcExample)),
                         *tmp_egs_buffer;
        iter_params_t prefetching_iter_params, computing_iter_params;
        pthread_t prefetching_thread;

        bool gpu_instantiated = false;

        receive_iter_params(&computing_iter_params, 0);
        if(computing_iter_params.archive != COMM_DIE) {
            start_egs_prefetching(examples_rspecifier, &computing_iter_params, computing_egs_buffer, &prefetching_thread);
            KALDI_LOG << "Before join";
            pthread_join(prefetching_thread, NULL);
            KALDI_LOG << "After join";
        }

        while(computing_iter_params.iter_no != COMM_DIE) {
            receive_iter_params(&prefetching_iter_params, 0);

            if(prefetching_iter_params.iter_no != COMM_DIE)
                start_egs_prefetching(examples_rspecifier, &prefetching_iter_params, prefetching_egs_buffer, &prefetching_thread);

#if HAVE_CUDA==1
            if(!gpu_instantiated) {
                CuDevice::Instantiate().SelectGpuId(use_gpu);
                gpu_instantiated = true;
            }
#endif
            
            SetLearningRate(computing_iter_params.learning_rate, nnet);

            KALDI_LOG << "Slave " << my_rank << " processes archive " << computing_iter_params.archive << ", current iter "
                      << computing_iter_params.iter_no << ", frame shift " << computing_iter_params.frame_shift << ", learning rate " << computing_iter_params.learning_rate;

            MPI_Recv(&nnet_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            receive_nnet(nnet_size, 0, nnet);

            if(!trainer) trainer = new NnetCctcTrainer(train_config, trans_model, nnet);
            else trainer->setupNewIteration(nnet);

            int example_cnt = computing_iter_params.egs_cnt;
                
            KALDI_LOG << "< iteration " << computing_iter_params.iter_no << " slave training on reading " << computing_iter_params.egs_cnt <<
                         " examples after " << (time(NULL) - start) << " seconds";

            for(int current_example = 0; current_example != example_cnt; current_example++)
                trainer->Train(computing_egs_buffer[current_example]);

            trainer->PrintTotalStats();

            int nnet_size = write_nnet_to_buffer(nnet, &nnet_buffer_ostream);
            KALDI_LOG << " slave sending " << nnet_size;
            send_nnet(nnet_size, &nnet_buffer_ostream, 0);
            reset_buffer(&nnet_buffer_ostream);

#if HAVE_CUDA == 1
            CuDevice::Instantiate().PrintProfile();
#endif

            if(prefetching_iter_params.archive != COMM_DIE) {
                KALDI_LOG << "Before join";
                pthread_join(prefetching_thread, NULL);
                KALDI_LOG << "After join";
            }

            computing_iter_params = prefetching_iter_params;

            tmp_egs_buffer = computing_egs_buffer;
            computing_egs_buffer = prefetching_egs_buffer;
            prefetching_egs_buffer = tmp_egs_buffer;
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
