#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <chrono>
#include <thread>
#include <future>  
#include "qpl/qpl.h"

#include "utils.hpp" // For argument parsing

constexpr uint32_t input_bit_width     = 8;
constexpr uint32_t value_to_find       = 48;

std::vector<uint8_t> benchmark_scan(qpl_path_t execution_path,
                                    std::span<const uint8_t> src)
{
    uint32_t job_size = 0;
    qpl_status status = qpl_get_job_size(execution_path, &job_size);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_get_job_size failed: " + std::to_string(status));
    auto job_buffer = std::make_unique<uint8_t[]>(job_size);
    auto* job = reinterpret_cast<qpl_job*>(job_buffer.get());
    status = qpl_init_job(execution_path, job);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_init_job failed: " + std::to_string(status));

    std::vector<uint8_t> dst;
    dst.resize((src.size() + 7) / 8));
  
    job->op                 = qpl_op_scan_eq;
    job->next_in_ptr        = const_cast<uint8_t*>(src.data());
    job->available_in       = static_cast<uint32_t>(src.size());
    job->src1_bit_width     = input_bit_width;
    job->num_input_elements = static_cast<uint32_t>(src.size());
    job->next_out_ptr       = dst.data();
    job->available_out      = static_cast<uint32_t>(dst.size());
    job->out_bit_width      = qpl_ow_nom;
    job->param_low          = value_to_find;
    job->flags              = QPL_FLAG_OMIT_CHECKSUMS | QPL_FLAG_OMIT_AGGREGATES;

    status = qpl_execute_job(job);
    qpl_fini_job(job);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_execute_job failed: " + std::to_string(status));

    return dst;
}

void benchmark_multithreaded(qpl_path_t execution_path, uint32_t source_size, size_t num_threads) {
    std::vector<uint8_t> source;
    insert_random(&source, source_size);
    source.back() = value_to_find;                    // ensure at least one match
    std::vector<uint8_t> destination;
    destination.reserve((source_size + 7) / 8);

    const size_t chunk_size = (source_size + num_threads - 1) / num_threads;
    std::span<const uint8_t> all{source};
    std::vector<std::future<std::vector<uint8_t>>> futures;
    std::vector<size_t> counts;                       // elements in each chunk

    for (size_t i = 0; i < num_threads; ++i) {
        const size_t start = std::min<size_t>(source_size, i * chunk_size);
        const size_t count = std::min<size_t>(chunk_size, source_size - start);

        counts.push_back(count);
        futures.push_back(std::async(std::launch::async, benchmark_scan,
                                     execution_path, all.subspan(start, count)));
    }

    size_t total_bits = 0;
    for (size_t i = 0; i < futures.size(); ++i) {     // original order
        auto bits = futures[i].get();
        const size_t count = counts[i];
        const size_t shift = total_bits % 8;          // bits already used in destination's last byte

        if (count % 8) bits.back() &= uint8_t((1u << (count % 8)) - 1);  // clear padding bits

        if (shift == 0) {
            destination.insert(destination.end(), bits.begin(), bits.end());
        } else {
            for (uint8_t b : bits) {
                destination.back() |= uint8_t(b << shift);          // fill the gap
                destination.push_back(uint8_t(b >> (8 - shift)));   // leftover bits start a new byte
            }
        }
        total_bits += count;
        destination.resize((total_bits + 7) / 8);     // drop the spare byte if it isn't needed
    }
}


void insert_random(std::vector<uint8_t>* vec,
                   size_t size,
                   size_t low = 0,
                   size_t high = 255) {
    vec->resize(size);

    std::mt19937 gen{ std::random_device{}() };
    std::uniform_int_distribution<int> dist(low, high);

    for (auto& byte : *vec) {
        byte = static_cast<uint8_t>(dist(gen));
    }
}


int main(int argc, char** argv) {
    std::cout << "Intel(R) Query Processing Library version is " << qpl_get_library_version() << ".\n";
    qpl_path_t execution_path = qpl_path_software;
    if (parse_execution_path(argc, argv, &execution_path, 0) != 0) {
        return 1;
    }

    std::vector<uint32_t> source_sizes = {128 * 1024 * 1024,
                                          256 * 1024 * 1024, 512 * 1024 * 1024, 1 * 1024 * 1024 * 1024};

    std::vector<int> thread = {1, 2, 4, 8, 16, 32, 64, 96, 128};
    for (uint32_t size : source_sizes) {
        for (auto num_threads: thread) {
            benchmark_multithreaded(execution_path, size, num_threads);
        }
    }
    return 0;
}
