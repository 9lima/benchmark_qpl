#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <chrono>
#include <thread>
#include <bit>
#include <bitset>
#include <atomic>
#include <future>  
#include "qpl/qpl.h"


void insert_random(std::vector<uint8_t>* vec, size_t size, size_t low = 0, size_t high = 255);
void benchmark_scan(qpl_path_t execution_path, std::span<const uint8_t> src, uint64_t * count);
void benchmark_multithreaded(qpl_path_t execution_path, std::vector<uint8_t>& source, size_t num_threads);
void cpu_find_bits(const std::vector<uint8_t>& vec, size_t num_threads);

constexpr uint32_t input_bit_width = 8;
constexpr uint32_t value_to_find = 48;

void benchmark_scan(qpl_path_t execution_path, std::span<const uint8_t> src, uint64_t * count)
{
    uint32_t job_size = 0;
    qpl_status status = qpl_get_job_size(execution_path, &job_size);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_get_job_size failed: " + std::to_string(status));
    auto job_buffer = std::make_unique<uint8_t[]>(job_size);
    auto * job = reinterpret_cast<qpl_job *>(job_buffer.get());
    status = qpl_init_job(execution_path, job);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_init_job failed: " + std::to_string(status));

    const uint64_t num_elements = (uint64_t(src.size()) * 8) / input_bit_width;
    std::vector<uint8_t> dst((num_elements + 7) / 8);

    job->op                 = qpl_op_scan_eq;
    job->next_in_ptr        = const_cast<uint8_t *>(src.data());
    job->available_in       = static_cast<uint32_t>(src.size());
    job->src1_bit_width     = input_bit_width;
    job->num_input_elements = static_cast<uint32_t>(num_elements);
    job->next_out_ptr       = dst.data();
    job->available_out      = static_cast<uint32_t>(dst.size());
    job->out_bit_width      = qpl_ow_nom;
    job->param_low          = value_to_find;
    job->flags              = QPL_FLAG_OMIT_CHECKSUMS;

    status = qpl_execute_job(job);
    const uint32_t matches = job->sum_value;
    qpl_fini_job(job);
    if (status != QPL_STS_OK)
        throw std::runtime_error("qpl_execute_job failed: " + std::to_string(status));

    std::atomic_ref<uint64_t>(*count).fetch_add(matches, std::memory_order_relaxed);
}

void benchmark_multithreaded(qpl_path_t execution_path, std::vector<uint8_t>& source, size_t num_threads)
{    
    const size_t source_size = source.size();
    if (source_size == 0 || num_threads == 0)
        return;

    uint64_t count{0};
    const size_t chunk_size = (source_size + num_threads - 1) / num_threads;
    std::span<const uint8_t> all{source};
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    auto start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_threads; ++i)
    {
        const size_t s = std::min<size_t>(source_size, i * chunk_size);
        const size_t e = std::min<size_t>(source_size, s + chunk_size);
        if (s >= e)
            continue;
        futures.push_back(std::async(std::launch::async, benchmark_scan,
                                     execution_path, all.subspan(s, e - s), &count));
    }

    for (auto & f : futures)
        f.get();
    auto end_time = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    double seconds         = elapsed_ms.count() / 1000.0;
    double throughput_mbps = (source_size / (1024.0 * 1024.0)) / seconds;
    std::cout << "Threads: " << num_threads
              << "  Time: " << elapsed_ms.count() << " ms"
              << "  Throughput: " << throughput_mbps << " MB/s\n"
              << "  Matched Items: " << count << "\n";
}

void cpu_find_bits(const std::vector<uint8_t>& vec, size_t num_threads)
{
    const size_t n = vec.size();
    if (n == 0 || num_threads == 0) return;

    const size_t chunk = (n + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    uint64_t count{0};

    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < num_threads; ++i) {
        const size_t s = std::min(n, i * chunk);
        const size_t e = std::min(n, s + chunk);
        if (s >= e) continue;
        threads.emplace_back([&vec, &count, s, e] {
            uint64_t local = 0;
            for (size_t j = s; j < e; ++j)
                local += (vec[j] == value_to_find);
            std::atomic_ref<uint64_t>(count).fetch_add(local, std::memory_order_relaxed);
        });
    }
    for (auto& t : threads) t.join();
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Linear Search CPU Threads: " << num_threads << "  Time: " << ms << " ms"
              << "  Throughput: " << (n / (1024.0 * 1024.0)) / (ms / 1000.0)
              << " MB/s  Matched Items: " << count << "\n";
}



void insert_random(std::vector<uint8_t>* vec, size_t size, size_t low = 0, size_t high = 255) {
    vec->clear();
    vec->resize(size);
    std::mt19937 gen{ std::random_device{}() };
    std::uniform_int_distribution<int> dist(low, high);
    
    for (auto& byte : *vec) {
        byte = static_cast<uint8_t>(dist(gen));
    }
    vec->back() = value_to_find;
}

int main(int argc, char** argv) {
    std::cout << "Intel(R) Query Processing Library version is " << qpl_get_library_version() << ".\n";
    qpl_path_t execution_path = qpl_path_hardware;
    if (parse_execution_path(argc, argv, &execution_path, 0) != 0) {
        return 1;
    }
    
    std::vector<uint8_t> source;
    std::vector<uint32_t> source_sizes = {128 * 1024 * 1024, 256 * 1024 * 1024, 512 * 1024 * 1024, 1 * 1024 * 1024 * 1024};
    std::vector<size_t> thread_counts = {1, 2, 4, 8, 16, 32, 64, 96, 128};
        
    for (uint32_t source_size : source_sizes) {
        insert_random(&source, source_size);
        for (size_t num_threads : thread_counts) {
            benchmark_multithreaded(execution_path, source, num_threads);
            cpu_find_bits(source, num_threads);
        }
    }    
    return 0;
}
