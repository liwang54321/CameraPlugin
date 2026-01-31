#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "CameraPluginOrin.hpp"
int main(int argc, char **argv) {
    CameraPluginParams params;
    params.enable_rgb_dump = false;
    params.enable_rtp_dump = false;
    params.enable_yuv_dump = false;
    params.enable_cpu_sink = false;
    CameraPluginOrin plugin(params);
    if (plugin.loadConfig(argv[1]) != 0) {
        std::cout << "open " << argv[1] << " failed\n";
        return -1;
    }
    bool file_save = argc > 2 ? true : false;

    if (plugin.registerStreamCallback(
            0, [&](uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
                std::cout << "Get Camera " << cameraId << " Size " << size << " timestamp "
                          << timestamp << "\n";
                if (file_save) {

                    static uint32_t count = 0;
                    void *data = 0;
                    data = malloc(size);
                    if (data == nullptr) {
                        std::cout << "malloc data failed size " << size << "\n";
                        return -1;
                    }

                    auto result =
                        cudaMemcpy(data, payload, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                    if (result != cudaSuccess) {
                        std::cout << "Cuda memcpy failed " << result << std::endl;
                        return -1;
                    }

                    std::ofstream ofs("out_" + std::to_string(cameraId) + "_" +
                                          std::to_string(count++) + ".raw",
                                      std::ios::trunc);
                    ofs.write((char *)data, size);
                    free(data);
                }
            })) {
        std::cout << "register  stream callback failed\n";
        return -1;
    }

    if (plugin.registerStreamCallback(
            1, [&](uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
                std::cout << "Get Camera " << cameraId << " Size " << size << " timestamp "
                          << timestamp << "\n";
                if (file_save) {

                    static uint32_t count = 0;
                    void *data = 0;
                    data = malloc(size);
                    if (data == nullptr) {
                        std::cout << "malloc data failed size " << size << "\n";
                        return -1;
                    }

                    auto result =
                        cudaMemcpy(data, payload, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                    if (result != cudaSuccess) {
                        std::cout << "Cuda memcpy failed " << result << std::endl;
                        return -1;
                    }

                    std::ofstream ofs("out_" + std::to_string(cameraId) + "_" +
                                          std::to_string(count++) + ".raw",
                                      std::ios::trunc);
                    ofs.write((char *)data, size);
                    free(data);
                }
            })) {
        std::cout << "register  stream callback failed\n";
        return -1;
    }

    if (plugin.registerStreamCallback(
            2, [&](uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
                std::cout << "Get Camera " << cameraId << " Size " << size << " timestamp "
                          << timestamp << "\n";
                if (file_save) {

                    static uint32_t count = 0;
                    void *data = 0;
                    data = malloc(size);
                    if (data == nullptr) {
                        std::cout << "malloc data failed size " << size << "\n";
                        return -1;
                    }

                    auto result =
                        cudaMemcpy(data, payload, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                    if (result != cudaSuccess) {
                        std::cout << "Cuda memcpy failed " << result << std::endl;
                        return -1;
                    }

                    std::ofstream ofs("out_" + std::to_string(cameraId) + "_" +
                                          std::to_string(count++) + ".raw",
                                      std::ios::trunc);
                    ofs.write((char *)data, size);
                    free(data);
                }
            })) {
        std::cout << "register  stream callback failed\n";
        return -1;
    }

    if (plugin.registerStreamCallback(
            3, [&](uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
                std::cout << "Get Camera " << cameraId << " Size " << size << " timestamp "
                          << timestamp << "\n";
                if (file_save) {

                    static uint32_t count = 0;
                    void *data = 0;
                    data = malloc(size);
                    if (data == nullptr) {
                        std::cout << "malloc data failed size " << size << "\n";
                        return -1;
                    }

                    auto result =
                        cudaMemcpy(data, payload, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
                    if (result != cudaSuccess) {
                        std::cout << "Cuda memcpy failed " << result << std::endl;
                        return -1;
                    }

                    std::ofstream ofs("out_" + std::to_string(cameraId) + "_" +
                                          std::to_string(count++) + ".raw",
                                      std::ios::trunc);
                    ofs.write((char *)data, size);
                    free(data);
                }
            })) {
        std::cout << "register  stream callback failed\n";
        return -1;
    }

    if (plugin.open(0) != 0) {
        std::cout << "plugin open failed\n";
    }

    if (plugin.open(1) != 0) {
        std::cout << "plugin open failed\n";
    }

    if (plugin.open(2) != 0) {
        std::cout << "plugin open failed\n";
    }
#if 0
    if(plugin.open(3) != 0) {
        std::cout << "plugin open failed\n";
    }
#endif
    if (plugin.open()) {
        std::cout << "open camera plugin failed\n";
        return -1;
    }
    while(true) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
    plugin.close();

    return 0;
}
