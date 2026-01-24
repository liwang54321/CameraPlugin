#include <iostream>
#include <thread>

#include "CameraPlugin.hpp"
int main(int argc, char **argv) {
    CameraPluginParams params;
    params.enable_rgb_dump = false;
    params.enable_rtp_dump = false;
    params.enable_yuv_dump = false;
    CameraPlugin plugin(params);
    if (plugin.loadConfig(argv[1]) != 0) {
        std::cout << "open " << argv[1] << " failed\n";
        return -1;
    }

    if (plugin.registerStreamCallback(
            0, [](uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size) {
                std::cout << "Get " << cameraId << "\n";
            })) {
        std::cout << "register  stream callback failed\n";
        return -1;
    }

    if (plugin.open()) {
        std::cout << "open camera plugin failed\n";
        return -1;
    }

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    plugin.close();

    return 0;
}
