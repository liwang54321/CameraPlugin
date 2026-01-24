#pragma once

#include "CameraPluginIf.hpp"
#include <memory>

using CameraEventCallback = std::function<void(uint32_t cameraId, uint32_t cameraStatus)>;

using CameraCallback =
    std::function<void(uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size)>;

struct CameraPluginParams {
    bool enable_yuv_dump;
    bool enable_rgb_dump;
    bool enable_rtp_dump;
};

class CameraPluginImpl;
class CameraPlugin : public ICameraPlugin {
public:
    CameraPlugin(const CameraPluginParams &params);
    ~CameraPlugin(void);

    int8_t loadConfig(const std::string &filename) override;
    int8_t open(void) override;
    int8_t close(void) override;
    int8_t open(uint32_t cameraId) override;
    int8_t close(uint32_t cameraId) override;
    int8_t registerStatusCallback(CameraEventCallback function) override;
    int8_t startStream(uint32_t cameraId) override;
    int8_t stopStream(uint32_t cameraId) override;
    int8_t registerStreamCallback(uint32_t cameraId, CameraCallback function) override;

private:
    std::unique_ptr<CameraPluginImpl> impl_;
};
