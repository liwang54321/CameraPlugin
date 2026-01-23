#ifndef ICAMERA_PLUGIN_H
#define ICAMERA_PLUGIN_H

#include <vector>
#include <functional>
#include <cstdint>

class ICameraPlugin
{
public:
    virtual ~ICameraPlugin(void) = default;

    /**
     * @brief Load configuration from file
     *
     * @param filename Configuration file path
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t loadConfig(const std::string &filename) = 0;

    /**
     * @brief Open all cameras
     *
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t open() = 0;

    /**
     * @brief Close all cameras
     *
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t close() = 0;

    /**
     * @brief Open camera
     *
     * @param cameraId Camera identifier
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t open(uint32_t cameraId) = 0;

    /**
     * @brief Close camera
     *
     * @param cameraId Camera identifier
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t close(uint32_t cameraId) = 0;

    /**
     * @brief Register camera status callback function
     *
     * @param function Callback function for camera status changes
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t registerStatusCallback(std::function<void(uint32_t cameraId, uint32_t cameraStatus)> function) = 0;

    /**
     * @brief Start real-time video stream for camera
     *
     * @param cameraId Camera identifier
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t startStream(uint32_t cameraId) = 0;

    /**
     * @brief Stop real-time video stream for camera
     *
     * @param cameraId Camera identifier
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t stopStream(uint32_t cameraId) = 0;

    /**
     * @brief Register stream data callback function
     *
     * @param cameraId Camera identifier
     * @param function Callback function for stream data
     * @return int8_t 0:success, other value:fail
     */
    virtual int8_t registerStreamCallback(uint32_t cameraId, std::function<void(uint32_t cameraId, uint64_t timestamp, const uint8_t *payload, size_t size)> function) = 0;
};

#endif // ICAMERA_PLUGIN_H
