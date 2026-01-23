#pragma once

#include <memory>
#include <string>
#include <vector>
#include <optional>

struct CameraConfig {
    uint32_t id;
    std::string name;
    std::string type;
    std::string rtp_ip;
    uint16_t port;
    std::string dev_device;
    std::string test_file;
    uint32_t width;
    uint32_t height;
    uint8_t fps;
};

struct CameraConfigs {
    std::vector<CameraConfig> config;
};

class ConfigParserImpl;
class ConfigParser {
   public:
    ConfigParser(const std::string& config);

    ~ConfigParser(void);

    bool Parser(void);

    std::optional<CameraConfigs> GetConfig(void);

private:
    std::unique_ptr<ConfigParserImpl> impl_;
};
