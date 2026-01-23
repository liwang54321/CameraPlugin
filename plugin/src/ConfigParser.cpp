#include "ConfigParser.hpp"

#include <syslog.h>
#include <fstream>
#include <atomic>
#include <filesystem>
#include <nlohmann/json.hpp>


struct CameraConfigTotal {
    CameraConfigs configs;
};

static inline void from_json(const nlohmann::json& j, CameraConfig& c) {
    j.at("id").get_to(c.id);
    j.at("name").get_to(c.name);
    j.at("type").get_to(c.type);
    j.at("rtp_ip").get_to(c.rtp_ip);
    j.at("port").get_to(c.port);
    j.at("dev_device").get_to(c.dev_device);
    j.at("test_file").get_to(c.test_file);
    j.at("width").get_to(c.width);
    j.at("height").get_to(c.height);
    j.at("fps").get_to(c.fps);
}

static inline void from_json(const nlohmann::json& j, CameraConfigs& c) {
    j.at("cameras").get_to(c.config);
}

static inline void from_json(const nlohmann::json& j, CameraConfigTotal& c) {
    j.at("CameraConfig").get_to(c.configs);
}

class ConfigParserImpl {
   public:
    ConfigParserImpl(const std::string& config) : config_{config} {}

    ~ConfigParserImpl(void) {}

    bool Parser(void) {
        std::error_code ec;
        auto ret = std::filesystem::exists(config_, ec);
        if (!ret) {
            syslog(LOG_ERR, "Config file %s does not exist", config_.c_str());
            return false;
        }

        std::ifstream ifs(config_);
        if(!ifs.is_open()) {
            syslog(LOG_ERR, "Failed to open file %s\n", config_);
            return false;
        }
        CameraConfigTotal total;
        try {
            total = nlohmann::json::parse(ifs);
        } catch (const std::exception& e) {
            syslog(LOG_ERR, "Failed To parser json %s\n", e.what());
            return false;
        }

        configs_ = total.configs;
        is_parsed_.store(true);
        return true;
    }

    std::optional<CameraConfigs> GetConfig(void) {
        if (!is_parsed_) {
            return std::nullopt;
        }
        return configs_;
    }

   private:
    std::string config_;
    CameraConfigs configs_;
    std::atomic_bool is_parsed_{false};
};

ConfigParser::ConfigParser(const std::string& config)
    : impl_{std::make_unique<ConfigParserImpl>(config)} {}

ConfigParser::~ConfigParser(void) {}

bool ConfigParser::Parser(void) { return impl_->Parser(); }

std::optional<CameraConfigs> ConfigParser::GetConfig(void) {
    return impl_->GetConfig();
}
