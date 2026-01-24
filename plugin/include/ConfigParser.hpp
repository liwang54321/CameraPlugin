#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "CConfig.hpp"

class ConfigParserImpl;
class ConfigParser {
public:
    ConfigParser(const std::string &config);
    ~ConfigParser(void);

    bool Parser(void);
    std::optional<CameraConfigs> GetConfig(void);

private:
    std::unique_ptr<ConfigParserImpl> impl_;
};
