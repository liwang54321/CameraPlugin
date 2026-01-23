#include "CameraPlugin.hpp"

#include <syslog.h>

#include <filesystem>
#include <map>

#include "ConfigParser.hpp"
#include "CManager.hpp"
#include "CStatusManager.hpp"
#include "CCmdLineParser.hpp"

struct CameraStatus {
    CameraConfig config;
    bool is_open = false;
};

class CameraPluginImpl {
   public:
    CameraPluginImpl(const CameraPluginParams& params)
        : params_{params} {

          };

    ~CameraPluginImpl(void) {

    };

    int8_t loadConfig(const std::string& filename) {
        ConfigParser parser(filename);
        if (!parser.Parser()) {
            syslog(LOG_ERR, "Camera Plugin failed to load config %s\n",
                   filename.c_str());
            return -1;
        }

        config_ = parser.GetConfig();
        if (config_ == std::nullopt) {
            syslog(LOG_ERR, "Camera Plugin Parser Config failed %s\n",
                   filename.c_str());
            return -1;
        }

        for (auto item : config_->config) {
            maps_[item.id].config = item;
            maps_[item.id].is_open = false;
        }
        return 0;
    }

    int8_t open(void) {
        for (auto& [k, v] : maps_) {
            if (v.is_open) {
                continue;
            }

            if (v.config.type != "RTP") {
                syslog(LOG_ERR, "Camera Plugin unkown video type %s\n",
                       v.config.type.c_str());
                return -1;
            }


            syslog(LOG_INFO, "Camera Plugin Opened: Camera Id: %d\n", k);
        }

        NvError error = NvError_Success;
        uint32_t argc = 0;
        error = cmdline.Parse(argc, argv, appCfg);
        std::shared_ptr<CAppCfg> appCfg = std::make_shared<CAppCfg>();
        // appCfg->

        return 0;
    }

    int8_t close(void) { return 0; }
    int8_t open(uint32_t cameraId) { return 0; }
    int8_t close(uint32_t cameraId) { return 0; }
    int8_t registerStatusCallback(CameraEventCallback function) {
        camera_event_callback_ = std::move(function);
        return 0;
    }
    int8_t startStream(uint32_t cameraId) { return 0; }
    int8_t stopStream(uint32_t cameraId) { return 0; }

    int8_t registerStreamCallback(uint32_t cameraId, CameraCallback function) {
        camera_callback_ = std::move(function);
        return 0;
    }

   private:
    CameraPluginParams params_;
    CameraEventCallback camera_event_callback_{nullptr};
    CameraCallback camera_callback_{nullptr};
    std::optional<CameraConfigs> config_;
    std::map<uint8_t, CameraStatus> maps_;
};

CameraPlugin::CameraPlugin(const CameraPluginParams& params)
    : impl_{std::make_unique<CameraPluginImpl>(params)} {}

CameraPlugin::~CameraPlugin(void) {}

int8_t CameraPlugin::loadConfig(const std::string& filename) {
    return impl_->loadConfig(filename);
}

int8_t CameraPlugin::open(void) { return impl_->open(); }

int8_t CameraPlugin::close(void) { return impl_->close(); }

int8_t CameraPlugin::open(uint32_t cameraId) { return impl_->open(cameraId); }

int8_t CameraPlugin::close(uint32_t cameraId) { return impl_->close(cameraId); }

int8_t CameraPlugin::registerStatusCallback(CameraEventCallback function) {
    return impl_->registerStatusCallback(function);
}

int8_t CameraPlugin::startStream(uint32_t cameraId) {
    return impl_->startStream(cameraId);
}

int8_t CameraPlugin::stopStream(uint32_t cameraId) {
    return impl_->stopStream(cameraId);
}

int8_t CameraPlugin::registerStreamCallback(uint32_t cameraId,
                                            CameraCallback function) {
    return impl_->registerStreamCallback(cameraId, function);
}
