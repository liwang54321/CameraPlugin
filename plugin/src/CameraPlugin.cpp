#include "CameraPlugin.hpp"

#include <filesystem>
#include <map>
#include <atomic>
#include "CCmdLineParser.hpp"
#include "CManager.hpp"
#include "CStatusManager.hpp"
#include "ConfigParser.hpp"

constexpr uint8_t kMaxVideoChannel = 16;

class CameraPluginImpl {
public:
    CameraPluginImpl(const CameraPluginParams &params)
        : params_{params} {

          };

    ~CameraPluginImpl(void) {

    };

    int8_t loadConfig(const std::string &filename) {
        ConfigParser parser(filename);
        if (!parser.Parser()) {
            LOG_ERR("Camera Plugin failed to load config %s\n", filename.c_str());
            return -1;
        }

        config_ = parser.GetConfig();
        if (config_ == std::nullopt) {
            LOG_ERR("Camera Plugin Parser Config failed %s\n", filename.c_str());
            return -1;
        }

        for (auto item : config_->config) {
            maps_[item.id].config = item;
            maps_[item.id].is_open = false;
        }
        return 0;
    }

    int8_t open(void) {
        for (auto &[k, v] : maps_) {
            if (v.is_open) {
                continue;
            }

            if (v.config.type != "RTP") {
                LOG_ERR("Camera Plugin unkown video type %s\n", v.config.type.c_str());
                return -1;
            }

            LOG_INFO("Camera Plugin Opened: Camera Id: %d\n", k);
        }

        NvError error = NvError_Success;
        uint32_t argc = 7;
        auto mask = GenerateBinaryString(maps_);
        std::cout << "mask :" << mask << std::endl;
        char *args[10] = {"CameraPlugin",
                          "-c",
                          "SIM_IMX623_30FPS_CPHY_x4",
                          "-m",
                          (char *)mask.c_str(),
                          "-p",
                          "FileSrc=type=4:path=./"
                          "FileSource_Test_Data:width=1920:height=1280,Cuda=width="
                          "1920:height=1280"};
        bool is_debug = getenv("SIPL_DEBUG") == nullptr ? false : true;
        if (is_debug) {
            args[7] = "-K";
            args[8] = "-r";
            args[9] = "10";
            argc = 10;
        }
        appCfg_ = std::make_shared<CAppCfg>();
        for (const auto &[k, _] : maps_) {
            appCfg_->RegisterCameraPlugin(k, camera_callback_[k]);
        }
        appCfg_->RegisterModuleInfo(maps_);
        appCfg_->SetCpuOutPut(params_.enable_cpu_sink);
        CCmdLineParser cmdline;
        error = cmdline.Parse(argc, args, appCfg_);
        if (NvError_EndOfFile == error) {
            return 0;
        }

        auto quitStatus = EventStatus::OK;
        auto quitStatusSet = [&quitStatus](CChannel *pChannel, EventStatus event) {
            if (EventStatus::ERROR != quitStatus) {
                // skip to update following event error if errors happened
                quitStatus = event;
            }
        };
        manager_main_ = std::make_unique<CManager>(appCfg_, quitStatusSet);
        error = manager_main_->Init();
        if (error != NvError_Success) {
            LOG_ERR("CManager init failed!\n");
            manager_main_->Quit();
        } else {
            manager_main_->Run();
        }

        return 0;
    }

    int8_t close(void) {
        manager_main_->DeInit();
        manager_main_.reset();
        return 0;
    }
    int8_t open(uint32_t cameraId) {
        mask_ |= 1 << cameraId;
        return 0;
    }
    int8_t close(uint32_t cameraId) {
        mask_ |= ~(1 << cameraId);
        return 0;
    }
    int8_t registerStatusCallback(CameraEventCallback function) {
        camera_event_callback_ = std::move(function);
        return 0;
    }
    int8_t startStream(uint32_t cameraId) { return 0; }
    int8_t stopStream(uint32_t cameraId) { return 0; }

    int8_t registerStreamCallback(uint32_t cameraId, CameraCallback function) {
        if (cameraId >= kMaxVideoChannel) {
            return -1;
        }

        camera_callback_[cameraId] = std::move(function);
        return 0;
    }

private:
    std::string GenerateBinaryString(const std::map<uint8_t, CameraStatus>& map) {
        constexpr int mapping[16] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};

        std::string result = "0000 0000 0000 0000";

        for (const auto &[num, _] : map) {
            if (num >= 0 && num <= 15) {
                for (int i = 0; i < 16; i++) {
                    if (mapping[i] == num) {
                        int str_pos = i + i / 4;
                        result[str_pos] = '1';
                        break;
                    }
                }
            }
        }

        return result;
    }

private:
    CameraPluginParams params_;
    CameraEventCallback camera_event_callback_{nullptr};
    CameraCallback camera_callback_[kMaxVideoChannel]{nullptr};
    std::optional<CameraConfigs> config_;
    std::map<uint8_t, CameraStatus> maps_;
    std::unique_ptr<CManager> manager_main_;
    std::unique_ptr<CManager> manager_[kMaxVideoChannel];
    uint16_t mask_;
    std::shared_ptr<CAppCfg> appCfg_;
};

CameraPlugin::CameraPlugin(const CameraPluginParams &params)
    : impl_{std::make_unique<CameraPluginImpl>(params)} {}

CameraPlugin::~CameraPlugin(void) {}

int8_t CameraPlugin::loadConfig(const std::string &filename) { return impl_->loadConfig(filename); }

int8_t CameraPlugin::open(void) { return impl_->open(); }

int8_t CameraPlugin::close(void) { return impl_->close(); }

int8_t CameraPlugin::open(uint32_t cameraId) { return impl_->open(cameraId); }

int8_t CameraPlugin::close(uint32_t cameraId) { return impl_->close(cameraId); }

int8_t CameraPlugin::registerStatusCallback(CameraEventCallback function) {
    return impl_->registerStatusCallback(function);
}

int8_t CameraPlugin::startStream(uint32_t cameraId) { return impl_->startStream(cameraId); }

int8_t CameraPlugin::stopStream(uint32_t cameraId) { return impl_->stopStream(cameraId); }

int8_t CameraPlugin::registerStreamCallback(uint32_t cameraId, CameraCallback function) {
    return impl_->registerStreamCallback(cameraId, function);
}
