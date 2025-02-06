#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <BoTSORT.h>
#include <crow.h>
#include <track.h>

#include <boost/bimap.hpp>
#include <opencv2/opencv.hpp>

#define VERSION "0.2.0"

// Structure to parse multipart form-data (image + JSON)
struct MultipartData {
    std::string context;
    cv::Mat image;
    std::string json;
};

#ifdef DEBUG
std::ostream &operator<<(std::ostream &os,
                         const crow::multipart::header &header) {
    os << header.value << "(params: ";
    for (const auto &param : header.params) {
        os << param.first << "=" << param.second << ", ";
    }
    os << "...)";
    return os;
}

void debug_multipart(const crow::multipart::message &messageMultipart) {
    std::cout << "=== Multipart Debug Info ===" << std::endl;
    if (messageMultipart.parts.empty()) {
        std::cout << "No parts found in multipart message!" << std::endl;
        return;
    }
    std::cout << "Boundary: " << messageMultipart.boundary << std::endl;
    for (size_t i = 0; i < messageMultipart.parts.size(); i++) {
        const auto &part = messageMultipart.parts[i];
        std::cout << "\nPart " << i + 1 << " of "
                  << messageMultipart.parts.size() << std::endl;
        std::cout << "Headers:" << std::endl;
        for (const auto &header : part.headers) {
            std::cout << "\t" << header.first << ": " << header.second
                      << std::endl;
        }
        std::cout << "Content length: " << part.body.size() << " bytes"
                  << std::endl;
        if (!part.body.empty()) {
            std::cout << "First 20 bytes (hex): ";
            for (size_t j = 0; j < std::min<size_t>(20, part.body.size());
                 ++j) {
                std::cout << std::hex << std::setw(2) << std::setfill('0')
                          << static_cast<int>(
                                 static_cast<unsigned char>(part.body[j]))
                          << " ";
            }
            std::cout << std::dec << std::endl;
        }
    }
}
#endif

// Function to parse multipart form-data (image + JSON)
MultipartData parse_multipart(const crow::request &req) {
    MultipartData data;

    crow::multipart::message messageMultipart(req);

#ifdef DEBUG
    debug_multipart(messageMultipart);
#endif

    auto partFrame = messageMultipart.get_part_by_name("frame");
    if (!partFrame.body.empty()) {
        std::vector<uchar> dataImage(partFrame.body.begin(),
                                     partFrame.body.end());
        data.image = cv::imdecode(dataImage, cv::IMREAD_COLOR);
    } else {
        std::cerr << "WARNING: HTTP requset has no frame!" << std::endl;
    }

    auto partJson = messageMultipart.get_part_by_name("json");
    if (!partJson.body.empty()) {
        data.json = partJson.body;
    }

    auto partContext = messageMultipart.get_part_by_name("context");
    if (!partContext.body.empty()) {
        data.context = partContext.body;
    }

    return data;
}

// Function to initialize tracker for single-instance usage
std::unique_ptr<BoTSORT> initialize_tracker(const std::string &config_path) {
    return std::make_unique<BoTSORT>(config_path);
}

// Functions to create a tracker. Note that on the first creation
// TensorRT will convert ONNX model to TRT engine representation.
// This may take a long time so that client may fail with timeout
// error. Thus either initial start is required, some locking logic,
// or special build procedure
std::shared_ptr<BoTSORT> create_tracker(const std::string &config_path) {
    return std::make_shared<BoTSORT>(config_path);
}

std::shared_ptr<BoTSORT> create_tracker(const std::string &config_tracker,
                                        const std::string &config_gmc,
                                        const std::string &config_reid,
                                        const std::string &model_reid) {
    return std::make_shared<BoTSORT>(config_tracker, config_gmc, config_reid,
                                     model_reid);
}

std::unordered_map<std::string, std::string> parse_config_args(int argc,
                                                               char *argv[]) {
    std::unordered_map<std::string, std::string> configs;
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        const std::string prefix = "--path-";
        if (arg.substr(0, prefix.size()) == prefix) {
            auto pos = arg.find('=');
            if (pos != std::string::npos && pos + 1 < arg.size()) {
                std::string key =
                    arg.substr(prefix.size(), pos - prefix.size());
                std::string value = arg.substr(pos + 1);
                // Store into the dict
                configs[key] = value;
            }
        }
    }
    return configs;
}

#define KEY_TRACKER "config-tracker"
#define KEY_GMC "config-gmc"
#define KEY_REID "config-reid"
#define KEY_MODEL "model-reid"

#define PATH_TRACKER "/usr/local/etc/botsort/tracker.ini"
#define PATH_GMC "/usr/local/etc/botsort/gmc.ini"
#define PATH_REID "/usr/local/etc/botsort/reid.ini"
#define PATH_MODEL "/workspace/botsort/mobilenetv2_x1_4_msmt17.onnx"

// Main function
int main(int argc, char *argv[]) {
    crow::SimpleApp app;
    app.loglevel(crow::LogLevel::Warning);

    auto configs = parse_config_args(argc, argv);

    // Default config paths
    if (configs.find(KEY_TRACKER) == configs.end()) {
        configs[KEY_TRACKER] = PATH_TRACKER;
    }
    if (configs.find(KEY_GMC) == configs.end()) {
        configs[KEY_GMC] = PATH_GMC;
    }
    if (configs.find(KEY_REID) == configs.end()) {
        configs[KEY_REID] = PATH_REID;
    }
    if (configs.find(KEY_MODEL) == configs.end()) {
        configs[KEY_MODEL] = PATH_MODEL;
    }

    std::unordered_map<size_t, boost::bimap<std::string, uint8_t>> classes;
    std::unordered_map<int, std::shared_ptr<BoTSORT>> trackers;

    CROW_ROUTE(app, "/status").methods(crow::HTTPMethod::GET)([]() {
        crow::json::wvalue response;
        response["name"] = "TrackerX";
        response["type"] = "detector";
        response["path"] = "/";
        response["version"] = VERSION;
        response["output"]["types"] = crow::json::wvalue::list({"track"});
        return response;
    });

    CROW_ROUTE(app, "/").methods(
        crow::HTTPMethod::POST)([&classes, &configs,
                                 &trackers](const crow::request &req) {
        try {
            auto data = parse_multipart(req);
            if (data.image.empty()) {
                throw std::runtime_error("Failed to decode image");
            }

            auto json_data = crow::json::load(data.json);
            if (!json_data) {
                throw std::runtime_error("Invalid JSON in request");
            }

            size_t id_camera = 0;
            try {
                // Get camera ID from context
                id_camera = crow::json::load(data.context)["camera"]["id"].i();
            } catch (const std::exception &e) {
                std::cerr << "WARNING: Cannot get camera ID from context"
                          << std::endl;
            }

            if (classes.find(id_camera) == classes.end()) {
                // Create bidirectional map for the camera classes
                boost::bimap<std::string, uint8_t> bimap_camera;
                classes[id_camera] = std::move(bimap_camera);
                std::cout << "New class bimap created for camera " << id_camera
                          << std::endl;
            }

            if (trackers.find(id_camera) == trackers.end()) {
                // Create a dedicated tracker for the camera
                trackers[id_camera] =
                    create_tracker(configs[KEY_TRACKER], configs[KEY_GMC],
                                   configs[KEY_REID], configs[KEY_MODEL]);
                std::cout << "New tracker created for camera " << id_camera
                          << std::endl;
            }

            std::vector<Detection> detections;
            for (const auto &item : json_data) {
                Detection det;
                // Since item["bbox"] is tlwh, but normalized (real value from 0
                // to 1), it's necessary to scale it to the actual image
                // coordinates in pixels
                det.bbox_tlwh = cv::Rect(
                    static_cast<int>(
                        std::round(item["bbox"][0].d() * data.image.cols)),
                    static_cast<int>(
                        std::round(item["bbox"][1].d() * data.image.rows)),
                    static_cast<int>(
                        std::round(item["bbox"][2].d() * data.image.cols)),
                    static_cast<int>(
                        std::round(item["bbox"][3].d() * data.image.rows)));
                // Class probability may be used as-is
                det.confidence = static_cast<float>(item["prob"].d());
                // Since item["type"] is a string value, it's necessary to map
                // it to an integer value. Assume here that the class names are
                // unique and must be mapped to unique integers. For simplicity,
                // just increment the last used integer for each new class name
                std::string key = item["type"].s();
                uint8_t value = 1;
                auto index = classes[id_camera].left.find(key);
                if (index == classes[id_camera].left.end()) {
                    if (!classes[id_camera].right.empty()) {
                        value = classes[id_camera].right.rbegin()->first + 1;
                    }
                    classes[id_camera].insert(
                        boost::bimap<std::string, uint8_t>::value_type(key,
                                                                       value));
                }
                det.class_id =
                    static_cast<int>(classes[id_camera].left.find(key)->second);
                detections.push_back(det);
            }

            // Do the tracking
            auto tracks = trackers[id_camera]->track(detections, data.image);

            // Prepare response from the tracking results
            std::vector<crow::json::wvalue> result;
            for (const auto &track : tracks) {
                crow::json::wvalue item;
                auto class_id = track->get_class_id();
                item["type"] = classes[id_camera].right.find(class_id)->second;
                item["prob"] = track->get_score();

                auto bbox = track->get_tlwh();
                item["bbox"] = crow::json::wvalue::list(
                    {bbox[0] / data.image.cols, bbox[1] / data.image.rows,
                     bbox[2] / data.image.cols, bbox[3] / data.image.rows});

                item["data"] = crow::json::wvalue::object();
                item["data"]["track_id"] = track->track_id;
                item["data"]["state"] = track->state;

                result.push_back(item);
            }

            crow::json::wvalue response_ok;
            response_ok["predicts"] = std::move(result);
            response_ok["version"] = VERSION;
            response_ok["time"] = static_cast<double>(cv::getTickCount()) /
                                  cv::getTickFrequency();

            return crow::response(response_ok.dump());
        } catch (const std::exception &e) {
            crow::json::wvalue response_error;
            response_error["error"] = std::string("Error: ") + e.what();
            return crow::response(400, response_error.dump());
        }
    });

    app.port(5000).multithreaded().run();
    return 0;
}
