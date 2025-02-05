#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <BoTSORT.h>
#include <crow.h>
#include <track.h>

#include <opencv2/opencv.hpp>

#define VERSION "0.2.0"

// Structure to parse multipart form-data (image + JSON)
struct MultipartData {
    std::string context;
    cv::Mat image;
    std::string json;
};

// Function to parse multipart form-data (image + JSON)
MultipartData parse_multipart(const crow::request &req) {
    MultipartData data;

    crow::multipart::message messageMultipart(req);

    auto partFrame = messageMultipart.get_part_by_name("frame");
    if (!partFrame.body.empty()) {
        std::vector<uchar> dataImage(partFrame.body.begin(),
                                     partFrame.body.end());
        data.image = cv::imdecode(dataImage, cv::IMREAD_COLOR);
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

// Function to create a tracker to be used in a collection
std::shared_ptr<BoTSORT> create_tracker(const std::string &config_path) {
    return std::make_shared<BoTSORT>(config_path);
}

// Main function
int main() {
    crow::SimpleApp app;

    const std::string CONFIG_FILE = "config/tracker.ini";

    std::unordered_map<int, std::shared_ptr<BoTSORT>> trackers;

    CROW_ROUTE(app, "/status").methods(crow::HTTPMethod::GET)([]() {
        crow::json::wvalue response;
        response["name"] = "TrackerX";
        response["type"] = "detector";
        response["path"] = "/";
        response["version"] = VERSION;
        response["output"]["types"] = crow::json::wvalue::list({"track"});
        response["image_types"] = crow::json::wvalue::list({"-"});
        return response;
    });

    CROW_ROUTE(app, "/").methods(crow::HTTPMethod::POST)(
        [&trackers, &CONFIG_FILE](const crow::request &req) {
            try {
                int id_camera = std::stoi("1"); // TODO: get from req["cam_id"]

                if (trackers.find(id_camera) == trackers.end()) {
                    trackers[id_camera] = create_tracker(CONFIG_FILE);
                }

                auto data = parse_multipart(req);
                if (data.image.empty()) {
                    throw std::runtime_error("Failed to decode image");
                }

                std::vector<Detection> detections;
                auto json_data = crow::json::load(data.json);
                if (!json_data) {
                    throw std::runtime_error("Invalid JSON in request");
                }

                for (const auto &item : json_data) {
                    Detection det;
                    det.bbox_tlwh =
                        cv::Rect(static_cast<int>(item["bbox"][0].i()),
                                 static_cast<int>(item["bbox"][1].i()),
                                 static_cast<int>(item["bbox"][2].i()),
                                 static_cast<int>(item["bbox"][3].i()));
                    det.confidence = static_cast<float>(item["prob"].d());
                    det.class_id = static_cast<int>(item["type"].i());
                    detections.push_back(det);
                }

                auto tracks =
                    trackers[id_camera]->track(detections, data.image);

                std::vector<crow::json::wvalue> result;
                for (const auto &track : tracks) {
                    crow::json::wvalue track_json;
                    track_json["track_id"] = track->track_id;
                    auto bbox = track->get_tlwh();
                    track_json["bbox"] = crow::json::wvalue::list(
                        {bbox[0], bbox[1], bbox[2], bbox[3]});
                    result.push_back(track_json);
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
