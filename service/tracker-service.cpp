#include <crow.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "BoTSORT.h"
#include "track.h"

#define VERSION "0.0.2"

// Function to parse multipart form-data (image + JSON)
struct MultipartData {
    cv::Mat image;
    std::string json;
};

MultipartData parse_multipart(const crow::request &req) {
    MultipartData data;

    auto content_type = req.get_header_value("Content-Type");
    auto boundary_pos = content_type.find("boundary=");
    if (boundary_pos == std::string::npos) {
        throw std::runtime_error("Boundary not found in Content-Type header");
    }

    std::string boundary = "--" + content_type.substr(boundary_pos + 9);
    std::string body = req.body;

    size_t pos = 0;
    while ((pos = body.find(boundary, pos)) != std::string::npos) {
        size_t start = body.find("\r\n\r\n", pos) + 4;
        size_t end = body.find(boundary, start) - 2;
        std::string part = body.substr(start, end - start);

        if (part.find("Content-Disposition: form-data; name=\"frame\"") != std::string::npos) {
            size_t img_start = part.find("\r\n\r\n") + 4;
            std::vector<uchar> img_data(part.begin() + img_start, part.end());
            data.image = cv::imdecode(img_data, cv::IMREAD_COLOR);
        } else if (part.find("Content-Disposition: form-data; name=\"json\"") != std::string::npos) {
            size_t json_start = part.find("\r\n\r\n") + 4;
            data.json = part.substr(json_start);
        }

        pos = end + boundary.size();
    }

    return data;
}

std::unique_ptr<BoTSORT> initialize_tracker(const std::string &config_path) {
    return std::make_unique<BoTSORT>(config_path);
}

int main() {
    crow::SimpleApp app;

    const std::string CONFIG_FILE = "config/tracker.ini";
    auto tracker = initialize_tracker(CONFIG_FILE);

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

    CROW_ROUTE(app, "/").methods(crow::HTTPMethod::POST)([&tracker](const crow::request &req) {
        try {
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
                det.bbox_tlwh = cv::Rect(
                    static_cast<int>(item["bbox"][0].i()),
                    static_cast<int>(item["bbox"][1].i()),
                    static_cast<int>(item["bbox"][2].i()),
                    static_cast<int>(item["bbox"][3].i()));
                det.confidence = static_cast<float>(item["prob"].d());
                det.class_id = static_cast<int>(item["type"].i());
                detections.push_back(det);
            }

            auto tracks = tracker->track(detections, data.image);

            std::vector<crow::json::wvalue> result;
            for (const auto &track : tracks) {
                crow::json::wvalue track_json;
                track_json["track_id"] = track->track_id;
                auto bbox = track->get_tlwh();
                track_json["bbox"] = crow::json::wvalue::list({bbox[0], bbox[1], bbox[2], bbox[3]});
                result.push_back(track_json);
            }

            crow::json::wvalue response;
            response["predicts"] = result;
            response["version"] = VERSION;
            response["time"] = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();

            return response;
        } catch (const std::exception &e) {
            return crow::response(400, std::string("Error: ") + e.what());
        }
    });

    app.port(5000).multithreaded().run();
    return 0;
}
