#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "adaptive_sorting_bringup/sorting_colors.hpp"
#include "adaptive_sorting_bringup/sorting_layout.hpp"
#include "adaptive_sorting_bringup/sorting_validation.hpp"
#include "adaptive_sorting_bringup/srv/get_sorting_observation.hpp"
#include "adaptive_sorting_bringup/srv/get_sorting_scene_config.hpp"
#include "adaptive_sorting_bringup/srv/present_bin_color_labels.hpp"
#include "adaptive_sorting_bringup/srv/present_experiment_status.hpp"
#include "adaptive_sorting_bringup/srv/present_feedback.hpp"
#include "adaptive_sorting_bringup/srv/present_observation.hpp"

namespace {
constexpr char kWorldFrame[] = "world";
namespace colors = adaptive_sorting_bringup::colors;
namespace layout = adaptive_sorting_bringup::layout;

geometry_msgs::msg::Pose pose(double x, double y, double z) {
  geometry_msgs::msg::Pose result;
  result.position.x = x;
  result.position.y = y;
  result.position.z = z;
  result.orientation.w = 1.0;
  return result;
}

std_msgs::msg::ColorRGBA color(float red, float green, float blue,
                               float alpha = 1.0F) {
  std_msgs::msg::ColorRGBA result;
  result.r = red;
  result.g = green;
  result.b = blue;
  result.a = alpha;
  return result;
}

shape_msgs::msg::SolidPrimitive box(double x, double y, double z) {
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
  primitive.dimensions = {x, y, z};
  return primitive;
}

shape_msgs::msg::SolidPrimitive cylinder(double height, double radius) {
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::CYLINDER;
  primitive.dimensions = {height, radius};
  return primitive;
}

shape_msgs::msg::SolidPrimitive sphere(double radius) {
  shape_msgs::msg::SolidPrimitive primitive;
  primitive.type = shape_msgs::msg::SolidPrimitive::SPHERE;
  primitive.dimensions = {radius};
  return primitive;
}

moveit_msgs::msg::CollisionObject object(const std::string &id) {
  moveit_msgs::msg::CollisionObject result;
  result.header.frame_id = kWorldFrame;
  result.id = id;
  result.operation = moveit_msgs::msg::CollisionObject::ADD;
  return result;
}

moveit_msgs::msg::ObjectColor
object_color(const std::string &id, const std_msgs::msg::ColorRGBA &value) {
  moveit_msgs::msg::ObjectColor result;
  result.id = id;
  result.color = value;
  return result;
}

moveit_msgs::msg::CollisionObject make_bin(
    int index, int count, const layout::SortingLayout &sorting_layout) {
  const auto center_y = sorting_layout.bin_y(index, count);
  auto result = object("bin" + std::to_string(index + 1));

  result.primitives.push_back(
      box(layout::kBinDepth, layout::kBinWidth, layout::kWallThickness));
  result.primitive_poses.push_back(
      pose(sorting_layout.bin_x, center_y, layout::kWallThickness / 2.0));

  const auto wall_z = layout::kBinHeight / 2.0;
  const auto side_x =
      sorting_layout.bin_x +
      (layout::kBinDepth - layout::kWallThickness) / 2.0;
  result.primitives.push_back(
      box(layout::kWallThickness, layout::kBinWidth, layout::kBinHeight));
  result.primitive_poses.push_back(pose(side_x, center_y, wall_z));
  result.primitives.push_back(
      box(layout::kWallThickness, layout::kBinWidth, layout::kBinHeight));
  result.primitive_poses.push_back(
      pose(2.0 * sorting_layout.bin_x - side_x, center_y, wall_z));

  const auto side_y =
      center_y + (layout::kBinWidth - layout::kWallThickness) / 2.0;
  result.primitives.push_back(
      box(layout::kBinDepth - 2.0 * layout::kWallThickness,
          layout::kWallThickness, layout::kBinHeight));
  result.primitive_poses.push_back(pose(sorting_layout.bin_x, side_y, wall_z));
  result.primitives.push_back(
      box(layout::kBinDepth - 2.0 * layout::kWallThickness,
          layout::kWallThickness, layout::kBinHeight));
  result.primitive_poses.push_back(
      pose(sorting_layout.bin_x, 2.0 * center_y - side_y, wall_z));
  return result;
}

visualization_msgs::msg::Marker marker(const rclcpp::Time &stamp,
                                       const std::string &name_space, int id,
                                       int type) {
  visualization_msgs::msg::Marker result;
  result.header.frame_id = kWorldFrame;
  result.header.stamp = stamp;
  result.ns = name_space;
  result.id = id;
  result.type = type;
  result.action = visualization_msgs::msg::Marker::ADD;
  result.pose.orientation.w = 1.0;
  return result;
}

visualization_msgs::msg::Marker delete_marker(const rclcpp::Time &stamp,
                                              const std::string &name_space,
                                              int id) {
  auto result = marker(stamp, name_space, id,
                       visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
  result.action = visualization_msgs::msg::Marker::DELETE;
  return result;
}

} // namespace

class SortingScenePublisher : public rclcpp::Node {
public:
  SortingScenePublisher() : Node("sorting_scene_publisher") {
    bin_count_ = declare_parameter<int>("bin_count", 5);
    research_question_ =
        declare_parameter<std::string>("research_question", "RQ--");
    light_1_on_ = declare_parameter<bool>("light_1_on", false);
    light_2_on_ = declare_parameter<bool>("light_2_on", false);
    context_light_count_ = declare_parameter<int>("context_light_count", 2);
    show_bin_color_labels_ =
        declare_parameter<bool>("show_bin_color_labels", true);
    color_target_bins_ = declare_parameter<bool>("color_target_bins", false);
    sorting_layout_ = {
        declare_parameter<double>("pickup_x"),
        declare_parameter<double>("pickup_y"),
        declare_parameter<double>("bin_x"),
        declare_parameter<double>("bin_y_offset"),
        declare_parameter<double>("bin_spacing"),
        declare_parameter<double>("sphere_radius"),
        declare_parameter<double>("pickup_sphere_center_z"),
        declare_parameter<double>("bin_sphere_center_z"),
    };
    adaptive_sorting_bringup::validate_bin_count(bin_count_);
    if (context_light_count_ < 0 || context_light_count_ > 2) {
      throw std::invalid_argument("context_light_count must be between 0 and 2");
    }
    adaptive_sorting_bringup::validate_layout(sorting_layout_);
    const auto supported_colors = colors::supported_names();
    bin_color_names_.assign(supported_colors.begin(),
                            supported_colors.begin() + bin_count_);

    marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/rviz_visual_tools", rclcpp::QoS(1).reliable());
    observation_service_ =
        create_service<adaptive_sorting_bringup::srv::PresentObservation>(
            "present_sorting_observation",
            [this](const adaptive_sorting_bringup::srv::PresentObservation::
                       Request::SharedPtr request,
                   adaptive_sorting_bringup::srv::PresentObservation::Response::
                       SharedPtr response) {
              present_observation(*request, *response);
            });
    perception_service_ =
        create_service<adaptive_sorting_bringup::srv::GetSortingObservation>(
            "observe_sorting_scene",
            [this](const adaptive_sorting_bringup::srv::GetSortingObservation::
                       Request::SharedPtr,
                   adaptive_sorting_bringup::srv::GetSortingObservation::
                       Response::SharedPtr response) {
              response->sphere_color = sphere_color_name_;
              response->light_1_on = light_1_on_;
              response->light_2_on = light_2_on_;
            });
    scene_config_service_ =
        create_service<adaptive_sorting_bringup::srv::GetSortingSceneConfig>(
            "get_sorting_scene_config",
            [this](const adaptive_sorting_bringup::srv::GetSortingSceneConfig::
                       Request::SharedPtr,
                   adaptive_sorting_bringup::srv::GetSortingSceneConfig::
                       Response::SharedPtr response) {
              response->bin_count = bin_count_;
              response->context_light_count = context_light_count_;
              response->show_bin_color_labels = show_bin_color_labels_;
              response->color_target_bins = color_target_bins_;
              response->supported_colors = colors::supported_names();
            });
    feedback_service_ =
        create_service<adaptive_sorting_bringup::srv::PresentFeedback>(
            "present_sorting_feedback",
            [this](const adaptive_sorting_bringup::srv::PresentFeedback::
                       Request::SharedPtr request,
                   adaptive_sorting_bringup::srv::PresentFeedback::Response::
                       SharedPtr response) {
              present_feedback(*request, *response);
            });
    experiment_status_service_ =
        create_service<adaptive_sorting_bringup::srv::PresentExperimentStatus>(
            "present_experiment_status",
            [this](const adaptive_sorting_bringup::srv::
                       PresentExperimentStatus::Request::SharedPtr request,
                   adaptive_sorting_bringup::srv::PresentExperimentStatus::
                       Response::SharedPtr response) {
              present_experiment_status(*request, *response);
            });
    bin_color_labels_service_ =
        create_service<adaptive_sorting_bringup::srv::PresentBinColorLabels>(
            "present_bin_color_labels",
            [this](const adaptive_sorting_bringup::srv::PresentBinColorLabels::
                       Request::SharedPtr request,
                   adaptive_sorting_bringup::srv::PresentBinColorLabels::
                       Response::SharedPtr response) {
              present_bin_color_labels(*request, *response);
            });

    publish_collision_scene();
    publish_markers();
    marker_timer_ = create_wall_timer(std::chrono::seconds(1),
                                      [this]() { publish_markers(); });
  }

private:
  void publish_collision_scene() {
    std::vector<moveit_msgs::msg::CollisionObject> objects;
    std::vector<moveit_msgs::msg::ObjectColor> colors;

    auto table = object("work_table");
    table.primitives.push_back(box(1.0, 1.3, 0.09));
    table.primitive_poses.push_back(pose(0.38, 0.0, layout::kTableTop - 0.045));
    objects.push_back(table);
    colors.push_back(object_color(table.id, color(0.65F, 0.67F, 0.70F)));

    auto pickup = object("pickup_pedestal");
    pickup.primitives.push_back(cylinder(0.04, 0.065));
    pickup.primitive_poses.push_back(
        pose(sorting_layout_.pickup_x, sorting_layout_.pickup_y, 0.02));
    objects.push_back(pickup);
    colors.push_back(object_color(pickup.id, color(0.18F, 0.20F, 0.22F)));

    auto ball = object("sorting_sphere");
    ball.primitives.push_back(sphere(sorting_layout_.sphere_radius));
    ball.primitive_poses.push_back(
        pose(sorting_layout_.pickup_x, sorting_layout_.pickup_y,
             sorting_layout_.pickup_sphere_center_z));
    objects.push_back(ball);
    // Keep the collision geometry used by trajectory preplanning, but do not
    // imply an observation before the experiment client presents the first one.
    colors.push_back(object_color(ball.id, color(0.0F, 0.0F, 0.0F, 0.0F)));

    for (int index = 0; index < bin_count_; ++index) {
      auto bin = make_bin(index, bin_count_, sorting_layout_);
      auto bin_color = color(0.28F, 0.30F, 0.34F, 0.92F);
      if (color_target_bins_) {
        // Color-addressed targets use the same color order as parse_bin_index.
        bin_color = colors::sphere_color(colors::kSortingColors[index].first);
        bin_color.a = 0.92F;
      }
      colors.push_back(object_color(bin.id, bin_color));
      objects.push_back(std::move(bin));
    }

    if (!planning_scene_.applyCollisionObjects(objects, colors)) {
      throw std::runtime_error(
          "failed to apply adaptive sorting collision scene");
    }
    RCLCPP_INFO(get_logger(),
                "Published sorting scene with %d bins and a transparent "
                "placeholder sphere",
                bin_count_);
  }

  void publish_markers() {
    visualization_msgs::msg::MarkerArray markers;
    const auto stamp = now();

    for (int index = 0; index < bin_count_; ++index) {
      // Color-target bins are identified by color, not by number.
      if (color_target_bins_) {
        markers.markers.push_back(delete_marker(stamp, "bin_labels", index));
      } else {
        auto label = marker(stamp, "bin_labels", index,
                            visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
        label.pose = pose(sorting_layout_.bin_x,
                          sorting_layout_.bin_y(index, bin_count_),
                          layout::kBinHeight + 0.035);
        label.scale.z = 0.05;
        label.color = color(0.08F, 0.08F, 0.08F);
        label.text = std::to_string(index + 1);
        markers.markers.push_back(label);
      }

      // Color-target names are agent-facing, so the display toggle does not
      // hide them; they stay fixed because label updates are rejected.
      if ((show_bin_color_labels_ || color_target_bins_) &&
          !bin_color_names_[index].empty()) {
        auto color_label =
            marker(stamp, "bin_color_labels", index,
                   visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
        color_label.pose = pose(
            sorting_layout_.bin_x + layout::kBinDepth / 2.0 + 0.045,
            sorting_layout_.bin_y(index, bin_count_), 0.03);
        color_label.scale.z = 0.028;
        color_label.color = colors::sphere_color(bin_color_names_[index]);
        color_label.text = bin_color_names_[index];
        markers.markers.push_back(color_label);
      } else {
        markers.markers.push_back(
            delete_marker(stamp, "bin_color_labels", index));
      }
    }

    auto pickup_label = marker(
        stamp, "pickup", 0, visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
    pickup_label.pose =
        pose(sorting_layout_.pickup_x, sorting_layout_.pickup_y, 0.16);
    pickup_label.scale.z = 0.045;
    pickup_label.color = color(0.08F, 0.08F, 0.08F);
    pickup_label.text = "PICKUP";
    markers.markers.push_back(pickup_label);

    const std::vector<bool> light_states = {light_1_on_, light_2_on_};
    for (int light_number = 1; light_number <= 2; ++light_number) {
      if (light_number <= context_light_count_) {
        add_light_markers(
            markers, stamp, light_number,
            sorting_layout_.pickup_x - 0.06 - 0.10 * light_number,
            sorting_layout_.pickup_y, light_states[light_number - 1]);
      } else {
        for (int offset = 0; offset < 3; ++offset) {
          markers.markers.push_back(delete_marker(stamp, "context_lights",
                                                  light_number * 10 + offset));
        }
      }
    }

    const auto status_color = color(0.08F, 0.08F, 0.08F);
    const auto last_color = last_reward_ > 0   ? color(0.10F, 0.75F, 0.18F)
                            : last_reward_ < 0 ? color(0.85F, 0.08F, 0.08F)
                                               : status_color;
    const std::vector<std::pair<std::string, std_msgs::msg::ColorRGBA>>
        status_lines = {
            {research_question_ + "|" + agent_name_, status_color},
            {total_phases_ > 0
                 ? "PHASE:" + std::to_string(phase_) + "/" +
                       std::to_string(total_phases_)
                 : "PHASE:--",
             status_color},
            {"TRIAL:" + std::to_string(trial_) + "/" +
                 std::to_string(total_trials_),
             status_color},
            {rolling_accuracy_ < 0.0
                 ? "ROLLING_ACC:--"
                 : "ROLLING_ACC:" +
                       std::to_string(
                           static_cast<int>(rolling_accuracy_ * 100.0 + 0.5)) +
                       "%",
             status_color},
            {last_reward_ > 0   ? "LAST:CORRECT"
             : last_reward_ < 0 ? "LAST:WRONG"
                                : "LAST:--",
             last_color},
        };
    for (std::size_t index = 0; index < status_lines.size(); ++index) {
      auto status = marker(stamp, "experiment_status", static_cast<int>(index),
                           visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
      status.pose =
          pose(0.28 - 0.05 * static_cast<double>(index), -0.40, 0.05);
      status.scale.z = 0.030;
      status.scale.x = status.scale.z * 0.6;
      status.color = status_lines[index].second;
      status.text = status_lines[index].first;
      markers.markers.push_back(status);
    }

    marker_publisher_->publish(markers);
  }

  void present_observation(
      const adaptive_sorting_bringup::srv::PresentObservation::Request &request,
      adaptive_sorting_bringup::srv::PresentObservation::Response &response) {
    if (!colors::is_supported(request.sphere_color)) {
      response.success = false;
      response.error = "unsupported sphere color: " + request.sphere_color;
      return;
    }
    const auto requested_color = colors::sphere_color(request.sphere_color);

    sphere_color_name_ = request.sphere_color;
    light_1_on_ = request.light_1_on;
    light_2_on_ = request.light_2_on;

    auto ball = object("sorting_sphere");
    ball.primitives.push_back(sphere(sorting_layout_.sphere_radius));
    ball.primitive_poses.push_back(
        pose(sorting_layout_.pickup_x, sorting_layout_.pickup_y,
             sorting_layout_.pickup_sphere_center_z));
    if (!planning_scene_.applyCollisionObjects(
            {ball}, {object_color(ball.id, requested_color)})) {
      response.success = false;
      response.error = "failed to reset the sorting sphere";
      return;
    }

    publish_markers();
    response.success = true;
    RCLCPP_INFO(get_logger(), "Presented %s sphere (light_1=%s, light_2=%s)",
                sphere_color_name_.c_str(), light_1_on_ ? "on" : "off",
                light_2_on_ ? "on" : "off");
  }

  void present_feedback(
      const adaptive_sorting_bringup::srv::PresentFeedback::Request &request,
      adaptive_sorting_bringup::srv::PresentFeedback::Response &response) {
    if (request.reward != -1 && request.reward != 1) {
      response.success = false;
      response.error = "feedback reward must be -1 or 1";
      return;
    }
    last_reward_ = request.reward;
    publish_markers();
    response.success = true;
    RCLCPP_INFO(get_logger(), "Presented sorting feedback: %d", last_reward_);
  }

  void present_experiment_status(
      const adaptive_sorting_bringup::srv::PresentExperimentStatus::Request
          &request,
      adaptive_sorting_bringup::srv::PresentExperimentStatus::Response
          &response) {
    if (request.research_question.empty() || request.agent.empty()) {
      response.success = false;
      response.error = "research_question and agent must not be empty";
      return;
    }
    if (request.total_trials <= 0 || request.trial < 0 ||
        request.trial > request.total_trials) {
      response.success = false;
      response.error = "trial must be between 0 and total_trials";
      return;
    }
    if (request.total_phases <= 0 || request.phase <= 0 ||
        request.phase > request.total_phases) {
      response.success = false;
      response.error = "phase must be between 1 and total_phases";
      return;
    }
    if ((request.rolling_accuracy < 0.0 || request.rolling_accuracy > 1.0) &&
        request.rolling_accuracy != -1.0) {
      response.success = false;
      response.error = "rolling_accuracy must be -1 or between 0 and 1";
      return;
    }
    if (request.last_reward != -1 && request.last_reward != 0 &&
        request.last_reward != 1) {
      response.success = false;
      response.error = "last_reward must be -1, 0, or 1";
      return;
    }

    research_question_ = request.research_question;
    agent_name_ = request.agent;
    trial_ = request.trial;
    total_trials_ = request.total_trials;
    phase_ = request.phase;
    total_phases_ = request.total_phases;
    rolling_accuracy_ = request.rolling_accuracy;
    last_reward_ = request.last_reward;
    publish_markers();
    response.success = true;
  }

  void present_bin_color_labels(
      const adaptive_sorting_bringup::srv::PresentBinColorLabels::Request
          &request,
      adaptive_sorting_bringup::srv::PresentBinColorLabels::Response
          &response) {
    if (color_target_bins_) {
      response.success = false;
      response.error = "color-target bins have fixed color labels";
      return;
    }
    if (request.bin_colors.size() > static_cast<std::size_t>(bin_count_)) {
      response.success = false;
      response.error = "bin_colors cannot contain more than " +
                       std::to_string(bin_count_) + " entries";
      return;
    }
    for (const auto &name : request.bin_colors) {
      if (!name.empty() && !colors::is_supported(name)) {
        response.success = false;
        response.error = "unsupported bin label color: " + name;
        return;
      }
    }

    bin_color_names_.assign(bin_count_, "");
    std::copy(request.bin_colors.begin(), request.bin_colors.end(),
              bin_color_names_.begin());
    publish_markers();
    response.success = true;
    RCLCPP_INFO(get_logger(), "Updated human-only bin color labels");
  }

  static void add_light_markers(visualization_msgs::msg::MarkerArray &markers,
                                const rclcpp::Time &stamp, int light_number,
                                double x, double y, bool is_on) {
    auto base = marker(stamp, "context_lights", light_number * 10,
                       visualization_msgs::msg::Marker::CYLINDER);
    base.pose = pose(x, y, 0.018);
    base.scale.x = 0.06;
    base.scale.y = 0.06;
    base.scale.z = 0.036;
    base.color = color(0.16F, 0.17F, 0.18F);
    markers.markers.push_back(base);

    auto lens = marker(stamp, "context_lights", light_number * 10 + 1,
                       visualization_msgs::msg::Marker::CYLINDER);
    lens.pose = pose(x, y, 0.041);
    lens.scale.x = 0.045;
    lens.scale.y = 0.045;
    lens.scale.z = 0.012;
    lens.color =
        is_on ? color(0.15F, 1.00F, 0.25F) : color(0.18F, 0.24F, 0.19F);
    markers.markers.push_back(lens);

    auto label = marker(stamp, "context_lights", light_number * 10 + 2,
                        visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
    label.pose = pose(x, y, 0.10);
    label.scale.z = 0.03;
    label.color = color(0.08F, 0.08F, 0.08F);
    label.text = "L" + std::to_string(light_number);
    markers.markers.push_back(label);
  }

  int bin_count_;
  layout::SortingLayout sorting_layout_{};
  std::string sphere_color_name_;
  bool light_1_on_;
  bool light_2_on_;
  int context_light_count_;
  bool show_bin_color_labels_;
  bool color_target_bins_;
  std::vector<std::string> bin_color_names_;
  std::string research_question_;
  std::string agent_name_ = "AGENT--";
  int trial_ = 0;
  int total_trials_ = 0;
  int phase_ = 0;
  int total_phases_ = 0;
  double rolling_accuracy_ = -1.0;
  int last_reward_ = 0;
  moveit::planning_interface::PlanningSceneInterface planning_scene_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      marker_publisher_;
  rclcpp::Service<adaptive_sorting_bringup::srv::PresentObservation>::SharedPtr
      observation_service_;
  rclcpp::Service<adaptive_sorting_bringup::srv::GetSortingObservation>::
      SharedPtr perception_service_;
  rclcpp::Service<adaptive_sorting_bringup::srv::GetSortingSceneConfig>::
      SharedPtr scene_config_service_;
  rclcpp::Service<adaptive_sorting_bringup::srv::PresentFeedback>::SharedPtr
      feedback_service_;
  rclcpp::Service<adaptive_sorting_bringup::srv::PresentExperimentStatus>::
      SharedPtr experiment_status_service_;
  rclcpp::Service<adaptive_sorting_bringup::srv::PresentBinColorLabels>::
      SharedPtr bin_color_labels_service_;
  rclcpp::TimerBase::SharedPtr marker_timer_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<SortingScenePublisher>());
  } catch (const std::exception &error) {
    RCLCPP_ERROR(rclcpp::get_logger("sorting_scene_publisher"), "%s",
                 error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
