#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.hpp>
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#include <moveit/robot_state/robot_state.hpp>
#include <moveit/utils/moveit_error_code.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

#include "adaptive_sorting_bringup/sorting_colors.hpp"
#include "adaptive_sorting_bringup/sorting_execution_state.hpp"
#include "adaptive_sorting_bringup/sorting_layout.hpp"
#include "adaptive_sorting_bringup/sorting_validation.hpp"
#include "adaptive_sorting_bringup/srv/execute_primitive.hpp"

namespace
{
constexpr char kWorldFrame[] = "world";
constexpr double kDownwardQuaternionX = 0.9238795;
constexpr double kDownwardQuaternionY = -0.3826834;
constexpr char kPlanningGroup[] = "panda_arm";
constexpr unsigned int kPlanningAttempts = 3;
namespace colors = adaptive_sorting_bringup::colors;
namespace layout = adaptive_sorting_bringup::layout;

template<typename T>
T required_parameter(const rclcpp::Node & node, const std::string & name)
{
  T value;
  if (!node.get_parameter(name, value)) {
    throw std::invalid_argument("required ROS parameter is missing: " + name);
  }
  return value;
}

geometry_msgs::msg::Pose pose(double x, double y, double z)
{
  geometry_msgs::msg::Pose result;
  result.position.x = x;
  result.position.y = y;
  result.position.z = z;
  result.orientation.w = 1.0;
  return result;
}

geometry_msgs::msg::Pose motion_pose(double x, double y, double z)
{
  auto result = pose(x, y, z);
  result.orientation.x = kDownwardQuaternionX;
  result.orientation.y = kDownwardQuaternionY;
  result.orientation.z = 0.0;
  result.orientation.w = 0.0;
  return result;
}
}  // namespace

class SortingPrimitiveExecutor
{
public:
  explicit SortingPrimitiveExecutor(const rclcpp::Node::SharedPtr & node)
  : node_(node), move_group_(node, kPlanningGroup)
  {
    bin_count_ = required_parameter<int>(*node_, "bin_count");
    velocity_scaling_ = required_parameter<double>(*node_, "velocity_scaling");
    acceleration_scaling_ =
      required_parameter<double>(*node_, "acceleration_scaling");
    cache_trajectories_ = required_parameter<bool>(*node_, "cache_trajectories");
    cache_start_tolerance_ =
      required_parameter<double>(*node_, "cache_start_tolerance");
    planning_time_ = required_parameter<double>(*node_, "planning_time");
    pickup_hover_z_ = required_parameter<double>(*node_, "pickup_hover_z");
    bin_hover_z_ = required_parameter<double>(*node_, "bin_hover_z");
    sorting_layout_ = {
      required_parameter<double>(*node_, "pickup_x"),
      required_parameter<double>(*node_, "pickup_y"),
      required_parameter<double>(*node_, "bin_x"),
      required_parameter<double>(*node_, "bin_y_offset"),
      required_parameter<double>(*node_, "bin_spacing"),
      required_parameter<double>(*node_, "sphere_radius"),
      required_parameter<double>(*node_, "pickup_sphere_center_z"),
      required_parameter<double>(*node_, "bin_sphere_center_z"),
    };
    adaptive_sorting_bringup::validate_bin_count(bin_count_);
    adaptive_sorting_bringup::validate_layout(sorting_layout_);
    adaptive_sorting_bringup::validate_motion_parameters(
      velocity_scaling_, acceleration_scaling_, cache_start_tolerance_,
      planning_time_, pickup_hover_z_, bin_hover_z_);

    move_group_.setPoseReferenceFrame(kWorldFrame);
    move_group_.setPlanningTime(planning_time_);
    move_group_.setNumPlanningAttempts(kPlanningAttempts);
    move_group_.setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_.setMaxAccelerationScalingFactor(acceleration_scaling_);

    service_group_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    if (cache_trajectories_) {
      cache_timer_ = node_->create_wall_timer(
        std::chrono::seconds(1), [this]() {warm_trajectory_cache();}, service_group_);
      RCLCPP_INFO(
        node_->get_logger(),
        "Preplanning fixed-scene trajectories using %u attempts at velocity %.2f, "
        "acceleration %.2f",
        kPlanningAttempts, velocity_scaling_, acceleration_scaling_);
    } else {
      create_service();
    }
  }

private:
  using Request = adaptive_sorting_bringup::srv::ExecutePrimitive::Request;
  using Response = adaptive_sorting_bringup::srv::ExecutePrimitive::Response;
  using Plan = moveit::planning_interface::MoveGroupInterface::Plan;
  using TargetSetter = std::function<void()>;

  struct MotionMetrics
  {
    double planning_seconds = 0.0;
    double execution_seconds = 0.0;
    bool cache_hit = false;
  };

  struct CachedTrajectory
  {
    Plan plan;
    std::vector<std::string> joint_names;
    std::vector<double> start_positions;
  };

  void create_service()
  {
    service_ = node_->create_service<
      adaptive_sorting_bringup::srv::ExecutePrimitive>(
      "execute_sorting_primitive",
      [this](
        const adaptive_sorting_bringup::srv::ExecutePrimitive::Request::SharedPtr request,
        adaptive_sorting_bringup::srv::ExecutePrimitive::Response::SharedPtr response)
      {
        execute(*request, *response);
      },
      rclcpp::ServicesQoS(), service_group_);
    RCLCPP_INFO(
      node_->get_logger(), "Sorting primitives ready for %d bins (end effector: %s)",
      bin_count_, move_group_.getEndEffectorLink().c_str());
  }

  void execute(const Request & request, Response & response)
  {
    const auto started = std::chrono::steady_clock::now();
    MotionMetrics metrics;
    try {
      if (request.primitive == "pick") {
        execute_pick(metrics);
      } else if (request.primitive == "return_home") {
        execute_return_home(metrics);
      } else {
        const int bin_index = adaptive_sorting_bringup::parse_bin_index(
          request.primitive, bin_count_);
        execute_place(bin_index, request.sphere_color, metrics);
      }
      response.success = true;
    } catch (const std::exception & error) {
      response.success = false;
      response.error = error.what();
      RCLCPP_ERROR(
        node_->get_logger(), "Primitive %s failed: %s",
        request.primitive.c_str(), error.what());
    }
    response.duration_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    response.planning_duration_seconds = metrics.planning_seconds;
    response.execution_duration_seconds = metrics.execution_seconds;
    response.cache_hit = metrics.cache_hit;
    RCLCPP_INFO(
      node_->get_logger(), "%s timing: total=%.3fs plan=%.3fs execute=%.3fs cache=%s",
      request.primitive.c_str(), response.duration_seconds,
      response.planning_duration_seconds, response.execution_duration_seconds,
      response.cache_hit ? "hit" : "miss");
  }

  void execute_pick(MotionMetrics & metrics)
  {
    execution_state_.require_pick();
    move_to_pose(
      "pick", sorting_layout_.pickup_x, sorting_layout_.pickup_y,
      pickup_hover_z_, metrics);
    planning_scene_.removeCollisionObjects({"sorting_sphere"});
    execution_state_.pick_succeeded();
    RCLCPP_INFO(node_->get_logger(), "Executed pick primitive");
  }

  void execute_place(
    int bin_index, const std::string & color_name, MotionMetrics & metrics)
  {
    execution_state_.require_place();
    if (!colors::is_supported(color_name)) {
      throw std::invalid_argument("unsupported sphere color: " + color_name);
    }
    const auto primitive = "place_to_bin" + std::to_string(bin_index + 1);
    move_to_pose(
      primitive, sorting_layout_.bin_x,
      sorting_layout_.bin_y(bin_index, bin_count_), bin_hover_z_, metrics);
    place_sphere_in_bin(bin_index, color_name);
    execution_state_.place_succeeded(bin_index);
    RCLCPP_INFO(
      node_->get_logger(), "Executed place_to_bin%d primitive", bin_index + 1);
  }

  void execute_return_home(MotionMetrics & metrics)
  {
    const int placed_bin = execution_state_.require_return_home();
    move_to_named_target(
      "return_home_from_bin_" + std::to_string(placed_bin + 1),
      "ready", metrics);
    execution_state_.return_home_succeeded();
  }

  void move_to_pose(
    const std::string & cache_key, double x, double y, double z,
    MotionMetrics & metrics)
  {
    const auto target = motion_pose(x, y, z);
    plan_and_execute(
      cache_key,
      [this, target]() {
        if (!move_group_.setPoseTarget(target)) {
          throw std::runtime_error("pose target is outside the Panda joint bounds");
        }
      },
      metrics);
  }

  void move_to_named_target(
    const std::string & cache_key, const std::string & target,
    MotionMetrics & metrics)
  {
    plan_and_execute(
      cache_key,
      [this, target]() {
        if (!move_group_.setNamedTarget(target)) {
          throw std::runtime_error("unknown Panda named target: " + target);
        }
      },
      metrics);
    RCLCPP_INFO(node_->get_logger(), "Executed return_home primitive");
  }

  void plan_and_execute(
    const std::string & cache_key, const TargetSetter & set_target,
    MotionMetrics & metrics)
  {
    const auto current_state = move_group_.getCurrentState(2.0);
    if (!current_state) {
      throw std::runtime_error("current robot state is unavailable");
    }

    Plan plan;
    const auto cached = trajectory_cache_.find(cache_key);
    if (cache_trajectories_ && cached != trajectory_cache_.end() &&
      cache_start_matches(cached->second, *current_state))
    {
      plan = cached->second.plan;
      metrics.cache_hit = true;
    } else {
      if (cached != trajectory_cache_.end()) {
        RCLCPP_WARN(
          node_->get_logger(), "Cached %s start state differs; replanning",
          cache_key.c_str());
      }
      const auto planning_started = std::chrono::steady_clock::now();
      try {
        plan = plan_from_state(*current_state, set_target);
      } catch (...) {
        metrics.planning_seconds = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - planning_started).count();
        throw;
      }
      metrics.planning_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_started).count();
      if (cache_trajectories_) {
        trajectory_cache_[cache_key] = make_cached_trajectory(plan, *current_state);
      }
    }

    const auto execution_started = std::chrono::steady_clock::now();
    if (move_group_.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      metrics.execution_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - execution_started).count();
      throw std::runtime_error("trajectory controller execution failed");
    }
    metrics.execution_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - execution_started).count();
  }

  Plan plan_from_state(
    const moveit::core::RobotState & start_state,
    const TargetSetter & set_target)
  {
    move_group_.setStartState(start_state);
    try {
      set_target();
    } catch (...) {
      move_group_.clearPoseTargets();
      throw;
    }

    Plan plan;
    const auto result = move_group_.plan(plan);
    move_group_.clearPoseTargets();
    if (result != moveit::core::MoveItErrorCode::SUCCESS) {
      throw std::runtime_error("MoveIt planning failed");
    }
    return plan;
  }

  CachedTrajectory make_cached_trajectory(
    const Plan & plan, const moveit::core::RobotState & start_state) const
  {
    CachedTrajectory cached;
    cached.plan = plan;
    cached.joint_names = move_group_.getJointNames();
    cached.start_positions.reserve(cached.joint_names.size());
    for (const auto & joint_name : cached.joint_names) {
      cached.start_positions.push_back(start_state.getVariablePosition(joint_name));
    }
    return cached;
  }

  bool cache_start_matches(
    const CachedTrajectory & cached,
    const moveit::core::RobotState & current_state) const
  {
    for (std::size_t index = 0; index < cached.joint_names.size(); ++index) {
      if (std::abs(
          current_state.getVariablePosition(cached.joint_names[index]) -
          cached.start_positions[index]) > cache_start_tolerance_)
      {
        return false;
      }
    }
    return true;
  }

  moveit::core::RobotState trajectory_end_state(
    const moveit::core::RobotState & start_state, const Plan & plan) const
  {
    const auto & trajectory = plan.trajectory.joint_trajectory;
    if (trajectory.points.empty()) {
      throw std::runtime_error("planned trajectory contains no points");
    }
    const auto & positions = trajectory.points.back().positions;
    if (positions.size() != trajectory.joint_names.size()) {
      throw std::runtime_error("planned trajectory has inconsistent joint positions");
    }

    moveit::core::RobotState result(start_state);
    for (std::size_t index = 0; index < trajectory.joint_names.size(); ++index) {
      result.setVariablePosition(trajectory.joint_names[index], positions[index]);
    }
    result.update();
    return result;
  }

  moveit::core::RobotState preplan(
    const std::string & cache_key,
    const moveit::core::RobotState & start_state,
    const TargetSetter & set_target)
  {
    const auto started = std::chrono::steady_clock::now();
    const auto plan = plan_from_state(start_state, set_target);
    trajectory_cache_[cache_key] = make_cached_trajectory(plan, start_state);
    const auto planning_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started).count();
    RCLCPP_INFO(
      node_->get_logger(), "Cached %s in %.3fs", cache_key.c_str(), planning_seconds);
    return trajectory_end_state(start_state, plan);
  }

  void warm_trajectory_cache()
  {
    cache_timer_->cancel();
    const auto warmup_started = std::chrono::steady_clock::now();
    try {
      const auto current_state = move_group_.getCurrentState(2.0);
      if (!current_state) {
        throw std::runtime_error("current robot state is unavailable");
      }

      moveit::core::RobotState ready_state(*current_state);
      for (const auto & [joint_name, value] : move_group_.getNamedTargetValues("ready")) {
        ready_state.setVariablePosition(joint_name, value);
      }
      ready_state.update();

      const auto pickup_target = motion_pose(
        sorting_layout_.pickup_x, sorting_layout_.pickup_y, pickup_hover_z_);
      const auto pickup_state = preplan(
        "pick", ready_state,
        [this, pickup_target]() {
          if (!move_group_.setPoseTarget(pickup_target)) {
            throw std::runtime_error("pickup target is outside the Panda joint bounds");
          }
        });

      for (int bin_index = 0; bin_index < bin_count_; ++bin_index) {
        const auto bin_number = bin_index + 1;
        const auto bin_target = motion_pose(
          sorting_layout_.bin_x,
          sorting_layout_.bin_y(bin_index, bin_count_), bin_hover_z_);
        const auto bin_state = preplan(
          "place_to_bin" + std::to_string(bin_number), pickup_state,
          [this, bin_target]() {
            if (!move_group_.setPoseTarget(bin_target)) {
              throw std::runtime_error("bin target is outside the Panda joint bounds");
            }
          });
        preplan(
          "return_home_from_bin_" + std::to_string(bin_number), bin_state,
          [this]() {
            if (!move_group_.setNamedTarget("ready")) {
              throw std::runtime_error("unknown Panda named target: ready");
            }
          });
      }
      RCLCPP_INFO(
        node_->get_logger(),
        "Trajectory cache ready with %zu fixed-scene paths in %.3fs",
        trajectory_cache_.size(),
        std::chrono::duration<double>(
          std::chrono::steady_clock::now() - warmup_started).count());
    } catch (const std::exception & error) {
      trajectory_cache_.clear();
      RCLCPP_WARN(
        node_->get_logger(),
        "Trajectory preplanning failed (%s); paths will be cached on first use",
        error.what());
    }
    move_group_.setStartStateToCurrentState();
    move_group_.clearPoseTargets();
    create_service();
  }

  void place_sphere_in_bin(int bin_index, const std::string & color_name)
  {
    moveit_msgs::msg::CollisionObject ball;
    ball.header.frame_id = kWorldFrame;
    ball.id = "sorting_sphere";
    ball.operation = moveit_msgs::msg::CollisionObject::ADD;

    shape_msgs::msg::SolidPrimitive sphere;
    sphere.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    sphere.dimensions = {sorting_layout_.sphere_radius};
    ball.primitives.push_back(sphere);
    ball.primitive_poses.push_back(
      pose(sorting_layout_.bin_x,
        sorting_layout_.bin_y(bin_index, bin_count_),
        sorting_layout_.bin_sphere_center_z));

    moveit_msgs::msg::ObjectColor object_color;
    object_color.id = ball.id;
    object_color.color = colors::sphere_color(color_name);
    if (!planning_scene_.applyCollisionObjects({ball}, {object_color})) {
      throw std::runtime_error("failed to place sphere in planning scene");
    }
  }

  rclcpp::Node::SharedPtr node_;
  int bin_count_;
  double velocity_scaling_;
  double acceleration_scaling_;
  bool cache_trajectories_;
  double cache_start_tolerance_;
  double planning_time_;
  double pickup_hover_z_;
  double bin_hover_z_;
  layout::SortingLayout sorting_layout_{};
  adaptive_sorting_bringup::SortingExecutionState execution_state_;
  moveit::planning_interface::MoveGroupInterface move_group_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_;
  std::map<std::string, CachedTrajectory> trajectory_cache_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<adaptive_sorting_bringup::srv::ExecutePrimitive>::SharedPtr service_;
  rclcpp::TimerBase::SharedPtr cache_timer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    const auto options =
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true);
    const auto node = rclcpp::Node::make_shared("sorting_primitive_executor", options);
    const auto primitive_executor = std::make_shared<SortingPrimitiveExecutor>(node);
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception & error) {
    RCLCPP_ERROR(
      rclcpp::get_logger("sorting_primitive_executor"), "%s", error.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}
