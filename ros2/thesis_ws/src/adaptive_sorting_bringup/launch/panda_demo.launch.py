"""Launch Panda MoveIt with mock hardware and the sorting scene."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def _load_sorting_parameters(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        scene = config["scene"]
        motion = config["motion"]
        cache_trajectories = motion["cache_trajectories"]
        if not isinstance(cache_trajectories, bool):
            raise TypeError("motion.cache_trajectories must be a boolean")
        scene_parameters = {
            name: float(scene[name])
            for name in (
                "pickup_x",
                "pickup_y",
                "bin_x",
                "bin_y_offset",
                "bin_spacing",
                "sphere_radius",
                "pickup_sphere_center_z",
                "bin_sphere_center_z",
            )
        }
        motion_parameters = {
            "pickup_hover_z": float(motion["pickup_hover_z"]),
            "bin_hover_z": float(motion["bin_hover_z"]),
            "planning_time": float(motion["planning_time"]),
            "velocity_scaling": float(motion["velocity_scaling"]),
            "acceleration_scaling": float(motion["acceleration_scaling"]),
            "cache_trajectories": cache_trajectories,
            "cache_start_tolerance": float(motion["cache_start_tolerance"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid sorting parameter file: {path}") from exc
    return scene_parameters, motion_parameters


def generate_launch_description() -> LaunchDescription:
    use_rviz = LaunchConfiguration("use_rviz")
    use_sorting_scene = LaunchConfiguration("use_sorting_scene")
    bin_count = LaunchConfiguration("bin_count")
    research_question = LaunchConfiguration("research_question")
    light_1_on = LaunchConfiguration("light_1_on")
    light_2_on = LaunchConfiguration("light_2_on")
    context_light_count = LaunchConfiguration("context_light_count")
    show_bin_color_labels = LaunchConfiguration("show_bin_color_labels")
    color_target_bins = LaunchConfiguration("color_target_bins")
    velocity_scaling = LaunchConfiguration("velocity_scaling")
    acceleration_scaling = LaunchConfiguration("acceleration_scaling")
    cache_trajectories = LaunchConfiguration("cache_trajectories")

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(
            file_path="config/panda.urdf.xacro",
            mappings={"ros2_control_hardware_type": "mock_components"},
        )
        .robot_description_semantic(file_path="config/panda.srdf")
        .planning_scene_monitor(
            publish_robot_description=True,
            publish_robot_description_semantic=True,
        )
        .trajectory_execution(file_path="config/gripper_moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    # The experiment visualizes executed controller motion, not MoveIt's
    # separate planned-path animation.  Publishing it also makes disabled
    # MoveIt RViz displays report that they have no loaded robot model.
    ompl_config = moveit_config.planning_pipelines["ompl"]
    ompl_config["response_adapters"] = [
        adapter
        for adapter in ompl_config["response_adapters"]
        if adapter != "default_planning_response_adapters/DisplayMotionPath"
    ]

    panda_config_dir = Path(
        get_package_share_directory("moveit_resources_panda_moveit_config")
    )
    controller_config = panda_config_dir / "config" / "ros2_controllers.yaml"
    bringup_share_dir = Path(
        get_package_share_directory("adaptive_sorting_bringup")
    )
    scene_parameters, motion_parameters = _load_sorting_parameters(
        bringup_share_dir / "config" / "sorting_parameters.yaml"
    )
    rviz_config = bringup_share_dir / "config" / "sorting.rviz"
    package_library_dir = Path(get_package_prefix("adaptive_sorting_bringup")) / "lib"
    existing_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    package_runtime_env = {
        "LD_LIBRARY_PATH": (
            f"{package_library_dir}{os.pathsep}{existing_library_path}"
            if existing_library_path
            else str(package_library_dir)
        )
    }

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict()],
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", str(rviz_config)],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        condition=IfCondition(use_rviz),
    )
    world_to_panda = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="world_to_panda",
        output="log",
        arguments=[
            "--x",
            "0.0",
            "--y",
            "0.0",
            "--z",
            "0.0",
            "--roll",
            "0.0",
            "--pitch",
            "0.0",
            "--yaw",
            "0.0",
            "--frame-id",
            "world",
            "--child-frame-id",
            "panda_link0",
        ],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )
    ros2_control = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[str(controller_config)],
        remappings=[
            ("/controller_manager/robot_description", "/robot_description")
        ],
    )
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
    )
    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_arm_controller", "-c", "/controller_manager"],
    )
    hand_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["panda_hand_controller", "-c", "/controller_manager"],
    )
    sorting_scene = Node(
        package="adaptive_sorting_bringup",
        executable="sorting_scene_publisher",
        output="screen",
        additional_env=package_runtime_env,
        parameters=[
            {
                "bin_count": ParameterValue(bin_count, value_type=int),
                "research_question": research_question,
                "light_1_on": ParameterValue(light_1_on, value_type=bool),
                "light_2_on": ParameterValue(light_2_on, value_type=bool),
                "context_light_count": ParameterValue(
                    context_light_count, value_type=int
                ),
                "show_bin_color_labels": ParameterValue(
                    show_bin_color_labels, value_type=bool
                ),
                "color_target_bins": ParameterValue(
                    color_target_bins, value_type=bool
                ),
                **scene_parameters,
            }
        ],
        condition=IfCondition(use_sorting_scene),
    )
    primitive_executor = Node(
        package="adaptive_sorting_bringup",
        executable="sorting_primitive_executor",
        output="screen",
        additional_env=package_runtime_env,
        parameters=[
            moveit_config.to_dict(),
            {
                "bin_count": ParameterValue(bin_count, value_type=int),
                "velocity_scaling": ParameterValue(
                    velocity_scaling, value_type=float
                ),
                "acceleration_scaling": ParameterValue(
                    acceleration_scaling, value_type=float
                ),
                "cache_trajectories": ParameterValue(
                    cache_trajectories, value_type=bool
                ),
                "cache_start_tolerance": motion_parameters[
                    "cache_start_tolerance"
                ],
                "planning_time": motion_parameters["planning_time"],
                "pickup_hover_z": motion_parameters["pickup_hover_z"],
                "bin_hover_z": motion_parameters["bin_hover_z"],
                **scene_parameters,
            },
        ],
        condition=IfCondition(use_sorting_scene),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz with the Panda MoveIt configuration.",
            ),
            DeclareLaunchArgument(
                "use_sorting_scene",
                default_value="true",
                description="Add the sorting table, pickup area, sphere, and bins.",
            ),
            DeclareLaunchArgument(
                "bin_count",
                default_value="5",
                description="Number of destination bins to display (1-7).",
            ),
            DeclareLaunchArgument(
                "research_question",
                default_value="RQ--",
                description="Research question shown before a client connects.",
            ),
            DeclareLaunchArgument(
                "light_1_on",
                default_value="false",
                description="State of the task-relevant context light.",
            ),
            DeclareLaunchArgument(
                "light_2_on",
                default_value="false",
                description="State of the irrelevant context light.",
            ),
            DeclareLaunchArgument(
                "context_light_count",
                default_value="2",
                description="Number of context lights shown in RViz (0-2).",
            ),
            DeclareLaunchArgument(
                "show_bin_color_labels",
                default_value="true",
                description="Show human-only color-to-bin labels in RViz.",
            ),
            DeclareLaunchArgument(
                "color_target_bins",
                default_value="false",
                description=(
                    "Color each bin as its fixed color target instead of "
                    "numbering it (experiment 4b)."
                ),
            ),
            DeclareLaunchArgument(
                "velocity_scaling",
                default_value=str(motion_parameters["velocity_scaling"]),
                description="MoveIt velocity scaling in the interval (0, 1].",
            ),
            DeclareLaunchArgument(
                "acceleration_scaling",
                default_value=str(motion_parameters["acceleration_scaling"]),
                description="MoveIt acceleration scaling in the interval (0, 1].",
            ),
            DeclareLaunchArgument(
                "cache_trajectories",
                default_value=(
                    "true" if motion_parameters["cache_trajectories"] else "false"
                ),
                description="Preplan and reuse fixed-scene sorting trajectories.",
            ),
            world_to_panda,
            robot_state_publisher,
            move_group,
            ros2_control,
            joint_state_broadcaster,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=joint_state_broadcaster,
                    on_exit=[arm_controller],
                )
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=arm_controller,
                    on_exit=[
                        hand_controller,
                        primitive_executor,
                    ],
                )
            ),
            sorting_scene,
            rviz,
        ]
    )
