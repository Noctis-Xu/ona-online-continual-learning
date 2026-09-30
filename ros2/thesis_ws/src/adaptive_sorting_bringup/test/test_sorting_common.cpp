#include <stdexcept>

#include <gtest/gtest.h>

#include "adaptive_sorting_bringup/sorting_colors.hpp"
#include "adaptive_sorting_bringup/sorting_execution_state.hpp"
#include "adaptive_sorting_bringup/sorting_layout.hpp"
#include "adaptive_sorting_bringup/sorting_validation.hpp"

namespace colors = adaptive_sorting_bringup::colors;
namespace layout = adaptive_sorting_bringup::layout;
using adaptive_sorting_bringup::parse_bin_index;
using adaptive_sorting_bringup::ExecutionState;
using adaptive_sorting_bringup::SortingExecutionState;
using adaptive_sorting_bringup::validate_bin_count;
using adaptive_sorting_bringup::validate_layout;
using adaptive_sorting_bringup::validate_motion_parameters;

TEST(SortingLayout, CentersBinsInStableOrder)
{
  const layout::SortingLayout sorting_layout{
    0.34, 0.45, 0.56, -0.10, 0.15, 0.035, 0.075, 0.05};
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(0, 1), -0.10);
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(0, 5), 0.20);
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(2, 5), -0.10);
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(4, 5), -0.40);
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(0, 7), 0.35);
  EXPECT_DOUBLE_EQ(sorting_layout.bin_y(6, 7), -0.55);
}

TEST(SortingColors, ExposesOneCanonicalPalette)
{
  EXPECT_TRUE(colors::is_supported("red"));
  EXPECT_TRUE(colors::is_supported("violet"));
  EXPECT_FALSE(colors::is_supported("purple"));
  EXPECT_EQ(colors::supported_names().size(), 7U);

  const auto blue = colors::sphere_color("blue");
  EXPECT_FLOAT_EQ(blue.r, 0.05F);
  EXPECT_FLOAT_EQ(blue.g, 0.25F);
  EXPECT_FLOAT_EQ(blue.b, 0.90F);
  EXPECT_FLOAT_EQ(blue.a, 1.0F);
}

TEST(SortingValidation, ParsesAvailableBinPrimitives)
{
  EXPECT_EQ(parse_bin_index("place_to_bin_red", 7), 0);
  EXPECT_EQ(parse_bin_index("place_to_bin_violet", 7), 6);
  EXPECT_THROW(parse_bin_index("place_to_bin_violet", 5), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_bin_unknown", 7), std::invalid_argument);
  EXPECT_EQ(parse_bin_index("place_to_bin1", 7), 0);
  EXPECT_EQ(parse_bin_index("place_to_bin7", 7), 6);
  EXPECT_THROW(parse_bin_index("place_to_bin_1", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_bin0", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_bin8", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_binx", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_bin1_extra", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("pick", 7), std::invalid_argument);
  EXPECT_THROW(parse_bin_index("place_to_bin6", 5), std::invalid_argument);
}

TEST(SortingValidation, RejectsInvalidParameters)
{
  EXPECT_NO_THROW(validate_bin_count(1));
  EXPECT_NO_THROW(validate_bin_count(7));
  EXPECT_THROW(validate_bin_count(0), std::invalid_argument);
  EXPECT_THROW(validate_bin_count(8), std::invalid_argument);

  EXPECT_NO_THROW(validate_motion_parameters(1.0, 0.5, 0.005, 5.0, 0.3, 0.34));
  EXPECT_THROW(
    validate_motion_parameters(0.0, 0.5, 0.005, 5.0, 0.3, 0.34),
    std::invalid_argument);
  EXPECT_THROW(
    validate_motion_parameters(0.5, 1.1, 0.005, 5.0, 0.3, 0.34),
    std::invalid_argument);
  EXPECT_THROW(
    validate_motion_parameters(0.5, 0.5, 0.0, 5.0, 0.3, 0.34),
    std::invalid_argument);

  const layout::SortingLayout valid_layout{
    0.34, 0.45, 0.56, -0.10, 0.15, 0.035, 0.075, 0.05};
  auto invalid_layout = valid_layout;
  invalid_layout.bin_sphere_center_z = 0.02;
  EXPECT_NO_THROW(validate_layout(valid_layout));
  EXPECT_THROW(validate_layout(invalid_layout), std::invalid_argument);
}

TEST(SortingExecutionState, AllowsOnlySuccessfulOrderedTransitions)
{
  SortingExecutionState state;
  EXPECT_EQ(state.value(), ExecutionState::Ready);
  EXPECT_THROW(state.require_place(), std::runtime_error);
  EXPECT_THROW(state.require_return_home(), std::runtime_error);

  state.require_pick();
  state.pick_succeeded();
  EXPECT_EQ(state.value(), ExecutionState::HoldingSphere);
  EXPECT_THROW(state.require_pick(), std::runtime_error);
  EXPECT_THROW(state.place_succeeded(-1), std::invalid_argument);
  EXPECT_EQ(state.value(), ExecutionState::HoldingSphere);

  state.require_place();
  state.place_succeeded(4);
  EXPECT_EQ(state.value(), ExecutionState::PlacedInBin);
  EXPECT_EQ(state.require_return_home(), 4);

  state.return_home_succeeded();
  EXPECT_EQ(state.value(), ExecutionState::Ready);
  EXPECT_THROW(state.require_return_home(), std::runtime_error);
}
