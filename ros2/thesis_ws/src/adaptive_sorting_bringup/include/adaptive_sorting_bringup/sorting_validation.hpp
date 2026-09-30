#ifndef ADAPTIVE_SORTING_BRINGUP__SORTING_VALIDATION_HPP_
#define ADAPTIVE_SORTING_BRINGUP__SORTING_VALIDATION_HPP_

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

#include "adaptive_sorting_bringup/sorting_layout.hpp"
#include "adaptive_sorting_bringup/sorting_colors.hpp"

namespace adaptive_sorting_bringup
{
inline constexpr int kMaximumBinCount = 7;

inline void validate_bin_count(int bin_count)
{
  if (bin_count < 1 || bin_count > kMaximumBinCount) {
    throw std::invalid_argument("bin_count must be between 1 and 7");
  }
}

inline void validate_motion_parameters(
  double velocity_scaling, double acceleration_scaling,
  double cache_start_tolerance, double planning_time,
  double pickup_hover_z, double bin_hover_z)
{
  if (velocity_scaling <= 0.0 || velocity_scaling > 1.0 ||
    acceleration_scaling <= 0.0 || acceleration_scaling > 1.0)
  {
    throw std::invalid_argument(
            "velocity_scaling and acceleration_scaling must be in (0, 1]");
  }
  if (cache_start_tolerance <= 0.0) {
    throw std::invalid_argument("cache_start_tolerance must be positive");
  }
  if (planning_time <= 0.0 || pickup_hover_z <= 0.0 || bin_hover_z <= 0.0) {
    throw std::invalid_argument(
            "planning_time and motion hover heights must be positive");
  }
}

inline void validate_layout(const layout::SortingLayout & layout)
{
  if (layout.bin_spacing <= 0.0 || layout.sphere_radius <= 0.0) {
    throw std::invalid_argument(
            "bin_spacing and sphere_radius must be positive");
  }
  if (layout.pickup_sphere_center_z < layout.sphere_radius ||
    layout.bin_sphere_center_z < layout.sphere_radius)
  {
    throw std::invalid_argument(
            "sphere center heights must not be below sphere_radius");
  }
}

inline int parse_bin_index(std::string_view primitive, int bin_count)
{
  validate_bin_count(bin_count);
  constexpr std::string_view prefix = "place_to_bin";
  if (primitive.substr(0, prefix.size()) != prefix) {
    throw std::invalid_argument(
            "unknown sorting primitive: " + std::string(primitive));
  }

  const auto suffix = primitive.substr(prefix.size());
  // Resolve a semantic target directly to its location in the default scene.
  // The selected sphere color is deliberately not involved in this lookup.
  if (!suffix.empty() && suffix.front() == '_') {
    for (int index = 0; index < bin_count; ++index) {
      if (suffix.substr(1) == colors::kSortingColors[index].first) {
        return index;
      }
    }
    throw std::invalid_argument("unknown color target: " + std::string(primitive));
  }

  int bin_number = 0;
  const auto result = std::from_chars(
    suffix.data(), suffix.data() + suffix.size(), bin_number);
  if (suffix.empty() || result.ec != std::errc() ||
    result.ptr != suffix.data() + suffix.size() ||
    bin_number < 1 || bin_number > kMaximumBinCount)
  {
    throw std::invalid_argument(
            "unknown sorting primitive: " + std::string(primitive));
  }
  if (bin_number > bin_count) {
    throw std::invalid_argument(
            std::string(primitive) + " is unavailable with bin_count=" +
            std::to_string(bin_count));
  }
  return bin_number - 1;
}
}  // namespace adaptive_sorting_bringup

#endif  // ADAPTIVE_SORTING_BRINGUP__SORTING_VALIDATION_HPP_
