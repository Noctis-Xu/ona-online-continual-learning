#ifndef ADAPTIVE_SORTING_BRINGUP__SORTING_COLORS_HPP_
#define ADAPTIVE_SORTING_BRINGUP__SORTING_COLORS_HPP_

#include <array>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <std_msgs/msg/color_rgba.hpp>

namespace adaptive_sorting_bringup::colors
{
using Rgb = std::array<float, 3>;
using ColorDefinition = std::pair<std::string_view, Rgb>;

inline constexpr std::array<ColorDefinition, 7> kSortingColors = {{
  {"red", {0.90F, 0.05F, 0.05F}},
  {"orange", {1.00F, 0.35F, 0.02F}},
  {"yellow", {1.00F, 0.75F, 0.02F}},
  {"green", {0.12F, 0.65F, 0.12F}},
  {"blue", {0.05F, 0.25F, 0.90F}},
  {"indigo", {0.20F, 0.05F, 0.55F}},
  {"violet", {0.55F, 0.10F, 0.75F}},
}};

inline std_msgs::msg::ColorRGBA rgba(
  float red, float green, float blue, float alpha = 1.0F)
{
  std_msgs::msg::ColorRGBA result;
  result.r = red;
  result.g = green;
  result.b = blue;
  result.a = alpha;
  return result;
}

inline bool is_supported(std::string_view name)
{
  for (const auto & definition : kSortingColors) {
    if (definition.first == name) {
      return true;
    }
  }
  return false;
}

inline std_msgs::msg::ColorRGBA sphere_color(std::string_view name)
{
  for (const auto & [candidate, rgb] : kSortingColors) {
    if (candidate == name) {
      return rgba(rgb[0], rgb[1], rgb[2]);
    }
  }
  return rgba(0.55F, 0.55F, 0.55F);
}

inline std::vector<std::string> supported_names()
{
  std::vector<std::string> result;
  result.reserve(kSortingColors.size());
  for (const auto & definition : kSortingColors) {
    result.emplace_back(definition.first);
  }
  return result;
}
}  // namespace adaptive_sorting_bringup::colors

#endif  // ADAPTIVE_SORTING_BRINGUP__SORTING_COLORS_HPP_
