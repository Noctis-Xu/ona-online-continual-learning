#ifndef ADAPTIVE_SORTING_BRINGUP__SORTING_LAYOUT_HPP_
#define ADAPTIVE_SORTING_BRINGUP__SORTING_LAYOUT_HPP_

namespace adaptive_sorting_bringup::layout
{
constexpr double kTableTop = -0.01;
constexpr double kBinDepth = 0.14;
constexpr double kBinWidth = 0.13;
constexpr double kBinHeight = 0.16;
constexpr double kWallThickness = 0.012;

struct SortingLayout
{
  double pickup_x;
  double pickup_y;
  double bin_x;
  double bin_y_offset;
  double bin_spacing;
  double sphere_radius;
  double pickup_sphere_center_z;
  double bin_sphere_center_z;

  double bin_y(int index, int count) const
  {
    return ((static_cast<double>(count) - 1.0) / 2.0 -
           static_cast<double>(index)) * bin_spacing + bin_y_offset;
  }
};
}  // namespace adaptive_sorting_bringup::layout

#endif  // ADAPTIVE_SORTING_BRINGUP__SORTING_LAYOUT_HPP_
