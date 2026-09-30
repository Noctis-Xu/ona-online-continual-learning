#ifndef ADAPTIVE_SORTING_BRINGUP__SORTING_EXECUTION_STATE_HPP_
#define ADAPTIVE_SORTING_BRINGUP__SORTING_EXECUTION_STATE_HPP_

#include <optional>
#include <stdexcept>

namespace adaptive_sorting_bringup
{
enum class ExecutionState
{
  Ready,
  HoldingSphere,
  PlacedInBin,
};

class SortingExecutionState
{
public:
  ExecutionState value() const
  {
    return state_;
  }

  void require_pick() const
  {
    if (state_ != ExecutionState::Ready) {
      throw std::runtime_error("pick primitive requires the ready state");
    }
  }

  void pick_succeeded()
  {
    require_pick();
    state_ = ExecutionState::HoldingSphere;
  }

  void require_place() const
  {
    if (state_ != ExecutionState::HoldingSphere) {
      throw std::runtime_error("place primitive requires a successful pick first");
    }
  }

  void place_succeeded(int bin_index)
  {
    require_place();
    if (bin_index < 0) {
      throw std::invalid_argument("placed bin index must not be negative");
    }
    state_ = ExecutionState::PlacedInBin;
    placed_bin_ = bin_index;
  }

  int require_return_home() const
  {
    if (state_ != ExecutionState::PlacedInBin || !placed_bin_.has_value()) {
      throw std::runtime_error("return_home requires a successful place first");
    }
    return *placed_bin_;
  }

  void return_home_succeeded()
  {
    require_return_home();
    state_ = ExecutionState::Ready;
    placed_bin_.reset();
  }

private:
  ExecutionState state_ = ExecutionState::Ready;
  std::optional<int> placed_bin_;
};
}  // namespace adaptive_sorting_bringup

#endif  // ADAPTIVE_SORTING_BRINGUP__SORTING_EXECUTION_STATE_HPP_
