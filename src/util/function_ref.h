#pragma once
#include <memory>
#include <type_traits>
#include <utility>

namespace nibble {

template <class Sig>
class FunctionRef;

// Non-owning reference to a callable: a pointer plus a trampoline, never allocates.
// The callable (an object, not a bare function) must outlive every call. Not nullable.
template <class R, class... Args>
class FunctionRef<R(Args...)> {
 public:
  template <class F>
    requires(std::is_object_v<std::remove_reference_t<F>> && !std::is_same_v<std::remove_cvref_t<F>, FunctionRef> && std::is_invocable_r_v<R, F&, Args...>)
  FunctionRef(F&& f) noexcept  // NOLINT: implicit by design, like std::function_ref
      : obj_(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
        call_([](void* o, Args... a) -> R { return (*static_cast<std::add_pointer_t<F>>(o))(std::forward<Args>(a)...); }) {}

  R operator()(Args... args) const { return call_(obj_, std::forward<Args>(args)...); }

 private:
  void* obj_;
  R (*call_)(void*, Args...);
};

}  // namespace nibble
