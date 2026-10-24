#pragma once
// A tiny non-owning callable reference (a stand-in for C++26 std::function_ref), used where a callback is
// only called while the caller is on the stack, e.g. thread-pool task bodies. Unlike std::function it never
// allocates, which keeps the decode loop's "no allocation after warm-up" guarantee.
#include <memory>
#include <type_traits>
#include <utility>

namespace nibble {

// Primary template, declared only; the partial specialization below takes a signature like void(int, int).
template <class Sig>
class FunctionRef;

// Non-owning reference to a callable: a pointer plus a trampoline, never allocates.
// The callable must outlive every call. Function lvalues are rejected; function-pointer
// variables are accepted. Not nullable.
template <class R, class... Args>
class FunctionRef<R(Args...)> {
 public:
  // Binds to f (any object callable as R(Args...), lvalue or rvalue); keeps only its address, so f must
  // stay alive for as long as this reference is called. Deliberately implicit so a lambda can be passed
  // directly as an argument.
  template <class F>
    requires(std::is_object_v<std::remove_reference_t<F>> && !std::is_same_v<std::remove_cvref_t<F>, FunctionRef> && std::is_invocable_r_v<R, F&, Args...>)
  FunctionRef(F&& f) noexcept  // NOLINT: implicit by design, like std::function_ref
      : obj_(const_cast<void*>(static_cast<const void*>(std::addressof(f)))),
        call_([](void* o, Args... a) -> R { return (*static_cast<std::add_pointer_t<F>>(o))(std::forward<Args>(a)...); }) {}

  // Calls the referenced callable with the given arguments.
  R operator()(Args... args) const { return call_(obj_, std::forward<Args>(args)...); }

 private:
  void* obj_;
  R (*call_)(void*, Args...);
};

}  // namespace nibble
