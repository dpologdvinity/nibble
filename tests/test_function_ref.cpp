#include <catch2/catch_test_macros.hpp>

#include "util/function_ref.h"

namespace {
struct Adder {
  int k;
  int operator()(int x) const { return x + k; }
};
int call_with_temp(nibble::FunctionRef<int(int)> f) { return f(1); }
}  // namespace

TEST_CASE("FunctionRef calls lambdas and functors") {
  int base = 10;
  auto lam = [&](int x) { return x + base; };
  nibble::FunctionRef<int(int)> a(lam);
  CHECK(a(1) == 11);
  base = 20;  // non-owning: sees the live capture
  CHECK(a(1) == 21);
  Adder add{5};
  nibble::FunctionRef<int(int)> b(add);
  CHECK(b(1) == 6);
  CHECK(call_with_temp(Adder{7}) == 8);  // rvalue lives until the end of the full expression
}

TEST_CASE("FunctionRef forwards reference arguments and void returns") {
  int sum = 0;
  auto f = [&](int& out, int v) { out += v; sum += v; };
  nibble::FunctionRef<void(int&, int)> r(f);
  int o = 1;
  r(o, 3);
  CHECK(o == 4);
  CHECK(sum == 3);
}
