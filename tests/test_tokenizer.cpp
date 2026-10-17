#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nibble/error.h"
#include "tiny_model.h"
#include "tokenizer/tokenizer.h"

using nibble::tok::Tokenizer;
using Ids = std::vector<int32_t>;
using Catch::Matchers::ContainsSubstring;

static Tokenizer make(const TinyTokOpts& o = {}) {
  auto bytes = tiny_tokenizer_gguf(o);
  return Tokenizer::from_gguf(nibble::gguf::File::parse(bytes));  // File non-owning; Tokenizer copies
}

TEST_CASE("bpe merges", "[tokenizer]") {
  auto t = make();
  CHECK(t.encode("hel", false) == Ids{257});
  CHECK(t.encode("hello", false) == Ids{257, 'l', 'o'});
  CHECK(t.encode(" w", false) == Ids{258});
  CHECK(t.encode("", false).empty());
  CHECK(t.n_vocab() == 261);
  CHECK(t.find("hel") == 257);
  CHECK(t.find("nope") == -1);
}

TEST_CASE("merge priority", "[tokenizer]") {
  TinyTokOpts o;
  o.merges = {"e l", "h e"};
  o.extra = {"el"};  // id 261
  auto t = make(o);
  CHECK(t.encode("hel", false) == Ids{'h', 261});
}

TEST_CASE("specials", "[tokenizer]") {
  auto t = make();
  CHECK(t.encode("a<|ctl|>b", true) == Ids{'a', 259, 'b'});
  for (auto id : t.encode("a<|ctl|>b", false)) CHECK(id != 259);
  CHECK(t.encode("x<tool>", false) == Ids{'x', 260});
  CHECK(t.decode_token(258) == " w");
  CHECK(t.decode_token(259) == "");
  CHECK(t.decode_token(260) == "<tool>");
  CHECK(t.eos() == 259);
  CHECK(t.is_eog(259));
  CHECK_FALSE(t.is_eog(260));
}

TEST_CASE("tokenizer errors", "[tokenizer]") {
  auto t = make();
  CHECK_THROWS_AS(t.encode("\xFF", false), nibble::Error);
  { TinyTokOpts o; o.pre = "deepseek-llm"; CHECK_THROWS_WITH(make(o), ContainsSubstring("deepseek-llm")); }
  { TinyTokOpts o; o.model = "llama"; CHECK_THROWS_WITH(make(o), ContainsSubstring("llama")); }
  { TinyTokOpts o; o.drop_type = true; CHECK_THROWS_AS(make(o), nibble::Error); }
  { TinyTokOpts o; o.merges = {"h zz"}; CHECK_THROWS_WITH(make(o), ContainsSubstring("zz")); }
  { TinyTokOpts o; o.merges = {"nospace"}; CHECK_THROWS_AS(make(o), nibble::Error); }
}
