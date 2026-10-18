#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "nibble/error.h"
#include "tiny_model.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"

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
  { TinyTokOpts o; o.model = "bert"; CHECK_THROWS_WITH(make(o), ContainsSubstring("bert")); }
  { TinyTokOpts o; o.drop_type = true; CHECK_THROWS_AS(make(o), nibble::Error); }
  { TinyTokOpts o; o.merges = {"h zz"}; CHECK_THROWS_WITH(make(o), ContainsSubstring("zz")); }
  { TinyTokOpts o; o.merges = {"nospace"}; CHECK_THROWS_AS(make(o), nibble::Error); }
}

TEST_CASE("llama-bpe pre accepted, bos and add_bos", "[tokenizer]") {
  { auto t = make(); CHECK(t.bos() == -1); CHECK_FALSE(t.add_bos()); }  // qwen2 default
  for (const char* pre : {"llama-bpe", "llama3", "llama-v3"}) {
    TinyTokOpts o; o.pre = pre; o.bos = 260;
    auto t = make(o);
    CHECK(t.bos() == 260);
    CHECK(t.add_bos());  // llama.cpp default for LLAMA3 pre
    o.add_bos = 0;
    CHECK_FALSE(make(o).add_bos());
  }
  { TinyTokOpts o; o.add_bos = 1; CHECK(make(o).add_bos()); }
  { TinyTokOpts o; o.bos = 9999; CHECK_THROWS_WITH(make(o), ContainsSubstring("bos_token_id")); }
  { TinyTokOpts o; o.bos_as_string = true; CHECK_THROWS_AS(make(o), nibble::Error); }
}

TEST_CASE("llama3 digits split in 3 and whole-word vocab hit", "[tokenizer]") {
  TinyTokOpts o;
  o.pre = "llama-bpe";
  o.extra = {"12", "123"};  // 261, 262
  auto t = make(o);
  CHECK(t.encode("12345", false) == Ids{262, '4' , '5'});
  CHECK(t.encode("1234", false) == Ids{262, '4'});
}

TEST_CASE("eog by name", "[tokenizer]") {
  TinyTokOpts o;
  o.extra = {"<|eot_id|>", "<|eom_id|>", "</s>", "<|end_of_text|>", "plain"};  // 261..265
  auto t = make(o);
  for (int id : {261, 262, 263, 264}) CHECK(t.is_eog(id));
  CHECK_FALSE(t.is_eog(265));
}

// ---- SentencePiece ("llama") ----
static const std::string SP = "\xE2\x96\x81";  // U+2581

static Tokenizer make_spm(int add_space_prefix = -1) {
  TinyTokOpts o;
  o.spm = true;
  o.add_space_prefix = add_space_prefix;
  return make(o);
}

TEST_CASE("spm space prefix and escaping", "[tokenizer][spm]") {
  auto t = make_spm();
  CHECK(t.add_bos());
  CHECK(t.bos() == 1);
  CHECK(t.encode("hi there", false) == Ids{t.find(SP + "hi"), t.find(SP + "there")});
  CHECK(t.encode("", false).empty());
  auto np = make_spm(0);
  CHECK(np.encode("hi", false) == Ids{np.find("hi")});
}

TEST_CASE("spm highest score merge wins", "[tokenizer][spm]") {
  auto t = make_spm();
  // SP+"a" (-20) is a candidate but "ab" (-1) outranks it; "bc" (-2) is then gone.
  CHECK(t.encode("abc", false) == Ids{t.find(SP), t.find("ab"), t.find("c")});
}

TEST_CASE("spm byte fallback", "[tokenizer][spm]") {
  auto t = make_spm();
  CHECK(t.encode("\xC3\xA9", false) == Ids{t.find(SP), 3 + 0xC3, 3 + 0xA9});
}

TEST_CASE("spm space prefix after special", "[tokenizer][spm]") {
  auto t = make_spm();
  CHECK(t.encode("a</s>b", true) == Ids{t.find(SP + "a"), 2, t.find(SP + "b")});
  CHECK(t.encode("</s>", true) == Ids{2});
}

TEST_CASE("spm decode", "[tokenizer][spm]") {
  auto t = make_spm();
  CHECK(t.decode_token(t.find(SP + "hi")) == " hi");
  CHECK(t.decode_token(0) == "");
  CHECK(t.decode_token(1) == "");
  nibble::tok::Utf8Streamer st;
  std::string s = st.push(t.decode_token(3 + 0xC3));
  CHECK(s.empty());
  s += st.push(t.decode_token(3 + 0xA9));
  CHECK(s == "\xC3\xA9");
}

TEST_CASE("spm rejects bad scores", "[tokenizer][spm]") {
  TinyTokOpts o;
  o.spm = true;
  o.bad_scores = 1;
  CHECK_THROWS_WITH(make(o), ContainsSubstring("scores length"));
  o.bad_scores = 2;
  CHECK_THROWS_WITH(make(o), ContainsSubstring("scores is not a float array"));
}
