// SentencePiece BPE merge loop (see spm.h). Line references (llama-vocab.cpp:N) point at the code ported.
#include "tokenizer/spm.h"

#include <queue>

namespace nibble::tok {

namespace {

// A symbol: byte range of the text plus doubly linked neighbours (-1 at the ends). Merging a pair absorbs the
// right symbol into the left one (right.len becomes 0).
struct Sym {
  size_t off, len;
  int prev, next;
};
// A candidate merge of two adjacent symbols, ordered by the vocabulary score of the merged text.
struct Bigram {
  int left, right;
  float score;
  size_t size;  // combined byte length; detects stale entries
};
// llama-vocab.cpp:99-102: highest score first, ties to the lowest left index.
struct BigramCmp {
  bool operator()(const Bigram& l, const Bigram& r) const {
    return l.score < r.score || (l.score == r.score && l.left > r.left);
  }
};

}  // namespace

// Algorithm: one symbol per UTF-8 character; queue every adjacent pair whose concatenation is a vocabulary
// token; repeatedly pop the best-scoring pair (skipping stale ones), merge it and queue the new
// neighbouring pairs; finally emit each remaining symbol as its token, or as byte tokens if the symbol is
// not in the vocabulary.
void spm_encode(const SpmVocab& v, std::string_view text, std::vector<int32_t>& out) {
  // Split into UTF-8 code points without validating; a truncated tail is clamped (llama-vocab.cpp:120-131).
  std::vector<Sym> syms;
  for (size_t off = 0; off < text.size();) {
    const unsigned char c = static_cast<unsigned char>(text[off]);
    size_t n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    n = std::min(n, text.size() - off);
    syms.push_back({off, n, static_cast<int>(syms.size()) - 1, -1});
    off += n;
  }
  for (size_t k = 0; k + 1 < syms.size(); ++k) syms[k].next = static_cast<int>(k + 1);

  std::priority_queue<Bigram, std::vector<Bigram>, BigramCmp> q;
  // llama-vocab.cpp:204-232: queue the pair if its concatenation is a vocab token.
  auto try_add = [&](int l, int r) {
    if (l < 0 || r < 0) return;
    const size_t size = syms[l].len + syms[r].len;
    auto it = v.ids.find(std::string(text.substr(syms[l].off, size)));
    if (it != v.ids.end()) q.push({l, r, v.scores[static_cast<size_t>(it->second)], size});
  };
  for (int k = 1; k < static_cast<int>(syms.size()); ++k) try_add(k - 1, k);  // :133-136

  while (!q.empty()) {  // :139-167
    const Bigram b = q.top();
    q.pop();
    Sym& L = syms[b.left];
    Sym& R = syms[b.right];
    if (L.len == 0 || R.len == 0 || L.len + R.len != b.size) continue;  // already merged
    L.len += R.len;
    R.len = 0;
    L.next = R.next;
    if (R.next >= 0) syms[R.next].prev = b.left;
    try_add(L.prev, b.left);
    try_add(b.left, L.next);
  }

  // resegment (:171-202). llama.cpp's rev_merge branch is unreachable: its keys are vocab tokens, and
  // a symbol that is not a vocab token is never a key.
  for (int k = 0; k >= 0 && !syms.empty(); k = syms[k].next) {
    const std::string s(text.substr(syms[k].off, syms[k].len));
    auto it = v.ids.find(s);
    if (it != v.ids.end()) {
      out.push_back(it->second);
      continue;
    }
    for (unsigned char byte : s) {
      const int32_t id = v.byte_ids[byte];
      out.push_back(id >= 0 ? id : v.unk);
    }
  }
}

}  // namespace nibble::tok
