#pragma once
// Chat-template formatting behind nibble::Chat (declared in nibble/nibble.h): picks the format from the
// model's chat template and builds each turn's prompt as a list of parts, marking which parts are special
// markup (parsed as control tokens) and which are user text (never parsed, so users cannot inject markup).
#include <string>
#include <string_view>
#include <vector>

#include "nibble/nibble.h"

namespace nibble {

// The prompt layouts nibble knows: ChatML (Qwen), Llama 3 header format, Zephyr (TinyLlama).
enum class ChatFormat { ChatML, Llama3, Zephyr };

// Picked from markers in the model's chat template ("<|im_start|>" ChatML,
// "<|start_header_id|>" Llama3, "<|assistant|>" Zephyr), else "qwen2" gives ChatML;
// otherwise throws Error("no chat template for architecture <a>") with <a> escaped.
ChatFormat chat_format_for(std::string_view arch, std::string_view chat_template);

// Qwen's default system prompt, used when Chat gets an empty one.
inline constexpr std::string_view kQwenDefaultSystem =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// The prompt parts of one chat turn. The first turn opens with the system
// message (Zephyr omits it when empty, as its template does); later turns open
// with the end-of-turn markup, closing the previous assistant reply (its end
// token is never forwarded into the cache). Markup parts are special; system and
// user text are not. BOS is never included. The views point into the arguments.
std::vector<PromptPart> chat_turn(ChatFormat, std::string_view system, std::string_view user, bool first);

// State of one conversation. Chat detects misuse of the Session behind its back ("chat context lost") by
// remembering what the Session looked like after the previous send.
struct Chat::Impl {
  Session& session;
  ChatFormat format;
  std::string system;
  bool started = false;  // the first turn's prompt has been prefilled
  bool full = false;     // a send returned ContextFull
  // session.n_past() and session.generations() as the last send left them;
  // expected_past is -1 once the cache was lost. Both are checked at the top of
  // send while started, so a direct generate call that leaves n_past unchanged
  // (max_tokens <= 0, or a throw) still counts as lost context.
  int64_t expected_past = 0;
  uint64_t expected_gens = 0;

  Impl(Session& s, ChatFormat f, std::string sys) : session(s), format(f), system(std::move(sys)) {}
};

}  // namespace nibble
