#pragma once
// ChatML formatting for nibble::Chat (declared in nibble/nibble.h).
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "nibble/nibble.h"

namespace nibble {

enum class ChatFormat { ChatML };

// ChatML for "qwen2"; otherwise throws Error("no chat template for architecture <a>")
// with <a> escaped by escape_control.
ChatFormat chat_format_for(std::string_view arch);

// Qwen's default system prompt, used when Chat gets an empty one.
inline constexpr std::string_view kQwenDefaultSystem =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

// The prompt parts of one ChatML turn. The first turn opens with the system
// message; later turns open with "<|im_end|>\n", closing the previous assistant
// reply (its end token is never forwarded into the cache). Markup parts are
// special; system and user text are not. The views point into the arguments.
std::vector<PromptPart> chatml_turn(std::string_view system, std::string_view user, bool first);

struct Chat::Impl {
  Session& session;
  std::string system;
  bool started = false;  // the first turn's prompt has been prefilled
  bool full = false;     // a send returned ContextFull
  std::vector<std::pair<std::string, bool>> last_parts;  // owned copy, for tests

  Impl(Session& s, std::string sys) : session(s), system(std::move(sys)) {}
};

// Test hook: the parts the last send passed to Session::generate.
struct ChatTestAccess {
  static const std::vector<std::pair<std::string, bool>>& last_parts(const Chat& c) { return c.impl_->last_parts; }
};

}  // namespace nibble
