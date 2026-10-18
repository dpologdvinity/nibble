#include "chat.h"

#include "util/escape.h"

namespace nibble {

ChatFormat chat_format_for(std::string_view arch) {
  if (arch == "qwen2") return ChatFormat::ChatML;
  throw Error("no chat template for architecture " + escape_control(arch));
}

std::vector<PromptPart> chatml_turn(std::string_view system, std::string_view user, bool first) {
  if (first)
    return {{"<|im_start|>system\n", true},
            {system, false},
            {"<|im_end|>\n<|im_start|>user\n", true},
            {user, false},
            {"<|im_end|>\n<|im_start|>assistant\n", true}};
  return {{"<|im_end|>\n<|im_start|>user\n", true}, {user, false}, {"<|im_end|>\n<|im_start|>assistant\n", true}};
}

namespace {
std::string default_system(ChatFormat) { return std::string(kQwenDefaultSystem); }
}  // namespace

Chat::Chat(Session& s, std::string system_prompt) {
  const ChatFormat fmt = chat_format_for(s.architecture());
  if (system_prompt.empty()) system_prompt = default_system(fmt);
  impl_ = std::make_unique<Impl>(s, std::move(system_prompt));
}
Chat::Chat(Chat&&) noexcept = default;
Chat& Chat::operator=(Chat&&) noexcept = default;
Chat::~Chat() = default;

StopReason Chat::send(std::string_view user, const GenParams& gp,
                      const std::function<bool(std::string_view)>& on_token) {
  if (!impl_) throw Error("use of moved-from Chat");
  Impl& c = *impl_;
  if (c.started && c.session.n_past() != c.expected_past) throw Error("chat context lost");
  if (c.full) return StopReason::ContextFull;
  // Records how this send left the cache, on return or throw. The prompt counts
  // as sent once prefilled (max_tokens <= 0 and early ContextFull skip that). A
  // started conversation always has n_past() > 0, so 0 means the Session reset
  // its cache (a forward threw): the history is gone and the next send throws.
  auto sync = [&c] {
    if (c.session.n_prompt_tokens() > 0) c.started = true;
    if (c.started) c.expected_past = c.session.n_past() > 0 ? c.session.n_past() : -1;
  };
  GenParams p = gp;
  p.append = c.started;
  StopReason r;
  try {
    r = c.session.generate(chatml_turn(c.system, user, !c.started), p, on_token);
  } catch (...) {  // e.g. on_token threw: the tokens cached so far stay valid
    sync();
    throw;
  }
  sync();
  if (r == StopReason::ContextFull) c.full = true;
  return r;
}

}  // namespace nibble
