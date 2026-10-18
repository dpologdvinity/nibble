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
  if (c.full) return StopReason::ContextFull;
  const std::vector<PromptPart> parts = chatml_turn(c.system, user, !c.started);
  c.last_parts.clear();
  for (const PromptPart& p : parts) c.last_parts.emplace_back(std::string(p.text), p.special);
  GenParams p = gp;
  p.append = c.started;
  const StopReason r = c.session.generate(parts, p, on_token);
  // max_tokens <= 0 returns before prefilling; then the turn did not happen.
  if (c.session.n_prompt_tokens() > 0) c.started = true;
  if (r == StopReason::ContextFull) c.full = true;
  return r;
}

}  // namespace nibble
