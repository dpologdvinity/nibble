// Chat-template selection, per-turn prompt construction and Chat::send (the conversation state machine).
#include "chat.h"

#include "util/escape.h"

namespace nibble {

ChatFormat chat_format_for(std::string_view arch, std::string_view tmpl) {
  auto has = [&](std::string_view m) { return tmpl.find(m) != std::string_view::npos; };
  if (has("<|im_start|>")) return ChatFormat::ChatML;
  if (has("<|start_header_id|>")) return ChatFormat::Llama3;
  if (has("<|assistant|>")) return ChatFormat::Zephyr;
  if (arch == "qwen2") return ChatFormat::ChatML;
  throw Error("no chat template for architecture " + escape_control(arch));
}

namespace {
// Jinja |trim on ASCII whitespace; the Llama 3.2 template trims system and user content.
std::string_view trim(std::string_view s) {
  const size_t b = s.find_first_not_of(" \t\n\r\f\v");
  if (b == std::string_view::npos) return {};
  return s.substr(b, s.find_last_not_of(" \t\n\r\f\v") - b + 1);
}
}  // namespace

// Layouts reproduce the models' official Jinja templates. Llama 3 hard-codes the template's default date
// lines ("Today Date: 26 Jul 2024") because the official template does when no date is passed.
std::vector<PromptPart> chat_turn(ChatFormat f, std::string_view system, std::string_view user, bool first) {
  switch (f) {
    case ChatFormat::ChatML:
      if (first)
        return {{"<|im_start|>system\n", true},
                {system, false},
                {"<|im_end|>\n<|im_start|>user\n", true},
                {user, false},
                {"<|im_end|>\n<|im_start|>assistant\n", true}};
      return {{"<|im_end|>\n<|im_start|>user\n", true}, {user, false}, {"<|im_end|>\n<|im_start|>assistant\n", true}};
    case ChatFormat::Llama3:
      if (first)
        return {{"<|start_header_id|>system<|end_header_id|>\n\n"
                 "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                 true},
                {trim(system), false},
                {"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", true},
                {trim(user), false},
                {"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", true}};
      return {{"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", true},
              {trim(user), false},
              {"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", true}};
    case ChatFormat::Zephyr:
      if (first && !system.empty())
        return {{"<|system|>\n", true},
                {system, false},
                {"</s>\n<|user|>\n", true},
                {user, false},
                {"</s>\n<|assistant|>\n", true}};
      if (first) return {{"<|user|>\n", true}, {user, false}, {"</s>\n<|assistant|>\n", true}};
      return {{"</s>\n<|user|>\n", true}, {user, false}, {"</s>\n<|assistant|>\n", true}};
  }
  throw Error("bad chat format");
}

namespace {
// Only Qwen has a default system prompt; the Llama 3 and Zephyr ones are empty.
std::string default_system(ChatFormat f) {
  return f == ChatFormat::ChatML ? std::string(kQwenDefaultSystem) : std::string();
}
}  // namespace

// Chooses the format (throws if the model has no known chat template) and substitutes the model's default
// system prompt when none is given.
Chat::Chat(Session& s, std::string system_prompt) {
  const ChatFormat fmt = chat_format_for(s.architecture(), s.chat_template());
  if (system_prompt.empty()) system_prompt = default_system(fmt);
  impl_ = std::make_unique<Impl>(s, fmt, std::move(system_prompt));
}
Chat::Chat(Chat&&) noexcept = default;
Chat& Chat::operator=(Chat&&) noexcept = default;
Chat::~Chat() = default;

// Sends one turn. The first turn starts the Session fresh (append = false); later turns append, so only the
// new tokens are prefilled. `sync` records the Session state after every call, thrown or not, so the
// next call can tell whether the cache still matches the conversation.
StopReason Chat::send(std::string_view user, const GenParams& gp,
                      const std::function<bool(std::string_view)>& on_token) {
  if (!impl_) throw Error("use of moved-from Chat");
  Impl& c = *impl_;
  if (c.started && (c.session.n_past() != c.expected_past || c.session.generations() != c.expected_gens))
    throw Error("chat context lost");
  if (c.full) return StopReason::ContextFull;
  // Records how this send left the cache, on return or throw. The prompt counts
  // as sent once prefilled (max_tokens <= 0 and early ContextFull skip that). A
  // started conversation always has n_past() > 0, so 0 means the Session reset
  // its cache (a forward threw): the history is gone and the next send throws.
  auto sync = [&c] {
    if (c.session.n_prompt_tokens() > 0) c.started = true;
    if (c.started) c.expected_past = c.session.n_past() > 0 ? c.session.n_past() : -1;
    c.expected_gens = c.session.generations();
  };
  GenParams p = gp;
  p.append = c.started;
  StopReason r;
  try {
    r = c.session.generate(chat_turn(c.format, c.system, user, !c.started), p, on_token);
  } catch (...) {  // e.g. on_token threw: the tokens cached so far stay valid
    sync();
    throw;
  }
  sync();
  if (r == StopReason::ContextFull) c.full = true;
  return r;
}

}  // namespace nibble
